# =============================================================================
# wellcast: command-line interface
# -----------------------------------------------------------------------------
# Module:      wellcast.cli
# Repository:  https://github.com/olaflaitinen/wellcast
# Author:      Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Copyright:   Copyright (c) 2026 Olaf Yunus Laitinen Imanov
# License:     Mozilla Public License 2.0
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# =============================================================================

"""Command-line interface, installed as the ``wellcast`` command.

Commands::

    wellcast --version
    wellcast qc <file>                 # data-quality report
    wellcast dca <file> --well <name>  # decline fit and EUR
    wellcast demo                      # write a synthetic data set to CSV
    wellcast app                       # launch the Streamlit app

Exit codes: 0 success, 1 a wellcast error (bad data, failed fit, missing
extra), 2 invalid command-line usage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from wellcast.__about__ import __version__
from wellcast._validation import require_extra
from wellcast.clean import qc_report
from wellcast.datasets import make_synthetic_field
from wellcast.dca import DeclineModel
from wellcast.exceptions import DataValidationError, WellcastError
from wellcast.io import read_production

__all__ = ["build_parser", "find_app", "main"]


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def _format_value(value: object) -> object:
    # Six significant figures are plenty for engineering output in a terminal.
    return f"{value:.6g}" if isinstance(value, float) else value


def _print_frame(frame: pd.DataFrame | pd.Series, fmt: str) -> None:
    if fmt == "csv":
        frame.to_csv(sys.stdout)
        return
    if isinstance(frame, pd.Series):
        frame = frame.map(_format_value)
    with pd.option_context("display.width", 120, "display.max_rows", 500):
        print(frame.to_string())


def _cmd_qc(args: argparse.Namespace) -> int:
    df = read_production(args.file, well=args.default_well)
    report = qc_report(df, max_hours=args.max_hours)
    _print_frame(report.round(2), args.format)
    return 0


def _cmd_dca(args: argparse.Namespace) -> int:
    df = read_production(args.file, well=args.default_well)
    if args.rate_column not in df.columns:
        raise DataValidationError(
            f"column {args.rate_column!r} not found in {args.file}"
        )
    wells = sorted(df["well"].unique())
    if args.well is None:
        if len(wells) != 1:
            raise DataValidationError(
                f"the file has {len(wells)} wells; choose one with --well "
                f"({', '.join(wells[:10])}{', ...' if len(wells) > 10 else ''})"
            )
        well = wells[0]
    else:
        well = args.well
        if well not in wells:
            raise DataValidationError(f"well {well!r} not found in {args.file}")
    rates = df.loc[df["well"] == well, args.rate_column]
    model = DeclineModel(args.model).fit(rates)
    summary = model.summary(econ_limit=args.econ_limit)
    if args.bootstrap:
        model.bootstrap(args.bootstrap, seed=args.seed)
        quantiles = model.eur_quantiles(args.econ_limit)
        for name, value in quantiles.items():
            summary[f"eur_{name}"] = value
    summary.name = well
    _print_frame(summary, args.format)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    field = make_synthetic_field(
        n_wells=args.wells,
        years=args.years,
        seed=args.seed,
        n_injectors=args.injectors,
    )
    output = Path(args.output)
    field.production.to_csv(output)
    print(
        f"wrote {len(field.production)} rows for {field.production['well'].nunique()} "
        f"wells to {output}"
    )
    if args.labels:
        field.labels.to_csv(args.labels)
        print(f"wrote ground-truth labels to {args.labels}")
    return 0


def find_app() -> Path:
    """Locate the Streamlit entry script.

    Returns
    -------
    pathlib.Path
        ``streamlit_app.py`` shipped inside the installed package, or the
        repository copy when running from a source checkout.

    Raises
    ------
    WellcastError
        If neither location exists.
    """
    package_dir = Path(__file__).resolve().parent
    candidates = (
        package_dir / "_app" / "streamlit_app.py",
        package_dir.parent.parent / "app" / "streamlit_app.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise WellcastError("the Streamlit app files are not installed")


def _cmd_app(args: argparse.Namespace) -> int:
    require_extra("streamlit", "app")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(find_app()),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        command += ["--server.headless", "true"]
    return subprocess.run(command, check=False).returncode


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``wellcast`` command.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``qc``, ``dca``, ``demo`` and ``app`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="wellcast",
        description="Oil and gas well performance analysis.",
    )
    parser.add_argument(
        "--version", action="version", version=f"wellcast {__version__}"
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    def add_input(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("file", help="CSV or Excel production file")
        sub.add_argument(
            "--default-well",
            metavar="NAME",
            help="well name for files without a well column",
        )
        sub.add_argument(
            "--format", choices=("text", "csv"), default="text", help="output format"
        )

    qc = commands.add_parser("qc", help="print a data-quality report")
    add_input(qc)
    qc.add_argument(
        "--max-hours",
        type=float,
        default=24.0,
        help="upper bound of producing hours per row (default 24, daily data)",
    )
    qc.set_defaults(handler=_cmd_qc)

    dca = commands.add_parser("dca", help="fit a decline curve and print the EUR")
    add_input(dca)
    dca.add_argument("--well", help="well to analyze (required for multi-well files)")
    dca.add_argument(
        "--rate-column", default="oil_rate", help="rate column (default oil_rate)"
    )
    dca.add_argument(
        "--model",
        choices=("auto", "exponential", "hyperbolic", "harmonic"),
        default="auto",
        help="Arps model (default auto, chosen by AIC)",
    )
    dca.add_argument(
        "--econ-limit",
        type=float,
        default=10.0,
        help="economic limit rate for EUR (default 10)",
    )
    dca.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="add P90/P50/P10 EUR from N bootstrap samples",
    )
    dca.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    dca.set_defaults(handler=_cmd_dca)

    demo = commands.add_parser("demo", help="write a synthetic data set to CSV")
    demo.add_argument("--output", default="wellcast_demo.csv", help="output CSV path")
    demo.add_argument("--labels", help="also write ground-truth labels to this CSV")
    demo.add_argument("--wells", type=int, default=5, help="number of producers")
    demo.add_argument("--injectors", type=int, default=1, help="number of injectors")
    demo.add_argument("--years", type=int, default=5, help="years of daily data")
    demo.add_argument("--seed", type=int, default=42, help="random seed")
    demo.set_defaults(handler=_cmd_demo)

    app = commands.add_parser("app", help="launch the interactive Streamlit app")
    app.add_argument(
        "--port", type=int, default=8501, help="server port (default 8501)"
    )
    app.add_argument("--headless", action="store_true", help="do not open a browser")
    app.set_defaults(handler=_cmd_app)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``wellcast`` command.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name. Default: ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status.

    Examples
    --------
    >>> main(["--version"])
    Traceback (most recent call last):
    ...
    SystemExit: 0
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        status: int = args.handler(args)
    except (WellcastError, OSError) as exc:
        print(f"wellcast: error: {exc}", file=sys.stderr)
        return 1
    return status


# =============================================================================
# End of module: wellcast.cli
# =============================================================================
