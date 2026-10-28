# =============================================================================
# wellcast: tests for the command-line interface
# -----------------------------------------------------------------------------
# Module:      tests/test_cli.py
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

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from wellcast import __version__, cli
from wellcast.exceptions import WellcastError


@pytest.fixture(scope="module")
def demo_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "demo.csv"
    assert (
        cli.main(["demo", "--wells", "2", "--years", "2", "--output", str(path)]) == 0
    )
    return path


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"wellcast {__version__}"


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert "<command>" in capsys.readouterr().out


def test_demo_writes_data_and_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data, labels = tmp_path / "d.csv", tmp_path / "l.csv"
    status = cli.main(
        [
            "demo",
            "--wells",
            "1",
            "--injectors",
            "0",
            "--years",
            "1",
            "--seed",
            "3",
            "--output",
            str(data),
            "--labels",
            str(labels),
        ]
    )
    assert status == 0
    out = capsys.readouterr().out
    assert "for 1 wells" in out
    assert "labels" in out
    frame = pd.read_csv(data)
    assert set(frame["well"]) == {"P-01"}
    assert "anomaly_type" in pd.read_csv(labels).columns


def test_qc_text_and_csv(demo_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["qc", str(demo_csv)]) == 0
    text = capsys.readouterr().out
    assert "missing_pct" in text
    assert "P-01" in text
    assert cli.main(["qc", str(demo_csv), "--format", "csv", "--max-hours", "48"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(",well,column")


def test_dca_single_well_with_bootstrap(
    demo_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "dca",
        str(demo_csv),
        "--well",
        "P-01",
        "--econ-limit",
        "20",
        "--bootstrap",
        "20",
    ]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    for key in ("model", "qi", "eur", "eur_p90", "eur_p10"):
        assert key in out


def test_dca_csv_and_single_well_file(
    tmp_path: Path, demo_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = pd.read_csv(demo_csv)
    single = tmp_path / "single.csv"
    frame[frame["well"] == "P-02"].drop(columns="well").to_csv(single, index=False)
    status = cli.main(
        [
            "dca",
            str(single),
            "--default-well",
            "W",
            "--model",
            "exponential",
            "--format",
            "csv",
        ]
    )
    assert status == 0
    out = capsys.readouterr().out
    assert "exponential" in out
    assert out.splitlines()[0] == ",W"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "choose one with --well"),
        (["--well", "X-99"], "not found"),
        (["--well", "I-01"], "at least 4"),
        (["--well", "P-01", "--rate-column", "nope"], "'nope' not found"),
    ],
)
def test_dca_errors(
    demo_csv: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    assert cli.main(["dca", str(demo_csv), *extra]) == 1
    assert message in capsys.readouterr().err


def test_missing_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["qc", str(tmp_path / "missing.csv")]) == 1
    assert "wellcast: error:" in capsys.readouterr().err


def test_invalid_usage() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["dca"])
    assert exc.value.code == 2


def test_app_launches_streamlit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "streamlit_app.py"
    script.write_text("")
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setitem(sys.modules, "streamlit", object())
    monkeypatch.setattr(cli, "find_app", lambda: script)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["app", "--port", "8600", "--headless"]) == 0
    command = calls[0]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[4] == str(script)
    assert command[-4:] == ["--server.port", "8600", "--server.headless", "true"]


def test_app_without_streamlit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", None)
    assert cli.main(["app"]) == 1
    assert "wellcast[app]" in capsys.readouterr().err


def test_find_app_reports_missing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_module = tmp_path / "site" / "wellcast" / "cli.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(fake_module))
    with pytest.raises(WellcastError, match="not installed"):
        cli.find_app()


def test_python_dash_m_entry_point(demo_csv: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "wellcast", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"wellcast {__version__}"


# =============================================================================
# End of module: tests/test_cli.py
# =============================================================================
