# =============================================================================
# wellcast: Data page: upload, column mapping and quality control
# -----------------------------------------------------------------------------
# Module:      app/pages/data.py
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

"""Upload production data, map its columns to the canonical schema and run QC."""

from __future__ import annotations

import io as _io

import pandas as pd
import streamlit as st

from wellcast.clean import detect_shut_ins, qc_report
from wellcast.exceptions import WellcastError
from wellcast.io import CANONICAL_SCHEMA, DEFAULT_ALIASES, from_dataframe
from wellcast.units import get_unit, list_units

import common

st.title("Data")
st.write(
    "Upload daily or monthly production data as CSV or Excel, map its columns to the "
    "wellcast schema and review data quality. Or start with the synthetic demo field."
)

if st.button("Load demo data", type="primary", key="load_demo_page"):
    common.load_demo()
    st.rerun()

# -----------------------------------------------------------------------------
# Upload and column mapping
# -----------------------------------------------------------------------------

upload = st.file_uploader("Production file", type=["csv", "xlsx", "xls"])
if upload is not None:
    try:
        if upload.name.lower().endswith(".csv"):
            raw = pd.read_csv(upload)
        else:
            raw = pd.read_excel(_io.BytesIO(upload.getvalue()))
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
    st.caption(f"{len(raw):,} rows, {raw.shape[1]} columns")
    st.dataframe(raw.head(20), width="stretch")

    normalized = {
        c: "".join(ch for ch in str(c).upper() if ch.isalnum()) for c in raw.columns
    }
    detected = {
        DEFAULT_ALIASES[n]: c for c, n in normalized.items() if n in DEFAULT_ALIASES
    }
    none = "(not present)"
    options = [none, *map(str, raw.columns)]

    st.subheader("Column mapping")
    fields = [("date", "-", "Production date")] + [
        (spec.name, spec.unit, spec.description) for spec in CANONICAL_SCHEMA
    ]
    canonical_units = {spec.name: spec.unit for spec in CANONICAL_SCHEMA}
    mapping: dict[str, str] = {}
    units: dict[str, str] = {}
    columns = st.columns(3)
    for i, (name, unit, description) in enumerate(fields):
        with columns[i % 3]:
            default = detected.get(name)
            choice = st.selectbox(
                f"{name}",
                options,
                index=options.index(str(default)) if default is not None else 0,
                help=description,
                key=f"map_{name}",
            )
            if choice != none:
                mapping[choice] = name
                if unit not in ("-", "in/64", "h"):
                    choices = list_units(get_unit(unit).quantity)
                    units[name] = str(
                        st.selectbox(
                            f"{name} unit",
                            choices,
                            index=choices.index(unit),
                            key=f"unit_{name}",
                            label_visibility="collapsed",
                        )
                    )
    default_well = st.text_input(
        "Well name (only for single-well files without a well column)"
    )
    dayfirst = st.checkbox("Dates are day first (31/01/2024)")
    if st.button("Apply mapping", type="primary"):
        try:
            df = from_dataframe(
                raw,
                mapping,
                units={k: v for k, v in units.items() if v != canonical_units[k]},
                well=default_well or None,
                dayfirst=dayfirst,
            )
        except WellcastError as exc:
            st.error(str(exc))
        else:
            common.set_data(df, upload.name)
            st.success(f"Loaded {len(df):,} rows for {df['well'].nunique()} wells.")

# -----------------------------------------------------------------------------
# Quality control
# -----------------------------------------------------------------------------

df = common.get_data()
if df is not None:
    st.header("Quality control")
    shut_in = (
        detect_shut_ins(df)
        if {"oil_rate", "on_stream_hours"} & set(df.columns)
        else None
    )
    report = qc_report(df)
    a, b, c, d = st.columns(4)
    a.metric("Wells", df["well"].nunique())
    b.metric("Rows", f"{len(df):,}")
    c.metric("Shut-in rows", "n/a" if shut_in is None else f"{int(shut_in.sum()):,}")
    d.metric("Impossible values", f"{int(report['impossible'].fillna(0).sum()):,}")
    st.write(
        "Counts per well and column. Flatlines are runs of 5 or more identical "
        "non-zero readings; impossible values break physical limits (negative "
        "pressure, water cut outside 0 to 1, more than 24 producing hours a day)."
    )
    if not st.toggle("Show columns without any data", value=False):
        # Injectors have no oil rate, producers no injection rate: those
        # columns are absent by design, not a data-quality problem.
        report = report[report["missing"] < report["n"]]
    st.dataframe(
        report,
        width="stretch",
        hide_index=True,
        column_config={
            "missing_pct": st.column_config.NumberColumn("missing %", format="%.1f")
        },
    )
    st.download_button(
        "Download QC report (CSV)",
        report.to_csv(index=False),
        "wellcast_qc.csv",
        "text/csv",
    )

# =============================================================================
# End of module: app/pages/data.py
# =============================================================================
