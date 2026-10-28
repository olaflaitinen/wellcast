# =============================================================================
# wellcast: shared state, assets and helpers for the Streamlit app
# -----------------------------------------------------------------------------
# Module:      app/common.py
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

"""Helpers shared by the pages of the wellcast Streamlit app."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import wellcast
from wellcast.datasets import make_synthetic_field

APP_DIR = Path(__file__).resolve().parent

# Session-state keys.
DATA = "production"
LABELS = "labels"
SOURCE = "source"
RESULTS = "results"


# -----------------------------------------------------------------------------
# Assets
# -----------------------------------------------------------------------------


def asset(name: str) -> Path | None:
    """Return a brand asset from the installed app or the repository."""
    for base in (APP_DIR / "assets", APP_DIR.parent / "assets"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


# -----------------------------------------------------------------------------
# Data in session state
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner="Simulating a synthetic field...")
def demo_field(
    n_wells: int = 5, years: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    field = make_synthetic_field(n_wells=n_wells, years=years, seed=seed)
    return field.production, field.labels


def set_data(df: pd.DataFrame, source: str, labels: pd.DataFrame | None = None) -> None:
    st.session_state[DATA] = df
    st.session_state[SOURCE] = source
    st.session_state[RESULTS] = {}
    if labels is None:
        st.session_state.pop(LABELS, None)
    else:
        st.session_state[LABELS] = labels


def load_demo() -> None:
    production, labels = demo_field()
    set_data(production, "synthetic demo field (seed 42)", labels)


def get_data() -> pd.DataFrame | None:
    data: pd.DataFrame | None = st.session_state.get(DATA)
    return data


def results() -> dict[str, Any]:
    store: dict[str, Any] = st.session_state.setdefault(RESULTS, {})
    return store


def require_data() -> pd.DataFrame:
    """Return the loaded data, or offer the demo data and stop the page."""
    df = get_data()
    if df is None:
        st.info(
            "No data loaded yet. Upload a file on the Data page or start with the demo."
        )
        if st.button("Load demo data", type="primary", key="load_demo_inline"):
            load_demo()
            st.rerun()
        st.stop()
    return df


# -----------------------------------------------------------------------------
# Widgets and display
# -----------------------------------------------------------------------------


def wells_with(df: pd.DataFrame, column: str) -> list[str]:
    """Wells that have at least one positive value in ``column``."""
    if column not in df.columns:
        return []
    has = df[column].fillna(0) > 0
    return sorted(df.loc[has, "well"].astype(str).unique())


def well_selector(options: list[str], key: str, label: str = "Well") -> str:
    if not options:
        st.warning("No well in the data has the required measurements.")
        st.stop()
    return str(st.selectbox(label, options, key=key))


def show(fig: Any) -> None:
    # theme=None keeps the wellcast plotly template instead of Streamlit's.
    st.plotly_chart(fig, theme=None, width="stretch")


def sidebar() -> None:
    with st.sidebar:
        df = get_data()
        if df is None:
            st.caption("No data loaded")
        else:
            st.caption(f"Data: {st.session_state.get(SOURCE, 'uploaded file')}")
            st.caption(
                f"{df['well'].nunique()} wells, {len(df):,} rows, "
                f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
            )
        if st.button("Load demo data", key="load_demo_sidebar", width="stretch"):
            load_demo()
            st.rerun()
        st.caption(f"wellcast {wellcast.__version__}, MPL-2.0")


# -----------------------------------------------------------------------------
# HTML report
# -----------------------------------------------------------------------------

_REPORT_CSS = """
body { font-family: Inter, Helvetica, Arial, sans-serif; color: #0b0b0b;
       background: #fcfcfb; margin: 0 auto; max-width: 1100px; padding: 24px; }
h1 { font-size: 26px; margin-bottom: 4px; }
h2 { font-size: 19px; margin-top: 36px; padding-bottom: 6px;
     border-bottom: 1px solid #e6e5e0; }
p.meta { color: #52514e; margin-top: 0; }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0 16px 0; }
th, td { padding: 4px 10px; border-bottom: 1px solid #e6e5e0; text-align: right; }
th { color: #52514e; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
footer { color: #52514e; font-size: 12px; margin-top: 40px; }
"""


def build_report(
    title: str,
    source: str,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, Any],
    *,
    self_contained: bool = False,
) -> str:
    """Assemble a standalone HTML report from tables and plotly figures."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title><style>{_REPORT_CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="meta">Data: {html.escape(source)}. Generated {stamp} with '
        f"wellcast {wellcast.__version__}.</p>",
    ]
    for heading, table in tables.items():
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(
            table.to_html(border=0, float_format=lambda v: f"{v:,.4g}", na_rep="")
        )
    include = "inline" if self_contained else "cdn"
    for heading, fig in figures.items():
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include))
        include = False  # plotly.js is needed once per document
    parts.append(
        "<footer>Created with wellcast (https://github.com/olaflaitinen/wellcast), "
        "Mozilla Public License 2.0.</footer></body></html>"
    )
    return "\n".join(str(p) for p in parts)


# =============================================================================
# End of module: app/common.py
# =============================================================================
