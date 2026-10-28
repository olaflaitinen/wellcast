# =============================================================================
# wellcast: Report page: standalone HTML report export
# -----------------------------------------------------------------------------
# Module:      app/pages/report.py
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

"""Export a standalone HTML report of the analysis."""

from __future__ import annotations

import streamlit as st

from wellcast.clean import qc_report
from wellcast.dca import DeclineModel, fit_many
from wellcast.diagnostics import detect_anomalies
from wellcast.exceptions import MissingDependencyError, WellcastError
from wellcast.plotting import production_history

import common

st.title("Report")
st.write("Collect the main results into one HTML file to share or archive.")
df = common.require_data()

title = st.text_input("Title", "Well performance report")
a, b = st.columns(2)
econ_limit = float(
    a.number_input("Economic limit for EUR (per day)", min_value=0.1, value=20.0)
)
max_wells = int(
    b.number_input("Decline charts (wells)", min_value=0, max_value=20, value=5)
)
include = st.multiselect(
    "Sections",
    ["Data quality", "Decline and EUR", "Anomalies", "Production charts"],
    default=["Data quality", "Decline and EUR", "Anomalies", "Production charts"],
)
self_contained = st.checkbox(
    "Embed the charting library (works offline, adds about 4 MB)", value=False
)

if st.button("Build report", type="primary"):
    tables = {}
    figures = {}
    with st.spinner("Building report..."):
        if "Data quality" in include:
            tables["Data quality"] = qc_report(df).round(2)
        producers = common.wells_with(df, "oil_rate")
        if "Decline and EUR" in include and producers:
            eur = fit_many(df[df["well"].isin(producers)], econ_limit=econ_limit)
            columns = [
                c
                for c in ("model", "qi", "di", "b", "eur", "remaining", "status")
                if c in eur
            ]
            tables["Decline and EUR (oil)"] = eur[columns]
            for well in producers[:max_wells]:
                rates = df.loc[df["well"] == well, "oil_rate"]
                try:
                    model = DeclineModel().fit(rates)
                except WellcastError:
                    continue
                figures[f"Decline {well}"] = model.plot(rates, econ_limit=econ_limit)
        if "Anomalies" in include:
            try:
                tables["Anomalies"] = detect_anomalies(df)
            except MissingDependencyError:
                tables["Anomalies (rules only)"] = detect_anomalies(df, method="rules")
        if "Production charts" in include:
            for well in sorted(df["well"].unique())[:max_wells]:
                figures[f"Production history {well}"] = production_history(df, well)
        html = common.build_report(
            title,
            str(st.session_state.get(common.SOURCE, "uploaded data")),
            tables,
            figures,
            self_contained=self_contained,
        )
    st.session_state["report_html"] = html

html = st.session_state.get("report_html")
if html:
    st.success(f"Report ready ({len(html) / 1e6:.1f} MB).")
    st.download_button(
        "Download HTML report", html, "wellcast_report.html", "text/html"
    )

# =============================================================================
# End of module: app/pages/report.py
# =============================================================================
