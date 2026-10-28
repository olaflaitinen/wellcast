# =============================================================================
# wellcast: Decline page: per-well Arps fits and EUR
# -----------------------------------------------------------------------------
# Module:      app/pages/decline.py
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

"""Fit Arps declines for every well and inspect one well in detail."""

from __future__ import annotations

import streamlit as st

from wellcast.dca import DeclineModel, fit_many
from wellcast.exceptions import WellcastError

import common

st.title("Decline")
df = common.require_data()

rate_columns = [c for c in ("oil_rate", "gas_rate", "water_rate") if c in df.columns]
if not rate_columns:
    st.warning("The data has no rate column.")
    st.stop()
a, b, c, d = st.columns(4)
rate_column = str(a.selectbox("Rate", rate_columns))
model_name = str(
    b.selectbox("Arps model", ["auto", "exponential", "hyperbolic", "harmonic"])
)
econ_limit = float(
    c.number_input("Economic limit (per day)", min_value=0.1, value=20.0)
)
n_boot = int(
    d.number_input("Bootstrap samples", min_value=0, max_value=1000, value=200, step=50)
)

producers = common.wells_with(df, rate_column)
table = fit_many(
    df[df["well"].isin(producers)],
    rate_column=rate_column,
    model=model_name,
    econ_limit=econ_limit,  # type: ignore[arg-type]
)
st.subheader("EUR by well")
st.caption(
    "EUR from the start of decline to the economic limit, in the rate's volume unit."
)
shown = (
    table[["model", "qi", "di", "b", "eur", "remaining", "status"]].copy()
    if "eur" in table
    else table
)
st.dataframe(
    shown,
    width="stretch",
    column_config={
        "qi": st.column_config.NumberColumn("qi", format="%.0f"),
        "di": st.column_config.NumberColumn("di (1/yr)", format="%.3f"),
        "b": st.column_config.NumberColumn("b", format="%.2f"),
        "eur": st.column_config.NumberColumn("EUR", format="%.0f"),
        "remaining": st.column_config.NumberColumn("remaining", format="%.0f"),
    },
)
st.download_button(
    "Download EUR table (CSV)", table.to_csv(), "wellcast_eur.csv", "text/csv"
)
common.results()["decline"] = {
    "table": table,
    "rate": rate_column,
    "econ_limit": econ_limit,
}

st.subheader("Well detail")
well = common.well_selector(producers, key="decline_well")
rates = df.loc[df["well"] == well, rate_column]
try:
    model = DeclineModel(model_name).fit(rates)  # type: ignore[arg-type]
    if n_boot:
        model.bootstrap(n_boot)
except WellcastError as exc:
    st.error(str(exc))
    st.stop()
if n_boot:
    quantiles = model.eur_quantiles(econ_limit)
    a, b, c = st.columns(3)
    a.metric("P90 EUR (low)", f"{quantiles['p90']:,.0f}")
    b.metric("P50 EUR (best)", f"{quantiles['p50']:,.0f}")
    c.metric("P10 EUR (high)", f"{quantiles['p10']:,.0f}")
common.show(model.plot(rates, econ_limit=econ_limit, title=f"{well}: {model}"))

# =============================================================================
# End of module: app/pages/decline.py
# =============================================================================
