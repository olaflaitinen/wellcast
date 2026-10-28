# =============================================================================
# wellcast: Explore page: production history per well
# -----------------------------------------------------------------------------
# Module:      app/pages/explore.py
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

"""Explore production history and key ratios per well."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wellcast.features import add_physics_features
from wellcast.plotting import production_history

import common

st.title("Explore")
df = common.require_data()

# Producers first, so the page opens on a well with production.
producers = common.wells_with(df, "oil_rate")
wells = producers + sorted(set(df["well"].astype(str)) - set(producers))
well = common.well_selector(wells, key="explore_well")
data = df[df["well"].astype(str) == well]
common.show(production_history(df, well))

features = add_physics_features(data)
latest = features.dropna(
    subset=[c for c in ("liquid_rate",) if c in features.columns]
).tail(30)
cols = st.columns(4)
injected = data["water_inj_rate"].fillna(0).sum() if "water_inj_rate" in data else 0.0
if injected > 0:
    cols[0].metric("Cumulative injection (bbl)", f"{injected:,.0f}")
elif "cum_oil" in features.columns:
    cols[0].metric("Cumulative oil (bbl)", f"{features['cum_oil'].iloc[-1]:,.0f}")
if "gor" in latest.columns and latest["gor"].notna().any():
    cols[1].metric("GOR, last 30 days (scf/STB)", f"{latest['gor'].median():,.0f}")
if "water_cut" in latest.columns and latest["water_cut"].notna().any():
    cols[2].metric("Water cut, last 30 days", f"{latest['water_cut'].median():.1%}")
if "days_on_production" in features.columns:
    cols[3].metric(
        "Days on production", f"{features['days_on_production'].iloc[-1]:,.0f}"
    )

st.subheader("Field summary")
summary = []
for name, group in add_physics_features(df).groupby("well"):
    row: dict[str, object] = {
        "well": name,
        "first": group.index.min().date(),
        "last": group.index.max().date(),
    }
    for column in ("cum_oil", "cum_gas", "cum_water"):
        if column in group.columns:
            row[column] = group[column].iloc[-1]
    summary.append(row)
volume = {"format": "localized"}
st.dataframe(
    pd.DataFrame(summary).round(0),
    hide_index=True,
    width="stretch",
    column_config={
        "cum_oil": st.column_config.NumberColumn("oil (bbl)", **volume),
        "cum_gas": st.column_config.NumberColumn("gas (Mscf)", **volume),
        "cum_water": st.column_config.NumberColumn("water (bbl)", **volume),
    },
)

# =============================================================================
# End of module: app/pages/explore.py
# =============================================================================
