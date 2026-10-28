# =============================================================================
# wellcast: Diagnose page: IPR, Chan, Hall and anomaly diagnostics
# -----------------------------------------------------------------------------
# Module:      app/pages/diagnose.py
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

"""Inflow performance, water diagnostics, injectivity and anomalies."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wellcast.diagnostics import (
    chan_plot_data,
    classify_water_mechanism,
    detect_anomalies,
    hall_plot_data,
    hall_slope_change,
)
from wellcast.exceptions import MissingDependencyError, WellcastError
from wellcast.ipr import composite_productivity_index, ipr_curve, vogel_qmax
from wellcast.plotting import anomaly_overlay, chan_plot, hall_plot, ipr_plot

import common

st.title("Diagnose")
df = common.require_data()
ipr_tab, chan_tab, hall_tab, anomaly_tab = st.tabs(
    ["IPR", "Chan plot", "Hall plot", "Anomalies"]
)

# -----------------------------------------------------------------------------
# Inflow performance
# -----------------------------------------------------------------------------

with ipr_tab:
    st.write("Inflow performance from one stabilized well test.")
    a, b, c, d = st.columns(4)
    pr = float(
        a.number_input("Reservoir pressure (psi)", min_value=100.0, value=3500.0)
    )
    pb = float(b.number_input("Bubble point (psi)", min_value=0.0, value=2200.0))
    q_test = float(c.number_input("Test rate (STB/d)", min_value=1.0, value=1200.0))
    pwf_test = float(
        d.number_input("Test flowing pressure (psi)", min_value=0.0, value=2600.0)
    )
    try:
        if pwf_test >= pr:
            raise WellcastError(
                "the test flowing pressure must be below the reservoir pressure"
            )
        pb_eff = min(pb, pr)
        j = composite_productivity_index(q_test, pwf_test, pr, max(pb_eff, 1e-6))
        qmax = vogel_qmax(q_test, pwf_test, pr)
        curves = {
            "linear (constant J)": ipr_curve(
                "linear", pr, productivity_index=q_test / (pr - pwf_test)
            ),
            "Vogel": ipr_curve("vogel", pr, qmax=qmax),
        }
        if pb_eff > 0:
            curves["composite"] = ipr_curve(
                "composite", pr, bubble_point=pb_eff, productivity_index=j
            )
    except WellcastError as exc:
        st.error(str(exc))
    else:
        m1, m2 = st.columns(2)
        m1.metric("Composite J (STB/d/psi)", f"{j:.3f}")
        m2.metric("Vogel AOF (STB/d)", f"{qmax:,.0f}")
        common.show(ipr_plot(curves, operating_point=(q_test, pwf_test)))

# -----------------------------------------------------------------------------
# Chan plot
# -----------------------------------------------------------------------------

with chan_tab:
    producers = common.wells_with(df, "water_rate")
    if not producers:
        st.info("No well produces water.")
    else:
        chan = chan_plot_data(df[df["well"].isin(producers)])
        st.dataframe(classify_water_mechanism(chan), width="stretch")
        st.caption(
            "Screening per Chan (1995): a rising WOR derivative suggests channeling, a "
            "falling one coning. Confirm with completion and reservoir data."
        )
        well = common.well_selector(sorted(chan["well"].unique()), key="chan_well")
        common.show(chan_plot(chan, well))

# -----------------------------------------------------------------------------
# Hall plot
# -----------------------------------------------------------------------------

with hall_tab:
    injectors = common.wells_with(df, "water_inj_rate")
    if not injectors:
        st.info("No injection well in the data.")
    else:
        a, b = st.columns(2)
        injector = str(a.selectbox("Injector", injectors))
        p_res = float(
            b.number_input(
                "Average reservoir pressure (psi)", min_value=0.0, value=3000.0
            )
        )
        data = df[df["well"] == injector]
        pressure = "bhp" if data["bhp"].notna().any() else "whp"
        hall = hall_plot_data(data, pressure_column=pressure, reservoir_pressure=p_res)
        try:
            change = hall_slope_change(hall)
        except WellcastError:
            change = None
        if change is not None:
            st.metric(
                f"Slope change on {change['break_date']:%Y-%m-%d}",
                f"x{change['ratio']:.2f}",
                help=(
                    "Above 1: injectivity loss (plugging, skin). "
                    "Below 1: improvement or fracturing."
                ),
            )
        common.show(hall_plot(hall, change))

# -----------------------------------------------------------------------------
# Anomalies
# -----------------------------------------------------------------------------

with anomaly_tab:
    a, b = st.columns(2)
    method = str(a.selectbox("Detectors", ["both", "rules", "isolation_forest"]))
    contamination = float(b.slider("Forest contamination", 0.001, 0.05, 0.01, 0.001))
    try:
        anomalies = detect_anomalies(df, method=method, contamination=contamination)  # type: ignore[arg-type]
    except MissingDependencyError as exc:
        st.error(str(exc))
        st.stop()
    common.results()["anomalies"] = anomalies
    st.write(f"{len(anomalies)} flagged periods")
    st.dataframe(anomalies, hide_index=True, width="stretch")

    labels = st.session_state.get(common.LABELS)
    if labels is not None and len(anomalies):
        # Demo data carries ground truth: report how much of it was found.
        covered = pd.Series(False, index=range(len(df)))
        wells = df["well"].to_numpy()
        for _, period in anomalies.iterrows():
            covered |= (
                (wells == period["well"])
                & (df.index >= period["start"])
                & (df.index <= period["end"])
            )
        truth = labels.reset_index(drop=True)
        recall = {
            kind: covered[(truth["anomaly_type"] == kind).to_numpy()].mean()
            for kind in sorted(set(truth["anomaly_type"]) - {""})
        }
        st.caption(
            "Share of injected anomalies found (demo ground truth): "
            + ", ".join(f"{k} {v:.0%}" for k, v in recall.items())
        )
    signals = [
        c
        for c in ("bhp", "whp", "oil_rate", "gas_rate", "water_rate")
        if c in df.columns
    ]
    c, d = st.columns(2)
    well = str(c.selectbox("Well", sorted(df["well"].unique()), key="anomaly_well"))
    column = str(d.selectbox("Signal", signals))
    common.show(anomaly_overlay(df, anomalies, column, well))

# =============================================================================
# End of module: app/pages/diagnose.py
# =============================================================================
