# =============================================================================
# wellcast: Predict page: virtual flow metering with validation and SHAP
# -----------------------------------------------------------------------------
# Module:      app/pages/predict.py
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

"""Train a virtual flow meter, validate it and explain its predictions."""

from __future__ import annotations

import numpy as np
import streamlit as st

from wellcast.exceptions import MissingDependencyError, WellcastError
from wellcast.features import build_features
from wellcast.ml import Predictor, metrics_by_well
from wellcast.plotting import predicted_vs_actual, prediction_interval_plot

import common

st.title("Predict")
st.write(
    "Train a model that estimates a rate or pressure from the other measurements "
    "(virtual flow metering), for example bottomhole pressure after a gauge failure."
)
df = common.require_data()

targets = [
    c for c in ("bhp", "oil_rate", "gas_rate", "water_rate", "whp") if c in df.columns
]
if not targets:
    st.warning("The data has no rate or pressure column to predict.")
    st.stop()
left, middle, right = st.columns(3)
target = str(left.selectbox("Target", targets))
estimator = str(
    middle.selectbox(
        "Model", ["lightgbm", "hist_gradient_boosting", "random_forest", "ridge"]
    )
)
method = str(
    right.selectbox(
        "Validation",
        ["timeseries", "lowo"],
        format_func={
            "timeseries": "time-ordered splits",
            "lowo": "leave one well out",
        }.get,
    )
)
holdout = st.slider(
    "Hold-out period for the plots (last fraction of dates)", 0.1, 0.4, 0.2, 0.05
)

if st.button("Train and validate", type="primary"):
    data = build_features(df[df[target].notna().groupby(df["well"]).transform("any")])
    try:
        with st.spinner("Cross-validating..."):
            model = Predictor(target, estimator=estimator)  # type: ignore[arg-type]
            cv = model.cross_validate(data, method)  # type: ignore[arg-type]
        dates = np.sort(data.index.unique())
        cutoff = dates[int(len(dates) * (1 - holdout))]
        train, test = data[data.index < cutoff], data[data.index >= cutoff]
        with st.spinner("Training on the training period..."):
            model.fit(train)
    except (MissingDependencyError, WellcastError) as exc:
        st.error(str(exc))
        st.stop()
    common.results()["predict"] = {
        "model": model,
        "cv": cv,
        "test": test,
        "target": target,
    }

state = common.results().get("predict")
if state is not None and state["target"] == target:
    model, cv, test = state["model"], state["cv"], state["test"]
    st.subheader("Out-of-sample metrics")
    overall = cv[cv["well"] == "all"][["mae", "rmse", "mape", "r2"]].mean()
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{overall['mae']:,.2f}")
    b.metric("RMSE", f"{overall['rmse']:,.2f}")
    c.metric("MAPE", f"{overall['mape']:.2f} %")
    d.metric("R2", f"{overall['r2']:.3f}")
    st.dataframe(
        cv[cv["well"] == "all"].drop(columns="well").round(3),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Metrics per well and fold"):
        st.dataframe(cv.round(3), hide_index=True, width="stretch")

    rows = test[test[target].notna()]
    if "on_stream_hours" in rows.columns:
        rows = rows[rows["on_stream_hours"] > 0]
    if len(rows):
        st.subheader("Hold-out period")
        interval = model.predict_interval(rows)
        st.dataframe(
            metrics_by_well(rows[target], interval["prediction"], rows["well"]).round(
                3
            ),
            width="stretch",
        )
        common.show(
            predicted_vs_actual(rows[target], interval["prediction"], rows["well"])
        )
        well = common.well_selector(sorted(rows["well"].unique()), key="predict_well")
        mask = (rows["well"] == well).to_numpy()
        common.show(
            prediction_interval_plot(
                rows.loc[mask, target],
                interval.loc[mask],
                title=f"{well}: {target}, 80 percent prediction interval",
            )
        )

    st.subheader("What drives the prediction")
    try:
        common.show(model.explain(rows if len(rows) else test, max_samples=300))
    except MissingDependencyError as exc:
        st.info(str(exc))

# =============================================================================
# End of module: app/pages/predict.py
# =============================================================================
