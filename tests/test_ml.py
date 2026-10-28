# =============================================================================
# wellcast: tests for the machine learning predictor
# -----------------------------------------------------------------------------
# Module:      tests/test_ml.py
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

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from wellcast.datasets import make_synthetic_field
from wellcast.exceptions import DataValidationError, FitError, MissingDependencyError
from wellcast.features import build_features
from wellcast.ml import Predictor, metrics_by_well, regression_metrics

FAST = {"n_estimators": 60}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    field = make_synthetic_field(n_wells=3, years=2, n_injectors=0)
    return build_features(field.production)


@pytest.fixture(scope="module")
def fitted(data: pd.DataFrame) -> Predictor:
    return Predictor("bhp", params=FAST).fit(data)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def test_regression_metrics_values() -> None:
    m = regression_metrics([100.0, 200.0, 300.0], [110.0, 190.0, 300.0])
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(20.0 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(200.0 / 3))
    assert m["mape"] == pytest.approx(100.0 * (0.1 + 0.05) / 3)
    assert m["r2"] == pytest.approx(1.0 - 200.0 / 20000.0)


def test_regression_metrics_edge_cases() -> None:
    empty = regression_metrics([np.nan], [1.0])
    assert empty["n"] == 0
    assert np.isnan(empty["mae"])
    zeros = regression_metrics([0.0, 0.0], [1.0, 1.0])
    assert np.isnan(zeros["mape"])
    assert np.isnan(zeros["r2"])
    with pytest.raises(DataValidationError, match="same length"):
        regression_metrics([1.0], [1.0, 2.0])


def test_metrics_by_well_without_wells() -> None:
    table = metrics_by_well(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), None)
    assert list(table.index) == ["all"]
    assert table.loc["all", "n"] == 2


# -----------------------------------------------------------------------------
# Fitting and prediction
# -----------------------------------------------------------------------------


def test_feature_selection_excludes_leaky_columns(fitted: Predictor) -> None:
    for leaky in ("bhp", "dp_tubing", "drawdown", "static_pressure", "well"):
        assert leaky not in fitted.features_
    assert "bhp_lag1" in fitted.features_
    # water_inj_rate is entirely NaN for producers and is skipped.
    assert "water_inj_rate" not in fitted.features_
    assert "fitted" in repr(fitted)


def test_virtual_flow_meter_accuracy(fitted: Predictor, data: pd.DataFrame) -> None:
    scores = fitted.score(data)
    assert set(scores.index) == {"P-01", "P-02", "P-03", "all"}
    assert scores.loc["all", "mape"] < 2.0
    assert scores.loc["all", "r2"] > 0.95
    prediction = fitted.predict(data)
    assert prediction.name == "bhp_pred"
    assert prediction.index.equals(data.index)


def test_prediction_interval(fitted: Predictor, data: pd.DataFrame) -> None:
    interval = fitted.predict_interval(data)
    assert list(interval.columns) == ["lower", "prediction", "upper"]
    assert (interval["lower"] <= interval["upper"]).all()
    flowing = data["on_stream_hours"] > 0
    inside = (data["bhp"] >= interval["lower"]) & (data["bhp"] <= interval["upper"])
    # Nominal 80 percent interval (quantiles 0.1 and 0.9).
    assert 0.65 < inside[flowing].mean() < 0.95


def test_explicit_features_and_rate_target(data: pd.DataFrame) -> None:
    model = Predictor(
        "oil_rate", features=["whp", "choke", "wht"], quantiles=None, params=FAST
    ).fit(data)
    assert model.features_ == ["whp", "choke", "wht"]
    with pytest.raises(FitError, match="quantile"):
        model.predict_interval(data)


@pytest.mark.parametrize(
    "estimator", ["hist_gradient_boosting", "random_forest", "ridge"]
)
def test_other_estimators(estimator: str, data: pd.DataFrame) -> None:
    params = {"n_estimators": 30} if estimator == "random_forest" else {}
    model = Predictor("bhp", estimator=estimator, params=params).fit(data)  # type: ignore[arg-type]
    assert model.score(data).loc["all", "r2"] > 0.9
    assert model.shap_values(data, max_samples=20).shape == (20, len(model.features_))
    assert (model.predict_interval(data.iloc[:50]).notna()).all().all()


def test_configuration_errors(data: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="unknown estimator"):
        Predictor(estimator="svm")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="quantiles"):
        Predictor(quantiles=(0.9, 0.1))
    with pytest.raises(DataValidationError, match="cannot be a feature"):
        Predictor("bhp", features=["bhp", "whp"]).fit(data)
    with pytest.raises(DataValidationError, match="no usable"):
        Predictor("q").fit(pd.DataFrame({"q": np.arange(20.0), "tag": ["x"] * 20}))
    with pytest.raises(FitError, match="at least 10"):
        Predictor("bhp").fit(data.iloc[:5])
    with pytest.raises(FitError, match="not fitted"):
        Predictor().predict(data)


def test_missing_ml_extra(monkeypatch: pytest.MonkeyPatch, data: pd.DataFrame) -> None:
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(MissingDependencyError, match=r"wellcast\[ml\]"):
        Predictor("bhp").fit(data)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def test_time_series_cross_validation(data: pd.DataFrame) -> None:
    cv = Predictor("bhp", params=FAST).cross_validate(data, "timeseries", n_splits=3)
    assert list(cv.columns) == ["fold", "well", "n", "mae", "rmse", "mape", "r2"]
    assert sorted(cv["fold"].unique()) == [0, 1, 2]
    totals = cv[cv["well"] == "all"]
    assert (totals["mape"] < 10.0).all()


def test_time_series_split_never_trains_on_the_future(data: pd.DataFrame) -> None:
    model = Predictor("bhp")
    rows = model._training_rows(data)
    for train, test in model._splits(rows, "timeseries", 3):
        assert rows.index[train].max() < rows.index[test].min()


def test_leave_one_well_out(data: pd.DataFrame) -> None:
    cv = Predictor("bhp", params=FAST).cross_validate(data, "lowo")
    per_well = cv[cv["well"] != "all"]
    assert sorted(per_well["well"]) == ["P-01", "P-02", "P-03"]
    # Each fold tests exactly one held-out well.
    assert (per_well.groupby("fold").size() == 1).all()


def test_cross_validation_errors(data: pd.DataFrame) -> None:
    one_well = data[data["well"] == "P-01"]
    with pytest.raises(DataValidationError, match="two wells"):
        Predictor("bhp").cross_validate(one_well, "lowo")
    with pytest.raises(DataValidationError, match="'well' column"):
        Predictor("bhp").cross_validate(one_well.drop(columns="well"), "lowo")
    with pytest.raises(DataValidationError, match="unknown validation"):
        Predictor("bhp").cross_validate(data, "kfold")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="distinct dates"):
        Predictor("bhp").cross_validate(one_well.iloc[:12], "timeseries", n_splits=20)


# -----------------------------------------------------------------------------
# Explanation and persistence
# -----------------------------------------------------------------------------


def test_tree_shap_is_additive(fitted: Predictor, data: pd.DataFrame) -> None:
    sample = data.iloc[:200]
    shap_values = fitted.shap_values(sample)
    assert list(shap_values.columns) == fitted.features_
    # TreeSHAP is exact: prediction - sum(contributions) is the same base
    # value for every row.
    base = fitted.predict(sample) - shap_values.sum(axis=1)
    assert base.std() < 1e-3 * abs(base.mean())


def test_explain_figure(fitted: Predictor, data: pd.DataFrame) -> None:
    fig = fitted.explain(data, max_samples=100, top=5)
    bars = fig.data[0]
    assert len(bars.y) == 5
    assert list(bars.x) == sorted(bars.x)
    assert "bhp" in fig.layout.title.text


def test_save_and_load(fitted: Predictor, data: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "vfm.joblib"
    fitted.save(path)
    restored = Predictor.load(path)
    pd.testing.assert_series_equal(restored.predict(data), fitted.predict(data))
    with pytest.raises(FitError, match="not fitted"):
        Predictor().save(tmp_path / "x.joblib")


def test_load_rejects_foreign_files_and_warns_on_version(
    fitted: Predictor, tmp_path: Path
) -> None:
    foreign = tmp_path / "foreign.joblib"
    joblib.dump({"something": 1}, foreign)
    with pytest.raises(DataValidationError, match="does not contain"):
        Predictor.load(foreign)
    old = tmp_path / "old.joblib"
    joblib.dump({"wellcast_version": "0.0.1", "predictor": fitted}, old)
    with pytest.warns(UserWarning, match="0.0.1"):
        Predictor.load(old)


# =============================================================================
# End of module: tests/test_ml.py
# =============================================================================
