# =============================================================================
# wellcast: tests for physics and leakage-safe features
# -----------------------------------------------------------------------------
# Module:      tests/test_features.py
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

import numpy as np
import pandas as pd
import pytest

from wellcast.datasets import make_synthetic_field
from wellcast.exceptions import DataValidationError
from wellcast.features import (
    FEATURE_SOURCES,
    add_lag_features,
    add_physics_features,
    add_rolling_features,
    build_features,
    features_using,
)


def _frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    return pd.DataFrame(
        {
            "well": "A",
            "oil_rate": [1000.0, 900.0, 0.0, 0.0, 800.0],
            "gas_rate": [500.0, 450.0, 0.0, 0.0, 480.0],
            "water_rate": [0.0, 100.0, 0.0, 0.0, 200.0],
            "whp": [500.0, 480.0, 900.0, 950.0, 470.0],
            "bhp": [3000.0, 2950.0, 3300.0, 3400.0, 2900.0],
            "choke": [32.0, 32.0, 32.0, 32.0, 40.0],
            "on_stream_hours": [24.0, 24.0, 0.0, 0.0, 12.0],
        },
        index=idx,
    )


# -----------------------------------------------------------------------------
# Physics features
# -----------------------------------------------------------------------------


def test_physics_feature_values() -> None:
    out = add_physics_features(_frame())
    row = out.iloc[1]
    assert row["liquid_rate"] == 1000.0
    assert row["gor"] == pytest.approx(500.0)  # 1000 * 450 / 900 scf/STB
    assert row["water_cut"] == pytest.approx(0.1)
    assert row["glr"] == pytest.approx(450.0)
    assert row["dp_tubing"] == pytest.approx(2470.0)
    # Bean area of a 32/64 in choke: pi/4 * 0.5^2 = 0.19635 in2.
    assert row["choke_normalized_rate"] == pytest.approx(1000.0 / (np.pi / 4 * 0.25))
    # Shut-in rows give NaN ratios, never inf.
    assert np.isnan(out["gor"].iloc[2])
    assert np.isnan(out["water_cut"].iloc[3])
    assert np.isfinite(
        out.select_dtypes("number").to_numpy()[
            ~np.isnan(out.select_dtypes("number").to_numpy())
        ]
    ).all()


def test_cumulative_and_days_on_production() -> None:
    out = add_physics_features(_frame())
    assert out["cum_oil"].tolist() == [1000.0, 1900.0, 1900.0, 1900.0, 2700.0]
    assert out["days_on_production"].tolist() == [1.0, 2.0, 2.0, 2.0, 2.5]
    no_hours = add_physics_features(_frame().drop(columns="on_stream_hours"))
    assert no_hours["days_on_production"].tolist() == [1.0, 2.0, 2.0, 2.0, 3.0]


def test_cumulative_uses_time_step() -> None:
    # Steps of 1, 1 and 2 days; the first row uses the median step (1 day).
    df = _frame().iloc[[0, 1, 2, 4]]
    out = add_physics_features(df)
    assert out["cum_oil"].iloc[-1] == pytest.approx(1000.0 + 900.0 + 0.0 + 800.0 * 2)


def test_static_pressure_from_build_up_is_past_only() -> None:
    out = add_physics_features(_frame())
    # Before the first shut-in the static pressure is unknown.
    assert np.isnan(out["static_pressure"].iloc[0])
    # The last build-up reading (3400 psi) is carried forward.
    assert out["static_pressure"].iloc[4] == 3400.0
    assert out["drawdown"].iloc[4] == pytest.approx(500.0)


def test_static_pressure_without_hours_uses_rates() -> None:
    out = add_physics_features(_frame().drop(columns="on_stream_hours"))
    assert out["static_pressure"].iloc[4] == 3400.0
    bare = add_physics_features(_frame()[["bhp"]])
    assert bare["static_pressure"].isna().all()


def test_reservoir_pressure_options() -> None:
    scalar = add_physics_features(_frame(), reservoir_pressure=3500.0)
    assert scalar["drawdown"].iloc[0] == 500.0
    mapping = add_physics_features(_frame(), reservoir_pressure={"A": 3600.0})
    assert mapping["drawdown"].iloc[0] == 600.0
    with pytest.raises(DataValidationError, match="missing for well"):
        add_physics_features(_frame(), reservoir_pressure={"B": 1.0})
    with pytest.raises(DataValidationError, match="well column"):
        add_physics_features(
            _frame().drop(columns="well"), reservoir_pressure={"A": 1.0}
        )


def test_physics_requires_datetime_index_and_skips_missing() -> None:
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        add_physics_features(_frame().reset_index(drop=True))
    out = add_physics_features(_frame()[["well", "whp"]])
    assert set(out.columns) == {"well", "whp"}


def test_features_using() -> None:
    assert features_using("oil_rate") == [
        name for name, src in FEATURE_SOURCES.items() if "oil_rate" in src
    ]
    assert features_using("nothing") == []


# -----------------------------------------------------------------------------
# Lag and rolling features
# -----------------------------------------------------------------------------


def _interleaved() -> pd.DataFrame:
    # Two wells stored row by row alternately, sharing dates.
    idx = pd.date_range("2024-01-01", periods=4, freq="D").repeat(2)
    return pd.DataFrame(
        {"well": ["A", "B"] * 4, "q": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]},
        index=idx,
    )


def test_lag_and_rolling_respect_wells_when_interleaved() -> None:
    df = _interleaved()
    lagged = add_lag_features(df, ["q"], [1, 2])
    assert lagged["q_lag1"].tolist()[2:] == [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
    rolled = add_rolling_features(df, ["q"], [2], ["mean", "max"])
    a = rolled[rolled["well"] == "A"]
    assert a["q_roll2_mean"].tolist()[1:] == [1.0, 1.5, 2.5]
    b = rolled[rolled["well"] == "B"]
    assert b["q_roll2_max"].tolist()[1:] == [10.0, 20.0, 30.0]


def test_features_are_leakage_safe() -> None:
    # Changing the value at time t must not change any lag or rolling feature
    # at times <= t.
    df = _interleaved()
    base = add_rolling_features(
        add_lag_features(df, ["q"], [1]), ["q"], [3], ["mean", "std"]
    )
    changed = df.copy()
    changed.iloc[4, 1] = 999.0  # well A, third date
    other = add_rolling_features(
        add_lag_features(changed, ["q"], [1]), ["q"], [3], ["mean", "std"]
    )
    feature_cols = [c for c in base.columns if c.startswith("q_")]
    pd.testing.assert_frame_equal(
        base[feature_cols].iloc[:5], other[feature_cols].iloc[:5]
    )
    assert base["q_lag1"].iloc[6] != other["q_lag1"].iloc[6]


def test_single_series_without_well_column() -> None:
    df = pd.DataFrame({"q": [1.0, 2.0, 3.0]})
    assert add_lag_features(df, ["q"], [1], by=None)["q_lag1"].tolist()[1:] == [
        1.0,
        2.0,
    ]
    out = add_rolling_features(df, ["q"], [2], ["median"], by=None)
    assert out["q_roll2_median"].tolist()[1:] == [1.0, 1.5]


@pytest.mark.parametrize(
    "call",
    [
        lambda df: add_lag_features(df, ["q"], [0]),
        lambda df: add_lag_features(df, ["missing"], [1]),
        lambda df: add_rolling_features(df, ["q"], [0]),
        lambda df: add_rolling_features(df, ["q"], [3], ["mode"]),
    ],
)
def test_invalid_lag_and_rolling_arguments(call: object) -> None:
    with pytest.raises(DataValidationError):
        call(_interleaved())  # type: ignore[operator]


def test_build_features_on_synthetic_field() -> None:
    field = make_synthetic_field(n_wells=2, years=1, n_injectors=0)
    out = build_features(field.production)
    assert len(out) == len(field.production)
    for name in ("gor", "drawdown", "bhp_lag7", "whp_roll30_mean", "cum_oil"):
        assert name in out.columns
    custom = build_features(field.production, lag_columns=[], windows=())
    assert not any("_lag" in c for c in custom.columns)


# =============================================================================
# End of module: tests/test_features.py
# =============================================================================
