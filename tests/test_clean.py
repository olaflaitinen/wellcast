# =============================================================================
# wellcast: tests for cleaning, outlier flags and QC
# -----------------------------------------------------------------------------
# Module:      tests/test_clean.py
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

from wellcast.clean import (
    detect_shut_ins,
    fill_gaps,
    flag_outliers,
    hampel_filter,
    iqr_outliers,
    qc_report,
    resample_regular,
    rolling_zscore_outliers,
)
from wellcast.exceptions import DataValidationError

# -----------------------------------------------------------------------------
# Shut-ins
# -----------------------------------------------------------------------------


def test_shut_ins_from_rates_and_hours(two_well_frame: pd.DataFrame) -> None:
    flags = detect_shut_ins(two_well_frame)
    assert flags.name == "shut_in"
    assert flags.sum() == 2
    hours_only = detect_shut_ins(two_well_frame[["on_stream_hours"]])
    assert hours_only.sum() == 2


def test_shut_ins_threshold() -> None:
    df = pd.DataFrame({"oil_rate": [0.5, 5.0]})
    assert detect_shut_ins(df, threshold=1.0).tolist() == [True, False]


def test_shut_ins_requires_columns() -> None:
    with pytest.raises(DataValidationError):
        detect_shut_ins(pd.DataFrame({"bhp": [1.0]}))


# -----------------------------------------------------------------------------
# Outlier detectors
# -----------------------------------------------------------------------------


def test_hampel_ignores_nan_and_gaps() -> None:
    s = pd.Series([np.nan] * 5 + [10.0, 10.1, 9.9, 80.0, 10.0, 10.2, 9.8])
    flags = hampel_filter(s, window=5)
    assert flags.tolist()[:5] == [False] * 5
    assert flags.iloc[8]
    assert flags.sum() == 1


@pytest.mark.parametrize(("window", "n_sigmas"), [(4, 3.0), (1, 3.0), (5, 0.0)])
def test_hampel_validates(window: int, n_sigmas: float) -> None:
    with pytest.raises(DataValidationError):
        hampel_filter(pd.Series([1.0, 2.0]), window=window, n_sigmas=n_sigmas)


def test_zscore_flat_series_has_no_flags() -> None:
    s = pd.Series([5.0] * 50)
    assert not rolling_zscore_outliers(s).any()
    with pytest.raises(DataValidationError):
        rolling_zscore_outliers(s, window=1)


def test_iqr_all_nan() -> None:
    assert not iqr_outliers(pd.Series([np.nan, np.nan])).any()


def test_flag_outliers_per_well() -> None:
    # 30 bbl/d is an outlier for well A but sits between the two wells'
    # levels, so only per-well grouping catches it.
    a_rates = [10.0, 11.0, 10.5, 30.0, 10.8, 10.1]
    b_rates = [500.0, 510.0, 505.0, 520.0, 498.0, 502.0]
    df = pd.DataFrame({"well": ["A"] * 6 + ["B"] * 6, "oil_rate": [*a_rates, *b_rates]})
    flags = flag_outliers(df, "oil_rate", "iqr")
    assert flags.tolist() == [False] * 3 + [True] + [False] * 8
    assert not flag_outliers(df, "oil_rate", "iqr", by=None).any()


def test_flag_outliers_duplicate_index_and_methods(
    two_well_frame: pd.DataFrame,
) -> None:
    # Two wells share dates, so the index is not unique.
    for method in ("hampel", "zscore", "iqr"):
        kwargs = {"window": 3} if method != "iqr" else {}
        flags = flag_outliers(two_well_frame, "bhp", method, **kwargs)  # type: ignore[arg-type]
        assert len(flags) == len(two_well_frame)
    with pytest.raises(DataValidationError, match="unknown method"):
        flag_outliers(two_well_frame, "bhp", "magic")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="missing"):
        flag_outliers(two_well_frame, "nope")


# -----------------------------------------------------------------------------
# Resampling and gap filling
# -----------------------------------------------------------------------------


def test_resample_two_wells(two_well_frame: pd.DataFrame) -> None:
    sparse = two_well_frame.iloc[[0, 3, 6, 11]]
    out = resample_regular(sparse)
    assert out["well"].tolist() == ["A"] * 4 + ["B"] * 6
    assert out.index.name == "date"
    weekly = resample_regular(two_well_frame, freq="W", agg="sum")
    assert weekly["oil_rate"].sum() == pytest.approx(two_well_frame["oil_rate"].sum())


def test_resample_single_series_without_well() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-04"])
    out = resample_regular(pd.DataFrame({"q": [1.0, 4.0]}, index=idx))
    assert "well" not in out.columns
    assert len(out) == 4


def test_resample_requires_datetime_index() -> None:
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        resample_regular(pd.DataFrame({"q": [1.0]}))


def test_fill_gaps_does_not_cross_wells() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.concat(
        [
            pd.DataFrame({"well": "A", "bhp": [1.0, 2.0, np.nan]}, index=idx),
            pd.DataFrame({"well": "B", "bhp": [np.nan, 5.0, 6.0]}, index=idx),
        ]
    )
    linear = fill_gaps(df)
    # Edge gaps are never extrapolated, even with another well next to them.
    assert np.isnan(linear["bhp"].iloc[2])
    assert np.isnan(linear["bhp"].iloc[3])
    ffill = fill_gaps(df, method="ffill")
    assert ffill["bhp"].iloc[2] == 2.0
    assert np.isnan(ffill["bhp"].iloc[3])
    bfill = fill_gaps(df, ["bhp"], method="bfill")
    assert bfill["bhp"].iloc[3] == 5.0
    assert np.isnan(bfill["bhp"].iloc[2])


def test_fill_gaps_validation() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"q": [1.0, np.nan]}, index=idx)
    with pytest.raises(DataValidationError, match="limit"):
        fill_gaps(df, limit=0)
    with pytest.raises(DataValidationError, match="unknown method"):
        fill_gaps(df, method="cubic")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        fill_gaps(df.reset_index(drop=True))


def test_fill_gaps_leaves_input_untouched() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"q": [1.0, np.nan, 3.0]}, index=idx)
    out = fill_gaps(df)
    assert out["q"].iloc[1] == 2.0
    assert np.isnan(df["q"].iloc[1])


# -----------------------------------------------------------------------------
# QC report
# -----------------------------------------------------------------------------


def test_qc_report_counts() -> None:
    df = pd.DataFrame(
        {
            "well": ["A"] * 8,
            "whp": [250.0] * 6 + [-3.0, np.nan],
            "water_cut": [0.1, 0.2, 1.4, 0.3, -0.1, 0.2, 0.2, 0.2],
            "on_stream_hours": [24.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0],
            "tag": ["x"] * 8,
        }
    )
    report = qc_report(df).set_index("column")
    assert list(report.columns) == [
        "well",
        "n",
        "missing",
        "missing_pct",
        "negative",
        "flatline",
        "impossible",
    ]
    assert report.loc["whp", "missing_pct"] == pytest.approx(12.5)
    assert report.loc["whp", "flatline"] == 6
    assert report.loc["whp", "impossible"] == 1
    assert report.loc["water_cut", "impossible"] == 2
    # Zero runs are shut-ins, not frozen gauges.
    assert report.loc["on_stream_hours", "flatline"] == 0
    assert report.loc["on_stream_hours", "impossible"] == 1
    assert "tag" not in report.index


def test_flatline_runs_break_at_zero_and_gap() -> None:
    df = pd.DataFrame({"whp": [5.0, 5.0, 0.0, 5.0, 5.0, 5.0, np.nan, 5.0, 5.0]})
    assert qc_report(df, flatline_min_run=3)["flatline"].iloc[0] == 3


def test_qc_report_custom_hours_and_no_rule() -> None:
    df = pd.DataFrame({"on_stream_hours": [30.0, 700.0], "custom": [1.0, 2.0]})
    report = qc_report(df, max_hours=744.0).set_index("column")
    assert "well" not in report.columns
    assert report.loc["on_stream_hours", "impossible"] == 0
    assert np.isnan(report.loc["custom", "impossible"])


def test_qc_report_multi_well(two_well_frame: pd.DataFrame) -> None:
    report = qc_report(two_well_frame)
    assert set(report["well"]) == {"A", "B"}
    assert len(report) == 2 * 4


# =============================================================================
# End of module: tests/test_clean.py
# =============================================================================
