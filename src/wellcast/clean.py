# =============================================================================
# wellcast: production data cleaning, outlier flagging and quality control
# -----------------------------------------------------------------------------
# Module:      wellcast.clean
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

"""Cleaning and quality control for field production data.

Outlier detectors *flag* suspicious values and return a boolean Series aligned
with the input. They never delete or overwrite data: the decision to drop,
replace or keep a flagged value belongs to the engineer.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from wellcast._validation import require_columns
from wellcast.exceptions import DataValidationError

__all__ = [
    "IMPOSSIBLE_RANGES",
    "detect_shut_ins",
    "fill_gaps",
    "flag_outliers",
    "hampel_filter",
    "iqr_outliers",
    "qc_report",
    "resample_regular",
    "rolling_zscore_outliers",
]

# Scale factor that makes the median absolute deviation a consistent
# estimator of the standard deviation for normally distributed data.
_MAD_TO_SIGMA = 1.4826

# Setpoints and uptime are legitimately constant for long periods (a choke left
# at 32/64, 24 producing hours every day), so runs of identical values in these
# columns are not frozen gauges and are not counted as flatlines.
_FLATLINE_EXEMPT = frozenset({"on_stream_hours", "choke"})

# Physically admissible ranges [low, high] in canonical units. Values outside
# are counted as "impossible" by qc_report. Temperatures are in degF, so the
# lower bound is absolute zero (-459.67 degF).
IMPOSSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "oil_rate": (0.0, np.inf),
    "gas_rate": (0.0, np.inf),
    "water_rate": (0.0, np.inf),
    "water_inj_rate": (0.0, np.inf),
    "gas_inj_rate": (0.0, np.inf),
    "whp": (0.0, np.inf),
    "bhp": (0.0, np.inf),
    "wht": (-459.67, np.inf),
    "bht": (-459.67, np.inf),
    "choke": (0.0, np.inf),
    "water_cut": (0.0, 1.0),
    "on_stream_hours": (0.0, 24.0),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _iter_groups(
    df: pd.DataFrame, by: str | None
) -> Iterator[tuple[str, pd.DataFrame]]:
    # Yield (well name, frame) pairs; a frame without the grouping column is
    # treated as one well named "all".
    if by is not None and by in df.columns:
        for name, group in df.groupby(by, sort=True):
            yield str(name), group
    else:
        yield "all", df


def _require_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("a DatetimeIndex is required")


# -----------------------------------------------------------------------------
# Shut-in detection
# -----------------------------------------------------------------------------


def detect_shut_ins(
    df: pd.DataFrame,
    rate_columns: Sequence[str] = ("oil_rate", "gas_rate", "water_rate"),
    *,
    threshold: float = 0.0,
    hours_column: str = "on_stream_hours",
) -> pd.Series:
    """Flag periods when a well is shut in.

    A row is a shut-in when its producing hours are zero, or when every rate
    column present in ``df`` is known and at or below ``threshold``. Missing
    rates are not treated as zero: a gap in reporting is not a shut-in.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table in the canonical schema.
    rate_columns : sequence of str, optional
        Candidate rate columns; those absent from ``df`` are ignored.
    threshold : float, optional
        Rate at or below which the well counts as not flowing. Default 0.
    hours_column : str, optional
        Producing-hours column, used when present.

    Returns
    -------
    pandas.Series
        Boolean flags aligned with ``df``, named ``shut_in``.

    Raises
    ------
    DataValidationError
        If none of ``rate_columns`` nor ``hours_column`` is present.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"oil_rate": [120.0, 0.0, None], "water_rate": [10.0, 0.0, 5.0]}
    ... )
    >>> detect_shut_ins(df).tolist()
    [False, True, False]
    """
    present = [c for c in rate_columns if c in df.columns]
    has_hours = hours_column in df.columns
    if not present and not has_hours:
        raise DataValidationError(
            f"none of {list(rate_columns)} or {hours_column!r} found in data"
        )
    flags: pd.Series = pd.Series(False, index=df.index, name="shut_in")
    if present:
        rates = df[present]
        flags |= rates.notna().all(axis=1) & (rates <= threshold).all(axis=1)
    if has_hours:
        flags |= df[hours_column].eq(0.0)
    return flags


# -----------------------------------------------------------------------------
# Outlier detectors (flag only)
# -----------------------------------------------------------------------------


def hampel_filter(
    series: pd.Series, window: int = 7, n_sigmas: float = 3.0
) -> pd.Series:
    r"""Flag outliers with the Hampel identifier.

    Parameters
    ----------
    series : pandas.Series
        Values in time order.
    window : int, optional
        Odd length of the centered window. Default 7.
    n_sigmas : float, optional
        Threshold in robust standard deviations. Default 3.

    Returns
    -------
    pandas.Series
        Boolean flags aligned with ``series``. NaN values are never flagged.

    Raises
    ------
    DataValidationError
        If ``window`` is not an odd integer >= 3 or ``n_sigmas`` <= 0.

    Notes
    -----
    A value $x_i$ is flagged when

    $$
    |x_i - m_i| > t \cdot 1.4826 \, \mathrm{MAD}_i
    $$

    where $m_i$ and $\mathrm{MAD}_i$ are the median and median
    absolute deviation of the centered window. Missing values inside the
    window are ignored. See Pearson, R. K. (2002), "Outliers in process
    modeling and identification", IEEE Trans. Control Syst. Technol. 10(1).

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([10.0, 10.2, 9.9, 10.1, 55.0, 10.0, 9.8, 10.1])
    >>> hampel_filter(s, window=5).tolist()
    [False, False, False, False, True, False, False, False]
    """
    if window < 3 or window % 2 == 0:
        raise DataValidationError("window must be an odd integer >= 3")
    if n_sigmas <= 0:
        raise DataValidationError("n_sigmas must be positive")
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    half = window // 2
    padded = np.pad(values, half, constant_values=np.nan)
    windows = sliding_window_view(padded, window)
    with warnings.catch_warnings():
        # All-NaN windows are legitimate (long gaps) and yield NaN, not a flag.
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(windows, axis=1)
        mad = np.nanmedian(np.abs(windows - median[:, None]), axis=1)
    deviation = np.abs(values - median)
    with np.errstate(invalid="ignore"):
        flags = deviation > n_sigmas * _MAD_TO_SIGMA * mad
    result: pd.Series = pd.Series(flags, index=series.index, name=series.name)
    return result


def rolling_zscore_outliers(
    series: pd.Series,
    window: int = 30,
    threshold: float = 3.0,
    min_periods: int | None = None,
) -> pd.Series:
    r"""Flag values far from a centered rolling mean.

    Parameters
    ----------
    series : pandas.Series
        Values in time order.
    window : int, optional
        Rolling window length in samples. Default 30.
    threshold : float, optional
        Absolute z-score above which a value is flagged. Default 3.
    min_periods : int, optional
        Minimum observations in the window. Default ``window // 2``.

    Returns
    -------
    pandas.Series
        Boolean flags aligned with ``series``.

    Notes
    -----
    $z_i = (x_i - \bar{x}_i) / s_i$ with the rolling mean
    $\bar{x}_i$ and standard deviation $s_i$. Windows with zero
    spread produce no flags.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> s = pd.Series(np.r_[np.tile([10.0, 11.0], 20), 40.0, np.tile([10.0, 11.0], 20)])
    >>> int(rolling_zscore_outliers(s, window=21).sum())
    1
    """
    if window < 2:
        raise DataValidationError("window must be >= 2")
    periods = max(2, window // 2) if min_periods is None else min_periods
    rolling = series.astype("float64").rolling(window, center=True, min_periods=periods)
    mean = rolling.mean()
    std = rolling.std()
    z = (series - mean) / std.where(std > 0)
    result: pd.Series = z.abs() > threshold
    result.name = series.name
    return result


def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    r"""Flag values outside Tukey's fences.

    Parameters
    ----------
    series : pandas.Series
        Values to test.
    k : float, optional
        Fence multiplier. Default 1.5 (3.0 for "far out" values).

    Returns
    -------
    pandas.Series
        Boolean flags aligned with ``series``.

    Notes
    -----
    Values below $Q_1 - k\,\mathrm{IQR}$ or above
    $Q_3 + k\,\mathrm{IQR}$ are flagged (Tukey, 1977, Exploratory Data
    Analysis).

    Examples
    --------
    >>> import pandas as pd
    >>> iqr_outliers(pd.Series([1.0, 2.0, 2.5, 3.0, 50.0])).tolist()
    [False, False, False, False, True]
    """
    clean = series.dropna()
    if clean.empty:
        empty: pd.Series = pd.Series(False, index=series.index, name=series.name)
        return empty
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1
    result: pd.Series = (series < q1 - k * iqr) | (series > q3 + k * iqr)
    result.name = series.name
    return result


def flag_outliers(
    df: pd.DataFrame,
    column: str,
    method: Literal["hampel", "zscore", "iqr"] = "hampel",
    *,
    by: str | None = "well",
    **kwargs: Any,
) -> pd.Series:
    """Apply an outlier detector to one column, separately for each well.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table, sorted by date within each well.
    column : str
        Column to test.
    method : {"hampel", "zscore", "iqr"}, optional
        Detector to use. Default ``"hampel"``.
    by : str or None, optional
        Grouping column. Default ``"well"``; ignored when absent.
    **kwargs
        Passed to the detector.

    Returns
    -------
    pandas.Series
        Boolean flags aligned with ``df``.

    Raises
    ------
    DataValidationError
        If ``column`` is missing or ``method`` is unknown.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"well": ["A"] * 5, "oil_rate": [1.0, 2.0, 2.5, 3.0, 50.0]})
    >>> flag_outliers(df, "oil_rate", "iqr").tolist()
    [False, False, False, False, True]
    """
    require_columns(df, [column])
    detectors = {
        "hampel": hampel_filter,
        "zscore": rolling_zscore_outliers,
        "iqr": iqr_outliers,
    }
    if method not in detectors:
        raise DataValidationError(f"unknown method {method!r}")
    detector: Any = detectors[method]
    df = df.reset_index(drop=True) if not df.index.is_unique else df
    parts = [detector(g[column], **kwargs) for _, g in _iter_groups(df, by)]
    result: pd.Series = pd.concat(parts).reindex(df.index).astype(bool)
    result.name = column
    return result


# -----------------------------------------------------------------------------
# Regularization and gap filling
# -----------------------------------------------------------------------------


def resample_regular(
    df: pd.DataFrame,
    freq: str = "D",
    agg: Literal["mean", "median", "sum", "first", "last", "min", "max"] = "mean",
    *,
    by: str | None = "well",
) -> pd.DataFrame:
    """Resample each well onto a regular time grid.

    Missing periods become NaN rows so gaps stay visible. Non-numeric columns
    other than ``by`` are dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a ``DatetimeIndex``.
    freq : str, optional
        Pandas offset alias. Default ``"D"`` (daily).
    agg : str, optional
        Aggregation within each period. Default ``"mean"``. Use ``"sum"`` for
        volumes, ``"mean"`` for rates and pressures.
    by : str or None, optional
        Grouping column. Default ``"well"``; ignored when absent.

    Returns
    -------
    pandas.DataFrame
        Regularly sampled data, wells stacked in order.

    Raises
    ------
    DataValidationError
        If the index is not a ``DatetimeIndex``.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.to_datetime(["2024-01-01", "2024-01-03"])
    >>> df = pd.DataFrame({"well": ["A", "A"], "oil_rate": [100.0, 90.0]}, index=idx)
    >>> resample_regular(df)["oil_rate"].tolist()
    [100.0, nan, 90.0]
    """
    _require_datetime_index(df)
    parts = []
    for name, group in _iter_groups(df, by):
        numeric = group.select_dtypes("number")
        out = numeric.resample(freq).agg(agg)
        if by is not None and by in df.columns:
            out.insert(0, by, name)
        parts.append(out)
    result: pd.DataFrame = pd.concat(parts)
    result.index.name = df.index.name
    return result


def fill_gaps(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: Literal["linear", "ffill", "bfill"] = "linear",
    limit: int = 3,
    *,
    by: str | None = "well",
) -> pd.DataFrame:
    """Fill short gaps within each well, leaving longer gaps untouched.

    Parameters
    ----------
    df : pandas.DataFrame
        Regularly sampled table with a ``DatetimeIndex``
        (see `resample_regular`).
    columns : sequence of str, optional
        Columns to fill. Default: all numeric columns.
    method : {"linear", "ffill", "bfill"}, optional
        ``"linear"`` interpolates in time between valid neighbours (never
        extrapolates); ``"ffill"`` and ``"bfill"`` propagate the last or next
        valid value. Default ``"linear"``.
    limit : int, optional
        Maximum number of consecutive missing values to fill. Default 3.
    by : str or None, optional
        Grouping column. Default ``"well"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with gaps filled.

    Raises
    ------
    DataValidationError
        If ``limit`` < 1, ``method`` is unknown, or the index is not a
        ``DatetimeIndex``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=5, freq="D")
    >>> df = pd.DataFrame({"bhp": [3000.0, np.nan, 2980.0, np.nan, np.nan]}, index=idx)
    >>> fill_gaps(df, limit=1)["bhp"].tolist()
    [3000.0, 2990.0, 2980.0, nan, nan]
    """
    _require_datetime_index(df)
    if limit < 1:
        raise DataValidationError("limit must be >= 1")
    if method not in ("linear", "ffill", "bfill"):
        raise DataValidationError(f"unknown method {method!r}")
    cols = (
        list(columns)
        if columns is not None
        else list(df.select_dtypes("number").columns)
    )
    require_columns(df, cols)
    out: pd.DataFrame = df.copy()
    out[cols] = out[cols].astype("float64")
    col_positions = out.columns.get_indexer(pd.Index(cols))
    if by is not None and by in out.columns:
        masks = [(out[by] == key).to_numpy() for key in out[by].unique()]
    else:
        masks = [np.ones(len(out), dtype=bool)]
    for mask in masks:
        block = out.loc[mask, cols]
        if method == "linear":
            filled = block.interpolate(method="time", limit=limit, limit_area="inside")
        elif method == "ffill":
            filled = block.ffill(limit=limit)
        else:
            filled = block.bfill(limit=limit)
        out.iloc[np.flatnonzero(mask), col_positions] = filled.to_numpy()
    return out


# -----------------------------------------------------------------------------
# Quality-control report
# -----------------------------------------------------------------------------


def _flatline_count(series: pd.Series, min_run: int) -> int:
    # A flatline is a run of identical, non-zero readings: typical of a frozen
    # gauge or a copied-forward allocation. Zero runs are shut-ins, not
    # flatlines, and are reported by detect_shut_ins instead.
    # Runs are found on the full series so that a gap or a zero breaks a run
    # (NaN never equals its neighbour, so each NaN is a run of its own).
    run_id = series.ne(series.shift()).cumsum()
    sizes = run_id.map(run_id.value_counts())
    counted = series.notna() & series.ne(0.0) & (sizes >= min_run)
    return int(counted.sum())


def qc_report(
    df: pd.DataFrame,
    *,
    by: str | None = "well",
    flatline_min_run: int = 5,
    max_hours: float = 24.0,
) -> pd.DataFrame:
    """Summarize data quality per well and column.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table in the canonical schema.
    by : str or None, optional
        Grouping column. Default ``"well"``; ignored when absent.
    flatline_min_run : int, optional
        Minimum run of identical non-zero readings counted as a flatline.
        Default 5.
    max_hours : float, optional
        Upper bound for ``on_stream_hours`` (24 for daily data). Default 24.

    Returns
    -------
    pandas.DataFrame
        Tidy table with one row per (well, column) and the columns ``well``,
        ``column``, ``n``, ``missing``, ``missing_pct``, ``negative``,
        ``flatline`` and ``impossible``. ``flatline`` is NaN for setpoint
        columns (``choke``, ``on_stream_hours``), which are constant by
        design. ``impossible`` counts values outside
        `IMPOSSIBLE_RANGES` (for example a negative pressure or a water
        cut outside [0, 1]); it is NaN for columns without a rule.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"well": ["A"] * 4, "bhp": [3000.0, -5.0, None, 2990.0]})
    >>> qc_report(df)[["column", "missing", "negative", "impossible"]]
      column  missing  negative  impossible
    0    bhp        1         1         1.0
    """
    ranges = dict(IMPOSSIBLE_RANGES)
    ranges["on_stream_hours"] = (0.0, max_hours)
    rows: list[dict[str, Any]] = []
    for name, group in _iter_groups(df, by):
        numeric = group.select_dtypes("number")
        for column in numeric.columns:
            s = numeric[column]
            n = len(s)
            missing = int(s.isna().sum())
            impossible: float = np.nan
            if column in ranges:
                low, high = ranges[column]
                impossible = float(((s < low) | (s > high)).sum())
            rows.append(
                {
                    "well": name,
                    "column": column,
                    "n": n,
                    "missing": missing,
                    "missing_pct": 100.0 * missing / n if n else np.nan,
                    "negative": int((s < 0).sum()),
                    "flatline": np.nan
                    if column in _FLATLINE_EXEMPT
                    else _flatline_count(s, flatline_min_run),
                    "impossible": impossible,
                }
            )
    columns = [
        "well",
        "column",
        "n",
        "missing",
        "missing_pct",
        "negative",
        "flatline",
        "impossible",
    ]
    report: pd.DataFrame = pd.DataFrame(rows, columns=columns)
    if by is None or by not in df.columns:
        report = report.drop(columns="well")
    return report


# =============================================================================
# End of module: wellcast.clean
# =============================================================================
