# =============================================================================
# wellcast: physics-informed and leakage-safe time-series features
# -----------------------------------------------------------------------------
# Module:      wellcast.features
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

"""Feature engineering for production data.

Two kinds of features are built:

* **Physics features** combine measurements of the same row (GOR, water cut,
  drawdown and so on). :data:`FEATURE_SOURCES` records which raw columns each
  one uses, so a model can drop features computed from its own target.
* **Lag and rolling features** summarize the past of a column within each
  well. They are leakage-safe by construction: the value at time ``t`` only
  uses rows strictly before ``t`` of the same well.

All functions expect the canonical schema of :mod:`wellcast.io`, sorted by
date within each well, and return a new DataFrame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from wellcast._validation import require_columns
from wellcast.exceptions import DataValidationError

__all__ = [
    "FEATURE_SOURCES",
    "add_lag_features",
    "add_physics_features",
    "add_rolling_features",
    "build_features",
    "features_using",
]

# Raw columns behind each physics feature. Lag and rolling features are not
# listed: they only use past values and cannot leak the current target.
FEATURE_SOURCES: dict[str, tuple[str, ...]] = {
    "liquid_rate": ("oil_rate", "water_rate"),
    "gor": ("gas_rate", "oil_rate"),
    "water_cut": ("water_rate", "oil_rate"),
    "glr": ("gas_rate", "oil_rate", "water_rate"),
    "dp_tubing": ("bhp", "whp"),
    "drawdown": ("bhp",),
    "static_pressure": ("bhp",),
    "choke_normalized_rate": ("oil_rate", "water_rate", "choke"),
    "cum_oil": ("oil_rate",),
    "cum_gas": ("gas_rate",),
    "cum_water": ("water_rate",),
    "days_on_production": ("on_stream_hours", "oil_rate", "water_rate"),
}


def features_using(column: str) -> list[str]:
    """Return physics features computed from ``column`` on the same row.

    Parameters
    ----------
    column : str
        Raw column, typically a model target such as ``"bhp"``.

    Returns
    -------
    list of str
        Feature names that must be excluded when predicting ``column``.

    Examples
    --------
    >>> features_using("bhp")
    ['dp_tubing', 'drawdown', 'static_pressure']
    """
    return [name for name, sources in FEATURE_SOURCES.items() if column in sources]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    # Ratios are undefined (NaN, not inf) when the denominator is zero, which
    # happens on every shut-in day.
    ratio: pd.Series = num / den.where(den > 0)
    return ratio


def _groups(df: pd.DataFrame, by: str | None) -> pd.Series | None:
    if by is not None and by in df.columns:
        return df[by]
    return None


def _per_well_cumsum(values: pd.Series, keys: pd.Series | None) -> pd.Series:
    filled = values.fillna(0.0)
    out: pd.Series = (
        filled.groupby(keys).cumsum() if keys is not None else filled.cumsum()
    )
    return out


# -----------------------------------------------------------------------------
# Physics features
# -----------------------------------------------------------------------------


def add_physics_features(
    df: pd.DataFrame,
    *,
    reservoir_pressure: float | Mapping[str, float] | None = None,
    by: str | None = "well",
) -> pd.DataFrame:
    r"""Add physics-informed features computed from each row.

    Features are added only when their source columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table with a ``DatetimeIndex``, sorted within each well.
    reservoir_pressure : float or mapping of str to float, optional
        Average reservoir pressure (psi), one value for all wells or one per
        well. When omitted, the static pressure is estimated as the last
        bottomhole pressure recorded during a shut-in (end of build-up),
        carried forward in time; rows before the first shut-in get NaN.
    by : str or None, optional
        Well column. Default ``"well"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the new columns:

        ``liquid_rate`` (bbl/d), ``gor`` (scf/STB), ``water_cut`` (fraction),
        ``glr`` (scf/STB), ``dp_tubing`` (bhp - whp, psi), ``static_pressure``
        and ``drawdown`` (psi), ``choke_normalized_rate`` (bbl/d per square
        inch of bean area), ``cum_oil``/``cum_water`` (bbl), ``cum_gas``
        (Mscf) and ``days_on_production``.

    Raises
    ------
    DataValidationError
        If the index is not a ``DatetimeIndex``.

    Notes
    -----
    .. math::

        \mathrm{GOR} = 1000 \frac{q_g}{q_o}, \quad
        f_w = \frac{q_w}{q_o + q_w}, \quad
        \Delta p = \bar{p}_r - p_{wf}

    Gas rates are in Mscf/d, hence the factor 1000. Cumulative volumes
    integrate daily-average rates over the time step to the previous row.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=2, freq="D")
    >>> df = pd.DataFrame(
    ...     {
    ...         "oil_rate": [800.0, 0.0],
    ...         "gas_rate": [400.0, 0.0],
    ...         "water_rate": [200.0, 0.0],
    ...     },
    ...     index=idx,
    ... )
    >>> out = add_physics_features(df)
    >>> out[["gor", "water_cut", "cum_oil"]]
                  gor  water_cut  cum_oil
    2024-01-01  500.0        0.2    800.0
    2024-01-02    NaN        NaN    800.0
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("a DatetimeIndex is required")
    out: pd.DataFrame = df.copy()
    cols = set(out.columns)
    keys = _groups(out, by)
    oil = out["oil_rate"] if "oil_rate" in cols else None
    water = out["water_rate"] if "water_rate" in cols else None
    gas = out["gas_rate"] if "gas_rate" in cols else None

    liquid = None
    if oil is not None and water is not None:
        liquid = oil + water
        out["liquid_rate"] = liquid
        out["water_cut"] = _safe_ratio(water, liquid)
    if gas is not None and oil is not None:
        out["gor"] = 1000.0 * _safe_ratio(gas, oil)
    if gas is not None and liquid is not None:
        out["glr"] = 1000.0 * _safe_ratio(gas, liquid)

    if {"bhp", "whp"} <= cols:
        out["dp_tubing"] = out["bhp"] - out["whp"]
    if "bhp" in cols:
        static = _static_pressure(out, keys, reservoir_pressure)
        out["static_pressure"] = static
        out["drawdown"] = static - out["bhp"]
    if liquid is not None and "choke" in cols:
        # Bean area in square inches: pi/4 (S/64)^2 with S in 64ths.
        area = np.pi / 4.0 * (out["choke"] / 64.0) ** 2
        out["choke_normalized_rate"] = _safe_ratio(liquid, area)

    # Time step to the previous row of the same well, days; the first row of
    # each well uses the median step of the whole table (one day for daily
    # data).
    index_days = pd.Series(
        (out.index - out.index.min()) / pd.Timedelta(days=1), index=out.index
    )
    step = index_days.groupby(keys).diff() if keys is not None else index_days.diff()
    median_step = float(step.median()) if step.notna().any() else 1.0
    step = step.fillna(median_step if median_step > 0 else 1.0)
    for name, rate in (("cum_oil", oil), ("cum_gas", gas), ("cum_water", water)):
        if rate is not None:
            out[name] = _per_well_cumsum(rate * step, keys)

    if "on_stream_hours" in cols:
        out["days_on_production"] = _per_well_cumsum(
            out["on_stream_hours"] / 24.0, keys
        )
    elif liquid is not None:
        flowing = (liquid.fillna(0.0) > 0).astype(float) * step
        out["days_on_production"] = _per_well_cumsum(flowing, keys)
    return out


def _static_pressure(
    df: pd.DataFrame,
    keys: pd.Series | None,
    reservoir_pressure: float | Mapping[str, float] | None,
) -> pd.Series:
    if isinstance(reservoir_pressure, Mapping):
        if keys is None:
            raise DataValidationError(
                "a per-well reservoir_pressure needs a well column"
            )
        missing = set(keys.unique()) - set(reservoir_pressure)
        if missing:
            raise DataValidationError(
                f"reservoir_pressure missing for well(s): {sorted(missing)}"
            )
        mapped: pd.Series = keys.map(reservoir_pressure).astype("float64")
        return mapped
    if reservoir_pressure is not None:
        constant: pd.Series = pd.Series(float(reservoir_pressure), index=df.index)
        return constant
    if "on_stream_hours" in df.columns:
        shut = df["on_stream_hours"].eq(0.0)
    else:
        rate_cols = [c for c in ("oil_rate", "water_rate") if c in df.columns]
        if not rate_cols:
            unknown: pd.Series = pd.Series(np.nan, index=df.index)
            return unknown
        shut = df[rate_cols].fillna(-1.0).le(0.0).all(axis=1) & df[
            rate_cols
        ].notna().all(axis=1)
    # The last reading of each build-up is the best static estimate; carry it
    # forward only (never backward, which would leak the future).
    readings = df["bhp"].where(shut)
    filled: pd.Series = (
        readings.groupby(keys).ffill() if keys is not None else readings.ffill()
    )
    return filled


# -----------------------------------------------------------------------------
# Lag and rolling features (past only)
# -----------------------------------------------------------------------------


def add_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int] = (1, 7),
    *,
    by: str | None = "well",
) -> pd.DataFrame:
    """Add lagged copies of columns, shifted within each well.

    Parameters
    ----------
    df : pandas.DataFrame
        Table sorted by date within each well.
    columns : sequence of str
        Columns to lag.
    lags : sequence of int, optional
        Positive lags in rows. Default (1, 7).
    by : str or None, optional
        Well column. Default ``"well"``.

    Returns
    -------
    pandas.DataFrame
        Copy with columns named ``<column>_lag<k>``.

    Raises
    ------
    DataValidationError
        If a column is missing or a lag is not a positive integer.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"well": ["A", "A", "B", "B"], "q": [1.0, 2.0, 3.0, 4.0]})
    >>> add_lag_features(df, ["q"], [1])["q_lag1"].tolist()
    [nan, 1.0, nan, 3.0]
    """
    require_columns(df, columns)
    if any(int(k) < 1 for k in lags):
        raise DataValidationError("lags must be positive integers")
    out: pd.DataFrame = df.copy()
    keys = _groups(out, by)
    for column in columns:
        grouped = out[column].groupby(keys) if keys is not None else out[column]
        for k in lags:
            out[f"{column}_lag{k}"] = grouped.shift(int(k))
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int] = (7, 30),
    stats: Sequence[str] = ("mean", "std"),
    *,
    by: str | None = "well",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add rolling statistics over strictly past rows of each well.

    The window ending at row ``t`` covers rows ``t - w`` to ``t - 1``, so the
    current value never enters its own feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Table sorted by date within each well.
    columns : sequence of str
        Columns to summarize.
    windows : sequence of int, optional
        Window lengths in rows. Default (7, 30).
    stats : sequence of str, optional
        Any of ``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"median"``.
        Default ("mean", "std").
    by : str or None, optional
        Well column. Default ``"well"``.
    min_periods : int, optional
        Minimum past observations required. Default 1.

    Returns
    -------
    pandas.DataFrame
        Copy with columns named ``<column>_roll<w>_<stat>``.

    Raises
    ------
    DataValidationError
        If a column is missing, a window is < 1 or a statistic is unknown.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"q": [1.0, 2.0, 3.0, 4.0]})
    >>> add_rolling_features(df, ["q"], [2], ["mean"], by=None)["q_roll2_mean"].tolist()
    [nan, 1.0, 1.5, 2.5]
    """
    require_columns(df, columns)
    allowed = {"mean", "std", "min", "max", "median"}
    unknown = set(stats) - allowed
    if unknown:
        raise DataValidationError(f"unknown statistic(s): {sorted(unknown)}")
    if any(int(w) < 1 for w in windows):
        raise DataValidationError("windows must be positive integers")
    out: pd.DataFrame = df.copy()
    keys = _groups(out, by)
    # Work on positions so that duplicate dates (several wells) and wells that
    # are not stored contiguously keep their row alignment.
    positions = pd.RangeIndex(len(out))
    key_values = None if keys is None else pd.Series(keys.to_numpy(), index=positions)
    for column in columns:
        series = pd.Series(out[column].to_numpy(dtype=np.float64), index=positions)
        past = (
            series.groupby(key_values).shift(1)
            if key_values is not None
            else series.shift(1)
        )
        for w in windows:
            for stat in stats:
                if key_values is not None:
                    rolled = getattr(
                        past.groupby(key_values).rolling(
                            int(w), min_periods=min_periods
                        ),
                        stat,
                    )()
                    rolled = rolled.reset_index(level=0, drop=True).sort_index()
                else:
                    rolled = getattr(
                        past.rolling(int(w), min_periods=min_periods), stat
                    )()
                out[f"{column}_roll{w}_{stat}"] = rolled.to_numpy()
    return out


def build_features(
    df: pd.DataFrame,
    *,
    lag_columns: Sequence[str] | None = None,
    lags: Sequence[int] = (1, 7),
    windows: Sequence[int] = (7, 30),
    reservoir_pressure: float | Mapping[str, float] | None = None,
    by: str | None = "well",
) -> pd.DataFrame:
    """Build physics, lag and rolling features in one call.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table in the canonical schema.
    lag_columns : sequence of str, optional
        Columns for lag and rolling-mean features. Default: the canonical rate
        and pressure columns present in ``df``.
    lags : sequence of int, optional
        Lags in rows. Default (1, 7).
    windows : sequence of int, optional
        Rolling-mean windows in rows. Default (7, 30).
    reservoir_pressure : float or mapping, optional
        Passed to :func:`add_physics_features`.
    by : str or None, optional
        Well column. Default ``"well"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with all features.

    Examples
    --------
    >>> from wellcast.datasets import make_synthetic_field
    >>> field = make_synthetic_field(n_wells=1, years=1, n_injectors=0)
    >>> features = build_features(field.production)
    >>> {"gor", "bhp_lag1", "oil_rate_roll7_mean"} <= set(features.columns)
    True
    """
    out = add_physics_features(df, reservoir_pressure=reservoir_pressure, by=by)
    if lag_columns is None:
        candidates = ("oil_rate", "gas_rate", "water_rate", "whp", "bhp", "choke")
        lag_columns = [c for c in candidates if c in df.columns]
    if lag_columns:
        out = add_lag_features(out, lag_columns, lags, by=by)
        out = add_rolling_features(out, lag_columns, windows, ("mean",), by=by)
    return out


# =============================================================================
# End of module: wellcast.features
# =============================================================================
