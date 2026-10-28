# =============================================================================
# wellcast: well and reservoir diagnostics and anomaly detection
# -----------------------------------------------------------------------------
# Module:      wellcast.diagnostics
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

"""Diagnostic plots and checks for wells and reservoirs.

* Chan plot: water-oil ratio and its time derivative on log-log axes,
  distinguishing water coning from channeling (Chan, 1995).
* Hall plot: integrated injection pressure against cumulative injection,
  revealing injectivity changes (Hall, 1963).
* Havlena-Odeh material balance in straight-line form (Havlena and Odeh,
  1963).
* Anomaly detection combining an Isolation Forest with physical rules.

Units follow the canonical schema: rates in bbl/d (gas Mscf/d), pressures in
psi, time in days.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from wellcast._validation import require_columns, require_extra
from wellcast.clean import IMPOSSIBLE_RANGES, hampel_filter
from wellcast.exceptions import DataValidationError, FitError

__all__ = [
    "chan_plot_data",
    "classify_water_mechanism",
    "detect_anomalies",
    "fit_ooip",
    "hall_plot_data",
    "hall_slope_change",
    "havlena_odeh",
]

FloatArray = npt.NDArray[np.float64]


def _require_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("a DatetimeIndex is required")


def _days(index: pd.DatetimeIndex) -> FloatArray:
    return np.asarray((index - index[0]) / pd.Timedelta(days=1), dtype=np.float64)


# -----------------------------------------------------------------------------
# Chan plot
# -----------------------------------------------------------------------------


def chan_plot_data(
    df: pd.DataFrame,
    *,
    by: str | None = "well",
    n_bins: int = 40,
) -> pd.DataFrame:
    r"""Water-oil ratio and its derivative for a Chan diagnostic plot.

    Daily WOR is aggregated into ``n_bins`` logarithmically spaced time bins
    (median per bin) before differentiation. Chan plots are read on log time,
    and derivatives of raw daily data are dominated by noise.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table with ``oil_rate`` and ``water_rate`` and a
        ``DatetimeIndex``.
    by : str or None, optional
        Well column. Default ``"well"``.
    n_bins : int, optional
        Number of log-spaced time bins per well. Default 40.

    Returns
    -------
    pandas.DataFrame
        Columns ``well``, ``t`` (days since the well's first record, bin
        median), ``wor`` (bin median) and ``wor_derivative`` (1/day), one row
        per non-empty bin with both rates positive. Plot both against ``t``
        on log-log axes.

    Raises
    ------
    DataValidationError
        If columns are missing, the index is not a ``DatetimeIndex`` or
        ``n_bins`` < 3.

    Notes
    -----
    $\mathrm{WOR} = q_w / q_o$ and
    $\mathrm{WOR}' = d\,\mathrm{WOR} / dt$. Chan, K. S. (1995), "Water
    control diagnostic plots", SPE-30775-MS.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=101, freq="D")
    >>> t = np.arange(101.0)
    >>> df = pd.DataFrame({"oil_rate": 100.0, "water_rate": t}, index=idx)
    >>> chan = chan_plot_data(df, n_bins=5)
    >>> np.allclose(chan["wor_derivative"], 0.01)
    True
    """
    require_columns(df, ["oil_rate", "water_rate"])
    _require_datetime_index(df)
    if n_bins < 3:
        raise DataValidationError("n_bins must be >= 3")
    parts = []
    groups = (
        df.groupby(by, sort=True)
        if by is not None and by in df.columns
        else [("all", df)]
    )
    for name, group in groups:
        t = _days(cast(pd.DatetimeIndex, group.index))
        both = (group["oil_rate"] > 0) & (group["water_rate"] > 0)
        flowing = both.to_numpy() & (t > 0)
        if flowing.sum() < 3:
            continue
        t_flow = t[flowing]
        wor = (group["water_rate"] / group["oil_rate"]).to_numpy()[flowing]
        edges = np.geomspace(np.min(t_flow), np.max(t_flow) * (1 + 1e-9), n_bins + 1)
        which = np.digitize(t_flow, edges) - 1
        binned = (
            pd.DataFrame({"bin": which, "t": t_flow, "wor": wor})
            .groupby("bin")
            .median()
        )
        if len(binned) < 3:
            continue
        derivative = np.gradient(binned["wor"].to_numpy(), binned["t"].to_numpy())
        parts.append(
            pd.DataFrame(
                {
                    "well": str(name),
                    "t": binned["t"].to_numpy(),
                    "wor": binned["wor"].to_numpy(),
                    "wor_derivative": derivative,
                }
            )
        )
    if not parts:
        empty: pd.DataFrame = pd.DataFrame(
            columns=["well", "t", "wor", "wor_derivative"]
        )
        return empty
    result: pd.DataFrame = pd.concat(parts, ignore_index=True)
    return result


def classify_water_mechanism(
    chan: pd.DataFrame,
    *,
    slope_threshold: float = 0.3,
    min_points: int = 10,
) -> pd.DataFrame:
    r"""Classify excess water production from Chan plot data.

    After breakthrough, the log-log slope of $\mathrm{WOR}'$ against
    time separates the mechanisms (Chan, 1995):

    * **channeling** (high-permeability layer or near-wellbore channel):
      $\mathrm{WOR}'$ keeps increasing, slope > ``slope_threshold``;
    * **coning** (bottom water drawn up to the perforations):
      $\mathrm{WOR}'$ decreases after the cone builds, slope
      < ``-slope_threshold``;
    * otherwise **normal displacement**.

    Parameters
    ----------
    chan : pandas.DataFrame
        Output of `chan_plot_data`.
    slope_threshold : float, optional
        Minimum absolute log-log slope treated as a trend. Default 0.3.
    min_points : int, optional
        Minimum points (time bins) with positive derivative needed.
        Default 10.

    Returns
    -------
    pandas.DataFrame
        Index ``well``; columns ``mechanism``, ``derivative_slope`` and
        ``n_points``. Wells without enough data get ``"insufficient data"``.

    Notes
    -----
    This is a screening heuristic. Chan's own guidance is to read the curves
    together with completion and reservoir information.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> t = np.linspace(10, 1000, 100)
    >>> chan = pd.DataFrame(
    ...     {"well": "A", "t": t, "wor": 1e-4 * t**2, "wor_derivative": 2e-4 * t}
    ... )
    >>> classify_water_mechanism(chan).loc["A", "mechanism"]
    'channeling'
    """
    require_columns(chan, ["well", "t", "wor_derivative"])
    rows = []
    for name, group in chan.groupby("well", sort=True):
        valid = group[(group["wor_derivative"] > 0) & (group["t"] > 0)]
        if len(valid) < min_points:
            rows.append((str(name), "insufficient data", np.nan, len(valid)))
            continue
        slope = float(
            np.polyfit(np.log(valid["t"]), np.log(valid["wor_derivative"]), 1)[0]
        )
        if slope > slope_threshold:
            mechanism = "channeling"
        elif slope < -slope_threshold:
            mechanism = "coning"
        else:
            mechanism = "normal displacement"
        rows.append((str(name), mechanism, slope, len(valid)))
    table = pd.DataFrame(
        rows, columns=["well", "mechanism", "derivative_slope", "n_points"]
    ).set_index("well")
    result: pd.DataFrame = table
    return result


# -----------------------------------------------------------------------------
# Hall plot
# -----------------------------------------------------------------------------


def hall_plot_data(
    df: pd.DataFrame,
    *,
    pressure_column: str = "bhp",
    rate_column: str = "water_inj_rate",
    reservoir_pressure: float = 0.0,
) -> pd.DataFrame:
    r"""Hall plot coordinates for one injection well.

    Parameters
    ----------
    df : pandas.DataFrame
        Injector history with a ``DatetimeIndex``, sorted in time.
    pressure_column : str, optional
        Injection pressure, psi. Default ``"bhp"``; ``"whp"`` gives the
        classic wellhead form.
    rate_column : str, optional
        Injection rate, bbl/d. Default ``"water_inj_rate"``.
    reservoir_pressure : float, optional
        Average reservoir pressure subtracted from the injection pressure,
        psi. Default 0 (the original Hall form).

    Returns
    -------
    pandas.DataFrame
        Columns ``cum_injection`` (bbl) and ``hall_integral`` (psi day),
        indexed by date.

    Raises
    ------
    DataValidationError
        If columns are missing or the index is not a ``DatetimeIndex``.

    Notes
    -----
    $$
    \mathrm{Hall} = \int_0^t (p_{wf} - \bar{p}_r)\,dt
    \quad \text{vs.} \quad W_i = \int_0^t i\,dt
    $$

    The slope equals $1/I$ (inverse injectivity index) for steady
    radial flow; an increasing slope indicates plugging or rising skin, a
    decreasing slope fracturing or stimulation. Hall, H. N. (1963), "How to
    analyze waterflood injection well performance", World Oil, October,
    128-130.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=3, freq="D")
    >>> df = pd.DataFrame(
    ...     {"bhp": [5000.0] * 3, "water_inj_rate": [1000.0] * 3}, index=idx
    ... )
    >>> hall_plot_data(df, reservoir_pressure=3000.0)["hall_integral"].tolist()
    [2000.0, 4000.0, 6000.0]
    """
    require_columns(df, [pressure_column, rate_column])
    _require_datetime_index(df)
    t = _days(cast(pd.DatetimeIndex, df.index))
    step = np.diff(t, prepend=t[0] - (np.median(np.diff(t)) if len(t) > 1 else 1.0))
    rate = df[rate_column].fillna(0.0).clip(lower=0.0).to_numpy()
    dp = (df[pressure_column] - reservoir_pressure).fillna(0.0).to_numpy()
    # Shut-in days contribute neither volume nor pressure-time.
    active = rate > 0
    table: pd.DataFrame = pd.DataFrame(
        {
            "cum_injection": np.cumsum(rate * step),
            "hall_integral": np.cumsum(np.where(active, dp, 0.0) * step),
        },
        index=df.index,
    )
    return table


def hall_slope_change(hall: pd.DataFrame, *, min_segment: int = 30) -> pd.Series:
    """Locate the strongest slope change on a Hall plot.

    Fits two straight lines meeting at a breakpoint and picks the breakpoint
    with the smallest total squared error.

    Parameters
    ----------
    hall : pandas.DataFrame
        Output of `hall_plot_data`.
    min_segment : int, optional
        Minimum points on each side of the breakpoint. Default 30.

    Returns
    -------
    pandas.Series
        ``break_date``, ``slope_before`` and ``slope_after`` (psi day/bbl)
        and ``ratio`` (after / before; above 1 suggests declining
        injectivity).

    Raises
    ------
    FitError
        If there are fewer than ``2 * min_segment`` points.
    """
    data = hall[hall["cum_injection"].diff().fillna(1.0) > 0]
    if len(data) < 2 * min_segment:
        raise FitError(f"need at least {2 * min_segment} injecting days")
    x = data["cum_injection"].to_numpy()
    y = data["hall_integral"].to_numpy()
    best: tuple[float, int, float, float] | None = None
    # Coarse search on at most 200 candidate breakpoints keeps this O(n).
    candidates = np.unique(
        np.linspace(min_segment, len(x) - min_segment, 200).astype(int)
    )
    for k in candidates:
        s1, c1 = np.polyfit(x[:k], y[:k], 1)
        s2, c2 = np.polyfit(x[k:], y[k:], 1)
        sse = float(
            np.sum((y[:k] - (s1 * x[:k] + c1)) ** 2)
            + np.sum((y[k:] - (s2 * x[k:] + c2)) ** 2)
        )
        if best is None or sse < best[0]:
            best = (sse, int(k), float(s1), float(s2))
    assert best is not None
    _, k, s1, s2 = best
    result: pd.Series = pd.Series(
        {
            "break_date": data.index[k],
            "slope_before": s1,
            "slope_after": s2,
            "ratio": s2 / s1 if s1 != 0 else np.nan,
        },
        name="hall",
    )
    return result


# -----------------------------------------------------------------------------
# Havlena-Odeh material balance
# -----------------------------------------------------------------------------


def havlena_odeh(
    pressure: npt.ArrayLike,
    np_cum: npt.ArrayLike,
    gp_cum: npt.ArrayLike,
    wp_cum: npt.ArrayLike,
    bo: npt.ArrayLike,
    bg: npt.ArrayLike,
    rs: npt.ArrayLike,
    *,
    boi: float,
    bgi: float,
    rsi: float,
    bw: float = 1.0,
    winj_cum: npt.ArrayLike = 0.0,
    gas_cap_ratio: float = 0.0,
    swi: float = 0.2,
    cw: float = 3e-6,
    cf: float = 4e-6,
) -> pd.DataFrame:
    r"""Material balance terms in Havlena-Odeh straight-line form.

    Parameters
    ----------
    pressure : array_like
        Average reservoir pressure at each survey, psia (first value is the
        initial pressure).
    np_cum, gp_cum, wp_cum : array_like
        Cumulative oil (STB), gas (scf) and water (STB) production.
    bo : array_like
        Oil formation volume factor, rb/STB.
    bg : array_like
        Gas formation volume factor, rb/scf.
    rs : array_like
        Solution gas-oil ratio, scf/STB.
    boi, bgi, rsi : float
        Initial Bo (rb/STB), Bg (rb/scf) and Rs (scf/STB).
    bw : float, optional
        Water formation volume factor, rb/STB. Default 1.0.
    winj_cum : array_like, optional
        Cumulative water injection, STB. Default 0.
    gas_cap_ratio : float, optional
        Initial gas cap to oil zone volume ratio m. Default 0.
    swi : float, optional
        Connate water saturation, fraction. Default 0.2.
    cw, cf : float, optional
        Water and formation compressibility, 1/psi. Defaults 3e-6, 4e-6.

    Returns
    -------
    pandas.DataFrame
        Columns ``F`` (underground withdrawal, rb), ``Eo``, ``Eg``, ``Efw``
        and ``Et`` (expansion terms, rb/STB) and ``F_over_Et`` (STB).

    Raises
    ------
    DataValidationError
        If array lengths differ.

    Notes
    -----
    With $R_p = G_p / N_p$:

    $$
    \begin{aligned}
    F &= N_p [B_o + (R_p - R_s) B_g] + W_p B_w - W_{inj} B_w \\
    E_o &= (B_o - B_{oi}) + (R_{si} - R_s) B_g \\
    E_g &= B_{oi} (B_g / B_{gi} - 1) \\
    E_{fw} &= (1 + m) B_{oi} \frac{c_w S_{wi} + c_f}{1 - S_{wi}} \Delta p \\
    F &= N (E_o + m E_g + E_{fw}) + W_e
    \end{aligned}
    $$

    Without aquifer influx $W_e$, $F$ against
    $E_t$ is a straight line through the origin with slope $N$
    (see `fit_ooip`) and $F/E_t$ is constant. Since
    $F/E_t = N + W_e/E_t$, influx puts the points above $N$ and
    makes the ratio vary with time, which signals aquifer support (or a
    wrong drive mechanism). Havlena, D. and Odeh, A. S. (1963), "The material
    balance as an equation of a straight line", JPT 15(8), 896-900,
    SPE-559-PA.
    """
    arrays = {
        name: np.atleast_1d(np.asarray(value, dtype=np.float64))
        for name, value in {
            "pressure": pressure,
            "np": np_cum,
            "gp": gp_cum,
            "wp": wp_cum,
            "bo": bo,
            "bg": bg,
            "rs": rs,
        }.items()
    }
    sizes = {a.size for a in arrays.values()}
    if len(sizes) != 1:
        raise DataValidationError("all material balance arrays must have equal length")
    p, n_p, g_p, w_p = arrays["pressure"], arrays["np"], arrays["gp"], arrays["wp"]
    bo_, bg_, rs_ = arrays["bo"], arrays["bg"], arrays["rs"]
    w_inj = np.broadcast_to(np.asarray(winj_cum, dtype=np.float64), p.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = np.where(n_p > 0, g_p / n_p, rs_)
        f = n_p * (bo_ + (rp - rs_) * bg_) + w_p * bw - w_inj * bw
        eo = (bo_ - boi) + (rsi - rs_) * bg_
        eg = boi * (bg_ / bgi - 1.0)
        efw = (1.0 + gas_cap_ratio) * boi * (cw * swi + cf) / (1.0 - swi) * (p[0] - p)
        et = eo + gas_cap_ratio * eg + efw
        ratio = np.where(et > 0, f / et, np.nan)
    table: pd.DataFrame = pd.DataFrame(
        {"F": f, "Eo": eo, "Eg": eg, "Efw": efw, "Et": et, "F_over_Et": ratio}
    )
    return table


def fit_ooip(mb: pd.DataFrame) -> pd.Series:
    """Original oil in place from a Havlena-Odeh table (no aquifer).

    Parameters
    ----------
    mb : pandas.DataFrame
        Output of `havlena_odeh`.

    Returns
    -------
    pandas.Series
        ``ooip`` (STB, least-squares slope through the origin of F against
        Et), ``r2`` (uncentered) and ``n_points``.

    Raises
    ------
    FitError
        If fewer than two surveys have positive expansion.
    """
    data = mb[(mb["Et"] > 0) & mb["F"].notna()]
    if len(data) < 2:
        raise FitError("need at least two surveys with positive expansion")
    x = data["Et"].to_numpy()
    y = data["F"].to_numpy()
    n = float(np.dot(x, y) / np.dot(x, x))
    r2 = 1.0 - float(np.sum((y - n * x) ** 2)) / float(np.sum(y**2))
    result: pd.Series = pd.Series(
        {"ooip": n, "r2": r2, "n_points": len(data)}, name="material_balance"
    )
    return result


# -----------------------------------------------------------------------------
# Anomaly detection
# -----------------------------------------------------------------------------

_SIGNAL_COLUMNS = ("oil_rate", "gas_rate", "water_rate", "whp", "bhp", "wht")


def _rule_flags(group: pd.DataFrame, flat_run: int) -> dict[str, pd.Series]:
    flags: dict[str, pd.Series] = {}
    flowing = (
        group["on_stream_hours"] > 0
        if "on_stream_hours" in group.columns
        else pd.Series(True, index=group.index)
    )
    for column, (low, high) in IMPOSSIBLE_RANGES.items():
        if column in group.columns:
            flags[f"impossible_{column}"] = (group[column] < low) | (
                group[column] > high
            )
    for column in ("whp", "bhp"):
        if column not in group.columns:
            continue
        # A pressure of exactly zero while the well flows is a transmitter
        # dropout, not a physical state.
        flags[f"dropout_{column}"] = flowing & group[column].eq(0.0)
        values = group[column].where(flowing)
        run = values.ne(values.shift()).cumsum()
        size = run.map(run.value_counts())
        flags[f"flatline_{column}"] = (
            values.notna() & values.ne(0.0) & (size >= flat_run)
        )
    for column in ("oil_rate", "gas_rate", "water_rate"):
        if column in group.columns and group[column].notna().sum() > 7:
            # Allocation and transcription errors are upward multiples of the
            # true rate; partial-uptime days and choke cuts are legitimate
            # drops. Require a Hampel outlier at least 50 percent above the
            # local median.
            local = group[column].rolling(7, center=True, min_periods=1).median()
            upward = group[column] > 1.5 * local
            flags[f"spike_{column}"] = (
                flowing & upward & hampel_filter(group[column], 7, 4.0)
            )
    return flags


def _periods(mask: pd.Series, max_gap: int) -> list[tuple[int, int]]:
    # Merge flagged positions closer than max_gap rows into [start, stop].
    positions = np.flatnonzero(mask.to_numpy())
    if positions.size == 0:
        return []
    periods = [[int(positions[0]), int(positions[0])]]
    for p in positions[1:]:
        if p - periods[-1][1] <= max_gap + 1:
            periods[-1][1] = int(p)
        else:
            periods.append([int(p), int(p)])
    return [(a, b) for a, b in periods]


def detect_anomalies(
    df: pd.DataFrame,
    *,
    method: Literal["both", "rules", "isolation_forest"] = "both",
    columns: Sequence[str] | None = None,
    by: str | None = "well",
    contamination: float = 0.01,
    flat_run: int = 5,
    max_gap: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Find anomalous periods with rules and an Isolation Forest.

    Rules (physically motivated, always explainable):

    * ``impossible_<col>``: value outside `wellcast.clean.IMPOSSIBLE_RANGES`;
    * ``dropout_<col>``: pressure exactly zero while the well is on stream;
    * ``flatline_<col>``: at least ``flat_run`` identical non-zero pressure
      readings while flowing (frozen gauge);
    * ``spike_<col>``: Hampel outlier (window 7, 4 sigma) in a rate that is
      also at least 50 percent above the local median.

    The Isolation Forest (Liu, Ting and Zhou, 2008) scores each producing day
    of each well on standardized signals and their day-to-day changes, and
    flags the most isolated ``contamination`` fraction. It catches
    multivariate inconsistencies that no single rule covers.

    Parameters
    ----------
    df : pandas.DataFrame
        Production table with a ``DatetimeIndex``, sorted within each well.
    method : {"both", "rules", "isolation_forest"}, optional
        Detectors to run. Default ``"both"``; the forest needs the ``ml``
        extra.
    columns : sequence of str, optional
        Signals for the forest. Default: canonical rates and pressures
        present.
    by : str or None, optional
        Well column. Default ``"well"``.
    contamination : float, optional
        Expected anomaly fraction for the forest. Default 0.01.
    flat_run : int, optional
        Minimum identical readings for a flatline. Default 5.
    max_gap : int, optional
        Flagged rows separated by at most this many clean rows are merged
        into one period. Default 1.
    random_state : int, optional
        Seed of the forest. Default 0.

    Returns
    -------
    pandas.DataFrame
        One row per period with ``well``, ``start``, ``end``, ``n_rows`` and
        ``detector`` (rule name or ``"isolation_forest"``), sorted by well
        and start.

    Raises
    ------
    DataValidationError
        If the index is not a ``DatetimeIndex`` or ``method`` is unknown.
    MissingDependencyError
        If the forest is requested without scikit-learn.

    Notes
    -----
    Slow gauge drift (a few psi per day) is hidden in day-to-day noise and is
    not reliably found by either detector. Monitor it instead through the
    residual between the measured value and a virtual flow meter prediction
    from `wellcast.ml.Predictor` trained on a trusted period.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=6, freq="D")
    >>> df = pd.DataFrame({"whp": [300.0, 0.0, 0.0, 305.0, 301.0, 299.0]}, index=idx)
    >>> detect_anomalies(df, method="rules")[["start", "end", "detector"]]
           start        end     detector
    0 2024-01-02 2024-01-03  dropout_whp
    """
    _require_datetime_index(df)
    if method not in ("both", "rules", "isolation_forest"):
        raise DataValidationError(f"unknown method {method!r}")
    if not 0.0 < contamination < 0.5:
        raise DataValidationError("contamination must be in (0, 0.5)")
    signals = (
        list(columns)
        if columns is not None
        else [c for c in _SIGNAL_COLUMNS if c in df.columns]
    )
    groups = (
        df.groupby(by, sort=True)
        if by is not None and by in df.columns
        else [("all", df)]
    )
    records: list[dict[str, Any]] = []
    forest_cls: Any = None
    preprocessing: Any = None
    if method in ("both", "isolation_forest"):
        forest_cls = require_extra("sklearn.ensemble", "ml").IsolationForest
        preprocessing = require_extra("sklearn.preprocessing", "ml")

    for name, group in groups:
        flags: dict[str, pd.Series] = {}
        if method in ("both", "rules"):
            flags.update(_rule_flags(group, flat_run))
        if forest_cls is not None and signals:
            flags["isolation_forest"] = _forest_flags(
                group, signals, forest_cls, preprocessing, contamination, random_state
            )
        for detector, mask in flags.items():
            for start, stop in _periods(mask.fillna(False).astype(bool), max_gap):
                records.append(
                    {
                        "well": str(name),
                        "start": group.index[start],
                        "end": group.index[stop],
                        "n_rows": stop - start + 1,
                        "detector": detector,
                    }
                )
    columns_out = ["well", "start", "end", "n_rows", "detector"]
    table = pd.DataFrame(records, columns=columns_out)
    if not table.empty:
        table = table.sort_values(["well", "start", "detector"], ignore_index=True)
    if by is None or by not in df.columns:
        table = table.drop(columns="well")
    result: pd.DataFrame = table
    return result


def _forest_flags(
    group: pd.DataFrame,
    signals: list[str],
    forest_cls: Any,
    preprocessing: Any,
    contamination: float,
    random_state: int,
) -> pd.Series:
    flags: pd.Series = pd.Series(False, index=group.index)
    present = [c for c in signals if c in group.columns and group[c].notna().any()]
    if not present:
        return flags
    flowing = (
        group["on_stream_hours"] > 0
        if "on_stream_hours" in group.columns
        else pd.Series(True, index=group.index)
    )
    values = group[present].astype("float64")
    # Levels and day-to-day changes: a frozen or drifting gauge is often
    # unremarkable in level but inconsistent with the other signals' changes.
    features = pd.concat([values, values.diff().add_suffix("_diff")], axis=1)
    usable = flowing & features.notna().all(axis=1)
    if usable.sum() < 20:
        return flags
    x = preprocessing.RobustScaler().fit_transform(features[usable].to_numpy())
    forest = forest_cls(contamination=contamination, random_state=random_state)
    labels = forest.fit_predict(x)
    flags[usable] = labels == -1
    return flags


# =============================================================================
# End of module: wellcast.diagnostics
# =============================================================================
