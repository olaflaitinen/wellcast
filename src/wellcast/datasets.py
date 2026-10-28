# =============================================================================
# wellcast: synthetic oil field generator with ground-truth labels
# -----------------------------------------------------------------------------
# Module:      wellcast.datasets
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

"""Synthetic production data for examples, tests and the demo app.

`make_synthetic_field` simulates daily surface measurements for a small
oil field. The model is deliberately simple but physically consistent, so the
analysis modules can be checked against known ground truth:

* Liquid potential follows an Arps hyperbolic decline in calendar time and is
  scaled by the choke opening.
* Bottomhole pressure follows a linear IPR, ``pwf = pr - q_liquid / J``, and
  reservoir pressure falls linearly with cumulative liquid withdrawal.
* Wellhead pressure follows the Gilbert (1954) choke equation.
* The producing GOR equals the solution GOR above the bubble point and rises
  once reservoir pressure drops below it.
* Water cut grows exponentially towards a plateau after breakthrough.
* Shut-ins show zero rates and an exponential pressure build-up.
* An injector loses injectivity at a known date (visible on a Hall plot).
* Sensor anomalies are injected and labelled.

No third-party data is bundled with wellcast; all examples use this generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset

from wellcast.exceptions import DataValidationError

__all__ = ["ANOMALY_TYPES", "SyntheticField", "make_synthetic_field"]

FloatArray = npt.NDArray[np.float64]

ANOMALY_TYPES: tuple[str, ...] = ("spike", "flatline", "drift", "dropout")

_START = pd.Timestamp("2020-01-01")
# Hydrostatic gradients (psi/ft): light oil column in the tubing, fresh water
# in the injector tubing.
_OIL_GRADIENT = 0.30
_WATER_GRADIENT = 0.433
_MIN_PWF = 200.0


@dataclass(frozen=True)
class SyntheticField:
    """Output of `make_synthetic_field`.

    Attributes
    ----------
    production : pandas.DataFrame
        Measurements in the canonical schema (see `wellcast.io`),
        indexed by ``date`` and sorted by well and date.
    labels : pandas.DataFrame
        Ground truth aligned row by row with ``production``: ``well``,
        ``shut_in``, ``anomaly``, ``anomaly_type`` (empty string when none)
        and ``water_breakthrough``.
    wells : pandas.DataFrame
        True parameters per well, indexed by well name.
    """

    production: pd.DataFrame
    labels: pd.DataFrame
    wells: pd.DataFrame


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


def _arps_hyperbolic(
    t_days: FloatArray, qi: float, di_year: float, b: float
) -> FloatArray:
    # Arps (1945) hyperbolic rate, nominal decline di in 1/yr.
    di = di_year / 365.25
    return qi / np.power(1.0 + b * di * t_days, 1.0 / b)


def _gilbert_whp(
    q_liquid: FloatArray, glr_mscf_bbl: FloatArray, choke_64: FloatArray
) -> FloatArray:
    # Gilbert (1954): p_wh = 435 R^0.546 q / S^1.89 with p in psi, q in bbl/d,
    # R in Mscf/bbl and S in 64ths of an inch.
    return 435.0 * np.power(glr_mscf_bbl, 0.546) * q_liquid / np.power(choke_64, 1.89)


def _events(
    rng: np.random.Generator, n_days: int, count: int, min_len: int, max_len: int
) -> list[tuple[int, int]]:
    # Non-overlapping [start, stop) day intervals, kept away from day 0 so the
    # first weeks of each well are clean.
    events: list[tuple[int, int]] = []
    occupied = np.zeros(n_days, dtype=bool)
    occupied[:30] = True
    attempts = 0
    while len(events) < count and attempts < 200:
        attempts += 1
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(30, max(31, n_days - length)))
        stop = min(start + length, n_days)
        if occupied[max(0, start - 5) : stop + 5].any():
            continue
        occupied[start:stop] = True
        events.append((start, stop))
    return sorted(events)


# -----------------------------------------------------------------------------
# Producers
# -----------------------------------------------------------------------------


def _simulate_producer(
    rng: np.random.Generator, name: str, horizon: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    offset = int(rng.integers(0, 180))
    dates = horizon[offset:]
    n = len(dates)
    t = np.arange(n, dtype=np.float64)

    qi = float(rng.uniform(1200.0, 2500.0))
    di = float(rng.uniform(0.35, 0.9))
    b = float(rng.uniform(0.3, 0.9))
    pi = float(rng.uniform(3800.0, 5000.0))
    pb = float(rng.uniform(0.45, 0.7) * pi)
    j_index = float(rng.uniform(2.0, 5.0))
    rs = float(rng.uniform(450.0, 900.0))
    depth = float(rng.uniform(7000.0, 9500.0))
    t_res = float(rng.uniform(180.0, 230.0))
    api = float(rng.uniform(28.0, 40.0))
    gas_gravity = float(rng.uniform(0.65, 0.85))
    wc_max = float(rng.uniform(0.4, 0.85))
    breakthrough = int(rng.uniform(0.25, 0.65) * n)
    tau = float(rng.uniform(120.0, 400.0))

    # Choke schedule in 64ths: start at 32, then 2 to 4 operator changes.
    choke = np.full(n, 32.0)
    for day in np.sort(
        rng.choice(np.arange(60, n - 30), size=int(rng.integers(2, 5)), replace=False)
    ):
        choke[day:] = float(rng.choice([24.0, 28.0, 36.0, 40.0, 48.0]))
    choke_factor = np.clip(np.power(choke / 32.0, 0.6), 0.5, 1.4)

    potential = _arps_hyperbolic(t, qi, di, b) * choke_factor
    water_cut = np.where(
        t >= breakthrough, wc_max * (1.0 - np.exp(-(t - breakthrough) / tau)), 0.0
    )

    # Uptime: shut-ins (0 h) and partial downtime days (6 to 20 h).
    hours = np.full(n, 24.0)
    shut_in = np.zeros(n, dtype=bool)
    for start, stop in _events(rng, n, int(rng.integers(3, 7)), 2, 15):
        shut_in[start:stop] = True
    hours[shut_in] = 0.0
    partial = (~shut_in) & (rng.random(n) < 0.02)
    hours[partial] = rng.uniform(6.0, 20.0, partial.sum())

    # Reservoir pressure falls with cumulative liquid; the constant is chosen
    # so pressure reaches 50 to 65 percent of initial at the end of the run.
    liquid_full = potential * hours / 24.0
    cum = np.cumsum(liquid_full)
    drop = rng.uniform(0.35, 0.5) * pi
    pr = pi - drop * cum / cum[-1]

    # Deliverability limit: the IPR cannot pull pwf below _MIN_PWF.
    liquid_rate = np.minimum(potential, j_index * (pr - _MIN_PWF))
    liquid_rate = liquid_rate * hours / 24.0
    pwf = pr - liquid_rate / j_index

    # Shut-in build-up: pwf relaxes towards pr with a 1.5 day time constant.
    for start, stop in _events_from_mask(shut_in):
        elapsed = np.arange(1, stop - start + 1, dtype=np.float64)
        p0 = pwf[start - 1]
        pwf[start:stop] = pr[start:stop] - (pr[start:stop] - p0) * np.exp(
            -elapsed / 1.5
        )

    oil = liquid_rate * (1.0 - water_cut)
    water = liquid_rate * water_cut
    gor = np.where(pr >= pb, rs, rs * (1.0 + 2.5 * (pb - pr) / pb))
    gas = oil * gor / 1000.0
    glr = np.where(liquid_rate > 0, gas / np.maximum(liquid_rate, 1e-9), rs / 1000.0)
    whp = _gilbert_whp(liquid_rate, np.maximum(glr, 0.05), choke)
    whp_static = np.maximum(pwf - _OIL_GRADIENT * depth, 50.0)
    # A flowing well cannot hold more wellhead pressure than the bottomhole
    # pressure minus a minimum flowing gradient (about 0.15 psi/ft).
    whp = np.minimum(whp, np.maximum(pwf - 0.15 * depth, 50.0))
    whp = np.where(shut_in, whp_static, np.maximum(whp, 50.0))
    wht = 70.0 + (t_res - 70.0) * liquid_rate / (liquid_rate + 600.0)
    bht = np.full(n, t_res)

    # Measurement noise: 3 percent on rates, about 0.3 percent on pressures.
    def noise(scale: float) -> FloatArray:
        return rng.normal(1.0, scale, n)

    oil = oil * noise(0.03)
    water = water * noise(0.03)
    gas = gas * noise(0.03)
    pwf = pwf * noise(0.003)
    whp = whp * noise(0.005)
    wht = wht + rng.normal(0.0, 1.0, n)
    bht = bht + rng.normal(0.0, 0.5, n)
    kinds = _inject_anomalies(rng, oil, pwf, whp, ~shut_in & ~partial)

    frame = pd.DataFrame(
        {
            "well": name,
            "oil_rate": oil,
            "gas_rate": gas,
            "water_rate": water,
            "water_inj_rate": np.nan,
            "whp": whp,
            "wht": wht,
            "bhp": pwf,
            "bht": bht,
            "choke": choke,
            "on_stream_hours": hours,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    labels = pd.DataFrame(
        {
            "well": name,
            "shut_in": shut_in,
            "anomaly": kinds != "",
            "anomaly_type": kinds,
            "water_breakthrough": t >= breakthrough,
        },
        index=frame.index,
    )

    truth: dict[str, object] = {
        "type": "producer",
        "start_date": dates[0],
        "qi": qi,
        "di": di,
        "b": b,
        "initial_pressure": pi,
        "bubble_point": pb,
        "productivity_index": j_index,
        "rs": rs,
        "api": api,
        "gas_gravity": gas_gravity,
        "reservoir_temperature": t_res,
        "depth": depth,
        "wc_max": wc_max,
        "breakthrough_date": dates[breakthrough],
        "injectivity_loss_date": pd.NaT,
    }
    return frame, labels, truth


def _events_from_mask(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    # Convert a boolean mask into [start, stop) runs of True values.
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _inject_anomalies(
    rng: np.random.Generator,
    oil: FloatArray,
    bhp: FloatArray,
    whp: FloatArray,
    eligible: npt.NDArray[np.bool_],
) -> npt.NDArray[np.str_]:
    # Corrupt the arrays in place with one anomaly of each type, placed on
    # normal producing days, and return the per-day anomaly type ("" if none).
    n = len(oil)
    kinds = np.full(n, "", dtype="<U8")
    candidates = np.flatnonzero(eligible[: n - 40])
    candidates = candidates[candidates > 60]
    used = np.zeros(n, dtype=bool)
    for kind in ANOMALY_TYPES:
        length = {
            "spike": 1,
            "flatline": int(rng.integers(10, 21)),
            "drift": 30,
            "dropout": int(rng.integers(2, 6)),
        }[kind]
        for _ in range(100):
            start = int(rng.choice(candidates))
            stop = start + length
            if (
                not used[max(0, start - 10) : stop + 10].any()
                and eligible[start:stop].all()
            ):
                break
        else:  # pragma: no cover - only reachable for pathologically short runs
            continue
        used[start:stop] = True
        if kind == "spike":
            # Allocation or transcription error: 2 to 3 times the true rate.
            oil[start:stop] *= rng.uniform(2.0, 3.0)
        elif kind == "flatline":
            # Frozen downhole gauge repeating its last reading.
            bhp[start:stop] = bhp[start - 1]
        elif kind == "drift":
            # Gauge drift of 4 psi/day in either direction.
            bhp[start:stop] += np.arange(1, length + 1) * float(rng.choice([-4.0, 4.0]))
        else:
            # Wellhead transmitter dropout reporting zero while flowing.
            whp[start:stop] = 0.0
        kinds[start:stop] = kind
    return kinds


# -----------------------------------------------------------------------------
# Injectors
# -----------------------------------------------------------------------------


def _simulate_injector(
    rng: np.random.Generator, name: str, horizon: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    # Water injector with constant surface pressure targets. The injectivity
    # index drops by 40 percent at a known date (formation plugging), which
    # appears as a slope increase on a Hall plot.
    offset = int(rng.integers(90, 240))
    dates = horizon[offset:]
    n = len(dates)
    depth = float(rng.uniform(7000.0, 9000.0))
    pr = float(rng.uniform(3000.0, 3800.0))
    ii = float(rng.uniform(6.0, 10.0))
    loss = int(rng.uniform(0.4, 0.6) * n)
    ii_series = np.where(np.arange(n) >= loss, 0.6 * ii, ii)

    whp = np.full(n, float(rng.uniform(1200.0, 1500.0)))
    for day in np.sort(rng.choice(np.arange(30, n - 30), size=2, replace=False)):
        whp[day:] = whp[day - 1] + float(rng.uniform(-150.0, 150.0))
    # Tubing friction is neglected: pwf = whp + water column.
    pwf = whp + _WATER_GRADIENT * depth
    rate = np.maximum(ii_series * (pwf - pr), 0.0)

    hours = np.full(n, 24.0)
    shut_in = np.zeros(n, dtype=bool)
    for start, stop in _events(rng, n, 2, 2, 7):
        shut_in[start:stop] = True
    hours[shut_in] = 0.0
    rate[shut_in] = 0.0
    whp = np.where(shut_in, np.maximum(pr - _WATER_GRADIENT * depth, 0.0) + 50.0, whp)
    pwf = whp + _WATER_GRADIENT * depth

    frame = pd.DataFrame(
        {
            "well": name,
            "oil_rate": np.nan,
            "gas_rate": np.nan,
            "water_rate": np.nan,
            "water_inj_rate": rate * rng.normal(1.0, 0.02, n),
            "whp": whp * rng.normal(1.0, 0.003, n),
            "wht": 80.0 + rng.normal(0.0, 1.0, n),
            "bhp": pwf * rng.normal(1.0, 0.002, n),
            "bht": np.nan,
            "choke": np.nan,
            "on_stream_hours": hours,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    labels = pd.DataFrame(
        {
            "well": name,
            "shut_in": shut_in,
            "anomaly": False,
            "anomaly_type": "",
            "water_breakthrough": False,
        },
        index=frame.index,
    )
    truth: dict[str, object] = {
        "type": "injector",
        "start_date": dates[0],
        "initial_pressure": pr,
        "injectivity_index": ii,
        "depth": depth,
        "injectivity_loss_date": dates[loss],
        "breakthrough_date": pd.NaT,
    }
    return frame, labels, truth


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def make_synthetic_field(
    n_wells: int = 5,
    years: int = 5,
    freq: str = "D",
    seed: int = 42,
    *,
    n_injectors: int = 1,
) -> SyntheticField:
    """Simulate a small oil field with known ground truth.

    Parameters
    ----------
    n_wells : int, optional
        Number of producers, named ``P-01``, ``P-02`` and so on. Default 5.
    years : int, optional
        Length of the simulated period in years, starting 2020-01-01.
        Default 5.
    freq : str, optional
        Output frequency as a pandas offset alias. Data are simulated daily
        and averaged (labels: any) to coarser frequencies. Default ``"D"``.
    seed : int, optional
        Seed of the random generator; equal seeds give identical output.
        Default 42.
    n_injectors : int, optional
        Number of water injectors, named ``I-01`` and so on. Default 1.

    Returns
    -------
    SyntheticField
        Production data, row-aligned labels and true well parameters.

    Raises
    ------
    DataValidationError
        If ``n_wells`` < 1, ``n_injectors`` < 0 or ``years`` < 1.

    Notes
    -----
    Units follow the canonical schema: bbl/d for liquids, Mscf/d for gas,
    psi for pressure, degF for temperature and 64ths of an inch for the choke.
    See the module docstring for the physical model.

    Examples
    --------
    >>> from wellcast.datasets import make_synthetic_field
    >>> field = make_synthetic_field(n_wells=2, years=1, n_injectors=0)
    >>> sorted(field.production["well"].unique())
    ['P-01', 'P-02']
    >>> field.labels.index.equals(field.production.index)
    True
    """
    if n_wells < 1:
        raise DataValidationError("n_wells must be >= 1")
    if n_injectors < 0:
        raise DataValidationError("n_injectors must be >= 0")
    if years < 1:
        raise DataValidationError("years must be >= 1")

    rng = np.random.default_rng(seed)
    horizon = pd.date_range(_START, periods=round(years * 365.25), freq="D")
    frames: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    truth: dict[str, dict[str, object]] = {}
    for i in range(n_wells):
        name = f"P-{i + 1:02d}"
        f, lab, tr = _simulate_producer(rng, name, horizon)
        frames.append(f)
        labels.append(lab)
        truth[name] = tr
    for i in range(n_injectors):
        name = f"I-{i + 1:02d}"
        f, lab, tr = _simulate_injector(rng, name, horizon)
        frames.append(f)
        labels.append(lab)
        truth[name] = tr

    production = pd.concat(frames)
    label_frame = pd.concat(labels)
    if to_offset(freq) != to_offset("D"):
        production, label_frame = _aggregate(production, label_frame, freq)

    wells = pd.DataFrame.from_dict(truth, orient="index")
    wells.index.name = "well"
    return SyntheticField(production=production, labels=label_frame, wells=wells)


def _aggregate(
    production: pd.DataFrame, labels: pd.DataFrame, freq: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Average measurements and take "any" over labels within each period.
    prod_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    for name, group in production.groupby("well", sort=False):
        agg = group.drop(columns="well").resample(freq).mean()
        agg.insert(0, "well", str(name))
        prod_parts.append(agg)
        lab = labels[labels["well"] == name]
        flags = lab[["shut_in", "anomaly", "water_breakthrough"]].resample(freq).max()
        kinds = (
            lab["anomaly_type"]
            .resample(freq)
            .agg(lambda s: ",".join(sorted({k for k in s if k})))
        )
        lab_agg = flags.astype(bool).assign(anomaly_type=kinds.astype(str))
        lab_agg.insert(0, "well", str(name))
        label_parts.append(
            lab_agg[
                ["well", "shut_in", "anomaly", "anomaly_type", "water_breakthrough"]
            ]
        )
    return pd.concat(prod_parts), pd.concat(label_parts)


# =============================================================================
# End of module: wellcast.datasets
# =============================================================================
