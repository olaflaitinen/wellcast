# =============================================================================
# wellcast: tests for well and reservoir diagnostics
# -----------------------------------------------------------------------------
# Module:      tests/test_diagnostics.py
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

import numpy as np
import pandas as pd
import pytest

from wellcast import pvt
from wellcast.datasets import make_synthetic_field
from wellcast.diagnostics import (
    chan_plot_data,
    classify_water_mechanism,
    detect_anomalies,
    fit_ooip,
    hall_plot_data,
    hall_slope_change,
    havlena_odeh,
)
from wellcast.exceptions import DataValidationError, FitError, MissingDependencyError


@pytest.fixture(scope="module")
def field():  # type: ignore[no-untyped-def]
    return make_synthetic_field()


# -----------------------------------------------------------------------------
# Chan plot
# -----------------------------------------------------------------------------


def _water_history(wor: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(wor), freq="D")
    return pd.DataFrame(
        {"well": "A", "oil_rate": 100.0, "water_rate": 100.0 * wor}, index=idx
    )


def test_chan_classifies_channeling_and_coning() -> None:
    t = np.arange(1500.0)
    # Channeling: WOR grows ever faster after breakthrough (derivative rises).
    channel = _water_history(np.where(t > 200, 1e-3 * np.exp((t - 200) / 300), 0.0))
    # Coning: WOR rises quickly then flattens as the cone stabilizes.
    cone = _water_history(np.where(t > 200, 2.0 * (1 - np.exp(-(t - 200) / 150)), 0.0))
    cone["well"] = "B"
    chan = chan_plot_data(pd.concat([channel, cone]))
    assert set(chan["well"]) == {"A", "B"}
    result = classify_water_mechanism(chan)
    assert result.loc["A", "mechanism"] == "channeling"
    assert result.loc["B", "mechanism"] == "coning"


def test_chan_normal_and_insufficient_data() -> None:
    t = np.arange(1, 1001.0)
    # WOR proportional to t gives a constant derivative: slope ~ 0.
    linear = chan_plot_data(_water_history(1e-3 * t))
    assert (
        classify_water_mechanism(linear).loc["A", "mechanism"] == "normal displacement"
    )
    short = classify_water_mechanism(chan_plot_data(_water_history(1e-3 * t), n_bins=5))
    assert short.loc["A", "mechanism"] == "insufficient data"


def test_chan_edge_cases() -> None:
    dry = _water_history(np.zeros(50))
    empty = chan_plot_data(dry)
    assert empty.empty
    assert list(empty.columns) == ["well", "t", "wor", "wor_derivative"]
    no_well = chan_plot_data(
        _water_history(np.linspace(0.01, 1, 100)).drop(columns="well")
    )
    assert set(no_well["well"]) == {"all"}
    with pytest.raises(DataValidationError, match="n_bins"):
        chan_plot_data(dry, n_bins=2)
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        chan_plot_data(dry.reset_index(drop=True))


# -----------------------------------------------------------------------------
# Hall plot
# -----------------------------------------------------------------------------


def test_hall_plot_recovers_injectivity_loss(field) -> None:  # type: ignore[no-untyped-def]
    injector = field.production[field.production["well"] == "I-01"]
    truth = field.wells.loc["I-01"]
    hall = hall_plot_data(injector, reservoir_pressure=truth["initial_pressure"])
    assert hall["cum_injection"].is_monotonic_increasing
    change = hall_slope_change(hall)
    # Slope before the loss is the inverse injectivity index; after a 40
    # percent injectivity loss the slope rises by 1/0.6.
    assert change["slope_before"] == pytest.approx(
        1.0 / truth["injectivity_index"], rel=0.02
    )
    assert change["ratio"] == pytest.approx(1.0 / 0.6, rel=0.03)
    assert abs((change["break_date"] - truth["injectivity_loss_date"]).days) <= 10


def test_hall_wellhead_form_and_errors(field) -> None:  # type: ignore[no-untyped-def]
    injector = field.production[field.production["well"] == "I-01"]
    hall = hall_plot_data(injector, pressure_column="whp")
    assert hall["hall_integral"].iloc[-1] > 0
    with pytest.raises(FitError, match="injecting days"):
        hall_slope_change(hall.iloc[:20])
    single = hall_plot_data(injector.iloc[:1])
    assert len(single) == 1


# -----------------------------------------------------------------------------
# Havlena-Odeh material balance
# -----------------------------------------------------------------------------


def _solution_gas_drive(
    n_ooip: float, aquifer: np.ndarray | None = None
) -> pd.DataFrame:
    # Consistent synthetic history built from the PVT correlations: below
    # the bubble point with no gas cap, F = N Et (+ We).
    t_res, api, gg, rsi = 200.0, 35.0, 0.75, 600.0
    pb = pvt.bubble_point_standing(rsi, gg, api, t_res)
    p = np.linspace(pb, 0.5 * pb, 8)
    rs = np.asarray(pvt.solution_gor_standing(p, gg, api, t_res))
    bo = np.asarray(pvt.oil_fvf_standing(rs, gg, api, t_res))
    z = np.asarray(pvt.z_factor(p, t_res, gg))
    bg = np.asarray(pvt.gas_fvf(p, t_res, z)) / 5.615  # rb/scf
    # Rsi is taken from the correlation itself, so the initial survey has
    # exactly zero expansion (Standing's 1.2048 exponent is a rounded 1/0.83).
    boi, bgi, rsi = float(bo[0]), float(bg[0]), float(rs[0])
    eo = (bo - boi) + (rsi - rs) * bg
    rp = rsi * np.linspace(1.0, 3.0, p.size)
    we = np.zeros_like(p) if aquifer is None else aquifer
    np_cum = (n_ooip * eo + we) / (bo + (rp - rs) * bg)
    return havlena_odeh(
        p,
        np_cum,
        rp * np_cum,
        np.zeros_like(p),
        bo,
        bg,
        rs,
        boi=boi,
        bgi=bgi,
        rsi=rsi,
        cw=0.0,
        cf=0.0,
    )


def test_havlena_odeh_recovers_ooip() -> None:
    mb = _solution_gas_drive(25e6)
    assert list(mb.columns) == ["F", "Eo", "Eg", "Efw", "Et", "F_over_Et"]
    fit = fit_ooip(mb)
    assert fit["ooip"] == pytest.approx(25e6, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n_points"] == 7  # the initial survey has zero expansion


def test_havlena_odeh_aquifer_signature() -> None:
    mb = _solution_gas_drive(25e6, aquifer=np.linspace(0.0, 2e6, 8))
    ratio = mb["F_over_Et"].dropna()
    # F/Et = N + We/Et: with influx every point lies above N and the ratio is
    # no longer constant, so a through-origin fit overstates the oil in place.
    assert (ratio > 25e6).all()
    assert ratio.std() > 0.01 * ratio.mean()
    assert fit_ooip(mb)["ooip"] > 25e6


def test_havlena_odeh_compressibility_and_errors() -> None:
    mb = havlena_odeh(
        [3000.0, 2900.0],
        [0.0, 1e4],
        [0.0, 6e6],
        [0.0, 0.0],
        [1.3, 1.31],
        [8e-4, 8.3e-4],
        [600.0, 600.0],
        boi=1.3,
        bgi=8e-4,
        rsi=600.0,
    )
    # Above the bubble point (Rs constant) Efw carries the expansion.
    assert mb["Efw"].iloc[1] > 0
    with pytest.raises(DataValidationError, match="equal length"):
        havlena_odeh(
            [1.0, 2.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], boi=1, bgi=1, rsi=1
        )
    with pytest.raises(FitError, match="two surveys"):
        fit_ooip(mb.iloc[:1])


# -----------------------------------------------------------------------------
# Anomalies
# -----------------------------------------------------------------------------


def _coverage(field, anomalies: pd.DataFrame) -> np.ndarray:  # type: ignore[no-untyped-def]
    prod = field.production
    covered = np.zeros(len(prod), dtype=bool)
    wells = prod["well"].to_numpy()
    for _, period in anomalies.iterrows():
        covered |= (
            (wells == period["well"])
            & (prod.index >= period["start"])
            & (prod.index <= period["end"])
        )
    return covered


def test_rules_find_injected_anomalies(field) -> None:  # type: ignore[no-untyped-def]
    found = detect_anomalies(field.production, method="rules")
    assert list(found.columns) == ["well", "start", "end", "n_rows", "detector"]
    covered = _coverage(field, found)
    labels = field.labels
    for kind in ("spike", "flatline", "dropout"):
        assert covered[(labels["anomaly_type"] == kind).to_numpy()].all(), kind
    normal = (~labels["anomaly"] & ~labels["shut_in"]).to_numpy()
    assert covered[normal].mean() < 0.005


def test_isolation_forest_adds_a_bounded_fraction(field) -> None:  # type: ignore[no-untyped-def]
    both = detect_anomalies(field.production, contamination=0.01)
    assert "isolation_forest" in set(both["detector"])
    normal = (~field.labels["anomaly"] & ~field.labels["shut_in"]).to_numpy()
    assert _coverage(field, both)[normal].mean() < 0.03
    forest_only = detect_anomalies(field.production, method="isolation_forest")
    assert set(forest_only["detector"]) == {"isolation_forest"}


def test_anomaly_edge_cases() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    clean = pd.DataFrame({"whp": np.linspace(300, 280, 30)}, index=idx)
    assert detect_anomalies(clean, method="rules").empty
    # Too few usable rows for the forest: no flags rather than an error.
    assert detect_anomalies(clean.iloc[:10], method="isolation_forest").empty
    assert detect_anomalies(clean, method="isolation_forest", columns=["bhp"]).empty
    negative = clean.assign(whp=-clean["whp"])
    found = detect_anomalies(negative, method="rules", max_gap=0)
    assert found["detector"].tolist() == ["impossible_whp"]
    assert found["n_rows"].tolist() == [30]
    with pytest.raises(DataValidationError, match="unknown method"):
        detect_anomalies(clean, method="magic")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="contamination"):
        detect_anomalies(clean, contamination=0.6)
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        detect_anomalies(clean.reset_index(drop=True))


def test_forest_requires_ml_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.raises(MissingDependencyError, match=r"wellcast\[ml\]"):
        detect_anomalies(pd.DataFrame({"whp": [1.0, 2.0, 3.0]}, index=idx))


# =============================================================================
# End of module: tests/test_diagnostics.py
# =============================================================================
