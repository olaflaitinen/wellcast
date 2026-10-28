# =============================================================================
# wellcast: tests for Arps decline curve analysis
# -----------------------------------------------------------------------------
# Module:      tests/test_dca.py
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
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad

from wellcast.datasets import make_synthetic_field
from wellcast.dca import (
    DeclineModel,
    arps_cumulative,
    arps_rate,
    arps_time_to_rate,
    fit_many,
)
from wellcast.exceptions import DataValidationError, FitError, MissingDependencyError

YEAR = 365.25

# -----------------------------------------------------------------------------
# Arps equations (Arps, 1945)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("b", [0.0, 0.3, 0.5, 1.0, 1.5])
def test_cumulative_equals_integral_of_rate(b: float) -> None:
    # Closed-form Np(t) must equal the numerical integral of q(t).
    t_end = 2000.0
    numeric, _ = quad(lambda t: float(arps_rate(t, 1000.0, 0.7, b)), 0.0, t_end)
    assert arps_cumulative(t_end, 1000.0, 0.7, b) == pytest.approx(numeric, rel=1e-8)


def test_textbook_identities() -> None:
    # Standard Arps relations between rate and cumulative (Arps, 1945):
    #   exponential: Np = (qi - q) / D
    #   harmonic:    Np = (qi / D) ln(qi / q)
    #   hyperbolic:  Np = qi^b / ((1 - b) D) (qi^(1-b) - q^(1-b))
    qi, di, t = 1000.0, 0.5, 900.0
    d = di / YEAR
    q_exp = arps_rate(t, qi, di, 0.0)
    assert arps_cumulative(t, qi, di, 0.0) == pytest.approx((qi - q_exp) / d)
    q_har = arps_rate(t, qi, di, 1.0)
    assert arps_cumulative(t, qi, di, 1.0) == pytest.approx(qi / d * np.log(qi / q_har))
    b = 0.4
    q_hyp = arps_rate(t, qi, di, b)
    expected = qi**b / ((1 - b) * d) * (qi ** (1 - b) - q_hyp ** (1 - b))
    assert arps_cumulative(t, qi, di, b) == pytest.approx(expected)


def test_exponential_decline_hand_calculation() -> None:
    # qi = 1000 bbl/d, di = 0.5/yr: after one year q = 1000 e^-0.5 = 606.53 and
    # the effective annual decline is 1 - e^-0.5 = 39.35 percent.
    assert arps_rate(YEAR, 1000.0, 0.5, 0.0) == pytest.approx(606.5307)
    model = DeclineModel.from_params(1000.0, 0.5, 0.0)
    assert model.summary()["de"] == pytest.approx(1 - np.exp(-0.5))


@pytest.mark.parametrize("b", [0.0, 0.5, 1.0])
def test_time_to_rate_inverts_rate(b: float) -> None:
    t = arps_time_to_rate(250.0, 1000.0, 0.8, b)
    assert arps_rate(t, 1000.0, 0.8, b) == pytest.approx(250.0)
    assert arps_time_to_rate(2000.0, 1000.0, 0.8, b) == 0.0


def test_b_near_limits_is_continuous() -> None:
    t = np.array([100.0, 1000.0])
    np.testing.assert_allclose(
        arps_rate(t, 1e3, 0.6, 1e-7), arps_rate(t, 1e3, 0.6, 0.0)
    )
    # For small b the hyperbolic rate differs from the exponential by about
    # b (D t)^2 / 2 in relative terms (second-order expansion of ln(1 + bDt)).
    dt_max = 0.6 / 365.25 * 1000.0
    np.testing.assert_allclose(
        arps_rate(t, 1e3, 0.6, 1e-3),
        arps_rate(t, 1e3, 0.6, 0.0),
        rtol=1.1 * 1e-3 * dt_max**2 / 2,
    )
    np.testing.assert_allclose(
        arps_cumulative(t, 1e3, 0.6, 1.0 + 1e-7), arps_cumulative(t, 1e3, 0.6, 1.0)
    )


@given(
    qi=st.floats(min_value=1.0, max_value=1e5),
    di=st.floats(min_value=0.01, max_value=5.0),
    b=st.floats(min_value=0.0, max_value=2.0),
    t1=st.floats(min_value=0.0, max_value=1e4),
    t2=st.floats(min_value=0.0, max_value=1e4),
)
def test_rate_decreases_and_cumulative_increases(
    qi: float, di: float, b: float, t1: float, t2: float
) -> None:
    lo, hi = sorted((t1, t2))
    assert arps_rate(hi, qi, di, b) <= arps_rate(lo, qi, di, b) * (1 + 1e-12)
    assert (
        arps_cumulative(lo, qi, di, b)
        <= arps_cumulative(hi, qi, di, b) * (1 + 1e-12) + 1e-9
    )
    assert arps_cumulative(hi, qi, di, b) <= qi * hi * (1 + 1e-9) + 1e-9


# -----------------------------------------------------------------------------
# Fitting and model selection
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b", "expected"), [(0.0, "exponential"), (0.6, "hyperbolic"), (1.0, "harmonic")]
)
def test_aic_selects_true_model(b: float, expected: str) -> None:
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 1500.0)
    q = arps_rate(t, 1200.0, 0.9, b) * rng.lognormal(0.0, 0.05, t.size)
    model = DeclineModel().fit(q, time=t)
    assert model.model_ == expected
    assert set(model.aic_) == {"exponential", "harmonic", "hyperbolic"}
    assert model.qi_ == pytest.approx(1200.0, rel=0.02)
    assert model.di_ == pytest.approx(0.9, rel=0.05)
    assert model.b_ == pytest.approx(b, abs=0.05)


def test_fit_on_dated_series_skips_ramp_up_and_gaps() -> None:
    dates = pd.date_range("2021-01-01", periods=900, freq="D")
    t = np.arange(900.0)
    q = np.where(t < 30, 40.0 * t, arps_rate(t - 30.0, 1200.0, 0.7, 0.4))
    q[100:110] = 0.0  # shut-in
    q[200:205] = np.nan  # reporting gap
    model = DeclineModel("hyperbolic").fit(pd.Series(q, index=dates))
    assert model.start_ == pd.Timestamp("2021-01-31")
    assert model.t_offset_ == 30.0
    assert model.n_obs_ == 870 - 15
    assert model.b_ == pytest.approx(0.4, abs=1e-3)
    predicted = model.predict(pd.DatetimeIndex(["2021-01-31", "2022-01-31"]))
    np.testing.assert_allclose(
        predicted, arps_rate([0.0, 365.0], 1200.0, 0.7, 0.4), rtol=1e-4
    )


def test_robust_loss_resists_spikes() -> None:
    t = np.arange(0.0, 600.0)
    q = arps_rate(t, 1000.0, 0.6, 0.0)
    q[::50] *= 3.0  # allocation spikes
    robust = DeclineModel("exponential").fit(q, time=t)
    plain = DeclineModel("exponential", loss="linear").fit(q, time=t)
    assert abs(robust.qi_ - 1000.0) < abs(plain.qi_ - 1000.0)
    assert robust.qi_ == pytest.approx(1000.0, rel=0.01)


def test_default_time_axis_and_forecast() -> None:
    q = arps_rate(np.arange(50.0), 500.0, 2.0, 0.0)
    model = DeclineModel("exponential", from_peak=False).fit(q)
    forecast = model.forecast(100, econ_limit=300.0)
    assert forecast["t"].iloc[0] == 50.0
    assert forecast["rate"].min() >= 300.0
    assert isinstance(forecast.index, pd.RangeIndex)
    assert forecast["cumulative"].is_monotonic_increasing


def test_eur_remaining_and_summary() -> None:
    field = make_synthetic_field(n_wells=1, n_injectors=0)
    rates = field.production["oil_rate"]
    model = DeclineModel().fit(rates)
    eur = model.eur(20.0)
    assert eur > float(model.cumulative(model.t_last_))
    assert model.remaining(20.0) == pytest.approx(
        eur - float(model.cumulative(model.t_last_))
    )
    assert model.eur(20.0, max_years=1.0) < eur
    assert model.remaining(1e6) == 0.0
    summary = model.summary(econ_limit=20.0)
    assert {"model", "qi", "di", "de", "b", "eur", "remaining", "aic"} <= set(
        summary.index
    )
    forecast = model.forecast(30)
    assert isinstance(forecast.index, pd.DatetimeIndex)
    assert forecast.index[0] == rates.index[-1] + pd.Timedelta(days=1)


# -----------------------------------------------------------------------------
# Uncertainty
# -----------------------------------------------------------------------------


def test_bootstrap_bands_are_ordered() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 800.0, 2.0)
    q = arps_rate(t, 900.0, 0.8, 0.5) * rng.lognormal(0.0, 0.1, t.size)
    model = DeclineModel("hyperbolic").fit(q, time=t).bootstrap(50, seed=1)
    bands = model.predict_interval(np.array([0.0, 500.0, 3000.0]))
    assert (bands["p90"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p10"]).all()
    eur = model.eur_quantiles(10.0)
    assert eur["p90"] <= eur["p50"] <= eur["p10"]
    capped = model.eur_quantiles(10.0, max_years=2.0)
    assert capped["p50"] < eur["p50"]
    # The same seed reproduces the same samples.
    again = DeclineModel("hyperbolic").fit(q, time=t).bootstrap(50, seed=1)
    pd.testing.assert_series_equal(again.eur_quantiles(10.0), eur)


def test_bootstrap_two_parameter_models() -> None:
    t = np.arange(0.0, 500.0)
    q = arps_rate(t, 800.0, 0.5, 1.0) * np.random.default_rng(0).lognormal(
        0, 0.05, t.size
    )
    model = DeclineModel("harmonic").fit(q, time=t).bootstrap(20)
    assert model.eur_quantiles(10.0)["p50"] > 0


def test_bootstrap_requirements() -> None:
    model = DeclineModel.from_params(1000.0, 0.5, 0.5)
    with pytest.raises(FitError, match="fitted on data"):
        model.bootstrap()
    with pytest.raises(FitError, match="bootstrap"):
        model.predict_interval([0.0])
    fitted = DeclineModel().fit(arps_rate(np.arange(30.0), 100.0, 1.0, 0.0))
    with pytest.raises(DataValidationError, match="n_samples"):
        fitted.bootstrap(5)


# -----------------------------------------------------------------------------
# Construction, errors and reporting
# -----------------------------------------------------------------------------


def test_from_params_names_and_dates() -> None:
    assert DeclineModel.from_params(1.0, 1.0, 1.0).model_ == "harmonic"
    assert DeclineModel.from_params(1.0, 1.0, 0.5).model_ == "hyperbolic"
    dated = DeclineModel.from_params(1000.0, 0.5, 0.0, start="2024-01-01")
    assert dated.predict(pd.DatetimeIndex(["2024-01-01"]))[0] == pytest.approx(1000.0)
    with pytest.raises(DataValidationError, match="start date"):
        DeclineModel.from_params(1000.0, 0.5, 0.0).predict(
            pd.DatetimeIndex(["2024-01-01"])
        )
    with pytest.raises(DataValidationError, match="non-negative"):
        DeclineModel.from_params(1000.0, 0.5, -0.1)


def test_unfitted_and_invalid_configuration() -> None:
    model = DeclineModel()
    assert "unfitted" in repr(model)
    for call in (
        lambda: model.predict([1.0]),
        lambda: model.summary(),
        lambda: model.eur(1.0),
    ):
        with pytest.raises(FitError, match="not fitted"):
            call()
    with pytest.raises(DataValidationError, match="unknown model"):
        DeclineModel("linear")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="b_bounds"):
        DeclineModel(b_bounds=(0.0, 1.0))


def test_fit_input_errors() -> None:
    with pytest.raises(FitError, match="at least 4"):
        DeclineModel().fit([100.0, 90.0, 0.0, np.nan, 80.0])
    with pytest.raises(DataValidationError, match="equal length"):
        DeclineModel().fit([1.0, 2.0, 3.0, 4.0], time=[0.0, 1.0])
    series = pd.Series(
        [4.0, 3.0, 2.0, 1.0], index=pd.date_range("2024-01-01", periods=4)
    )
    with pytest.raises(DataValidationError, match="either"):
        DeclineModel().fit(series, time=[0.0, 1.0, 2.0, 3.0])


def test_repr_of_fitted_model() -> None:
    model = DeclineModel.from_params(1000.0, 0.5, 0.0)
    assert repr(model) == "DeclineModel(exponential, qi=1000, di=0.5/yr, b=0)"


# -----------------------------------------------------------------------------
# Batch fitting and plotting
# -----------------------------------------------------------------------------


def test_fit_many_reports_failures() -> None:
    field = make_synthetic_field(n_wells=3)
    table = fit_many(field.production, econ_limit=20.0)
    assert table.index.name == "well"
    assert list(table.index) == ["I-01", "P-01", "P-02", "P-03"]
    assert (table.loc[["P-01", "P-02", "P-03"], "status"] == "ok").all()
    # The injector has no oil rate, so it fails with a clear message.
    assert "at least 4" in table.loc["I-01", "status"]
    assert (table.loc["P-01":, "eur"] > 0).all()
    row = table.loc["P-01"]
    rebuilt = DeclineModel.from_params(
        row["qi"], row["di"], row["b"], start=row["start"]
    )
    assert rebuilt.eur(20.0) == pytest.approx(row["eur"])


def test_fit_many_validation() -> None:
    df = pd.DataFrame({"well": ["A"], "oil_rate": [1.0]})
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        fit_many(df)
    with pytest.raises(DataValidationError, match="gas_rate"):
        fit_many(df, rate_column="gas_rate")


def test_plot_returns_figure() -> None:
    field = make_synthetic_field(n_wells=1, n_injectors=0)
    rates = field.production["oil_rate"]
    model = DeclineModel().fit(rates).bootstrap(20)
    fig = model.plot(rates, forecast_days=365)
    names = [trace.name for trace in fig.data]
    assert names[-1].endswith("fit")
    assert "observed" in names
    assert fig.layout.yaxis.type == "log"
    plain = DeclineModel.from_params(100.0, 0.5, 0.0).plot(title="t")
    assert plain.layout.title.text == "t"


def test_plot_without_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(MissingDependencyError, match=r"wellcast\[viz\]"):
        DeclineModel.from_params(100.0, 0.5, 0.0).plot()


# =============================================================================
# End of module: tests/test_dca.py
# =============================================================================
