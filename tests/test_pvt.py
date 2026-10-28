# =============================================================================
# wellcast: tests for PVT correlations
# -----------------------------------------------------------------------------
# Module:      tests/test_pvt.py
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
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wellcast import pvt
from wellcast._validation import RangeWarning
from wellcast.exceptions import DataValidationError

# -----------------------------------------------------------------------------
# Reference values
# -----------------------------------------------------------------------------


def test_hall_yarborough_reference_value() -> None:
    # Reference implementation value: zFactor R package (Reyes, A., 2017),
    # z.HallYarborough(pres.pr = 1.5, temp.pr = 2.0) = 0.9580002.
    assert pvt.z_factor_hall_yarborough(1.5, 2.0) == pytest.approx(0.9580002, abs=1e-6)


# Standing and Katz (1942) chart, digitized values as distributed with the
# zFactor R package (getStandingKatzMatrix, low-pressure range).
_SK_PPR = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
_SK_CHART = {
    1.2: [0.893, 0.657, 0.519, 0.565, 0.650, 0.741],
    1.3: [0.916, 0.756, 0.638, 0.633, 0.684, 0.759],
    1.5: [0.948, 0.859, 0.794, 0.770, 0.790, 0.836],
    2.0: [0.982, 0.956, 0.941, 0.937, 0.945, 0.969],
    3.0: [1.002, 1.009, 1.018, 1.029, 1.041, 1.056],
}


@pytest.mark.parametrize("tpr", sorted(_SK_CHART))
def test_hall_yarborough_matches_standing_katz_chart(tpr: float) -> None:
    z = pvt.z_factor_hall_yarborough(np.array(_SK_PPR), tpr)
    np.testing.assert_allclose(z, _SK_CHART[tpr], atol=0.01)


def test_standing_hand_calculation() -> None:
    # Hand calculation from Standing (1947) for T = 250 degF, Rs = 751 scf/STB,
    # gas gravity 0.851, 47.1 API:
    #   a = 0.00091*250 - 0.0125*47.1 = -0.36125
    #   pb = 18.2 * ((751/0.851)**0.83 * 10**a - 1.4) = 2181.3 psia
    #   Bo = 0.9759 + 1.2e-4 * (751*sqrt(0.851/0.7923) + 1.25*250)**1.2 = 1.5062
    assert pvt.bubble_point_standing(751.0, 0.851, 47.1, 250.0) == pytest.approx(
        2181.31, abs=0.05
    )
    assert pvt.oil_fvf_standing(751.0, 0.851, 47.1, 250.0) == pytest.approx(
        1.50616, abs=1e-4
    )


def test_vasquez_beggs_hand_calculation() -> None:
    # Vasquez and Beggs (1980), API > 30 coefficients, 100 psig separator:
    #   Rs = 0.0178 * 0.75 * 2000**1.187 * exp(23.931*35/660) = 393.51 scf/STB
    #   Bo = 1 + 4.67e-4*Rs + 140*(35/0.75)*(1.1e-5 + 1.337e-9*Rs) = 1.2591
    rs = pvt.solution_gor_vasquez_beggs(2000.0, 0.75, 35.0, 200.0)
    assert rs == pytest.approx(393.51, abs=0.01)
    assert pvt.oil_fvf_vasquez_beggs(rs, 0.75, 35.0, 200.0) == pytest.approx(
        1.2591, abs=1e-4
    )
    assert pvt.bubble_point_vasquez_beggs(rs, 0.75, 35.0, 200.0) == pytest.approx(
        2000.0
    )


def test_vasquez_beggs_heavy_oil_and_separator() -> None:
    # API <= 30 uses the second coefficient set; a 300 psia separator at
    # 100 degF raises the corrected gas gravity and therefore Rs.
    heavy = pvt.solution_gor_vasquez_beggs(2000.0, 0.75, 25.0, 200.0)
    assert heavy == pytest.approx(
        0.0362 * 0.75 * 2000.0**1.0937 * np.exp(25.724 * 25 / 660)
    )
    hp_sep = pvt.solution_gor_vasquez_beggs(
        2000.0, 0.75, 25.0, 200.0, separator_pressure=300.0, separator_temperature=100.0
    )
    assert hp_sep > heavy
    assert pvt.oil_fvf_vasquez_beggs(300.0, 0.75, 25.0, 200.0) > 1.0


def test_beggs_robinson_hand_calculation() -> None:
    # Beggs and Robinson (1975): X = 10**(3.0324 - 0.02023*35) * 200**-1.163,
    # mu_od = 10**X - 1 = 1.785 cP; live oil at Rs = 393.5 scf/STB:
    # A = 10.715*(493.5)**-0.515, B = 5.44*(543.5)**-0.338 -> 0.639 cP.
    mu_od = pvt.dead_oil_viscosity_beggs_robinson(35.0, 200.0)
    assert mu_od == pytest.approx(1.7854, abs=1e-4)
    assert pvt.live_oil_viscosity_beggs_robinson(mu_od, 393.5) == pytest.approx(
        0.6395, abs=1e-3
    )


def test_undersaturated_properties() -> None:
    # Above pb, Bo shrinks (oil compression) and viscosity grows.
    bo = pvt.oil_fvf_undersaturated(4000.0, 2000.0, 1.26, 393.5, 0.75, 35.0, 200.0)
    assert 1.2 < bo < 1.26
    assert (
        pvt.oil_fvf_undersaturated(1500.0, 2000.0, 1.26, 393.5, 0.75, 35.0, 200.0)
        == 1.26
    )
    mu = pvt.undersaturated_oil_viscosity_vasquez_beggs([2000.0, 4000.0], 2000.0, 0.62)
    assert mu[0] == pytest.approx(0.62)
    assert mu[1] > 0.62


def test_pseudo_critical_and_gas_fvf() -> None:
    # Standing (1977) natural gas: gamma 0.7 -> Tpc 389.4 degR, Ppc 669.1 psia.
    tpc, ppc = pvt.pseudo_critical_standing(0.7)
    assert tpc == pytest.approx(389.375)
    assert ppc == pytest.approx(669.125)
    # Ideal gas at standard conditions has Bg = 1 ft3/scf.
    assert pvt.gas_fvf(14.7, 60.0, 1.0) == pytest.approx(1.0)
    z = pvt.z_factor(2000.0, 180.0, 0.7)
    assert z == pytest.approx(
        pvt.z_factor_hall_yarborough(2000.0 / 669.125, 639.67 / 389.375)
    )


def test_gravity_conversions_round_trip() -> None:
    sg = pvt.api_to_specific_gravity(np.array([10.0, 35.0, 50.0]))
    np.testing.assert_allclose(pvt.specific_gravity_to_api(sg), [10.0, 35.0, 50.0])


def test_black_oil_table() -> None:
    table = pvt.black_oil_table(
        np.linspace(500.0, 5000.0, 10), 200.0, 35.0, 0.75, 500.0
    )
    pb = table.attrs["bubble_point"]
    assert pb == pytest.approx(pvt.bubble_point_standing(500.0, 0.75, 35.0, 200.0))
    above = table[table["pressure"] > pb]
    below = table[table["pressure"] <= pb]
    assert (above["rs"] == 500.0).all()
    assert below["rs"].is_monotonic_increasing
    # Bo peaks at the bubble point; viscosity is lowest there.
    assert above["bo"].is_monotonic_decreasing
    assert below["bo"].is_monotonic_increasing
    assert above["mu_o"].is_monotonic_increasing
    assert below["mu_o"].is_monotonic_decreasing
    assert (table["z"] > 0).all()
    assert table["bg"].is_monotonic_decreasing


# -----------------------------------------------------------------------------
# Validation and range warnings
# -----------------------------------------------------------------------------


def test_scalar_and_array_outputs() -> None:
    assert isinstance(pvt.bubble_point_standing(500.0, 0.75, 35.0, 200.0), float)
    out = pvt.bubble_point_standing([400.0, 500.0], 0.75, 35.0, 200.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


def test_range_warning_outside_correlation() -> None:
    with pytest.warns(RangeWarning, match="Standing"):
        pvt.bubble_point_standing(500.0, 0.75, 35.0, 320.0)
    with pytest.warns(RangeWarning, match="Hall-Yarborough"):
        pvt.z_factor_hall_yarborough(1.0, 0.9)
    with pytest.warns(RangeWarning, match="Beggs-Robinson"):
        pvt.dead_oil_viscosity_beggs_robinson(70.0, 200.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: pvt.bubble_point_standing(0.0, 0.75, 35.0, 200.0),
        lambda: pvt.oil_fvf_standing(-1.0, 0.75, 35.0, 200.0),
        lambda: pvt.live_oil_viscosity_beggs_robinson(1.0, -5.0),
        lambda: pvt.z_factor_hall_yarborough(-1.0, 1.5),
        lambda: pvt.gas_fvf(1000.0, 150.0, 0.0),
        lambda: pvt.solution_gor_standing(0.0, 0.75, 35.0, 200.0),
        lambda: pvt.specific_gravity_to_api(0.0),
    ],
)
def test_invalid_inputs(call: object) -> None:
    with pytest.raises(DataValidationError):
        call()  # type: ignore[operator]


# -----------------------------------------------------------------------------
# Property tests: monotonic physics
# -----------------------------------------------------------------------------

_api = st.floats(min_value=20.0, max_value=50.0)
_gg = st.floats(min_value=0.6, max_value=0.9)
_temp = st.floats(min_value=110.0, max_value=250.0)


@given(
    rs1=st.floats(min_value=50.0, max_value=1400.0),
    rs2=st.floats(min_value=50.0, max_value=1400.0),
    gg=_gg,
    api=_api,
    t=_temp,
)
def test_bo_and_pb_increase_with_rs(
    rs1: float, rs2: float, gg: float, api: float, t: float
) -> None:
    lo, hi = sorted((rs1, rs2))
    assert pvt.oil_fvf_standing(lo, gg, api, t) <= pvt.oil_fvf_standing(hi, gg, api, t)
    assert pvt.oil_fvf_vasquez_beggs(lo, gg, api, t) <= pvt.oil_fvf_vasquez_beggs(
        hi, gg, api, t
    )
    assert pvt.bubble_point_standing(lo, gg, api, t) <= pvt.bubble_point_standing(
        hi, gg, api, t
    )


@given(
    p1=st.floats(min_value=150.0, max_value=5000.0),
    p2=st.floats(min_value=150.0, max_value=5000.0),
    gg=_gg,
    api=_api,
    t=_temp,
)
def test_rs_increases_with_pressure(
    p1: float, p2: float, gg: float, api: float, t: float
) -> None:
    lo, hi = sorted((p1, p2))
    assert pvt.solution_gor_standing(lo, gg, api, t) <= pvt.solution_gor_standing(
        hi, gg, api, t
    )
    assert pvt.solution_gor_vasquez_beggs(
        lo, gg, api, t
    ) <= pvt.solution_gor_vasquez_beggs(hi, gg, api, t)


@given(rs=st.floats(min_value=20.0, max_value=2000.0), api=_api, t=_temp)
def test_live_oil_is_less_viscous_than_dead(rs: float, api: float, t: float) -> None:
    mu_od = pvt.dead_oil_viscosity_beggs_robinson(api, t)
    assert pvt.live_oil_viscosity_beggs_robinson(mu_od, rs) < mu_od


@given(rs=st.floats(min_value=50.0, max_value=1400.0), gg=_gg, api=_api, t=_temp)
def test_standing_rs_pb_round_trip(rs: float, gg: float, api: float, t: float) -> None:
    pb = pvt.bubble_point_standing(rs, gg, api, t)
    if 130.0 <= pb <= 7000.0:
        # 1.2048 is Standing's rounded value of 1/0.83, so allow 0.2 percent.
        assert pvt.solution_gor_standing(pb, gg, api, t) == pytest.approx(rs, rel=2e-3)


@given(
    ppr=st.floats(min_value=0.2, max_value=15.0),
    tpr=st.floats(min_value=1.05, max_value=3.0),
)
def test_z_factor_converges_to_physical_root(ppr: float, tpr: float) -> None:
    z = pvt.z_factor_hall_yarborough(ppr, tpr)
    assert 0.25 < z < 1.8
    # z approaches 1 as pressure goes to zero.
    assert abs(pvt.z_factor_hall_yarborough(1e-3, tpr) - 1.0) < 1e-3


# =============================================================================
# End of module: tests/test_pvt.py
# =============================================================================
