# =============================================================================
# wellcast: tests for IPR models and choke equations
# -----------------------------------------------------------------------------
# Module:      tests/test_ipr.py
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

from wellcast import ipr
from wellcast.exceptions import DataValidationError, FitError

# -----------------------------------------------------------------------------
# Textbook example: Vogel IPR
# -----------------------------------------------------------------------------
# Ahmed, T., Reservoir Engineering Handbook (Gulf Professional Publishing),
# chapter on oil well performance, Vogel's method example: a well in a
# saturated reservoir (pr = 2500 psig) tests 350 STB/d at pwf = 2000 psig.
# Published answers: qo,max = 1067.1 STB/d; qo = 441.7 STB/d at 1850 psig;
# constant J = 0.7 STB/d/psi gives 455 STB/d at 1850 psig. Vogel IPR table:
# about 218, 632, 845 and 990 STB/d at 2200, 1500, 1000 and 500 psig.


def test_vogel_textbook_example() -> None:
    qmax = ipr.vogel_qmax(350.0, 2000.0, 2500.0)
    assert qmax == pytest.approx(1067.07, abs=0.01)
    assert ipr.vogel_ipr(1850.0, 2500.0, qmax) == pytest.approx(441.68, abs=0.01)
    table = ipr.vogel_ipr(np.array([2200.0, 1500.0, 1000.0, 500.0]), 2500.0, qmax)
    np.testing.assert_allclose(table, [218, 632, 845, 990], atol=1.0)


def test_constant_j_textbook_example() -> None:
    j = ipr.productivity_index(350.0, 2500.0, 2000.0)
    assert j == pytest.approx(0.7)
    assert ipr.linear_ipr(1850.0, 2500.0, j) == pytest.approx(455.0)


# -----------------------------------------------------------------------------
# Composite and Fetkovich
# -----------------------------------------------------------------------------


def test_composite_ipr_hand_calculation() -> None:
    # pr = 4000, pb = 2000, J = 1.4: qb = 2800, AOF = 2800 + 1.4*2000/1.8.
    q = ipr.composite_ipr(np.array([4000.0, 3000.0, 2000.0, 0.0]), 4000.0, 2000.0, 1.4)
    np.testing.assert_allclose(q, [0.0, 1400.0, 2800.0, 2800.0 + 2800.0 / 1.8])


def test_composite_is_smooth_at_bubble_point() -> None:
    eps = 1e-3
    q = ipr.composite_ipr(
        np.array([2000.0 + eps, 2000.0, 2000.0 - eps]), 4000.0, 2000.0, 1.4
    )
    slopes = np.diff(q) / eps
    assert slopes[0] == pytest.approx(slopes[1], rel=1e-3)


def test_composite_productivity_index_both_branches() -> None:
    j_above = ipr.composite_productivity_index(1400.0, 3000.0, 4000.0, 2000.0)
    assert j_above == pytest.approx(1.4)
    q = float(ipr.composite_ipr(1500.0, 4000.0, 2000.0, 1.4))
    assert ipr.composite_productivity_index(q, 1500.0, 4000.0, 2000.0) == pytest.approx(
        1.4
    )
    with pytest.raises(DataValidationError, match="below"):
        ipr.composite_productivity_index(100.0, 4000.0, 4000.0, 2000.0)


def test_composite_rejects_bubble_point_above_reservoir() -> None:
    with pytest.raises(DataValidationError, match="bubble point"):
        ipr.composite_ipr(1000.0, 2000.0, 2500.0, 1.0)


def test_fetkovich_fit_recovers_parameters() -> None:
    pwf = np.array([3170.0, 2897.0, 2440.0, 2150.0])
    q = ipr.fetkovich_ipr(pwf, 3600.0, 4.5e-4, 0.85)
    c, n = ipr.fit_fetkovich(q, pwf, 3600.0)
    assert n == pytest.approx(0.85)
    assert c == pytest.approx(4.5e-4, rel=1e-6)


def test_fetkovich_fit_errors() -> None:
    with pytest.raises(DataValidationError, match="two"):
        ipr.fit_fetkovich([100.0], [2000.0], 2500.0)
    with pytest.raises(DataValidationError, match="differ"):
        ipr.fit_fetkovich([100.0, 110.0], [2000.0, 2000.0], 2500.0)
    with pytest.raises(FitError, match="not positive"):
        ipr.fit_fetkovich([100.0, 200.0], [1000.0, 2000.0], 2500.0)


# -----------------------------------------------------------------------------
# Curves and validation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("linear", {"productivity_index": 1.0}),
        ("vogel", {"qmax": 1000.0}),
        ("fetkovich", {"c": 1e-4, "n": 0.9}),
        ("composite", {"bubble_point": 1500.0, "productivity_index": 1.0}),
    ],
)
def test_ipr_curve_models(model: str, params: dict[str, float]) -> None:
    curve = ipr.ipr_curve(model, 2500.0, n_points=20, **params)  # type: ignore[arg-type]
    assert list(curve.columns) == ["pwf", "rate"]
    assert curve["rate"].iloc[0] == 0.0
    assert curve["rate"].is_monotonic_increasing


def test_ipr_curve_errors() -> None:
    with pytest.raises(DataValidationError, match="unknown"):
        ipr.ipr_curve("darcy", 2500.0, k=1.0)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="n_points"):
        ipr.ipr_curve("vogel", 2500.0, n_points=1, qmax=1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: ipr.linear_ipr(3000.0, 2500.0, 1.0),
        lambda: ipr.vogel_qmax(100.0, 2500.0, 2500.0),
        lambda: ipr.productivity_index(100.0, 2000.0, 2000.0),
        lambda: ipr.vogel_ipr(100.0, 2500.0, 0.0),
    ],
)
def test_invalid_inputs(call: object) -> None:
    with pytest.raises(DataValidationError):
        call()  # type: ignore[operator]


# -----------------------------------------------------------------------------
# Chokes
# -----------------------------------------------------------------------------


def test_gilbert_matches_original_form() -> None:
    # Gilbert (1954) published p = 435 R^0.546 q / S^1.89 with R in Mscf/STB;
    # the tabulated coefficient A = 10.0 (R in scf/STB) is the same equation
    # to within the rounding of A (0.1 percent).
    q, glr, size = 1600.0, 1000.0, 24.0
    original = 435.0 * (glr / 1000.0) ** 0.546 * q / size**1.89
    assert ipr.gilbert_choke(q, glr, size) == pytest.approx(original, rel=2e-3)


def test_choke_correlations_hand_calculation() -> None:
    # p = A q R^B / S^C for q = 1000 STB/d, R = 1000 scf/STB, S = 32/64 in.
    for method, (a, b, c) in ipr.CHOKE_COEFFICIENTS.items():
        expected = a * 1000.0 * 1000.0**b / 32.0**c
        assert ipr.choke_pressure(1000.0, 1000.0, 32.0, method) == pytest.approx(
            expected
        )  # type: ignore[arg-type]
    assert ipr.ros_choke(1000.0, 1000.0, 32.0) == pytest.approx(
        17.4 * 1000.0 * 1000.0**0.5 / 1024.0
    )
    assert ipr.baxendell_choke(1000.0, 1000.0, 32.0) < ipr.gilbert_choke(
        1000.0, 1000.0, 32.0
    )


def test_unknown_choke_method() -> None:
    with pytest.raises(DataValidationError, match="unknown choke method"):
        ipr.choke_pressure(1000.0, 1000.0, 32.0, "magic")  # type: ignore[arg-type]


@given(
    q=st.floats(min_value=1.0, max_value=2e4),
    glr=st.floats(min_value=50.0, max_value=1e4),
    size=st.floats(min_value=8.0, max_value=128.0),
    method=st.sampled_from(sorted(ipr.CHOKE_COEFFICIENTS)),
)
def test_choke_round_trip(q: float, glr: float, size: float, method: str) -> None:
    p = ipr.choke_pressure(q, glr, size, method)  # type: ignore[arg-type]
    assert ipr.choke_rate(p, glr, size, method) == pytest.approx(q)  # type: ignore[arg-type]


@given(
    pr=st.floats(min_value=500.0, max_value=8000.0),
    frac=st.floats(min_value=0.0, max_value=1.0),
    pb_frac=st.floats(min_value=0.1, max_value=1.0),
)
def test_ipr_rate_decreases_with_pwf(pr: float, frac: float, pb_frac: float) -> None:
    pwf = np.array([frac * pr, min(1.0, frac + 0.01) * pr])
    for q in (
        ipr.linear_ipr(pwf, pr, 1.0),
        ipr.vogel_ipr(pwf, pr, 1000.0),
        ipr.fetkovich_ipr(pwf, pr, 1e-4, 0.8),
        ipr.composite_ipr(pwf, pr, pb_frac * pr, 1.0),
    ):
        assert q[0] >= q[1] >= 0.0


# =============================================================================
# End of module: tests/test_ipr.py
# =============================================================================
