# =============================================================================
# wellcast: tests for unit conversions
# -----------------------------------------------------------------------------
# Module:      tests/test_units.py
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

import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wellcast import units
from wellcast.exceptions import UnitError
from wellcast.units import Unit, convert, get_unit, list_units, register_unit

# -----------------------------------------------------------------------------
# Reference values
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        # 1 bbl = 42 US gal = 0.158987294928 m3 (API MPMS Ch. 11).
        (1.0, "bbl/d", "m3/d", 0.158987294928),
        (1.0, "stb", "sm3", 0.158987294928),
        # Standard atmosphere: 14.6959 psi = 101.325 kPa = 1.01325 bar.
        (14.69594877551, "psi", "kPa", 101.325),
        (1.0, "atm", "bar", 1.01325),
        (1.0, "MPa", "psi", 145.0377377),
        # Temperature scale fixed points.
        (32.0, "degF", "degC", 0.0),
        (0.0, "degC", "K", 273.15),
        (491.67, "degR", "degF", 32.0),
        # Gas-oil ratio: 1 scf/stb = 0.1781076 sm3/sm3 (SPE metric standard).
        (1000.0, "scf/stb", "sm3/sm3", 178.1076),
        (1.0, "MMscf/d", "sm3/d", 28316.846592),
        (1.0, "Mscf/stb", "scf/stb", 1000.0),
        (64.0, "in/64", "in", 1.0),
        (1.0, "ft", "m", 0.3048),
        (48.0, "h", "d", 2.0),
        (25.0, "percent", "fraction", 0.25),
    ],
)
def test_reference_conversions(
    value: float, src: str, dst: str, expected: float
) -> None:
    assert convert(value, src, dst) == pytest.approx(expected, rel=1e-7)


def test_aliases_and_case() -> None:
    assert get_unit("BOPD").name == "bbl/d"
    assert get_unit("psia").name == "psi"
    assert get_unit("Mcf/d").name == "Mscf/d"
    assert get_unit("deg C".replace("deg ", "")).name == "degC"
    assert convert(1.0, "MMSCFD", "mscf/d") == pytest.approx(1000.0)


def test_unknown_unit() -> None:
    with pytest.raises(UnitError, match="unknown unit"):
        convert(1.0, "furlong", "m")


def test_quantity_mismatch() -> None:
    with pytest.raises(UnitError, match="pressure"):
        convert(1.0, "psi", "degF")


def test_list_units() -> None:
    assert "degF" in list_units("temperature")
    assert len(list_units()) > len(list_units("pressure"))
    assert list_units("no-such-quantity") == []


def test_register_unit_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    # Work on a copy so the test does not leak units into other tests.
    monkeypatch.setattr(units, "_REGISTRY", dict(units._REGISTRY))
    with pytest.raises(UnitError, match="already registered"):
        register_unit("psi", "pressure", 1.0)
    with pytest.raises(UnitError, match="positive"):
        register_unit("bogus", "pressure", 0.0)
    unit = register_unit("hPa", "pressure", 0.1, overwrite=True)
    assert unit == Unit("hPa", "pressure", 0.1)
    assert convert(10.0, "hPa", "kPa") == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Container handling
# -----------------------------------------------------------------------------


def test_scalar_returns_float() -> None:
    out = convert(1, "bar", "kPa")
    assert isinstance(out, float)
    assert out == 100.0


def test_array_and_list() -> None:
    np.testing.assert_allclose(
        convert(np.array([0.0, 100.0]), "degC", "degF"), [32.0, 212.0]
    )
    out = convert([0.0, 100.0], "degC", "degF")
    assert isinstance(out, np.ndarray)


def test_series_and_frame_preserved() -> None:
    s = pd.Series([1.0, 2.0], name="whp")
    out = convert(s, "bar", "kPa")
    assert isinstance(out, pd.Series)
    assert out.name == "whp"
    frame = convert(pd.DataFrame({"a": [1.0]}), "bar", "kPa")
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "a"] == 100.0


def test_identity_returns_copy() -> None:
    s = pd.Series([1.0])
    out = convert(s, "psi", "PSIA")
    out.iloc[0] = 99.0
    assert s.iloc[0] == 1.0
    assert convert(5.0, "psi", "psi") == 5.0


# -----------------------------------------------------------------------------
# Property tests
# -----------------------------------------------------------------------------

_QUANTITIES = ["pressure", "temperature", "volume_rate", "volume", "gas_liquid_ratio"]
_PAIRS = [
    pair for q in _QUANTITIES for pair in itertools.permutations(list_units(q), 2)
]


@given(
    pair=st.sampled_from(_PAIRS),
    value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
)
def test_round_trip(pair: tuple[str, str], value: float) -> None:
    src, dst = pair
    back = convert(convert(value, src, dst), dst, src)
    assert back == pytest.approx(value, rel=1e-9, abs=1e-6)


@given(
    a=st.floats(min_value=0.0, max_value=1e5),
    b=st.floats(min_value=0.0, max_value=1e5),
    pair=st.sampled_from(
        [p for p in _PAIRS if p[0] not in ("degC", "degF", "K", "degR")]
    ),
)
def test_linear_units_preserve_order(a: float, b: float, pair: tuple[str, str]) -> None:
    src, dst = pair
    if a < b:
        assert convert(a, src, dst) <= convert(b, src, dst)


# =============================================================================
# End of module: tests/test_units.py
# =============================================================================
