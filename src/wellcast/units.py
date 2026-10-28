# =============================================================================
# wellcast: oilfield unit conversions with a typed unit registry
# -----------------------------------------------------------------------------
# Module:      wellcast.units
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

"""Unit conversions for oilfield quantities.

Every unit belongs to a physical *quantity* (pressure, temperature, volume
rate and so on) and is defined by an affine map to the base unit of that
quantity::

    base = value * scale + offset

Only temperature units use a non-zero offset. Conversions between units of
different quantities raise :class:`~wellcast.exceptions.UnitError`.

Standard-condition volumes (``stb``, ``scf``, ``sm3``) are treated as plain
volumes. The small difference between oilfield standard conditions (60 degF,
14.696 psia) and metric standard conditions (15 degC, 101.325 kPa) is ignored,
as is common practice in production reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from wellcast.exceptions import UnitError

__all__ = [
    "Unit",
    "convert",
    "get_unit",
    "list_units",
    "register_unit",
]

_T = TypeVar("_T", pd.Series, pd.DataFrame)


@dataclass(frozen=True)
class Unit:
    """A unit defined by an affine map to the base unit of its quantity.

    Parameters
    ----------
    name : str
        Canonical symbol, for example ``"psi"``.
    quantity : str
        Physical quantity, for example ``"pressure"``.
    scale : float
        Multiplier to the base unit.
    offset : float, optional
        Additive term applied after scaling (temperatures only).
    """

    name: str
    quantity: str
    scale: float
    offset: float = 0.0


# -----------------------------------------------------------------------------
# Exact conversion constants
# -----------------------------------------------------------------------------

# 1 US oil barrel = 42 US gallons = 0.158987294928 m3 (exact by definition).
_BBL_M3 = 0.158987294928
# 1 ft3 = 0.3048**3 m3 = 0.028316846592 m3 (exact).
_FT3_M3 = 0.028316846592
# 1 psi = 6.894757293168361 kPa (lbf/in2 from the exact pound and inch).
_PSI_KPA = 6.894757293168361

# Base units: kPa (pressure), K (temperature), m3/d (volume rate),
# m3 (volume), sm3/sm3 (gas-liquid ratio), m (length), d (time),
# fraction (dimensionless ratio).
_BUILTIN_UNITS: tuple[Unit, ...] = (
    # Pressure (all absolute; gauge pressures are not converted implicitly).
    Unit("kPa", "pressure", 1.0),
    Unit("Pa", "pressure", 1.0e-3),
    Unit("MPa", "pressure", 1.0e3),
    Unit("bar", "pressure", 100.0),
    Unit("atm", "pressure", 101.325),
    Unit("psi", "pressure", _PSI_KPA),
    # Temperature.
    Unit("K", "temperature", 1.0),
    Unit("degC", "temperature", 1.0, 273.15),
    Unit("degF", "temperature", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    Unit("degR", "temperature", 5.0 / 9.0),
    # Volume rate (liquid and gas share the quantity).
    Unit("m3/d", "volume_rate", 1.0),
    Unit("sm3/d", "volume_rate", 1.0),
    Unit("bbl/d", "volume_rate", _BBL_M3),
    Unit("stb/d", "volume_rate", _BBL_M3),
    Unit("scf/d", "volume_rate", _FT3_M3),
    Unit("Mscf/d", "volume_rate", _FT3_M3 * 1.0e3),
    Unit("MMscf/d", "volume_rate", _FT3_M3 * 1.0e6),
    Unit("ksm3/d", "volume_rate", 1.0e3),
    # Volume.
    Unit("m3", "volume", 1.0),
    Unit("sm3", "volume", 1.0),
    Unit("bbl", "volume", _BBL_M3),
    Unit("stb", "volume", _BBL_M3),
    Unit("Mbbl", "volume", _BBL_M3 * 1.0e3),
    Unit("MMbbl", "volume", _BBL_M3 * 1.0e6),
    Unit("ft3", "volume", _FT3_M3),
    Unit("scf", "volume", _FT3_M3),
    Unit("Mscf", "volume", _FT3_M3 * 1.0e3),
    Unit("MMscf", "volume", _FT3_M3 * 1.0e6),
    Unit("Bscf", "volume", _FT3_M3 * 1.0e9),
    # Gas-liquid ratio (GOR, GLR).
    Unit("sm3/sm3", "gas_liquid_ratio", 1.0),
    Unit("scf/stb", "gas_liquid_ratio", _FT3_M3 / _BBL_M3),
    Unit("Mscf/stb", "gas_liquid_ratio", _FT3_M3 * 1.0e3 / _BBL_M3),
    # Length.
    Unit("m", "length", 1.0),
    Unit("ft", "length", 0.3048),
    Unit("in", "length", 0.0254),
    Unit("in/64", "length", 0.0254 / 64.0),
    # Time.
    Unit("d", "time", 1.0),
    Unit("h", "time", 1.0 / 24.0),
    Unit("yr", "time", 365.25),
    # Dimensionless ratio (water cut, saturation).
    Unit("fraction", "ratio", 1.0),
    Unit("percent", "ratio", 0.01),
)

# Accepted spellings, matched case-insensitively after removing spaces.
_ALIASES: dict[str, str] = {
    "psia": "psi",
    "kpaa": "kPa",
    "bara": "bar",
    "c": "degC",
    "celsius": "degC",
    "f": "degF",
    "fahrenheit": "degF",
    "kelvin": "K",
    "r": "degR",
    "rankine": "degR",
    "bopd": "bbl/d",
    "bwpd": "bbl/d",
    "blpd": "bbl/d",
    "bbl/day": "bbl/d",
    "stb/day": "stb/d",
    "m3/day": "m3/d",
    "sm3/day": "sm3/d",
    "mscfd": "Mscf/d",
    "mcf/d": "Mscf/d",
    "mcfd": "Mscf/d",
    "mmscfd": "MMscf/d",
    "scf/bbl": "scf/stb",
    "m3/m3": "sm3/sm3",
    "feet": "ft",
    "inch": "in",
    "64ths": "in/64",
    "hr": "h",
    "hours": "h",
    "day": "d",
    "days": "d",
    "year": "yr",
    "%": "percent",
    "frac": "fraction",
    "v/v": "fraction",
}

_REGISTRY: dict[str, Unit] = {}


def _key(symbol: str) -> str:
    return symbol.replace(" ", "").lower()


# -----------------------------------------------------------------------------
# Registry access
# -----------------------------------------------------------------------------


def register_unit(
    name: str,
    quantity: str,
    scale: float,
    offset: float = 0.0,
    *,
    overwrite: bool = False,
) -> Unit:
    """Add a unit to the registry.

    Parameters
    ----------
    name : str
        Unit symbol. Lookup is case-insensitive and ignores spaces.
    quantity : str
        Physical quantity the unit measures.
    scale : float
        Multiplier from this unit to the base unit of ``quantity``.
    offset : float, optional
        Additive term applied after scaling. Default 0.
    overwrite : bool, optional
        Allow replacing an existing unit. Default False.

    Returns
    -------
    Unit
        The registered unit.

    Raises
    ------
    UnitError
        If the unit exists and ``overwrite`` is False, or ``scale`` is not a
        positive finite number.

    Examples
    --------
    >>> from wellcast.units import convert, register_unit
    >>> _ = register_unit("kbbl/d", "volume_rate", 158.987294928, overwrite=True)
    >>> round(convert(1.0, "kbbl/d", "bbl/d"), 6)
    1000.0
    """
    if not (np.isfinite(scale) and scale > 0.0):
        raise UnitError(f"scale for unit {name!r} must be positive and finite")
    key = _key(name)
    if key in _REGISTRY and not overwrite:
        raise UnitError(f"unit {name!r} is already registered")
    unit = Unit(name, quantity, float(scale), float(offset))
    _REGISTRY[key] = unit
    return unit


def get_unit(symbol: str) -> Unit:
    """Look up a unit by symbol or alias.

    Parameters
    ----------
    symbol : str
        Unit symbol such as ``"psi"``, ``"bopd"`` or ``"degC"``.

    Returns
    -------
    Unit
        The registered unit.

    Raises
    ------
    UnitError
        If the symbol is unknown.

    Examples
    --------
    >>> from wellcast.units import get_unit
    >>> get_unit("BOPD").name
    'bbl/d'
    """
    key = _key(symbol)
    key = _key(_ALIASES.get(key, key))
    try:
        return _REGISTRY[key]
    except KeyError:
        raise UnitError(f"unknown unit {symbol!r}") from None


def list_units(quantity: str | None = None) -> list[str]:
    """Return registered unit symbols, optionally for one quantity.

    Parameters
    ----------
    quantity : str, optional
        Restrict the list to this quantity, for example ``"pressure"``.

    Returns
    -------
    list of str
        Sorted unit symbols.

    Examples
    --------
    >>> from wellcast.units import list_units
    >>> list_units("pressure")
    ['MPa', 'Pa', 'atm', 'bar', 'kPa', 'psi']
    """
    return sorted(
        u.name for u in _REGISTRY.values() if quantity is None or u.quantity == quantity
    )


# -----------------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------------


@overload
def convert(value: float, from_unit: str, to_unit: str) -> float: ...
@overload
def convert(
    value: npt.NDArray[Any], from_unit: str, to_unit: str
) -> npt.NDArray[np.float64]: ...
@overload
def convert(value: _T, from_unit: str, to_unit: str) -> _T: ...


def convert(value: Any, from_unit: str, to_unit: str) -> Any:
    """Convert a value between two units of the same quantity.

    Parameters
    ----------
    value : float, numpy.ndarray, pandas.Series or pandas.DataFrame
        Value(s) to convert. The container type is preserved.
    from_unit : str
        Unit of ``value``.
    to_unit : str
        Target unit.

    Returns
    -------
    float, numpy.ndarray, pandas.Series or pandas.DataFrame
        Converted value(s) in ``to_unit``.

    Raises
    ------
    UnitError
        If a unit is unknown or the units measure different quantities.

    Notes
    -----
    With ``base = value * scale + offset`` for each unit, the conversion is
    ``(value * s_from + o_from - o_to) / s_to``.

    Examples
    --------
    >>> from wellcast.units import convert
    >>> round(convert(1000.0, "psi", "bar"), 4)
    68.9476
    >>> convert(212.0, "degF", "degC")
    100.0
    >>> round(convert(1.0, "m3/d", "bbl/d"), 4)
    6.2898
    """
    src = get_unit(from_unit)
    dst = get_unit(to_unit)
    if src.quantity != dst.quantity:
        raise UnitError(
            f"cannot convert {src.name} ({src.quantity}) to {dst.name} ({dst.quantity})"
        )
    if src == dst:
        return value.copy() if hasattr(value, "copy") else value
    a = src.scale / dst.scale
    b = (src.offset - dst.offset) / dst.scale
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value * a + b
    if np.isscalar(value):
        return float(np.asarray(value, dtype=np.float64)) * a + b
    return np.asarray(value, dtype=np.float64) * a + b


for _unit in _BUILTIN_UNITS:
    register_unit(_unit.name, _unit.quantity, _unit.scale, _unit.offset)

# =============================================================================
# End of module: wellcast.units
# =============================================================================
