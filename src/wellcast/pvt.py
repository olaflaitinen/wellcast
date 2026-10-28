# =============================================================================
# wellcast: black-oil and gas PVT correlations in oilfield units
# -----------------------------------------------------------------------------
# Module:      wellcast.pvt
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

r"""Empirical PVT correlations for black oil and natural gas.

All functions use oilfield units: pressure in psia, temperature in degF,
solution gas-oil ratio in scf/STB, gas specific gravity relative to air and
oil gravity in degrees API. Inputs may be scalars or arrays; scalar inputs
return a ``float``.

Each correlation was fitted to a finite laboratory data set. Inputs outside
the published range are still evaluated, but a
:class:`~wellcast._validation.RangeWarning` is issued.

Notes
-----
Oil specific gravity and API gravity are related by
:math:`\gamma_o = 141.5 / (131.5 + \mathrm{API})`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from wellcast._validation import (
    RangeWarning,
    as_float_array,
    check_positive,
    warn_outside_range,
)
from wellcast.exceptions import DataValidationError

__all__ = [
    "RangeWarning",
    "api_to_specific_gravity",
    "black_oil_table",
    "bubble_point_standing",
    "bubble_point_vasquez_beggs",
    "dead_oil_viscosity_beggs_robinson",
    "gas_fvf",
    "live_oil_viscosity_beggs_robinson",
    "oil_fvf_standing",
    "oil_fvf_undersaturated",
    "oil_fvf_vasquez_beggs",
    "pseudo_critical_standing",
    "solution_gor_standing",
    "solution_gor_vasquez_beggs",
    "specific_gravity_to_api",
    "undersaturated_oil_viscosity_vasquez_beggs",
    "z_factor",
    "z_factor_hall_yarborough",
]

FloatArray = npt.NDArray[np.float64]
FloatOrArray = float | FloatArray

# Standard conditions used for Bg: 14.7 psia and 60 degF (519.67 degR).
_P_SC = 14.7
_T_SC_R = 519.67
_R_OFFSET = 459.67


def _out(value: FloatArray) -> FloatOrArray:
    # Return a Python float for scalar input and the array otherwise.
    return float(value) if value.ndim == 0 else value


# -----------------------------------------------------------------------------
# Published validity ranges
# -----------------------------------------------------------------------------

# Standing (1947): 105 California crude oil systems.
_STANDING = {
    "temperature": (100.0, 258.0),
    "rs": (20.0, 1425.0),
    "api": (16.5, 63.8),
    "gas_gravity": (0.59, 0.95),
    "pressure": (130.0, 7000.0),
}
# Vasquez and Beggs (1980): about 6000 measurements worldwide.
_VASQUEZ_BEGGS = {
    "temperature": (70.0, 295.0),
    "rs": (0.0, 2199.0),
    "api": (15.3, 59.5),
    "gas_gravity": (0.511, 1.351),
    "pressure": (50.0, 5250.0),
}
# Beggs and Robinson (1975): 600 oil systems.
_BEGGS_ROBINSON = {
    "temperature": (70.0, 295.0),
    "rs": (20.0, 2070.0),
    "api": (16.0, 58.0),
}
# Vasquez and Beggs undersaturated viscosity data set.
_VB_VISCOSITY = {"pressure": (141.0, 9515.0)}
# Hall and Yarborough (1973), fitted to the Standing-Katz chart.
_HALL_YARBOROUGH = {"tpr": (1.0, 3.0), "ppr": (0.0, 24.0)}


def _warn(ranges: dict[str, tuple[float, float]], name: str, **values: Any) -> None:
    for key, value in values.items():
        low, high = ranges[key]
        warn_outside_range(value, key, low, high, name)


# -----------------------------------------------------------------------------
# Gravity conversions
# -----------------------------------------------------------------------------


def api_to_specific_gravity(api: npt.ArrayLike) -> FloatOrArray:
    r"""Convert API gravity to oil specific gravity (water = 1).

    Parameters
    ----------
    api : array_like
        Oil gravity in degrees API.

    Returns
    -------
    float or numpy.ndarray
        Specific gravity at 60 degF.

    Notes
    -----
    :math:`\gamma_o = 141.5 / (131.5 + \mathrm{API})`.

    Examples
    --------
    >>> round(api_to_specific_gravity(10.0), 4)
    1.0
    """
    return _out(141.5 / (131.5 + as_float_array(api, "api")))


def specific_gravity_to_api(specific_gravity: npt.ArrayLike) -> FloatOrArray:
    """Convert oil specific gravity (water = 1) to API gravity.

    Parameters
    ----------
    specific_gravity : array_like
        Oil specific gravity at 60 degF.

    Returns
    -------
    float or numpy.ndarray
        Gravity in degrees API.

    Examples
    --------
    >>> round(specific_gravity_to_api(0.85), 2)
    34.97
    """
    sg = check_positive(specific_gravity, "specific_gravity")
    return _out(141.5 / sg - 131.5)


# -----------------------------------------------------------------------------
# Standing (1947)
# -----------------------------------------------------------------------------


def bubble_point_standing(
    rs: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
    api: npt.ArrayLike,
    temperature: npt.ArrayLike,
) -> FloatOrArray:
    r"""Bubble-point pressure from Standing's correlation.

    Parameters
    ----------
    rs : array_like
        Solution gas-oil ratio at the bubble point, scf/STB.
    gas_gravity : array_like
        Gas specific gravity (air = 1).
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Reservoir temperature, degF.

    Returns
    -------
    float or numpy.ndarray
        Bubble-point pressure, psia.

    Raises
    ------
    DataValidationError
        If ``rs`` or ``gas_gravity`` is not positive.

    Warns
    -----
    RangeWarning
        Outside 100-258 degF, 20-1425 scf/STB, 16.5-63.8 API or gas gravity
        0.59-0.95.

    Notes
    -----
    .. math::

        p_b = 18.2 \left[ (R_s / \gamma_g)^{0.83} \, 10^{a} - 1.4 \right],
        \quad a = 0.00091\,T - 0.0125\,\mathrm{API}

    Standing, M. B. (1947), "A pressure-volume-temperature correlation for
    mixtures of California oils and gases", Drilling and Production Practice,
    API, 275-287.

    Examples
    --------
    >>> round(bubble_point_standing(751.0, 0.851, 47.1, 250.0), 1)
    2181.3
    """
    rs_ = check_positive(rs, "rs")
    gg = check_positive(gas_gravity, "gas_gravity")
    api_ = as_float_array(api, "api")
    t = as_float_array(temperature, "temperature")
    _warn(_STANDING, "Standing", rs=rs_, gas_gravity=gg, api=api_, temperature=t)
    a = 0.00091 * t - 0.0125 * api_
    return _out(18.2 * (np.power(rs_ / gg, 0.83) * np.power(10.0, a) - 1.4))


def solution_gor_standing(
    pressure: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
    api: npt.ArrayLike,
    temperature: npt.ArrayLike,
    bubble_point: npt.ArrayLike | None = None,
) -> FloatOrArray:
    r"""Solution gas-oil ratio from Standing's correlation.

    Parameters
    ----------
    pressure : array_like
        Pressure, psia.
    gas_gravity : array_like
        Gas specific gravity (air = 1).
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Temperature, degF.
    bubble_point : array_like, optional
        Bubble-point pressure, psia. Above it the oil is undersaturated and
        Rs stays at its bubble-point value. If omitted, the oil is assumed
        saturated at every pressure.

    Returns
    -------
    float or numpy.ndarray
        Solution gas-oil ratio, scf/STB.

    Raises
    ------
    DataValidationError
        If ``pressure`` or ``gas_gravity`` is not positive.

    Notes
    -----
    Inverse of :func:`bubble_point_standing`:

    .. math::

        R_s = \gamma_g \left[ \left(\frac{p}{18.2} + 1.4\right)
        10^{x} \right]^{1.2048}, \quad x = 0.0125\,\mathrm{API} - 0.00091\,T

    Standing (1947), see :func:`bubble_point_standing`.

    Examples
    --------
    >>> round(solution_gor_standing(2181.3, 0.851, 47.1, 250.0), 0)
    751.0
    """
    p = check_positive(pressure, "pressure")
    gg = check_positive(gas_gravity, "gas_gravity")
    api_ = as_float_array(api, "api")
    t = as_float_array(temperature, "temperature")
    if bubble_point is not None:
        p = np.minimum(p, check_positive(bubble_point, "bubble_point"))
    _warn(_STANDING, "Standing", pressure=p, gas_gravity=gg, api=api_, temperature=t)
    x = 0.0125 * api_ - 0.00091 * t
    return _out(gg * np.power((p / 18.2 + 1.4) * np.power(10.0, x), 1.2048))


def oil_fvf_standing(
    rs: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
    api: npt.ArrayLike,
    temperature: npt.ArrayLike,
) -> FloatOrArray:
    r"""Saturated oil formation volume factor from Standing's correlation.

    Parameters
    ----------
    rs : array_like
        Solution gas-oil ratio, scf/STB.
    gas_gravity : array_like
        Gas specific gravity (air = 1).
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Temperature, degF.

    Returns
    -------
    float or numpy.ndarray
        Oil formation volume factor, bbl/STB, at or below the bubble point.

    Raises
    ------
    DataValidationError
        If ``rs`` is negative or ``gas_gravity`` is not positive.

    Notes
    -----
    .. math::

        B_o = 0.9759 + 0.000120 \left[ R_s \sqrt{\gamma_g / \gamma_o}
        + 1.25\,T \right]^{1.2}

    Standing (1947), see :func:`bubble_point_standing`.

    Examples
    --------
    >>> round(oil_fvf_standing(751.0, 0.851, 47.1, 250.0), 4)
    1.5062
    """
    rs_ = as_float_array(rs, "rs")
    if np.any(rs_ < 0):
        raise DataValidationError("rs must be non-negative")
    gg = check_positive(gas_gravity, "gas_gravity")
    api_ = as_float_array(api, "api")
    t = as_float_array(temperature, "temperature")
    _warn(_STANDING, "Standing", gas_gravity=gg, api=api_, temperature=t)
    gamma_o = 141.5 / (131.5 + api_)
    f = rs_ * np.sqrt(gg / gamma_o) + 1.25 * t
    return _out(0.9759 + 0.000120 * np.power(f, 1.2))


# -----------------------------------------------------------------------------
# Vasquez and Beggs (1980)
# -----------------------------------------------------------------------------


def _vb_corrected_gravity(
    gas_gravity: FloatArray,
    api: FloatArray,
    separator_pressure: float,
    separator_temperature: float,
) -> FloatArray:
    # Gas gravity normalized to a 100 psig (114.7 psia) separator.
    corrected: FloatArray = gas_gravity * (
        1.0
        + 5.912e-5 * api * separator_temperature * np.log10(separator_pressure / 114.7)
    )
    return corrected


def _vb_rs_coefficients(api: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    # Coefficient sets for API <= 30 and API > 30.
    heavy = api <= 30.0
    c1 = np.where(heavy, 0.0362, 0.0178)
    c2 = np.where(heavy, 1.0937, 1.1870)
    c3 = np.where(heavy, 25.7240, 23.931)
    return c1, c2, c3


def solution_gor_vasquez_beggs(
    pressure: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
    api: npt.ArrayLike,
    temperature: npt.ArrayLike,
    *,
    separator_pressure: float = 114.7,
    separator_temperature: float = 60.0,
) -> FloatOrArray:
    r"""Solution gas-oil ratio from the Vasquez-Beggs correlation.

    Parameters
    ----------
    pressure : array_like
        Pressure at or below the bubble point, psia.
    gas_gravity : array_like
        Gas specific gravity (air = 1) at separator conditions.
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Temperature, degF.
    separator_pressure : float, optional
        Separator pressure, psia. Default 114.7 (100 psig).
    separator_temperature : float, optional
        Separator temperature, degF. Default 60.

    Returns
    -------
    float or numpy.ndarray
        Solution gas-oil ratio, scf/STB.

    Raises
    ------
    DataValidationError
        If ``pressure`` or ``gas_gravity`` is not positive.

    Notes
    -----
    .. math::

        R_s = C_1 \gamma_{gs} p^{C_2}
        \exp\!\left(\frac{C_3\,\mathrm{API}}{T + 460}\right)

    with :math:`(C_1, C_2, C_3) = (0.0362, 1.0937, 25.7240)` for API <= 30
    and :math:`(0.0178, 1.1870, 23.931)` above, and the separator-corrected
    gravity
    :math:`\gamma_{gs} = \gamma_g [1 + 5.912\times10^{-5}\,\mathrm{API}\,
    T_{sep} \log_{10}(p_{sep}/114.7)]`.

    Vasquez, M. and Beggs, H. D. (1980), "Correlations for fluid physical
    property prediction", JPT 32(6), 968-970, SPE-6719-PA.

    Examples
    --------
    >>> round(solution_gor_vasquez_beggs(2000.0, 0.75, 35.0, 200.0), 1)
    393.5
    """
    p = check_positive(pressure, "pressure")
    gg = check_positive(gas_gravity, "gas_gravity")
    api_ = as_float_array(api, "api")
    t = as_float_array(temperature, "temperature")
    _warn(
        _VASQUEZ_BEGGS,
        "Vasquez-Beggs",
        pressure=p,
        gas_gravity=gg,
        api=api_,
        temperature=t,
    )
    ggs = _vb_corrected_gravity(gg, api_, separator_pressure, separator_temperature)
    c1, c2, c3 = _vb_rs_coefficients(api_)
    # The published correlation uses T + 460 (degR, rounded).
    return _out(c1 * ggs * np.power(p, c2) * np.exp(c3 * api_ / (t + 460.0)))


def bubble_point_vasquez_beggs(
    rs: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
    api: npt.ArrayLike,
    temperature: npt.ArrayLike,
    *,
    separator_pressure: float = 114.7,
    separator_temperature: float = 60.0,
) -> FloatOrArray:
    r"""Bubble-point pressure from the Vasquez-Beggs correlation.

    Parameters
    ----------
    rs : array_like
        Solution gas-oil ratio at the bubble point, scf/STB.
    gas_gravity : array_like
        Gas specific gravity (air = 1) at separator conditions.
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Temperature, degF.
    separator_pressure : float, optional
        Separator pressure, psia. Default 114.7.
    separator_temperature : float, optional
        Separator temperature, degF. Default 60.

    Returns
    -------
    float or numpy.ndarray
        Bubble-point pressure, psia.

    Raises
    ------
    DataValidationError
        If ``rs`` or ``gas_gravity`` is not positive.

    Notes
    -----
    Algebraic inverse of :func:`solution_gor_vasquez_beggs`:

    .. math::

        p_b = \left[\frac{R_s}{C_1 \gamma_{gs}}
        \exp\!\left(-\frac{C_3\,\mathrm{API}}{T + 460}\right)\right]^{1/C_2}

    Examples
    --------
    >>> round(bubble_point_vasquez_beggs(393.5, 0.75, 35.0, 200.0), 0)
    2000.0
    """
    rs_ = check_positive(rs, "rs")
    gg = check_positive(gas_gravity, "gas_gravity")
    api_ = as_float_array(api, "api")
    t = as_float_array(temperature, "temperature")
    _warn(
        _VASQUEZ_BEGGS, "Vasquez-Beggs", rs=rs_, gas_gravity=gg, api=api_, temperature=t
    )
    ggs = _vb_corrected_gravity(gg, api_, separator_pressure, separator_temperature)
    c1, c2, c3 = _vb_rs_coefficients(api_)
    base = rs_ / (c1 * ggs) * np.exp(-c3 * api_ / (t + 460.0))
    return _out(np.power(base, 1.0 / c2))


def oil_fvf_vasquez_beggs(
    rs: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
    api: npt.ArrayLike,
    temperature: npt.ArrayLike,
    *,
    separator_pressure: float = 114.7,
    separator_temperature: float = 60.0,
) -> FloatOrArray:
    r"""Saturated oil formation volume factor from Vasquez-Beggs.

    Parameters
    ----------
    rs : array_like
        Solution gas-oil ratio, scf/STB.
    gas_gravity : array_like
        Gas specific gravity (air = 1) at separator conditions.
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Temperature, degF.
    separator_pressure : float, optional
        Separator pressure, psia. Default 114.7.
    separator_temperature : float, optional
        Separator temperature, degF. Default 60.

    Returns
    -------
    float or numpy.ndarray
        Oil formation volume factor, bbl/STB.

    Raises
    ------
    DataValidationError
        If ``gas_gravity`` is not positive.

    Notes
    -----
    .. math::

        B_o = 1 + C_1 R_s + (T - 60) \frac{\mathrm{API}}{\gamma_{gs}}
        (C_2 + C_3 R_s)

    with :math:`(C_1, C_2, C_3) = (4.677\times10^{-4}, 1.751\times10^{-5},
    -1.811\times10^{-8})` for API <= 30 and :math:`(4.670\times10^{-4},
    1.100\times10^{-5}, 1.337\times10^{-9})` above. Vasquez and Beggs (1980).

    Examples
    --------
    >>> round(oil_fvf_vasquez_beggs(393.5, 0.75, 35.0, 200.0), 4)
    1.2591
    """
    rs_ = as_float_array(rs, "rs")
    gg = check_positive(gas_gravity, "gas_gravity")
    api_ = as_float_array(api, "api")
    t = as_float_array(temperature, "temperature")
    _warn(_VASQUEZ_BEGGS, "Vasquez-Beggs", gas_gravity=gg, api=api_, temperature=t)
    ggs = _vb_corrected_gravity(gg, api_, separator_pressure, separator_temperature)
    heavy = api_ <= 30.0
    c1 = np.where(heavy, 4.677e-4, 4.670e-4)
    c2 = np.where(heavy, 1.751e-5, 1.100e-5)
    c3 = np.where(heavy, -1.811e-8, 1.337e-9)
    return _out(1.0 + c1 * rs_ + (t - 60.0) * (api_ / ggs) * (c2 + c3 * rs_))


def oil_fvf_undersaturated(
    pressure: npt.ArrayLike,
    bubble_point: npt.ArrayLike,
    bo_bubble: npt.ArrayLike,
    rs_bubble: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
    api: npt.ArrayLike,
    temperature: npt.ArrayLike,
) -> FloatOrArray:
    r"""Oil formation volume factor above the bubble point.

    Uses the Vasquez-Beggs isothermal compressibility, integrated from the
    bubble point to ``pressure``. At or below the bubble point the value at
    the bubble point is returned.

    Parameters
    ----------
    pressure : array_like
        Pressure, psia.
    bubble_point : array_like
        Bubble-point pressure, psia.
    bo_bubble : array_like
        Formation volume factor at the bubble point, bbl/STB.
    rs_bubble : array_like
        Solution gas-oil ratio at the bubble point, scf/STB.
    gas_gravity : array_like
        Gas specific gravity (air = 1).
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Temperature, degF.

    Returns
    -------
    float or numpy.ndarray
        Oil formation volume factor, bbl/STB.

    Raises
    ------
    DataValidationError
        If a pressure is not positive.

    Notes
    -----
    With :math:`c_o = A / p` and
    :math:`A = 10^{-5}(-1433 + 5 R_{sb} + 17.2\,T - 1180\,\gamma_g
    + 12.61\,\mathrm{API})`:

    .. math:: B_o = B_{ob} \exp\left[-A \ln(p / p_b)\right]

    Vasquez and Beggs (1980).

    Examples
    --------
    >>> bo = oil_fvf_undersaturated(4000.0, 2000.0, 1.26, 393.5, 0.75, 35.0, 200.0)
    >>> round(bo, 4)
    1.2295
    """
    p = check_positive(pressure, "pressure")
    pb = check_positive(bubble_point, "bubble_point")
    bob = check_positive(bo_bubble, "bo_bubble")
    a = 1.0e-5 * (
        -1433.0
        + 5.0 * as_float_array(rs_bubble, "rs_bubble")
        + 17.2 * as_float_array(temperature, "temperature")
        - 1180.0 * as_float_array(gas_gravity, "gas_gravity")
        + 12.61 * as_float_array(api, "api")
    )
    ratio = np.maximum(p, pb) / pb
    return _out(bob * np.exp(-a * np.log(ratio)))


# -----------------------------------------------------------------------------
# Viscosity: Beggs and Robinson (1975), Vasquez and Beggs (1980)
# -----------------------------------------------------------------------------


def dead_oil_viscosity_beggs_robinson(
    api: npt.ArrayLike, temperature: npt.ArrayLike
) -> FloatOrArray:
    r"""Dead (gas-free) oil viscosity from Beggs and Robinson.

    Parameters
    ----------
    api : array_like
        Oil gravity, degrees API.
    temperature : array_like
        Temperature, degF.

    Returns
    -------
    float or numpy.ndarray
        Dead oil viscosity, cP.

    Raises
    ------
    DataValidationError
        If ``temperature`` is not positive.

    Notes
    -----
    .. math::

        \mu_{od} = 10^{X} - 1, \quad X = 10^{3.0324 - 0.02023\,\mathrm{API}}
        \, T^{-1.163}

    Beggs, H. D. and Robinson, J. R. (1975), "Estimating the viscosity of
    crude oil systems", JPT 27(9), 1140-1141, SPE-5434-PA. The correlation
    tends to overpredict below about 100 degF.

    Examples
    --------
    >>> round(dead_oil_viscosity_beggs_robinson(35.0, 200.0), 3)
    1.785
    """
    api_ = as_float_array(api, "api")
    t = check_positive(temperature, "temperature")
    _warn(_BEGGS_ROBINSON, "Beggs-Robinson", api=api_, temperature=t)
    x = np.power(10.0, 3.0324 - 0.02023 * api_) * np.power(t, -1.163)
    return _out(np.power(10.0, x) - 1.0)


def live_oil_viscosity_beggs_robinson(
    dead_viscosity: npt.ArrayLike, rs: npt.ArrayLike
) -> FloatOrArray:
    r"""Saturated (live) oil viscosity from Beggs and Robinson.

    Parameters
    ----------
    dead_viscosity : array_like
        Dead oil viscosity at the same temperature, cP.
    rs : array_like
        Solution gas-oil ratio, scf/STB.

    Returns
    -------
    float or numpy.ndarray
        Live oil viscosity at or below the bubble point, cP.

    Raises
    ------
    DataValidationError
        If ``dead_viscosity`` is not positive or ``rs`` is negative.

    Notes
    -----
    .. math::

        \mu_o = A \mu_{od}^{B}, \quad A = 10.715 (R_s + 100)^{-0.515},
        \quad B = 5.44 (R_s + 150)^{-0.338}

    Beggs and Robinson (1975).

    Examples
    --------
    >>> round(live_oil_viscosity_beggs_robinson(1.785, 393.5), 3)
    0.639
    """
    mu_od = check_positive(dead_viscosity, "dead_viscosity")
    rs_ = as_float_array(rs, "rs")
    if np.any(rs_ < 0):
        raise DataValidationError("rs must be non-negative")
    _warn(_BEGGS_ROBINSON, "Beggs-Robinson", rs=rs_[rs_ > 0])
    a = 10.715 * np.power(rs_ + 100.0, -0.515)
    b = 5.44 * np.power(rs_ + 150.0, -0.338)
    return _out(a * np.power(mu_od, b))


def undersaturated_oil_viscosity_vasquez_beggs(
    pressure: npt.ArrayLike,
    bubble_point: npt.ArrayLike,
    saturated_viscosity: npt.ArrayLike,
) -> FloatOrArray:
    r"""Oil viscosity above the bubble point from Vasquez and Beggs.

    Parameters
    ----------
    pressure : array_like
        Pressure, psia.
    bubble_point : array_like
        Bubble-point pressure, psia.
    saturated_viscosity : array_like
        Viscosity at the bubble point, cP.

    Returns
    -------
    float or numpy.ndarray
        Oil viscosity, cP. Equals ``saturated_viscosity`` at or below the
        bubble point.

    Raises
    ------
    DataValidationError
        If any input is not positive.

    Notes
    -----
    .. math::

        \mu_o = \mu_{ob} (p / p_b)^{m}, \quad
        m = 2.6\,p^{1.187} \exp(-11.513 - 8.98\times10^{-5} p)

    Vasquez and Beggs (1980).

    Examples
    --------
    >>> round(undersaturated_oil_viscosity_vasquez_beggs(4000.0, 2000.0, 0.62), 3)
    0.786
    """
    p = check_positive(pressure, "pressure")
    pb = check_positive(bubble_point, "bubble_point")
    mu_ob = check_positive(saturated_viscosity, "saturated_viscosity")
    _warn(_VB_VISCOSITY, "Vasquez-Beggs viscosity", pressure=p)
    p_eff = np.maximum(p, pb)
    m = 2.6 * np.power(p_eff, 1.187) * np.exp(-11.513 - 8.98e-5 * p_eff)
    return _out(mu_ob * np.power(p_eff / pb, m))


# -----------------------------------------------------------------------------
# Natural gas
# -----------------------------------------------------------------------------


def pseudo_critical_standing(
    gas_gravity: npt.ArrayLike,
) -> tuple[FloatOrArray, FloatOrArray]:
    r"""Pseudo-critical temperature and pressure of a natural gas.

    Parameters
    ----------
    gas_gravity : array_like
        Gas specific gravity (air = 1).

    Returns
    -------
    tpc : float or numpy.ndarray
        Pseudo-critical temperature, degR.
    ppc : float or numpy.ndarray
        Pseudo-critical pressure, psia.

    Raises
    ------
    DataValidationError
        If ``gas_gravity`` is not positive.

    Notes
    -----
    Standing's fit for natural gas systems (sweet, dry gas):

    .. math::

        T_{pc} = 168 + 325\gamma_g - 12.5\gamma_g^2, \quad
        p_{pc} = 677 + 15\gamma_g - 37.5\gamma_g^2

    Standing, M. B. (1977), Volumetric and Phase Behavior of Oil Field
    Hydrocarbon Systems, SPE.

    Examples
    --------
    >>> tpc, ppc = pseudo_critical_standing(0.7)
    >>> round(tpc, 2), round(ppc, 2)
    (389.38, 669.12)
    """
    gg = check_positive(gas_gravity, "gas_gravity")
    tpc = 168.0 + 325.0 * gg - 12.5 * gg**2
    ppc = 677.0 + 15.0 * gg - 37.5 * gg**2
    return _out(tpc), _out(ppc)


def z_factor_hall_yarborough(
    ppr: npt.ArrayLike,
    tpr: npt.ArrayLike,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> FloatOrArray:
    r"""Gas compressibility factor from the Hall-Yarborough equation of state.

    Parameters
    ----------
    ppr : array_like
        Pseudo-reduced pressure.
    tpr : array_like
        Pseudo-reduced temperature.
    tol : float, optional
        Convergence tolerance on the reduced density. Default 1e-12.
    max_iter : int, optional
        Maximum Newton iterations. Default 100.

    Returns
    -------
    float or numpy.ndarray
        Compressibility factor z.

    Raises
    ------
    DataValidationError
        If ``ppr`` or ``tpr`` is not positive.

    Warns
    -----
    RangeWarning
        Outside 1.0 <= Tpr <= 3.0 or Ppr > 24.

    Notes
    -----
    With :math:`t = 1/T_{pr}`, the reduced density :math:`y` solves

    .. math::

        -A p_{pr} + \frac{y + y^2 + y^3 - y^4}{(1-y)^3} - B y^2 + C y^D = 0

    where :math:`A = 0.06125\,t\,e^{-1.2(1-t)^2}`,
    :math:`B = t(14.76 - 9.76t + 4.58t^2)`,
    :math:`C = t(90.7 - 242.2t + 42.4t^2)` and :math:`D = 2.18 + 2.82t`.
    Then :math:`z = A p_{pr} / y`. Solved here by a damped Newton method.

    Hall, K. R. and Yarborough, L. (1973), "A new equation of state for
    Z-factor calculations", Oil and Gas Journal 71(25), 82-92.

    Examples
    --------
    >>> round(z_factor_hall_yarborough(1.5, 2.0), 4)
    0.958
    """
    ppr_ = check_positive(ppr, "ppr")
    tpr_ = check_positive(tpr, "tpr")
    _warn(_HALL_YARBOROUGH, "Hall-Yarborough", ppr=ppr_, tpr=tpr_)
    ppr_b, tpr_b = np.broadcast_arrays(ppr_, tpr_)
    t = 1.0 / tpr_b
    a = 0.06125 * t * np.exp(-1.2 * (1.0 - t) ** 2)
    b = t * (14.76 - 9.76 * t + 4.58 * t**2)
    c = t * (90.7 - 242.2 * t + 42.4 * t**2)
    d = 2.18 + 2.82 * t
    # Start from the ideal-gas density (z = 1).
    y = np.clip(a * ppr_b, 1e-10, 0.5)
    for _ in range(max_iter):
        f = (
            -a * ppr_b
            + (y + y**2 + y**3 - y**4) / (1.0 - y) ** 3
            - b * y**2
            + c * np.power(y, d)
        )
        df = (
            (1.0 + 4.0 * y + 4.0 * y**2 - 4.0 * y**3 + y**4) / (1.0 - y) ** 4
            - 2.0 * b * y
            + c * d * np.power(y, d - 1.0)
        )
        step = f / df
        # Keep y inside (0, 1): halve the step whenever it would leave.
        y_new = y - step
        y_new = np.where((y_new <= 0.0) | (y_new >= 1.0), y - 0.5 * step, y_new)
        y_new = np.clip(y_new, 1e-12, 1.0 - 1e-9)
        if np.all(np.abs(y_new - y) < tol):
            y = y_new
            break
        y = y_new
    return _out(a * ppr_b / y)


def z_factor(
    pressure: npt.ArrayLike,
    temperature: npt.ArrayLike,
    gas_gravity: npt.ArrayLike,
) -> FloatOrArray:
    """Gas compressibility factor from pressure, temperature and gravity.

    Combines :func:`pseudo_critical_standing` with
    :func:`z_factor_hall_yarborough`.

    Parameters
    ----------
    pressure : array_like
        Pressure, psia.
    temperature : array_like
        Temperature, degF.
    gas_gravity : array_like
        Gas specific gravity (air = 1).

    Returns
    -------
    float or numpy.ndarray
        Compressibility factor z.

    Raises
    ------
    DataValidationError
        If ``pressure`` or ``gas_gravity`` is not positive.

    Examples
    --------
    >>> round(z_factor(2000.0, 180.0, 0.7), 3)
    0.844
    """
    p = check_positive(pressure, "pressure")
    t_r = as_float_array(temperature, "temperature") + _R_OFFSET
    tpc, ppc = pseudo_critical_standing(gas_gravity)
    return z_factor_hall_yarborough(p / np.asarray(ppc), t_r / np.asarray(tpc))


def gas_fvf(
    pressure: npt.ArrayLike,
    temperature: npt.ArrayLike,
    z: npt.ArrayLike,
) -> FloatOrArray:
    r"""Gas formation volume factor.

    Parameters
    ----------
    pressure : array_like
        Pressure, psia.
    temperature : array_like
        Temperature, degF.
    z : array_like
        Gas compressibility factor.

    Returns
    -------
    float or numpy.ndarray
        Gas formation volume factor, reservoir ft3 per scf. Divide by 5.615
        for bbl/scf.

    Raises
    ------
    DataValidationError
        If ``pressure`` or ``z`` is not positive.

    Notes
    -----
    From the real gas law between reservoir and standard conditions
    (14.7 psia, 60 degF):

    .. math:: B_g = \frac{p_{sc}}{T_{sc}} \frac{z T}{p} = 0.02829 \frac{z T}{p}

    with :math:`T` in degR.

    Examples
    --------
    >>> round(gas_fvf(2000.0, 180.0, 0.844), 6)
    0.007636
    """
    p = check_positive(pressure, "pressure")
    zz = check_positive(z, "z")
    t_r = as_float_array(temperature, "temperature") + _R_OFFSET
    return _out(_P_SC / _T_SC_R * zz * t_r / p)


# -----------------------------------------------------------------------------
# Black-oil table
# -----------------------------------------------------------------------------


def black_oil_table(
    pressure: npt.ArrayLike,
    temperature: float,
    api: float,
    gas_gravity: float,
    rs_bubble: float,
) -> pd.DataFrame:
    """Tabulate black-oil properties over a pressure range.

    Below the bubble point Standing's Rs and Bo and the Beggs-Robinson live
    viscosity are used; above it Rs is constant and Bo and viscosity follow
    Vasquez-Beggs. Gas z and Bg are reported at every pressure.

    Parameters
    ----------
    pressure : array_like
        Pressures, psia.
    temperature : float
        Reservoir temperature, degF.
    api : float
        Oil gravity, degrees API.
    gas_gravity : float
        Gas specific gravity (air = 1).
    rs_bubble : float
        Solution gas-oil ratio at the bubble point, scf/STB.

    Returns
    -------
    pandas.DataFrame
        Columns ``pressure`` (psia), ``rs`` (scf/STB), ``bo`` (bbl/STB),
        ``mu_o`` (cP), ``z`` and ``bg`` (ft3/scf). The bubble point is stored
        in ``attrs["bubble_point"]``.

    Examples
    --------
    >>> table = black_oil_table([1000.0, 3000.0], 200.0, 35.0, 0.75, 500.0)
    >>> table[["pressure", "rs"]].round(0)
       pressure     rs
    0    1000.0  196.0
    1    3000.0  500.0
    """
    p = np.atleast_1d(check_positive(pressure, "pressure"))
    pb = float(bubble_point_standing(rs_bubble, gas_gravity, api, temperature))
    rs_sat = np.asarray(solution_gor_standing(p, gas_gravity, api, temperature, pb))
    # Above the bubble point all gas is in solution, so Rs equals rs_bubble.
    rs = np.where(p >= pb, rs_bubble, rs_sat)
    bo_sat = np.asarray(oil_fvf_standing(rs, gas_gravity, api, temperature))
    bob = float(oil_fvf_standing(rs_bubble, gas_gravity, api, temperature))
    bo = np.where(
        p > pb,
        oil_fvf_undersaturated(p, pb, bob, rs_bubble, gas_gravity, api, temperature),
        bo_sat,
    )
    mu_od = float(dead_oil_viscosity_beggs_robinson(api, temperature))
    mu_sat = np.asarray(live_oil_viscosity_beggs_robinson(mu_od, rs))
    mu_ob = float(live_oil_viscosity_beggs_robinson(mu_od, rs_bubble))
    mu = np.where(
        p > pb, undersaturated_oil_viscosity_vasquez_beggs(p, pb, mu_ob), mu_sat
    )
    z = np.asarray(z_factor(p, temperature, gas_gravity))
    bg = np.asarray(gas_fvf(p, temperature, z))
    table: pd.DataFrame = pd.DataFrame(
        {"pressure": p, "rs": rs, "bo": bo, "mu_o": mu, "z": z, "bg": bg}
    )
    table.attrs["bubble_point"] = pb
    result: pd.DataFrame = table
    return result


# =============================================================================
# End of module: wellcast.pvt
# =============================================================================
