# =============================================================================
# wellcast: inflow performance relationships and multiphase choke equations
# -----------------------------------------------------------------------------
# Module:      wellcast.ipr
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

r"""Inflow performance relationships (IPR) and critical-flow choke equations.

Units: pressures in psia (or consistently psig), rates in STB/d, gas-liquid
ratio in scf/STB and choke size in 64ths of an inch. Inputs may be scalars or
arrays; scalar inputs return a ``float``.

An IPR gives the rate a well delivers for a given flowing bottomhole pressure
$p_{wf}$ at average reservoir pressure $\bar{p}_r$.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from wellcast._validation import as_float_array, check_non_negative, check_positive
from wellcast.exceptions import DataValidationError, FitError

__all__ = [
    "CHOKE_COEFFICIENTS",
    "baxendell_choke",
    "choke_pressure",
    "choke_rate",
    "composite_ipr",
    "composite_productivity_index",
    "fetkovich_ipr",
    "fit_fetkovich",
    "gilbert_choke",
    "ipr_curve",
    "linear_ipr",
    "productivity_index",
    "ros_choke",
    "vogel_ipr",
    "vogel_qmax",
]

FloatArray = npt.NDArray[np.float64]
FloatOrArray = float | FloatArray
ChokeMethod = Literal["gilbert", "ros", "baxendell", "achong"]


def _out(value: FloatArray) -> FloatOrArray:
    return float(value) if value.ndim == 0 else value


def _check_pwf(pwf: FloatArray, pr: FloatArray) -> None:
    if np.any(pwf > pr):
        raise DataValidationError("flowing pressure must not exceed reservoir pressure")


# -----------------------------------------------------------------------------
# Single-phase (straight-line) IPR
# -----------------------------------------------------------------------------


def productivity_index(
    rate: npt.ArrayLike,
    reservoir_pressure: npt.ArrayLike,
    flowing_pressure: npt.ArrayLike,
) -> FloatOrArray:
    r"""Productivity index from a stabilized rate test.

    Parameters
    ----------
    rate : array_like
        Stabilized liquid rate, STB/d.
    reservoir_pressure : array_like
        Average reservoir pressure, psi.
    flowing_pressure : array_like
        Flowing bottomhole pressure during the test, psi.

    Returns
    -------
    float or numpy.ndarray
        Productivity index J, STB/d/psi.

    Raises
    ------
    DataValidationError
        If the drawdown is not positive or the rate is negative.

    Notes
    -----
    $$
    J = \frac{q}{\bar{p}_r - p_{wf}}
    $$

    Examples
    --------
    >>> productivity_index(350.0, 2500.0, 2000.0)
    0.7
    """
    q = check_non_negative(rate, "rate")
    drawdown = as_float_array(
        reservoir_pressure, "reservoir_pressure"
    ) - as_float_array(flowing_pressure, "flowing_pressure")
    if np.any(drawdown <= 0):
        raise DataValidationError("drawdown (reservoir - flowing pressure) must be > 0")
    return _out(q / drawdown)


def linear_ipr(
    pwf: npt.ArrayLike,
    reservoir_pressure: float,
    productivity_index: float,
) -> FloatOrArray:
    r"""Straight-line IPR for single-phase (undersaturated) inflow.

    Parameters
    ----------
    pwf : array_like
        Flowing bottomhole pressure, psi.
    reservoir_pressure : float
        Average reservoir pressure, psi.
    productivity_index : float
        Productivity index J, STB/d/psi.

    Returns
    -------
    float or numpy.ndarray
        Rate, STB/d.

    Raises
    ------
    DataValidationError
        If ``pwf`` exceeds the reservoir pressure or J is not positive.

    Notes
    -----
    $$
    q = J (\bar{p}_r - p_{wf})
    $$

    Examples
    --------
    >>> linear_ipr(1850.0, 2500.0, 0.7)
    455.0
    """
    p = check_non_negative(pwf, "pwf")
    pr = check_positive(reservoir_pressure, "reservoir_pressure")
    j = check_positive(productivity_index, "productivity_index")
    _check_pwf(p, pr)
    return _out(j * (pr - p))


# -----------------------------------------------------------------------------
# Vogel (1968)
# -----------------------------------------------------------------------------


def _vogel_fraction(ratio: FloatArray) -> FloatArray:
    return 1.0 - 0.2 * ratio - 0.8 * ratio**2


def vogel_qmax(
    rate: float,
    pwf: float,
    reservoir_pressure: float,
) -> float:
    r"""Absolute open flow potential from one test, using Vogel's IPR.

    Parameters
    ----------
    rate : float
        Stabilized test rate, STB/d.
    pwf : float
        Flowing bottomhole pressure during the test, psi.
    reservoir_pressure : float
        Average reservoir pressure (at or below the bubble point), psi.

    Returns
    -------
    float
        Maximum rate $q_{o,max}$ at $p_{wf} = 0$, STB/d.

    Raises
    ------
    DataValidationError
        If ``pwf`` is not below the reservoir pressure.

    Notes
    -----
    $$
    q_{o,max} = \frac{q_o}{1 - 0.2 (p_{wf}/\bar{p}_r)
    - 0.8 (p_{wf}/\bar{p}_r)^2}
    $$

    Vogel, J. V. (1968), "Inflow performance relationships for solution-gas
    drive wells", JPT 20(1), 83-92, SPE-1476-PA.

    Examples
    --------
    >>> round(vogel_qmax(350.0, 2000.0, 2500.0), 2)
    1067.07
    """
    pr = float(check_positive(reservoir_pressure, "reservoir_pressure"))
    p = float(check_non_negative(pwf, "pwf"))
    if p >= pr:
        raise DataValidationError("pwf must be below the reservoir pressure")
    q = float(check_non_negative(rate, "rate"))
    return q / float(_vogel_fraction(np.asarray(p / pr)))


def vogel_ipr(
    pwf: npt.ArrayLike,
    reservoir_pressure: float,
    qmax: float,
) -> FloatOrArray:
    r"""Vogel IPR for a saturated (solution-gas drive) reservoir.

    Parameters
    ----------
    pwf : array_like
        Flowing bottomhole pressure, psi.
    reservoir_pressure : float
        Average reservoir pressure, psi.
    qmax : float
        Absolute open flow potential, STB/d (see `vogel_qmax`).

    Returns
    -------
    float or numpy.ndarray
        Oil rate, STB/d.

    Raises
    ------
    DataValidationError
        If ``pwf`` exceeds the reservoir pressure.

    Notes
    -----
    $$
    \frac{q_o}{q_{o,max}} = 1 - 0.2 \frac{p_{wf}}{\bar{p}_r}
    - 0.8 \left(\frac{p_{wf}}{\bar{p}_r}\right)^2
    $$

    Vogel (1968).

    Examples
    --------
    >>> round(vogel_ipr(1850.0, 2500.0, 1067.07), 1)
    441.7
    """
    p = check_non_negative(pwf, "pwf")
    pr = check_positive(reservoir_pressure, "reservoir_pressure")
    qm = check_positive(qmax, "qmax")
    _check_pwf(p, pr)
    return _out(qm * _vogel_fraction(p / pr))


# -----------------------------------------------------------------------------
# Fetkovich (1973)
# -----------------------------------------------------------------------------


def fetkovich_ipr(
    pwf: npt.ArrayLike,
    reservoir_pressure: float,
    c: float,
    n: float,
) -> FloatOrArray:
    r"""Fetkovich back-pressure IPR.

    Parameters
    ----------
    pwf : array_like
        Flowing bottomhole pressure, psi.
    reservoir_pressure : float
        Average reservoir pressure, psi.
    c : float
        Performance coefficient, STB/d/psi^(2n).
    n : float
        Deliverability exponent, usually 0.5 (turbulent) to 1.0 (laminar).

    Returns
    -------
    float or numpy.ndarray
        Oil rate, STB/d.

    Raises
    ------
    DataValidationError
        If ``pwf`` exceeds the reservoir pressure or ``c``, ``n`` are not
        positive.

    Notes
    -----
    $$
    q_o = C (\bar{p}_r^2 - p_{wf}^2)^n
    $$

    Fetkovich, M. J. (1973), "The isochronal testing of oil wells",
    SPE-4529-MS.

    Examples
    --------
    >>> round(fetkovich_ipr(2000.0, 2500.0, 1e-4, 1.0), 1)
    225.0
    """
    p = check_non_negative(pwf, "pwf")
    pr = check_positive(reservoir_pressure, "reservoir_pressure")
    cc = check_positive(c, "c")
    nn = check_positive(n, "n")
    _check_pwf(p, pr)
    return _out(cc * np.power(pr**2 - p**2, nn))


def fit_fetkovich(
    rates: npt.ArrayLike,
    pwfs: npt.ArrayLike,
    reservoir_pressure: float,
) -> tuple[float, float]:
    r"""Fit Fetkovich's C and n to a multi-rate test.

    Parameters
    ----------
    rates : array_like
        Stabilized rates, STB/d (at least two).
    pwfs : array_like
        Corresponding flowing bottomhole pressures, psi.
    reservoir_pressure : float
        Average reservoir pressure, psi.

    Returns
    -------
    c : float
        Performance coefficient.
    n : float
        Deliverability exponent.

    Raises
    ------
    DataValidationError
        If fewer than two valid points are given or shapes differ.
    FitError
        If the fitted exponent is not positive.

    Notes
    -----
    Least squares on $\log q = \log C + n \log(\bar{p}_r^2 - p_{wf}^2)$.

    Examples
    --------
    >>> import numpy as np
    >>> pwf = np.array([2200.0, 1800.0, 1200.0])
    >>> q = 2e-3 * (2500.0**2 - pwf**2) ** 0.8
    >>> c, n = fit_fetkovich(q, pwf, 2500.0)
    >>> round(n, 3), round(c, 5)
    (0.8, 0.002)
    """
    q = check_positive(rates, "rates")
    p = check_non_negative(pwfs, "pwfs")
    pr = float(check_positive(reservoir_pressure, "reservoir_pressure"))
    if q.shape != p.shape or q.size < 2:
        raise DataValidationError("need at least two (rate, pwf) pairs of equal length")
    _check_pwf(p, np.asarray(pr))
    x = np.log(pr**2 - p**2)
    if np.ptp(x) == 0:
        raise DataValidationError("test pressures must differ")
    n, log_c = np.polyfit(x, np.log(q), 1)
    if n <= 0:
        raise FitError(f"fitted exponent n = {n:.3f} is not positive")
    return float(np.exp(log_c)), float(n)


# -----------------------------------------------------------------------------
# Composite IPR (undersaturated above, Vogel below the bubble point)
# -----------------------------------------------------------------------------


def composite_ipr(
    pwf: npt.ArrayLike,
    reservoir_pressure: float,
    bubble_point: float,
    productivity_index: float,
) -> FloatOrArray:
    r"""Composite (generalized Vogel) IPR for an undersaturated reservoir.

    Parameters
    ----------
    pwf : array_like
        Flowing bottomhole pressure, psi.
    reservoir_pressure : float
        Average reservoir pressure, psi, at or above the bubble point.
    bubble_point : float
        Bubble-point pressure, psi.
    productivity_index : float
        Productivity index above the bubble point, STB/d/psi.

    Returns
    -------
    float or numpy.ndarray
        Oil rate, STB/d.

    Raises
    ------
    DataValidationError
        If the bubble point exceeds the reservoir pressure or ``pwf`` exceeds
        the reservoir pressure.

    Notes
    -----
    With $q_b = J(\bar{p}_r - p_b)$:

    $$
    q = \begin{cases}
    J(\bar{p}_r - p_{wf}) & p_{wf} \ge p_b \\
    q_b + \frac{J p_b}{1.8}\left[1 - 0.2\frac{p_{wf}}{p_b}
    - 0.8\left(\frac{p_{wf}}{p_b}\right)^2\right] & p_{wf} < p_b
    \end{cases}
    $$

    The two branches join with equal slope at $p_b$. Brown, K. E.
    (1984), The Technology of Artificial Lift Methods, Vol. 4, PennWell.

    Examples
    --------
    >>> round(composite_ipr(0.0, 4000.0, 2000.0, 1.4), 1)
    4355.6
    """
    p = check_non_negative(pwf, "pwf")
    pr = float(check_positive(reservoir_pressure, "reservoir_pressure"))
    pb = float(check_positive(bubble_point, "bubble_point"))
    j = float(check_positive(productivity_index, "productivity_index"))
    if pb > pr:
        raise DataValidationError("bubble point must not exceed reservoir pressure")
    _check_pwf(p, np.asarray(pr))
    qb = j * (pr - pb)
    below = qb + j * pb / 1.8 * _vogel_fraction(p / pb)
    return _out(np.where(p >= pb, j * (pr - p), below))


def composite_productivity_index(
    rate: float,
    pwf: float,
    reservoir_pressure: float,
    bubble_point: float,
) -> float:
    r"""Productivity index for `composite_ipr` from one rate test.

    Parameters
    ----------
    rate : float
        Stabilized test rate, STB/d.
    pwf : float
        Flowing bottomhole pressure during the test, psi.
    reservoir_pressure : float
        Average reservoir pressure, psi.
    bubble_point : float
        Bubble-point pressure, psi.

    Returns
    -------
    float
        Productivity index J above the bubble point, STB/d/psi.

    Raises
    ------
    DataValidationError
        If ``pwf`` is not below the reservoir pressure.

    Notes
    -----
    For a test above the bubble point $J = q/(\bar{p}_r - p_{wf})$;
    below it

    $$
    J = \frac{q}{\bar{p}_r - p_b + \frac{p_b}{1.8}\left[1 - 0.2
    \frac{p_{wf}}{p_b} - 0.8\left(\frac{p_{wf}}{p_b}\right)^2\right]}
    $$

    Examples
    --------
    >>> round(composite_productivity_index(3000.0, 1500.0, 4000.0, 2000.0), 4)
    1.2273
    """
    pr = float(check_positive(reservoir_pressure, "reservoir_pressure"))
    pb = float(check_positive(bubble_point, "bubble_point"))
    p = float(check_non_negative(pwf, "pwf"))
    q = float(check_non_negative(rate, "rate"))
    if p >= pr:
        raise DataValidationError("pwf must be below the reservoir pressure")
    if p >= pb:
        return q / (pr - p)
    denominator = pr - pb + pb / 1.8 * float(_vogel_fraction(np.asarray(p / pb)))
    return q / denominator


def ipr_curve(
    model: Literal["linear", "vogel", "fetkovich", "composite"],
    reservoir_pressure: float,
    *,
    n_points: int = 50,
    **params: float,
) -> pd.DataFrame:
    """Tabulate an IPR curve from zero to reservoir pressure.

    Parameters
    ----------
    model : {"linear", "vogel", "fetkovich", "composite"}
        IPR model.
    reservoir_pressure : float
        Average reservoir pressure, psi.
    n_points : int, optional
        Number of pressures. Default 50.
    **params
        Model parameters: ``productivity_index`` (linear), ``qmax`` (vogel),
        ``c`` and ``n`` (fetkovich), ``bubble_point`` and
        ``productivity_index`` (composite).

    Returns
    -------
    pandas.DataFrame
        Columns ``pwf`` (psi, decreasing) and ``rate`` (STB/d).

    Raises
    ------
    DataValidationError
        If ``model`` is unknown or ``n_points`` < 2.

    Examples
    --------
    >>> curve = ipr_curve("vogel", 2500.0, n_points=3, qmax=1000.0)
    >>> curve["rate"].tolist()
    [0.0, 700.0, 1000.0]
    """
    if n_points < 2:
        raise DataValidationError("n_points must be >= 2")
    pwf = np.linspace(reservoir_pressure, 0.0, n_points)
    if model == "linear":
        rate = linear_ipr(pwf, reservoir_pressure, params["productivity_index"])
    elif model == "vogel":
        rate = vogel_ipr(pwf, reservoir_pressure, params["qmax"])
    elif model == "fetkovich":
        rate = fetkovich_ipr(pwf, reservoir_pressure, params["c"], params["n"])
    elif model == "composite":
        rate = composite_ipr(
            pwf,
            reservoir_pressure,
            params["bubble_point"],
            params["productivity_index"],
        )
    else:
        raise DataValidationError(f"unknown IPR model {model!r}")
    curve: pd.DataFrame = pd.DataFrame({"pwf": pwf, "rate": rate})
    return curve


# -----------------------------------------------------------------------------
# Critical-flow choke equations
# -----------------------------------------------------------------------------

# Coefficients (A, B, C) of p_wh = A * q * R^B / S^C with p_wh in psi, q in
# STB/d, R (gas-liquid ratio) in scf/STB and S (bean size) in 64ths of an
# inch, as tabulated by Beggs, H. D. (1991), Production Optimization Using
# Nodal Analysis, OGCI, and Brill and Mukherjee (1999), Multiphase Flow in
# Wells, SPE Monograph 17. Gilbert's original form, 435 R^0.546 with R in
# Mscf/STB, is the same equation (435 * 1000^-0.546 = 10.0).
CHOKE_COEFFICIENTS: dict[str, tuple[float, float, float]] = {
    "gilbert": (10.00, 0.546, 1.89),
    "ros": (17.40, 0.500, 2.00),
    "baxendell": (9.56, 0.546, 1.93),
    "achong": (3.82, 0.650, 1.88),
}


def _coefficients(method: str) -> tuple[float, float, float]:
    try:
        return CHOKE_COEFFICIENTS[method]
    except KeyError:
        raise DataValidationError(
            f"unknown choke method {method!r}; use one of {sorted(CHOKE_COEFFICIENTS)}"
        ) from None


def choke_pressure(
    rate: npt.ArrayLike,
    glr: npt.ArrayLike,
    choke_size: npt.ArrayLike,
    method: ChokeMethod = "gilbert",
) -> FloatOrArray:
    r"""Upstream (wellhead) pressure for critical two-phase flow through a choke.

    Parameters
    ----------
    rate : array_like
        Liquid rate, STB/d.
    glr : array_like
        Producing gas-liquid ratio, scf/STB.
    choke_size : array_like
        Bean size in 64ths of an inch.
    method : {"gilbert", "ros", "baxendell", "achong"}, optional
        Correlation. Default ``"gilbert"``.

    Returns
    -------
    float or numpy.ndarray
        Upstream pressure, psi.

    Raises
    ------
    DataValidationError
        If ``method`` is unknown, the rate is negative, or ``glr`` or
        ``choke_size`` is not positive.

    Notes
    -----
    $$
    p_{wh} = \frac{A\, q_L\, R^{B}}{S^{C}}
    $$

    The equations assume critical flow, roughly when the downstream pressure
    is below 0.588 times the upstream pressure. Gilbert, W. E. (1954),
    "Flowing and gas-lift well performance", Drilling and Production
    Practice, API, 126-157.

    Examples
    --------
    >>> round(choke_pressure(1000.0, 1000.0, 32.0), 1)
    621.3
    """
    a, b, c = _coefficients(method)
    q = check_non_negative(rate, "rate")
    r = check_positive(glr, "glr")
    s = check_positive(choke_size, "choke_size")
    return _out(a * q * np.power(r, b) / np.power(s, c))


def choke_rate(
    pressure: npt.ArrayLike,
    glr: npt.ArrayLike,
    choke_size: npt.ArrayLike,
    method: ChokeMethod = "gilbert",
) -> FloatOrArray:
    r"""Liquid rate through a choke for a given upstream pressure.

    Inverse of `choke_pressure`.

    Parameters
    ----------
    pressure : array_like
        Upstream (wellhead) pressure, psi.
    glr : array_like
        Producing gas-liquid ratio, scf/STB.
    choke_size : array_like
        Bean size in 64ths of an inch.
    method : {"gilbert", "ros", "baxendell", "achong"}, optional
        Correlation. Default ``"gilbert"``.

    Returns
    -------
    float or numpy.ndarray
        Liquid rate, STB/d.

    Raises
    ------
    DataValidationError
        If ``method`` is unknown or an input is out of range.

    Notes
    -----
    $$
    q_L = \frac{p_{wh} S^{C}}{A R^{B}}
    $$

    Examples
    --------
    >>> round(choke_rate(621.25, 1000.0, 32.0), 1)
    1000.0
    """
    a, b, c = _coefficients(method)
    p = check_non_negative(pressure, "pressure")
    r = check_positive(glr, "glr")
    s = check_positive(choke_size, "choke_size")
    return _out(p * np.power(s, c) / (a * np.power(r, b)))


def gilbert_choke(
    rate: npt.ArrayLike, glr: npt.ArrayLike, choke_size: npt.ArrayLike
) -> FloatOrArray:
    """Upstream pressure from Gilbert's choke equation.

    Shortcut for ``choke_pressure(..., method="gilbert")``; see
    `choke_pressure` for parameters and units.

    Parameters
    ----------
    rate : array_like
        Liquid rate, STB/d.
    glr : array_like
        Gas-liquid ratio, scf/STB.
    choke_size : array_like
        Bean size, 64ths of an inch.

    Returns
    -------
    float or numpy.ndarray
        Upstream pressure, psi.

    Examples
    --------
    >>> round(gilbert_choke(1000.0, 1000.0, 32.0), 1)
    621.3
    """
    return choke_pressure(rate, glr, choke_size, "gilbert")


def ros_choke(
    rate: npt.ArrayLike, glr: npt.ArrayLike, choke_size: npt.ArrayLike
) -> FloatOrArray:
    """Upstream pressure from Ros's choke equation.

    Shortcut for ``choke_pressure(..., method="ros")``. Ros, N. C. J. (1960),
    "An analysis of critical simultaneous gas/liquid flow through a
    restriction and its application to flowmetering", Applied Scientific
    Research 9, 374-388.

    Parameters
    ----------
    rate : array_like
        Liquid rate, STB/d.
    glr : array_like
        Gas-liquid ratio, scf/STB.
    choke_size : array_like
        Bean size, 64ths of an inch.

    Returns
    -------
    float or numpy.ndarray
        Upstream pressure, psi.

    Examples
    --------
    >>> round(ros_choke(1000.0, 1000.0, 32.0), 1)
    537.3
    """
    return choke_pressure(rate, glr, choke_size, "ros")


def baxendell_choke(
    rate: npt.ArrayLike, glr: npt.ArrayLike, choke_size: npt.ArrayLike
) -> FloatOrArray:
    """Upstream pressure from Baxendell's choke equation.

    Shortcut for ``choke_pressure(..., method="baxendell")``. Baxendell, P. B.
    (1957), "Bean performance: Lake Maracaibo wells", Internal Company Report,
    Shell Oil Co.

    Parameters
    ----------
    rate : array_like
        Liquid rate, STB/d.
    glr : array_like
        Gas-liquid ratio, scf/STB.
    choke_size : array_like
        Bean size, 64ths of an inch.

    Returns
    -------
    float or numpy.ndarray
        Upstream pressure, psi.

    Examples
    --------
    >>> round(baxendell_choke(1000.0, 1000.0, 32.0), 1)
    517.0
    """
    return choke_pressure(rate, glr, choke_size, "baxendell")


# =============================================================================
# End of module: wellcast.ipr
# =============================================================================
