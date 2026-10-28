# =============================================================================
# wellcast: exception hierarchy shared by all modules
# -----------------------------------------------------------------------------
# Module:      wellcast.exceptions
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

"""Exception classes raised by wellcast.

Every exception derives from :class:`WellcastError`, so callers can catch all
library errors with a single ``except`` clause. The concrete classes also
derive from the closest built-in exception (``ValueError`` or
``RuntimeError``) so existing code that catches those keeps working.
"""

from __future__ import annotations

__all__ = [
    "DataValidationError",
    "FitError",
    "MissingDependencyError",
    "UnitError",
    "WellcastError",
]


class WellcastError(Exception):
    """Base class for all wellcast errors.

    Examples
    --------
    >>> from wellcast.exceptions import FitError, WellcastError
    >>> issubclass(FitError, WellcastError)
    True
    """


class DataValidationError(WellcastError, ValueError):
    """Input data is missing, malformed or physically impossible.

    Raised for example when a required column is absent, a rate is negative,
    or a water cut falls outside the closed interval [0, 1].

    Examples
    --------
    >>> from wellcast.exceptions import DataValidationError
    >>> try:
    ...     raise DataValidationError("column 'oil_rate' is missing")
    ... except ValueError as exc:
    ...     print(exc)
    column 'oil_rate' is missing
    """


class FitError(WellcastError, RuntimeError):
    """A model or curve fit failed to converge or is not yet fitted.

    Examples
    --------
    >>> from wellcast.exceptions import FitError
    >>> isinstance(FitError("no convergence"), RuntimeError)
    True
    """


class UnitError(WellcastError, ValueError):
    """A unit is unknown or two units measure different physical quantities.

    Examples
    --------
    >>> from wellcast.exceptions import UnitError
    >>> isinstance(UnitError("cannot convert psi to degF"), ValueError)
    True
    """


class MissingDependencyError(WellcastError, ImportError):
    """An optional dependency needed by the requested feature is not installed.

    Parameters
    ----------
    package : str
        Name of the missing import, for example ``"lightgbm"``.
    extra : str
        Name of the wellcast extra that provides it, for example ``"ml"``.

    Examples
    --------
    >>> from wellcast.exceptions import MissingDependencyError
    >>> print(MissingDependencyError("lightgbm", "ml"))
    lightgbm is required for this feature.
    Install the ml extra: pip install 'wellcast[ml]'
    """

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"{package} is required for this feature. "
            f"Install the {extra} extra: pip install 'wellcast[{extra}]'"
        )


# =============================================================================
# End of module: wellcast.exceptions
# =============================================================================
