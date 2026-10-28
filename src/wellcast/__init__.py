# =============================================================================
# wellcast: public API re-exports and package version
# -----------------------------------------------------------------------------
# Module:      wellcast
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

"""wellcast: oil and gas well performance analysis.

Field data ingestion and QC, decline curve analysis, PVT, inflow performance,
well and reservoir diagnostics, and machine learning for virtual flow metering.

Examples
--------
>>> import wellcast
>>> wellcast.__version__
'0.1.0'
"""

from wellcast import clean, datasets, dca, io, ipr, pvt, units
from wellcast.__about__ import (
    __author__,
    __copyright__,
    __email__,
    __license__,
    __version__,
)
from wellcast.clean import qc_report
from wellcast.datasets import make_synthetic_field
from wellcast.dca import DeclineModel, fit_many
from wellcast.exceptions import (
    DataValidationError,
    FitError,
    MissingDependencyError,
    UnitError,
    WellcastError,
)
from wellcast.io import read_production
from wellcast.units import convert

__all__ = [
    "DataValidationError",
    "DeclineModel",
    "FitError",
    "MissingDependencyError",
    "UnitError",
    "WellcastError",
    "__author__",
    "__copyright__",
    "__email__",
    "__license__",
    "__version__",
    "clean",
    "convert",
    "datasets",
    "dca",
    "fit_many",
    "io",
    "ipr",
    "make_synthetic_field",
    "pvt",
    "qc_report",
    "read_production",
    "units",
]

# =============================================================================
# End of module: wellcast
# =============================================================================
