# =============================================================================
# wellcast: tests for package metadata, exceptions and shared validation
# -----------------------------------------------------------------------------
# Module:      tests/test_package.py
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

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wellcast
from wellcast import _validation as v
from wellcast.exceptions import (
    DataValidationError,
    FitError,
    MissingDependencyError,
    UnitError,
    WellcastError,
)

ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", wellcast.__version__)
    assert wellcast.__version__ == "0.1.0"


def test_author_metadata() -> None:
    assert wellcast.__author__ == "Olaf Yunus Laitinen Imanov"
    assert wellcast.__email__ == "yunus.z.imanov@helsinki.fi"
    assert wellcast.__license__ == "MPL-2.0"
    assert wellcast.__copyright__ == "Copyright (c) 2026 Olaf Yunus Laitinen Imanov"


def test_py_typed_marker_ships() -> None:
    assert (ROOT / "src" / "wellcast" / "py.typed").is_file()


def test_all_python_files_have_header_and_footer() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_headers.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "builtin"),
    [
        (DataValidationError, ValueError),
        (UnitError, ValueError),
        (FitError, RuntimeError),
        (MissingDependencyError, ImportError),
    ],
)
def test_exception_hierarchy(exc: type[Exception], builtin: type[Exception]) -> None:
    assert issubclass(exc, WellcastError)
    assert issubclass(exc, builtin)


def test_missing_dependency_message() -> None:
    err = MissingDependencyError("shap", "ml")
    assert err.package == "shap"
    assert err.extra == "ml"
    assert "pip install 'wellcast[ml]'" in str(err)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def test_as_float_array_rejects_text() -> None:
    with pytest.raises(DataValidationError, match="numeric"):
        v.as_float_array(["a"], "x")


def test_check_positive() -> None:
    np.testing.assert_array_equal(v.check_positive([1.0, np.nan], "p"), [1.0, np.nan])
    with pytest.raises(DataValidationError, match="strictly positive"):
        v.check_positive([1.0, 0.0], "p")


def test_check_non_negative() -> None:
    with pytest.raises(DataValidationError, match="non-negative"):
        v.check_non_negative(-1.0, "q")


def test_check_fraction() -> None:
    v.check_fraction([0.0, 1.0, np.nan], "wc")
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        v.check_fraction([1.2], "wc")


def test_warn_outside_range() -> None:
    with pytest.warns(v.RangeWarning, match="Standing"):
        v.warn_outside_range([90.0, 120.0], "T", 100.0, 258.0, "Standing")


def test_warn_outside_range_ignores_all_nan() -> None:
    v.warn_outside_range([np.nan], "T", 100.0, 258.0, "Standing")


def test_require_columns() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="b, c"):
        v.require_columns(df, ["a", "b", "c"])


def test_require_extra_missing() -> None:
    with pytest.raises(MissingDependencyError, match=r"wellcast\[xyz\]"):
        v.require_extra("wellcast_no_such_module", "xyz")


# =============================================================================
# End of module: tests/test_package.py
# =============================================================================
