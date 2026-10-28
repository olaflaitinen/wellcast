# =============================================================================
# wellcast: shared pytest fixtures and Hypothesis settings
# -----------------------------------------------------------------------------
# Module:      tests/conftest.py
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

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, settings

# Deadlines are disabled because CI runners (especially Windows and macOS) have
# noisy timing; correctness, not speed, is what the property tests check.
settings.register_profile(
    "wellcast",
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci", parent=settings.get_profile("wellcast"), max_examples=200
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "wellcast"))


@pytest.fixture
def two_well_frame() -> pd.DataFrame:
    """Small daily production table for two wells in the canonical schema."""
    idx = pd.date_range("2024-01-01", periods=6, freq="D", name="date")
    a = pd.DataFrame(
        {
            "well": "A",
            "oil_rate": [100.0, 98.0, 0.0, 97.0, 95.0, 94.0],
            "water_rate": [10.0, 11.0, 0.0, 12.0, 12.5, 13.0],
            "bhp": [3000.0, 2990.0, 3100.0, np.nan, 2970.0, 2965.0],
            "on_stream_hours": [24.0, 24.0, 0.0, 24.0, 24.0, 24.0],
        },
        index=idx,
    )
    b = a.assign(well="B", oil_rate=a["oil_rate"] * 2.0)
    return pd.concat([a, b])


# =============================================================================
# End of module: tests/conftest.py
# =============================================================================
