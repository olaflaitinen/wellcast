# =============================================================================
# wellcast: Hall plot, Chan diagnostics and anomaly detection
# -----------------------------------------------------------------------------
# Module:      examples/well_diagnostics.py
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

"""Injectivity, water production mechanism and anomalies on a synthetic field.

Run with ``python examples/well_diagnostics.py``.
"""

from __future__ import annotations

import wellcast
from wellcast import diagnostics as dg


def main() -> None:
    field = wellcast.make_synthetic_field(seed=42)
    production = field.production

    injector = production[production["well"] == "I-01"]
    truth = field.wells.loc["I-01"]
    hall = dg.hall_plot_data(injector, reservoir_pressure=truth["initial_pressure"])
    change = dg.hall_slope_change(hall)
    print(
        f"Hall plot: slope x{change['ratio']:.2f} on {change['break_date']:%Y-%m-%d} "
        f"(true injectivity loss {truth['injectivity_loss_date']:%Y-%m-%d})"
    )

    producers = production[production["well"].str.startswith("P")]
    print("\nWater production mechanism (Chan screening):")
    print(
        dg.classify_water_mechanism(dg.chan_plot_data(producers)).round(2).to_string()
    )

    anomalies = dg.detect_anomalies(production, method="rules")
    print(f"\n{len(anomalies)} anomalous periods found by the physical rules:")
    print(anomalies.groupby("detector").size().to_string())


if __name__ == "__main__":
    main()

# =============================================================================
# End of module: examples/well_diagnostics.py
# =============================================================================
