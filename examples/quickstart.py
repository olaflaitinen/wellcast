# =============================================================================
# wellcast: 60-second tour on a synthetic field
# -----------------------------------------------------------------------------
# Module:      examples/quickstart.py
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

"""Quickstart: synthetic data, quality control, decline curve and EUR.

Run with ``python examples/quickstart.py``.
"""

from __future__ import annotations

import wellcast


def main() -> None:
    field = wellcast.make_synthetic_field(n_wells=3, years=3, seed=42)
    production = field.production
    print(f"{production['well'].nunique()} wells, {len(production)} daily rows")

    report = wellcast.qc_report(production)
    print("\nColumns with data problems:")
    problems = report[(report["negative"] > 0) | (report["impossible"] > 0)]
    print(problems if len(problems) else "none")

    rates = production.loc[production["well"] == "P-01", "oil_rate"]
    model = wellcast.DeclineModel().fit(rates).bootstrap(100)
    print(f"\n{model}")
    eur = model.eur_quantiles(econ_limit=20.0)
    for name in ("p90", "p50", "p10"):
        print(f"EUR {name.upper()}: {eur[name]:,.0f} bbl")


if __name__ == "__main__":
    main()

# =============================================================================
# End of module: examples/quickstart.py
# =============================================================================
