# =============================================================================
# wellcast: field-wide Arps decline analysis and EUR table
# -----------------------------------------------------------------------------
# Module:      examples/decline_curve_analysis.py
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

"""Fit Arps declines for every well and write an EUR table.

Run with ``python examples/decline_curve_analysis.py [output.csv]``.
"""

from __future__ import annotations

import sys

import wellcast
from wellcast.dca import fit_many


def main(output: str | None = None) -> None:
    production = wellcast.make_synthetic_field(seed=7).production
    table = fit_many(production, econ_limit=15.0)
    columns = ["model", "qi", "di", "b", "eur", "remaining", "status"]
    print(table[columns].round(3).to_string())
    total = table.loc[table["status"] == "ok", "eur"].sum()
    print(f"\nField EUR (oil, decline period): {total:,.0f} bbl")
    if output:
        table.to_csv(output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

# =============================================================================
# End of module: examples/decline_curve_analysis.py
# =============================================================================
