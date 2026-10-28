# =============================================================================
# wellcast: black-oil PVT table and inflow performance curves
# -----------------------------------------------------------------------------
# Module:      examples/pvt_and_inflow.py
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

"""PVT properties over a pressure range and IPR curves from a well test.

Run with ``python examples/pvt_and_inflow.py``.
"""

from __future__ import annotations

import numpy as np

from wellcast import ipr, pvt


def main() -> None:
    # Reservoir fluid: 35 API oil, 0.75 gas gravity, 500 scf/STB at 200 degF.
    table = pvt.black_oil_table(
        np.linspace(500.0, 5000.0, 10), 200.0, 35.0, 0.75, 500.0
    )
    print(f"Bubble point (Standing): {table.attrs['bubble_point']:.0f} psia")
    print(table.round(4).to_string(index=False))

    # One stabilized test: 1200 STB/d at 2600 psi, reservoir 3500 psi.
    pr, pb, q_test, pwf_test = 3500.0, table.attrs["bubble_point"], 1200.0, 2600.0
    j = ipr.composite_productivity_index(q_test, pwf_test, pr, pb)
    qmax = ipr.vogel_qmax(q_test, pwf_test, pr)
    print(f"\nComposite productivity index: {j:.3f} STB/d/psi")
    print(f"Vogel absolute open flow:     {qmax:,.0f} STB/d")
    for pwf in (2000.0, 1000.0, 0.0):
        q = ipr.composite_ipr(pwf, pr, pb, j)
        print(f"  composite IPR at {pwf:6.0f} psi: {q:8,.0f} STB/d")

    # Wellhead pressure needed to flow 1500 STB/d through a 32/64 in choke.
    for method in ("gilbert", "ros", "baxendell"):
        p_wh = ipr.choke_pressure(1500.0, 800.0, 32.0, method)  # type: ignore[arg-type]
        print(f"  {method:9s} choke: {p_wh:6.0f} psi")


if __name__ == "__main__":
    main()

# =============================================================================
# End of module: examples/pvt_and_inflow.py
# =============================================================================
