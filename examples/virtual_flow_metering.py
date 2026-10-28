# =============================================================================
# wellcast: bottomhole pressure virtual flow meter
# -----------------------------------------------------------------------------
# Module:      examples/virtual_flow_metering.py
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

"""Train and validate a virtual flow meter for bottomhole pressure.

Requires the ``ml`` extra. Run with ``python examples/virtual_flow_metering.py``.
"""

from __future__ import annotations

import wellcast
from wellcast.features import build_features
from wellcast.ml import Predictor


def main() -> None:
    production = wellcast.make_synthetic_field(
        n_wells=3, years=3, n_injectors=0
    ).production
    data = build_features(production)
    model = Predictor("bhp", params={"n_estimators": 150})

    cv = model.cross_validate(data, "lowo")
    print("Leave-one-well-out validation (BHP, psi):")
    print(cv[cv["well"] != "all"].round(3).to_string(index=False))

    model.fit(data)
    shap = (
        model.shap_values(data, max_samples=200)
        .abs()
        .mean()
        .sort_values(ascending=False)
    )
    print("\nMost important features (mean |SHAP|, psi):")
    print(shap.head(5).round(2).to_string())


if __name__ == "__main__":
    main()

# =============================================================================
# End of module: examples/virtual_flow_metering.py
# =============================================================================
