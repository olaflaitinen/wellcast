# =============================================================================
# wellcast: verify that Python files carry the mandatory header and footer
# -----------------------------------------------------------------------------
# Module:      scripts/check_headers.py
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

"""Check the wellcast source header and footer on Python files.

Usage: ``python scripts/check_headers.py [FILE ...]``. Without arguments every
tracked ``*.py`` file outside virtual environments and build output is checked.
Exits with status 1 and lists offending files if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

RULE = "# " + "=" * 77
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "site", "__pycache__"}

# Lines every header must contain, in this order, after the opening rule.
REQUIRED_HEADER_LINES = (
    "# wellcast: ",
    "# Module:      ",
    "# Repository:  https://github.com/olaflaitinen/wellcast",
    "# Author:      Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>",
    "# Copyright:   Copyright (c) 2026 Olaf Yunus Laitinen Imanov",
    "# License:     Mozilla Public License 2.0",
    "# SPDX-License-Identifier: MPL-2.0",
)


def check_file(path: Path) -> list[str]:
    """Return a list of problems found in ``path`` (empty when it passes)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    problems: list[str] = []
    if not lines or lines[0] != RULE:
        problems.append("first line is not the header rule")
    head = lines[:16]
    position = 0
    for required in REQUIRED_HEADER_LINES:
        for offset, line in enumerate(head[position:], start=position):
            if line.startswith(required):
                position = offset + 1
                break
        else:
            problems.append(f"header line missing: {required.strip()}")
    tail = lines[-3:]
    if (
        len(tail) != 3
        or tail[0] != RULE
        or not tail[1].startswith("# End of module: ")
        or tail[2] != RULE
    ):
        problems.append("footer missing or malformed")
    return problems


def iter_python_files(root: Path) -> list[Path]:
    """Return every ``*.py`` file below ``root`` outside skipped directories."""
    return sorted(
        p
        for p in root.rglob("*.py")
        if not SKIP_DIRS.intersection(part for part in p.relative_to(root).parts)
    )


def main(argv: list[str]) -> int:
    """Check the given files (or the whole tree) and report failures."""
    root = Path(__file__).resolve().parent.parent
    files = [Path(a) for a in argv] if argv else iter_python_files(root)
    failed = 0
    for path in files:
        for problem in check_file(path):
            print(f"{path}: {problem}")
            failed += 1
    if not argv:
        print(f"checked {len(files)} file(s), {failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# =============================================================================
# End of module: scripts/check_headers.py
# =============================================================================
