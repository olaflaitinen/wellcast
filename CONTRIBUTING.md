# Contributing to wellcast

Thank you for helping improve wellcast. This guide explains how to set up a
development environment and what a contribution needs before it can be merged.
Everyone taking part follows the [Code of Conduct](https://github.com/olaflaitinen/wellcast/blob/main/CODE_OF_CONDUCT.md).

## Ways to contribute

- Report a bug or request a feature through
  [GitHub Issues](https://github.com/olaflaitinen/wellcast/issues).
- Ask questions in [GitHub Discussions](https://github.com/olaflaitinen/wellcast/discussions).
- Improve the documentation or tutorials.
- Add a correlation, model or diagnostic, with tests against published values.

## Development setup

```bash
git clone https://github.com/olaflaitinen/wellcast.git
cd wellcast
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all,dev,docs]"
pre-commit install
```

## Branching

Work on a feature branch created from `main`, for example
`feat/beggs-brill-vlp` or `fix/hall-plot-gaps`, and open a pull request
against `main`. Keep each pull request focused on one change.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`<type>: <short imperative summary>`, with one logical change per commit.

| Type | Use for |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fixes |
| `docs` | Documentation only |
| `test` | Tests only |
| `ci` | GitHub Actions and other automation |
| `build` | Packaging and dependencies |
| `refactor` | Code changes that neither fix a bug nor add a feature |
| `chore` | Maintenance |

Example: `feat: add Beggs-Brill vertical lift performance`.

## Code standards

- Full type hints; `mypy --strict` must pass on `src/`.
- `ruff check` and `ruff format` must pass (pre-commit runs both).
- Public functions and classes have NumPy-style docstrings with Parameters,
  Returns, Raises, Notes (equation and reference) and an Examples section that
  runs as a doctest.
- Comments use `#` and explain the engineering reasoning: units, assumptions,
  valid ranges and the source of each correlation.
- Plain ASCII typography: no em dashes, en dashes or emojis.

### Mandatory file header and footer

Every Python file, including tests, scripts and app pages, starts and ends
with the block below. Adapt the purpose and module path.
`python scripts/check_headers.py` verifies it, and pre-commit runs it.

```python
# =============================================================================
# wellcast: <one-line purpose of this module>
# -----------------------------------------------------------------------------
# Module:      wellcast.<module>
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

...

# =============================================================================
# End of module: wellcast.<module>
# =============================================================================
```

YAML and TOML files carry a short `#` header with the copyright and SPDX
lines.

## Tests

```bash
pytest                                  # tests and doctests, coverage at least 90 percent
pytest --nbmake notebooks               # execute the tutorial notebooks
```

- The suite treats warnings as errors.
- Correlations are tested against published example values, with the
  reference in a `#` comment.
- Physical invariants (monotonic properties, unit round trips) get Hypothesis
  property tests.
- Use `make_synthetic_field` for data; never commit third-party data.

## Documentation

```bash
mkdocs serve                            # live preview
mkdocs build --strict                   # what CI runs
```

User guide pages explain the engineering background with equations; the API
reference is generated from docstrings.

## Pull request checklist

- [ ] Tests added or updated, and `pytest` passes.
- [ ] `ruff check`, `ruff format --check` and `mypy` pass.
- [ ] New Python files carry the header and footer.
- [ ] Docstrings and the user guide are updated.
- [ ] `CHANGELOG.md` has an entry under the next version.
- [ ] Commit messages follow Conventional Commits.

## License of contributions

wellcast is licensed under the [Mozilla Public License 2.0](https://github.com/olaflaitinen/wellcast/blob/main/LICENSE). By
submitting a contribution you agree that it is licensed under the MPL-2.0, as
described in section 3 of the license, and you confirm that you have the right
to submit it.
