# Report: 3-ruff-precommit — Replace flake8 with ruff in pre-commit

Task: Replace flake8 with ruff for the pre-commit lint hook and remove the
flake8 dependency.

Branch: 3/ruff-precommit

Plan approved: 2026-10-22 (`docs/plans/3-ruff-precommit.md`)

## Outcome
Completed.

## Changes
- `.pre-commit-config.yaml`: flake8 hook replaced by
  `astral-sh/ruff-pre-commit` v0.17.2, hook `ruff-check` with
  `types_or: [python, pyi]` (notebooks not linted, as with flake8). Comment
  updated.
- `pyproject.toml`: `[tool.ruff]` `line-length = 88`; `[tool.ruff.lint]`
  `select = ["E", "F", "W"]`, `ignore = ["E501"]`. Dev extra: `flake8>=7.0`
  → `ruff>=0.17`.
- `.flake8` deleted.
- `uv.lock` regenerated: removed flake8 7.4.1, mccabe, pycodestyle,
  pyflakes; added ruff 0.17.2.
- `README.md`: dev-extras comment lists ruff instead of flake8.
- `odil_wave/metrics/__init__.py`, `odil_wave/operator/conditions.py`:
  missing final newlines added; `odil_wave/grid/frequency_selection.py` and
  `conditions.py` reformatted by black (see deviations).

## Deviations from the plan
- Step 3 also includes black reformatting of
  `odil_wave/grid/frequency_selection.py` and two expressions in
  `odil_wave/operator/conditions.py`. These were pre-existing black
  violations surfaced by `pre-commit run --all-files`; they are
  formatting-only and were required to meet the agreed acceptance criterion
  that the full pre-commit run passes. No separate approval was sought, as
  the change has no behavioural effect and uses the already-configured
  formatter.

## Verification
| Check/command | Configuration | Result | Evidence |
|---------------|---------------|--------|----------|
| `pre-commit run --all-files` | black 24.8.0, ruff-check v0.17.2 | Passed (both hooks) | after commit `be17f15` |
| `uv run ruff check --output-format concise odil_wave tests main.py` | `[tool.ruff]` in `pyproject.toml` | All checks passed | — |
| `uv run ruff check .` | same | 12 × E702 in `notebooks/*.ipynb`, 0 in `.py` | notebooks are outside the hook's `types_or` |
| `grep -rn flake8 --exclude-dir=.venv --exclude-dir=.git .` | — | Only hits: this task's plan/report and the untracked, stale `odil_wave.egg-info/` | — |
| `uv run python -m pytest tests/ -q` | Python 3.13 venv | 210 passed | — |

## Interpretation and limitations
- The ruff rule selection reproduces flake8's default pyflakes and
  non-preview pycodestyle checks; ruff's preview-only E1/E2/E3 whitespace
  rules are off, which black covers. Not verified by a side-by-side flake8
  run (flake8 was not installed in the venv).
- A bare `ruff check .` still reports the notebook findings, since the
  notebook exclusion lives in the pre-commit hook, not in `[tool.ruff]`.
- `odil_wave.egg-info/` is untracked build output and will refresh on the
  next `pip install -e .`.

## Knowledge updates
None: lint tooling is not a capability tracked in `docs/features.md`, and the
swap is routine maintenance rather than a consequential decision.

## Follow-up
Optional: lint notebooks (fix the 12 E702 findings) or add `extend-exclude`
for notebooks in `[tool.ruff]`; add a lint step to CI.
