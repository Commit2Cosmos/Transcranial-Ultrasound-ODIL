# Plan: 3-ruff-precommit — Replace flake8 with ruff in pre-commit

Task: Replace flake8 with ruff for the pre-commit lint hook and remove the
flake8 dependency.

Branch: 3/ruff-precommit

Status: Approved 2026-10-22

## Objective
Lint staged Python files with `ruff check` instead of flake8 in pre-commit,
keeping the current lint rule set, and drop flake8 from the project's
dependencies and configuration.

## Repository findings
- `.pre-commit-config.yaml`: two hooks, `black` (24.8.0) and `flake8`
  (7.0.0), plus a trailing comment naming both.
- `.flake8`: `max-line-length = 88`, `extend-ignore = E203, W503, E501`.
- `pyproject.toml` `[project.optional-dependencies].dev` lists
  `flake8>=7.0`; `uv.lock` pins `flake8` 7.4.1 (and its transitive
  `pycodestyle`/`pyflakes`/`mccabe`).
- `README.md:169` lists flake8 among the dev extras.
- CI (`.github/workflows/tests.yml`) runs pytest only; no lint step.
- No feature-register entry or decision record covers lint tooling.
- Dry run (`uvx ruff check --select E,F,W --ignore E501`, ruff 0.17.2):
  - `.py` files: 2 × W292 (missing final newline) in
    `odil_wave/metrics/__init__.py` and `odil_wave/operator/conditions.py`.
    flake8 flags these too, so they are pre-existing violations.
  - Notebooks: 12 findings (E702, W292) in `notebooks/*.ipynb`. flake8 never
    linted notebooks, but ruff's pre-commit hook does by default.

## Scope and non-goals
In scope:
- Swap the flake8 hook for `astral-sh/ruff-pre-commit` (`ruff-check`).
- Move the lint configuration into `pyproject.toml` `[tool.ruff]`, delete
  `.flake8`.
- Replace `flake8` with `ruff` in the `dev` extra; regenerate `uv.lock`.
- Update the README dev-extras comment and the pre-commit config comment.
- Fix the 2 pre-existing W292 violations in `.py` files (trailing newline
  only) so `pre-commit run --all-files` passes.

Non-goals:
- Replacing black with `ruff format` (black stays).
- Enabling additional ruff rule sets (isort, pyupgrade, bugbear, …).
- Linting notebooks or adding lint to CI.

## Proposed approach
`.pre-commit-config.yaml`:
```yaml
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.2
    hooks:
      - id: ruff-check
        types_or: [python, pyi]
```
`types_or` overrides the hook default (`[python, pyi, jupyter]`) so
notebooks stay unlinted, matching flake8's behaviour. No `--fix`: the hook
reports, as flake8 did.

`pyproject.toml`:
```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W"]
ignore = ["E501"]
```
Rule mapping: flake8's default checks are pycodestyle (E/W) + pyflakes (F),
so `select = ["E", "F", "W"]` reproduces them. E203 and W503 need no ignore
because ruff does not implement them. Ruff's E1/E2/E3 whitespace rules are
preview-only and stay off; black already enforces that formatting.

Adding `ruff` to the dev extra (rather than only removing flake8) keeps
`ruff check` runnable outside pre-commit, as `black` is today.

## Implementation steps
1. Commit this plan.
2. `build`/`chore`: replace the flake8 hook with ruff, add `[tool.ruff]`
   config, delete `.flake8`, swap `flake8` → `ruff` in the dev extra,
   `uv lock`, update the README line.
3. `chore`: add missing final newlines to the two
   `.py` files.
4. Report.

## Acceptance criteria
- No `flake8` reference remains in `.pre-commit-config.yaml`,
  `pyproject.toml`, `uv.lock`, `README.md`; `.flake8` is deleted.
- `pre-commit run --all-files` passes (black + ruff-check).
- `ruff check .` (with the project config) reports no findings in `.py`
  files; notebooks are not checked by the hook.
- `python -m pytest tests/ -q` still passes.

## Verification
- `pre-commit run --all-files`
- `uv run ruff check .`
- `grep -rn flake8 --exclude-dir=.venv --exclude-dir=.git .`
- `python -m pytest tests/ -q`

## Knowledge updates
None expected: lint tooling is not a project capability tracked in
`docs/features.md`, and the swap is routine tooling maintenance rather than a
consequential decision.

## Risks and open questions
- Notebooks excluded from linting (parity with flake8). Say if you would
  rather lint them; that would require fixing the 12 notebook findings.
- Fixing the two W292 violations is a small scope extension; drop step 3 if
  unwanted (the hook would then fail when those files are next staged, as
  flake8 would).
