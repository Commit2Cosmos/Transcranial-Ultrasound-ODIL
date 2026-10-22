# Report: 2-feature-inventory — Populate the feature register from the current repository

Task: Populate `docs/features.md` with the project's meaningful capabilities,
linked to implementation and verification, with proposed statuses, flagged
uncertainty, missing rationale identified, and retrospective decision
records where evidence supports them.

Branch: 2/feature-inventory

Plan approved: 2026-10-22 (amendment approved 2026-10-22)

## Outcome
Completed. Documentation only; no code, tests, configs or dependencies
changed.

## Changes
- `docs/features.md`: 17 entries (F-001–F-014 Supported, three flagged
  "(review)"; F-015–F-017 Removed), an evidence-type legend, a Notes section
  and a rewritten Inventory status.
- `docs/decisions/D-003-joint-formulation-only.md`,
  `D-004-remove-leapfrog-solver.md`, `D-005-helmholtz-observation-baseline.md`:
  retrospective, Proposed, rationale quoted from the maintainer's plan-review
  statements of 2026-10-22.
- `docs/plans/2-feature-inventory.md`: Plan changes section.

## Deviations from the plan
- Stride reference material (planned entry 15) dropped from the register.
  During implementation `stride/` was deleted in the maintainer's working
  tree; the maintainer chose to omit the entry, and renewed approval was given
  2026-10-22. The deletion itself is the maintainer's uncommitted change and
  is not part of this branch. Removed entries are therefore F-015–F-017.

## Verification
| Check/command | Configuration | Result | Evidence |
|---------------|---------------|--------|----------|
| Path check (scratch script, `python3 -I`): backticked paths and decision links in `docs/features.md` against `git ls-files`, or `git cat-file -e 7c130ad^:<path>` for Removed rows | branch head | 74 references; all resolve except `docs/reports/2-feature-inventory.md` (this file, created after the check) and `tests/test_optimisation.py` in Notes (confirmed present at `7c130ad^`, cited as removed there) | session output |
| F-ID check (same script) | — | F-001–F-017, unique, sequential | session output |
| `uv run --frozen python -m pytest tests/ -q` | Python 3.13.7, torch 2.14.1, numpy 2.5.4, scipy 1.18.1, macOS | 210 passed | session output |
| `git diff --name-only main` | after commits | only the plan, register, report and D-003–D-005 | session output |

Not run: notebooks, Stride reference, any inversion beyond the test suite's
tiny run, any numerical validation.

## Interpretation and limitations
- The passing suite shows that the cited tests currently pass; it is software
  evidence only. No capability has numerical-validation evidence.
- Supported for capabilities that predate the register is the inventory's
  proposal, confirmed in plan review; it is not a record of past maintenance.
- Capability boundaries are a judgement call.

What was inspected:
- Source: all `odil_wave` subpackages and their `__init__.py` exports;
  `odil_wave/experiment/config.py` (schema, validation, removed settings);
  `main.py`; `configs/default_inverse.yaml`; `pyproject.toml`.
- Tests: every `tests/test_*.py` module by test name, mapped to capabilities;
  searches for tests of `plots.py`, `envinfo.py` and failure capture (none).
- Docs: `README.md`, `odil_wave/experiment/README.md`, `TODO.md`, `info.md`,
  `docs/` templates and history.
- CI: `.github/workflows/tests.yml`.
- History: this repository's commit log, the `7c130ad` diff and pre-removal
  exports, deleted-file history, PRs #1–#13 (titles; PR #11's unapproved plan
  for the same inventory, used as a cross-check; PR #5's register version).

Merged or omitted candidates:
- `Wavefield` folded into F-007.
- `skull` profile: placeholder that raises `NotImplementedError`; not a
  capability (noted under F-004).
- Deprecated sparse operators (`2e75cbe`) and reproduction/PINN notebooks
  (`f3a7341`): removed material, not entered; rationale not recorded.
- Stride reference material: omitted (see Deviations).
- Agent-workflow infrastructure: out of scope by plan.

Missing historical rationale:
- `7c130ad` / PR #2 record no reason; rationale for F-015–F-017 and the
  Helmholtz baseline exists only as maintainer statements given after the
  fact (D-003–D-005). D-004 does not say in what respect the leapfrog solver
  was "not good enough".
- Sigmoid-bounded `c` latent and torch-free config resolution: one-clause
  README reasons only; no records.
- No design rationale is recorded for any other Supported entry.
- Removal of the Experimental status (`fb64f51`) and of PR #5's Source
  column (PR #5 is absent from current `main` history): no rationale
  recorded; not acted on.

Doc drift found (not fixed):
- `README.md` "Tests" refers to the deleted `tests/test_optimisation.py`.
- `odil_wave/experiment/README.md` cites `sandbox/odil_2d_freq_domain.ipynb`,
  which does not exist.
- `configs/default_inverse.yaml` and the `space_order` docstring in
  `config.py` list orders 2–8; the operator supports and tests order 10.
- `configs/default_inverse.yaml` lists `skull` as a profile option; it raises
  `NotImplementedError`.

Coverage gaps:
- Upstream PR discussions referenced by older merge commits were not read.
- Notebooks were not executed or read cell by cell.

## Knowledge updates
- Feature register: F-001–F-017 added.
- Decision records: D-003, D-004, D-005 added (Proposed). D-001 and D-002
  were not reused because those numbers appear in repository/PR history with
  different content.

## Follow-up
Optional:
- Accept or revise D-003–D-005.
- Confirm or resolve the three "(review)" statuses (F-005, F-013, F-014).
- Fix the doc drift listed above.
- Add tests for the joint optimiser beyond the tiny integration run, for
  multi-band continuation, for `plots.py` and for `envinfo.py`.
- If Stride is removed from `main`, no register change is needed (it has no
  entry).
