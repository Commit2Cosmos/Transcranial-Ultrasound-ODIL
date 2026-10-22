# Plan: 2-feature-inventory — Populate the feature register from the current repository

Task: Populate `docs/features.md` with the project's meaningful capabilities,
linked to implementation and available verification, with proposed lifecycle
statuses, flagged uncertainty, explicitly identified missing rationale, and
retrospective decision records only where evidence supports them.

Branch: 2/feature-inventory

Status: Approved 2026-10-22

## Objective
Replace the empty table in `docs/features.md` with an inventory of the
capabilities present at the head of `main` (`b4a1d10`). Explicit requirements:

- capability-level entries, not functions;
- each entry links to code/docs and to the verification that exists (or says
  none exists);
- statuses are proposals; uncertain ones are flagged for review;
- observed facts, proposed classifications and uncertainty are kept apart;
- missing historical rationale is stated, never invented;
- decision records are created only when a source states the choice and its
  reason;
- `docs/reports/2-feature-inventory.md` states what was inspected and the gaps.

## Repository findings
Observed at `b4a1d10`:

- `docs/features.md`: rules, statuses (Proposed / Supported / Removed /
  Rejected) and an empty table with columns
  `ID | Capability | Status | Code/docs | Verification | Decisions`.
- `docs/decisions/`: only `README.md` and `TEMPLATE.md`. D-001 (plan-first
  development) existed but was deleted in `5e3b081`; D-002 (agent task
  gates) existed only via PR #5. No numerical decisions have been recorded.
- Package `odil_wave` (~8.2k lines incl. `main.py`), version 0.1.0. Public
  exports (`odil_wave/__init__.py` and subpackage `__init__.py` files):
  - `grid`: `Grid` (PML, non-dimensionalisation), `FrequencySelection`;
  - `source`: `SourceSignal` (tone burst with gaussian/rectangular envelope,
    Ricker);
  - `geometry`: `AcquisitionGeometry` (elliptical receiver ring, evenly
    spaced source subset, gaussian/point source footprint, trace
    normalisation);
  - `models`: `VelocityModel` (homogeneous, overdensity, shepp_logan,
    shepp_logan_skull, skull; head/skull masks);
  - `wavefield`: `Wavefield`;
  - `operator`: `WaveEquation`, Laplacians of order 2–10, `Sponge`,
    Neumann mirror BCs, `HelmholtzSolver` with `HelmholtzFactorCache`;
  - `loss`: `InverseLoss`, `ForwardLoss`, `Regulariser`
    (tikhonov / tv_iso / tv_aniso), `LossConfig`, `LossTape`;
  - `optimisation`: `JointFreqODIL`, `SlownessLatent` (sigmoid-bounded
    squared-slowness latent);
  - `metrics`: `ssim`, `ssim_map`, `mse`, `mae`;
  - `experiment`: config schema/resolution/overrides/validation, problem
    building, `run_inverse` with band continuation and warm start, recorder
    (artifacts, `metrics.jsonl`, `failure.json`), env/git metadata, plots and
    `LiveVelocityView` callback.
- `main.py`: CLI (`--config`, `--override`, `--dry-run`);
  `configs/default_inverse.yaml` is the reference config.
- Tests: 11 modules, 182 test functions. `test_integration.py` runs a
  forward-to-loss gradient check and a tiny `run_inverse` asserting the model
  improves; `test_experiments.py::test_removed_settings_are_rejected` asserts
  that removed options (`lbfgsb`, `leapfrog_fft`, `time_order`,
  `diagnostics`, `pml_fill`, `c_grad_smooth_sigma`) no longer resolve.
  No direct tests found for `experiment/plots.py` or `envinfo.py`
  (to be confirmed during implementation).
- CI: `.github/workflows/tests.yml` runs `python -m pytest tests/ -q` on
  Python 3.12.
- Other material: `notebooks/main_odil.ipynb`,
  `notebooks/odil_wave_inverse_example.ipynb`.
- No report contains numerical-validation evidence (no reports exist).
- History (this repository):
  - `7c130ad` "clean: remove 2 optimiser odil" (PR #2) removed the
    block-coordinate `LBFGSB` optimiser, `UBlockHessian`, Helmholtz
    u-transform, frequency-continuation helper module, the leapfrog
    time-domain solver and `temporal.py`, `leapfrog_fft` observations, and
    `experiment/diagnostics.py`. The commit and PR state no rationale.
  - `2e75cbe` removed deprecated sparse operators; `f3a7341` removed
    reproduction notebooks (incl. a PINN notebook).
- Doc drift (reported, not fixed): README refers to the deleted
  `tests/test_optimisation.py`; `odil_wave/experiment/README.md` cites the
  non-existent `sandbox/odil_2d_freq_domain.ipynb`; the config comments say
  `space_order` 2|4|6|8 while README and `operator` offer order 10.

## Scope and non-goals
In scope:
- `docs/features.md`: table entries; a short evidence-type legend; a
  per-entry "Notes" section for flags and rationale gaps; and a rewritten
  "Inventory status" section (commit inspected, date, coverage limits,
  "missing entries do not imply absence"). Existing rules and statuses stay.
- `docs/reports/2-feature-inventory.md`.
- Retrospective decision records only if the evidence test below is met.

Non-goals:
- Changes to code, tests, configs, notebooks, dependencies, `.github/`,
  `CLAUDE.md`, README or `info.md`. Doc drift is reported only.
- Running inversions, notebooks or the Stride reference.
- Agent-workflow infrastructure as feature entries.
- Changing the register's column set or status vocabulary.

## Proposed approach
### Entries (provisional wording; final wording settled in implementation)
Implemented:

| # | Capability | Proposed status |
|---|------------|-----------------|
| 1 | 2D space/time grid with PML sponge and non-dimensionalisation; FFT frequency-bin selection | Supported |
| 2 | Source wavelets (tone burst, Ricker) and spectra | Supported |
| 3 | Ring transmission acquisition geometry (receiver ring, source subset, source footprint, receiver extraction, trace normalisation) | Supported |
| 4 | Built-in velocity phantoms and head/skull masks | Supported |
| 5 | Frequency-domain discrete wave operator: FD Laplacians (orders 2–10), PML sponge, Neumann BCs | Supported (review: order-10 exposure, see drift) |
| 6 | Helmholtz forward solver and synthetic observations on the truth model (inverse-crime baseline) | Supported |
| 7 | ODIL discrete loss (PDE + data + regulariser terms) with Tikhonov / TV regularisers | Supported |
| 8 | Joint full-space ODIL inversion over `u` and `c` with L-BFGS and bounded slowness latent (`JointFreqODIL`) | Supported |
| 9 | Low-to-high multi-frequency continuation with Helmholtz warm start | Supported |
| 10 | Reconstruction metrics: masked SSIM, MSE, MAE | Supported |
| 11 | Config-driven reproducible run layer (schema, overrides, validation, artifacts, metadata, failure capture, band callbacks) | Supported |
| 12 | CLI entry point `main.py` | Supported |
| 13 | Run analysis plots/animations and live band view | Supported (review: no direct tests found) |
| 14 | Example and main notebooks | Supported (review: notebooks vs. capability) |
| 15 | Stride external reference forward/FWI material | Supported (review: reference material, not part of the package) |

Removed:

| # | Capability | Status |
|---|------------|--------|
| 16 | Block-coordinate dual L-BFGS optimiser (`LBFGSB`, u-block Hessian, Helmholtz u-transform) | Removed (`7c130ad`, PR #2) |
| 17 | Leapfrog time-domain solver and `leapfrog_fft` observations | Removed (`7c130ad`, PR #2) |
| 18 | Block-coordinate diagnostics (`experiment/diagnostics.py`) | Removed (`7c130ad`, PR #2) |

`Wavefield` folds into 7/8. Deprecated sparse operators (`2e75cbe`) and the
removed reproduction/PINN notebooks (`f3a7341`) are noted in the report, not
entered, unless review asks otherwise. Removed upstream/pre-fork
capabilities are out of reach (see Risks).

### Status policy
- Supported ("maintained and intended for use") is proposed for capabilities
  that exist on `main`, are exported or reachable from the CLI/config, and are
  exercised by tests or documented as entry points. It is the agent's
  proposal and does **not** imply numerical validation; the register will say
  so.
- Proposed is not used: it means "under consideration", and no unimplemented
  proposal was found (TODO.md items are maintenance, not capabilities).
- Removed only where this repository's history shows the code was deleted.
- Rejected is not used: no recorded consideration-and-refusal was found.
- Uncertain classifications carry "(review)" in the Status cell, with the
  reason in Notes.

### Evidence columns
- Code/docs: module paths, plus README / experiment README sections.
- Verification: specific test modules labelled by evidence type — `unit`,
  `integration (tiny run)`, `config-rejection`, `CI` — or `none found`.
  Software tests are kept apart from numerical validity; the register states
  that no entry has a numerical-validation report.
- Decisions: `none recorded` where there are none. Removed entries cite the
  removal commit and PR #2 in Code/docs.
- Notes: observed facts vs. proposals vs. uncertainty, and "rationale not
  recorded" where applicable.

### Decision records
Test: a commit, PR, issue, in-repo document or explicit maintainer statement
states both the choice and its reason. Candidates:

| Candidate | Choice documented | Reason documented |
|-----------|-------------------|-------------------|
| Remove block-coordinate optimiser (`7c130ad`) | yes (diff, test) | maintainer statement, plan review 2026-10-22 |
| Remove leapfrog time-domain solver (`7c130ad`) | yes (diff, test) | maintainer statement, plan review 2026-10-22 |
| Inverse-crime (Helmholtz-only) observation baseline | yes (README: "deliberate") | maintainer statement, plan review 2026-10-22 |
| Sigmoid-bounded `c` latent | yes (README) | partial ("so its bounds hold structurally") |
| Torch-free lazy config resolution | yes (README) | partial ("so `--help` and `--dry-run` are instant") |

Maintainer statements (local session, 2026-10-22), to be quoted verbatim:
- block-coordinate: "this project focuses solely on the joint formulation";
- leapfrog: "not efficient enough";
- Helmholtz observations: "make the problem simpler at the start (will
  have to make non-inverse crime once results are reproduced)".

Records to create, all `Status: Proposed`, `Source: 2-feature-inventory`,
labelled retrospective (rationale stated after the change, not at the time
of `7c130ad`):
- `D-003-joint-formulation-only.md`: retain only `JointFreqODIL`; remove the
  block-coordinate `LBFGSB` optimiser, u-block Hessian, Helmholtz u-transform
  and block-coordinate diagnostics. Features: optimiser, removed optimiser,
  removed diagnostics entries.
- `D-004-remove-leapfrog-solver.md`: remove the leapfrog time-domain solver
  and `leapfrog_fft` observations. Records that "not good enough" is not
  further specified (accuracy, cost or stability unknown) and that a more
  sophisticated replacement is intended but not planned. Features: removed
  leapfrog entry.
- `D-005-helmholtz-observation-baseline.md`: generate observed data by a
  Helmholtz solve on the truth model with the inversion's own operator
  (inverse crime) to simplify the problem initially. Revisit when: good
  results are obtained on this baseline. Features: Helmholtz/observations
  entry. Notes the interaction with D-004 (the removed `leapfrog_fft` path was
  the existing non-inverse-crime observation route).

Numbering starts at D-003 so that D-001 and D-002, which appear in
repository/PR history with different content, are not reused.

The two "partial" items are implementation-level rather than consequential:
no records; the report lists them with the rationale fragment found.

## Implementation steps
1. Re-confirm findings at the branch head: exports, `config.py` validation
   (incl. whether `space_order=10` is accepted), test-to-capability mapping,
   and absence of plot/envinfo tests.
2. Write the table, legend, Notes and Inventory status in `docs/features.md`.
3. Write D-003, D-004, D-005 from the template and link them in the
   Decisions cells of the affected entries.
4. Run the checks under Verification.
5. Write `docs/reports/2-feature-inventory.md`: inspected sources, evidence
   summary, flagged classifications, merged/omitted candidates, missing
   rationale, doc drift, coverage limits.
6. Commit atomically (plan; register; report), push, open PR.

## Acceptance criteria
- `docs/features.md` has an entry for each capability above, or the report
  explains a merge/omission. IDs are `F-001`… sequential and unique.
- Every entry has non-empty Code/docs, Verification and Decisions cells;
  `none found` / `none recorded` are explicit.
- Every cited path exists at the head commit, or for Removed entries is cited
  with the removing commit.
- Verification cells label evidence type; the register states that no entry
  has numerical-validation evidence.
- Every "(review)" status has a reason in Notes; every missing rationale is
  stated as such.
- D-003–D-005 exist with status Proposed, quote the maintainer statements
  verbatim, and state no rationale beyond them.
- `git diff --name-only main` lists only the plan, `docs/features.md`, the
  report, and the three `docs/decisions/D-*` files.

## Verification
- Path check: scratch script (in the session scratchpad, run with
  `python -I`) extracting backticked paths from `docs/features.md` and
  checking them with `git ls-files` / `git cat-file -e <sha>:<path>`.
- ID check: sequential, unique F-IDs (same script).
- Scope check: `git diff --name-only main`.
- `python -m pytest tests/ -q` to record whether the cited tests currently
  pass (software evidence only). If torch/jax are unavailable, record as not
  run.
- Not run: inversions beyond the test suite's tiny run, notebooks, Stride,
  any numerical validation.

## Knowledge updates
- `docs/features.md`: initial inventory (main output).
- Decision records: D-003, D-004, D-005 (retrospective, Proposed; acceptance
  is a separate maintainer step per `docs/decisions/README.md`).

## Risks and open questions
1. ~~Status for implemented capabilities.~~ Resolved 2026-10-22: keep the
   current vocabulary (no Experimental); implemented capabilities are
   Supported, with "(review)" flags as listed.
2. ~~Rationale for removals and the inverse-crime baseline.~~ Resolved
   2026-10-22: maintainer supplied rationale; see Decision records.
3. ~~Removed granularity.~~ Resolved 2026-10-22: three separate Removed
   entries (optimiser, leapfrog, diagnostics).
4. **Task ID.** No task numbers exist in `docs/plans/`, `docs/reports/` or
   current branches, which strictly gives `1`; `1/local-agent-workflow` was
   already used (PRs #12, #13, branch deleted), so `2` was chosen to avoid
   reuse. PR #11's plan (issue #9) is treated as a superseded attempt under
   the old workflow, not resumed.

Risks:
- Capability boundaries are a judgement call.
- Pre-fork/upstream history is not inspectable from this repository, so
  older removed or rejected capabilities may be missed; the report will say so.
