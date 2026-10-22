# Feature register

This register tracks meaningful project capabilities, not individual
functions or every implementation detail.

Statuses:
- Proposed: under consideration.
- Supported: maintained and intended for use.
- Removed: previously implemented and subsequently removed.
- Rejected: considered but not adopted.

Verification evidence types:
- unit: module-level pytest tests of the component.
- integration: pytest tests that run several components together, including
  a tiny end-to-end inversion.
- config-rejection: tests asserting that a removed setting no longer resolves.
- CI: `.github/workflows/tests.yml` runs `python -m pytest tests/ -q`.
- none found: no test or report exercises the capability directly.

Tests establish software behaviour only. No entry has a numerical-validation
report (accuracy against an analytic or external reference, convergence or
reconstruction quality on a realistic problem).

"(review)" marks a status that is the inventory's proposal and needs
maintainer confirmation; the reason is under Notes.

| ID | Capability | Status | Code/docs | Verification | Decisions |
|----|------------|--------|-----------|--------------|-----------|
| F-001 | 2D space/time grid with PML sponge and non-dimensionalisation; FFT frequency-bin selection | Supported | `odil_wave/grid/grid.py`, `odil_wave/grid/frequency_selection.py`; README "Library structure" | unit: `tests/test_grid.py` | none recorded |
| F-002 | Source wavelets (tone burst with gaussian/rectangular envelope, Ricker) and their spectra | Supported | `odil_wave/source/source_signal.py` | unit: `tests/test_source.py` (tone burst only); Ricker used only as a fixture in `tests/test_operator.py` | none recorded |
| F-003 | Ring transmission acquisition geometry: elliptical receiver ring, evenly spaced source subset, gaussian/point source footprint, receiver sampling, trace normalisation | Supported | `odil_wave/geometry/acquisition_geometry.py` | unit: `tests/test_geometry.py`; per-receiver normalisation in `tests/test_loss.py` | none recorded |
| F-004 | Built-in velocity phantoms (homogeneous, overdensity, shepp_logan, shepp_logan_skull) with head and skull masks | Supported | `odil_wave/models/velocity_models.py` | unit: `tests/test_models.py` | none recorded |
| F-005 | Frequency-domain discrete wave operator: FD Laplacians of order 2–10, PML sponge, Neumann mirror boundary conditions | Supported (review) | `odil_wave/operator/spatial.py`, `odil_wave/operator/conditions.py`, `odil_wave/operator/utils.py`, `odil_wave/operator/base.py` | unit: `tests/test_operator.py` (all five orders) | none recorded |
| F-006 | Helmholtz forward solver, and synthetic observations from a Helmholtz solve on the truth model (inverse-crime baseline) | Supported | `odil_wave/operator/helmholtz.py`, `odil_wave/experiment/problem.py`; README "Important points" | unit: `tests/test_operator.py`; integration: `tests/test_experiments.py` (observed data satisfy the inverse operator on the truth) | [D-005](decisions/D-005-helmholtz-observation-baseline.md) |
| F-007 | ODIL discrete loss (PDE residual, data misfit, regulariser terms) with Tikhonov and TV regularisers, loss logging | Supported | `odil_wave/loss/`, `odil_wave/wavefield/base.py` | unit: `tests/test_loss.py`, `tests/test_wavefield.py`; integration: `tests/test_integration.py` (gradient flows forward to loss) | none recorded |
| F-008 | Joint full-space ODIL inversion over `u` and `c` with a single L-BFGS and a sigmoid-bounded slowness latent (`JointFreqODIL`) | Supported | `odil_wave/optimisation/joint_odil.py`, `odil_wave/optimisation/base.py`; README "Optimiser" | integration: `tests/test_integration.py` (tiny run improves the model); no unit tests found | [D-003](decisions/D-003-joint-formulation-only.md) |
| F-009 | Low-to-high multi-frequency continuation bands with Helmholtz warm start | Supported | `odil_wave/experiment/runner.py`, `odil_wave/experiment/problem.py`; README "Frequency continuation" | unit: `tests/test_experiments.py` (warm-start toggle); integration: `tests/test_integration.py` (single band only) | none recorded |
| F-010 | Reconstruction metrics: masked SSIM and SSIM map, MSE, MAE | Supported | `odil_wave/metrics/metrics.py`; `odil_wave/experiment/README.md` "Masked head-ROI SSIM" | unit: `tests/test_metrics.py` | none recorded |
| F-011 | Config-driven reproducible run layer: schema, defaults, dotted overrides, validation, run artifacts, environment/git metadata, failure capture, per-band callbacks | Supported | `odil_wave/experiment/config.py`, `odil_wave/experiment/runner.py`, `odil_wave/experiment/recorder.py`, `odil_wave/experiment/envinfo.py`, `configs/default_inverse.yaml`; `odil_wave/experiment/README.md` | unit: `tests/test_experiments.py`; integration: `tests/test_integration.py` (artifacts written) | none recorded |
| F-012 | CLI entry point with `--config`, `--override`, `--dry-run` | Supported | `main.py`; README "Running an inversion" | unit: `tests/test_main.py` | none recorded |
| F-013 | Run analysis from saved artifacts: history and recovery plots, band animation, live band view | Supported (review) | `odil_wave/experiment/plots.py`; `odil_wave/experiment/README.md` "Plotting & analysis" | none found | none recorded |
| F-014 | Example and main notebooks | Supported (review) | `notebooks/odil_wave_inverse_example.ipynb`, `notebooks/main_odil.ipynb`; README "Example notebook" | none found (not executed in tests or CI) | none recorded |
| F-015 | Block-coordinate dual L-BFGS optimiser (`LBFGSB`, `StepScheduler`, u-block Hessian, Helmholtz u-transform, frequency-continuation helper) | Removed | removed in `7c130ad` (PR #2): `odil_wave/optimisation/u_block_hessian.py`, `odil_wave/optimisation/helmholtz_utransform.py`, `odil_wave/optimisation/frequency_continuation.py`, `LBFGSB` from `odil_wave/optimisation/base.py`, `tests/test_optimisation.py` | config-rejection: `tests/test_experiments.py` (`lbfgsb`, `u_steps`, `clamp`) | [D-003](decisions/D-003-joint-formulation-only.md) |
| F-016 | Leapfrog time-domain solver and `leapfrog_fft` observations | Removed | removed in `7c130ad` (PR #2): `odil_wave/operator/leapfrog.py`, `odil_wave/operator/temporal.py` | config-rejection: `tests/test_experiments.py` (`leapfrog_fft`, `time_order`) | [D-004](decisions/D-004-remove-leapfrog-solver.md) |
| F-017 | Block-coordinate run diagnostics | Removed | removed in `7c130ad` (PR #2): `odil_wave/experiment/diagnostics.py` | config-rejection: `tests/test_experiments.py` (`diagnostics`) | [D-003](decisions/D-003-joint-formulation-only.md) |

## Notes

Observed facts are stated plainly; proposals and uncertainty are labelled.

- F-002: the Ricker wavelet has no dedicated test; it is only constructed as
  a fixture in `tests/test_operator.py`.
- F-004: a `skull` profile exists as a placeholder that raises
  `NotImplementedError` (`tests/test_models.py`), although the comments in
  `configs/default_inverse.yaml` list it as an option. It is not counted as a
  capability.
- F-005 (review): order 10 is implemented and unit-tested, and the README
  lists orders 2–10, but `configs/default_inverse.yaml` and the
  `space_order` docstring in `odil_wave/experiment/config.py` list only
  2 | 4 | 6 | 8. Confirm whether order 10 is intended for use.
- F-006: Helmholtz-only observations are a deliberate starting
  simplification, expected to change once good results are obtained (D-005).
  Since F-016 was removed, no non-inverse-crime observation path exists.
- F-008: the optimiser unit tests (`tests/test_optimisation.py`) were
  removed in `7c130ad` together with F-015; the joint optimiser is now
  covered only by the tiny integration run.
- F-009: no test runs more than one band.
- F-011: no test covers `odil_wave/experiment/envinfo.py` metadata or
  `failure.json` capture; config handling and artifact writing are tested.
- F-013 (review): no test imports `odil_wave/experiment/plots.py`. Supported
  is proposed because it is exported and documented as the analysis path.
- F-014 (review): notebooks are usage material rather than a library
  capability; kept as an entry so the documented entry points are traceable.
- F-015, F-016, F-017: the removal rationale was stated by the maintainer in
  the 2026-10-22 inventory review, after the change; `7c130ad` and PR #2
  record none. D-003 and D-004 are Proposed.

Rationale not recorded:
- Sigmoid-bounded `c` latent (F-008) and torch-free lazy config resolution
  (F-011, F-012): the README gives one-clause reasons only; judged
  implementation-level, so no decision records.
- All other Supported entries: no design rationale is recorded in the
  repository.
- Removal of deprecated sparse operators (`2e75cbe`) and reproduction/PINN
  notebooks (`f3a7341`): not entered as capabilities; no rationale recorded.

## Inventory status

Initial inventory completed 2026-10-22 at `main` commit `b4a1d10`
(task 2-feature-inventory; see `docs/reports/2-feature-inventory.md`).
Statuses for capabilities that predate this register are the inventory's
proposals, confirmed in plan review.

This repository's commit history was inspected; the upstream PR discussions
referenced by older merge commits were not, so earlier removed or rejected
capabilities may be missing.

Missing entries do not imply that a capability is absent.
