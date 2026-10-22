# Report: 5-optimiser-e2e-tests — Joint-optimiser tests, multi-band and E2E runs, centred defaults

Task:
- Add tests for the joint optimiser's latent map, its gradient and its
  objective at the truth.
- Add a 2-band continuation run test.
- Add end-to-end float32 runs on CPU and on MPS.
- Remove the unused analytic residual JVP.
- Make the acquisition ring and the overdensity anomaly default to the centre
  of the interior grid.

Branch: 5/optimiser-e2e-tests

Plan approved: 2026-10-22 (`docs/plans/5-optimiser-e2e-tests.md`)

## Outcome
Completed. All planned tests pass locally, including the MPS case.

## Changes
- **`278dfba` `fix:` default centres:**
  - New `Grid.interior_center` property.
  - `AcquisitionGeometry(ring_center=None)` now resolves to it, instead of
    defaulting to `(0, 0)`.
  - The overdensity `center` now defaults to it.
  - Docstrings updated.
  - New tests: `test_interior_center_is_interior_midpoint` (`test_grid.py`),
    `test_default_ring_center_is_interior_center` and
    `test_fixture_ring_lies_in_interior` (`test_geometry.py`), and
    `test_overdensity_default_center_is_interior_center` (`test_models.py`).
- **`c1c0319` `refactor:` JVP removal:** removed
  `JointFreqODIL.residual_jvp_analytic`, `SlownessLatent.dm_dz`,
  `JointFreqODIL.lap` and `JointFreqODIL._zero_pad_interior`. Nothing called
  any of them.
- **`6f2346a` `test:` `tests/test_joint_odil.py`:**
  - Latent round trip (`z_scale` 1 and 10) and bounds at `z = ±50`.
  - Directional central-difference vs autograd gradient for the u, z and
    joint blocks, plus a Tikhonov-regulariser case.
  - The objective and its PDE and data terms are below 1e-20 at the truth.
- **`28213c0` `test:` 2-band test:** `test_run_inverse_two_bands_chain_models`
  in `tests/test_integration.py`. It spies on `runner.run_frequency_band`.
  The tiny run config was factored into a helper.
- **`6645182` `test:` E2E:**
  - `tests/test_e2e.py` runs `main.main` on a 2-band tiny YAML config in
    float32, on `cpu` and `mps`.
  - It asserts the device and dtype in `metadata.json`, the completed
    status and band count, that the final error beats the starting model's,
    and the shape of the final `c`.
  - The `mps` marker is registered in `pyproject.toml`.
  - The tiny-config helper moved to `tests/conftest.py` as
    `tiny_run_config`.

No existing test assertion was changed. The only edits to existing tests
are the helper extraction in `test_integration.py`, where the config values
are unchanged.

## Deviations from the plan
- The tiny-config helper lives in `tests/conftest.py`, not in
  `tests/test_integration.py`, because both `test_integration.py` and
  `test_e2e.py` use it. The move is test-only and the config values are
  unchanged.

## Verification
| Check/command | Configuration | Result | Evidence |
|---------------|---------------|--------|----------|
| `python -m pytest tests/ -q -rs --durations=8` | Python 3.13.7, torch 2.14.1, macOS with MPS | 225 passed, 0 skipped, 7.6 s | slowest: e2e[mps] 5.8 s, e2e[cpu] 0.7 s |
| `python -m pytest -m mps -q -rs` | same | 1 passed, 224 deselected | MPS case ran (not skipped) |
| `python -m pytest tests/test_joint_odil.py -q` | float64 tiny problem | 8 passed | — |
| grep for `residual_jvp_analytic`, `dm_dz`, `_zero_pad_interior`, `.lap(` in `odil_wave tests notebooks main.py README.md docs/features.md` | `__pycache__` excluded | no matches | — |
| `pre-commit run --all-files` | black, ruff-check | Passed | at `6645182` |

CI was not run before the push. On `ubuntu-latest` the MPS case is expected
to be reported as skipped.

## Interpretation and limitations
- **Software correctness only.** The tests show that the gradient matches
  finite differences and that the objective vanishes at the truth in the
  inverse-crime setting. They do not show reconstruction quality on
  realistic problems.
- **Effect of the centre fix on existing tests.** These setups now sit
  inside the interior: the `geometry` and `wave_stack` fixtures, the rings
  built at `test_geometry.py:36` and `:70` and `test_operator.py:89`, and
  the ring and overdensity in `test_forward_to_loss_gradient_flows`.
  Previously, 5 of 8 receivers, one of the two sources and the overdensity
  sat in or near the PML. These tests pass unchanged, but they now exercise
  a non-degenerate geometry. For example, `wave_stack` previously had one
  source with zero interior RMS. Before the fix, the joint-optimiser
  objective at the truth on this geometry was 1e58 (scratch probe recorded
  in the plan).
- **MPS coverage.** On MPS only the optimiser and loss run on the device;
  the Helmholtz solves run on the CPU (scipy). The plan's timing probe puts
  them at about 10 % of band time or less.
- **Robustness of the improvement check.** "Final error below the starting
  error" holds on CPU and MPS in float32. It depends on optimiser behaviour
  and was observed in one run per device.
- **Behaviour change for outside callers.** Code that builds
  `AcquisitionGeometry` or an overdensity `VelocityModel` without a centre
  now gets an interior-centred one. In-repo config runs and the example
  notebook pass explicit centres and are unaffected.

## Knowledge updates
- `docs/features.md`:
  - Added the `e2e` evidence type.
  - F-003 and F-004: interior-centred defaults, new tests, and a Note on the
    change from `(0, 0)`.
  - F-008: unit and e2e evidence, and a Note on the JVP removal.
  - F-009: 2-band integration and e2e evidence; removed "no test runs more
    than one band".
  - F-012: e2e evidence.
  - Note that the MPS case is skipped in CI and that the Helmholtz solves
    stay on the CPU.
- No decision record: the default change fixes a degenerate default, and the
  removal deletes dead code. No design choice was made.

## Follow-up
Optional:
- Tests for the run layer's failure path (`failure.json`, a failing
  callback) and for `envinfo`.
- Smoke tests for `odil_wave/experiment/plots.py` (F-013).
