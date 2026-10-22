# Plan: 5-optimiser-e2e-tests — Joint-optimiser tests, multi-band and E2E runs, centred defaults

Task:
- Add tests for the joint optimiser's latent map, its gradient and its
  objective at the truth.
- Add a 2-band continuation run test.
- Add end-to-end float32 runs on CPU and on MPS.
- Remove the unused analytic residual JVP.
- Make the acquisition ring and the overdensity anomaly default to the centre
  of the interior grid.

Branch: 5/optimiser-e2e-tests

Status: Approved 2026-10-22

## Objective
1. `z_of_c(c_of_z(z))` round-trips, and `c_of_z` stays within
   `[c_min, c_max]`.
2. The gradient of `objective` matches finite differences.
3. `objective` is close to 0 at the truth.
4. A 2-band tiny run writes one `bands/` entry per band, and band 2 starts
   from band 1's final `c`.
5. The CLI runs a tiny inversion end to end in float32 on CPU and with the
   optimiser on MPS.
6. Remove `residual_jvp_analytic` and the helpers only it uses.
7. `AcquisitionGeometry` and the `overdensity` profile of `VelocityModel`
   default to the centre of the interior grid instead of `(0, 0)`.

## Repository findings
- `odil_wave/optimisation/joint_odil.py`:
  - `SlownessLatent` (l.111) has `m_of_z`, `c_of_z`, `dm_dz`,
    `sigmoid_deriv`, `z_of_m` and `z_of_c`. `z_of_m` clips the sigmoid
    fraction to `[logit_clip, 1 - logit_clip]` (default `1e-6`), so the
    round trip holds only for `|z_scale·z| ≲ 13.8`.
  - `JointFreqODIL.objective` (l.426) is the production loss.
    `init_params()` returns `(u_re, u_im, z_m)` from `u_init` and the
    starting model.
  - `residual_jvp_analytic` (l.393) was added in `eaca31b` together with the
    joint optimiser and has never been called.
  - The helpers used only by it are `SlownessLatent.dm_dz` (l.133),
    `JointFreqODIL.lap` (l.377) and `JointFreqODIL._zero_pad_interior`
    (l.381). `sigmoid_deriv` is used by `_model_stats` and stays.
  - No README or docs reference any of these.
  - No test references any `joint_odil` name.
- `odil_wave/experiment/runner.py`:
  - `run_inverse` calls the module-level `run_frequency_band(problem,
    current, ...)` per band, then sets `current = band_res.velocity_model`
    (l.611–615).
  - Each band writes `bands/band_{i:02d}_{label}/` containing
    `c_final.npy`, `c_final_metadata.json`, `band_config.yaml` and
    `band_summary.json`.
  - The only integration run (`tests/test_integration.py:70`) has one band.
- Devices:
  - `envinfo.resolve_device` and `Grid.__post_init__` fall back to CPU
    silently, with a `RuntimeWarning`.
  - `HelmholtzSolver.solve` factorises on the CPU with scipy `splu` and
    copies its results to `grid.device`.
  - Local torch is 2.14.1 with MPS available; CI (`ubuntu-latest`) has no
    MPS.
- `main.main(argv)` returns an exit code. `tests/test_main.py` covers only
  `--help`, `--dry-run` and config errors.
- **Centre defaults (observed):**
  - `AcquisitionGeometry(ring_center=(0.0, 0.0))` (`acquisition_geometry.py`
    l.92) and the overdensity `center` (`velocity_models.py:294`) both
    default to `(0, 0)`. On the tests' `[0, 0.05]²` interior, that point is
    the corner. As a result:
    - 5 of 8 receivers and one source are clamped into the PML.
    - That source has zero interior RMS, so `pde_scale` falls back to its
      `1e-30` floor and the objective at the truth reads `1e58`.
    - The anomaly sits in the PML, where `build_full_c` replaces it.
  - `build_problem` always passes an explicit centre: `"grid_center"` is the
    midpoint of `grid.extent`, which equals the interior midpoint because
    the PML is symmetric. Config-driven runs are therefore unaffected.
  - The example notebook passes `ring_center` and `center` explicitly.
  - Shepp–Logan profiles embed an interior-sized array and are already
    centred.
  - Test call sites that rely on the default ring centre:
    `tests/conftest.py:75`, `tests/conftest.py:97`,
    `tests/test_geometry.py:36`, `tests/test_geometry.py:70`,
    `tests/test_operator.py:89` and `tests/test_integration.py:34`. The
    overdensity at `tests/test_integration.py:40` relies on the default
    anomaly centre.
- Scratch probes (not committed):

  | Check | Result |
  |---|---|
  | Copy of `tests/` with all rings and that anomaly at the interior centre | All 206 tests that could run pass. The other 4, in `test_main.py`, failed only because they look for `configs/` relative to the copied folder. |
  | Objective at the truth, with the problem centred (float64, 20×20, PML 4, 40 kHz) | 1.07e-30 |
  | Directional central difference vs autograd gradient | rel. error 3e-8 at h=1e-5, 4e-10 at h=1e-6 |
  | Latent round trip on z∈[-3, 3] | max error 1.5e-14 |
  | Tiny 2-band (30 kHz → 40 kHz) float32 run, CPU | completes |
  | Tiny 2-band (30 kHz → 40 kHz) float32 run, MPS | completes; final `c` on `mps:0` |
- Timing probe (default config scaled, one frequency, 64 shots, float32;
  "objective eval" is one objective plus backward pass):

  | Interior | Device | Helmholtz: truth obs + warm start | Objective eval | Estimated L-BFGS per band (≥400 evals) |
  |---|---|---|---|---|
  | 125² | CPU | 0.9 s | 79 ms | ≥32 s |
  | 125² | MPS | 1.2 s | 27 ms | ≥11 s |
  | 500² | MPS | 8.7 s | 210 ms | ≥84 s |
  | 1000² | MPS | 36 s | 790 ms | ≥316 s |

  The Helmholtz solves stay on the CPU and take about 10 % of band time or
  less, so the test of the optimiser on MPS covers the part that matters
  for speed.

## Scope and non-goals
In scope:
- Library:
  - A `Grid.interior_center` property.
  - `AcquisitionGeometry(ring_center=None)` and the overdensity `center`
    both default to it.
  - Removal of `residual_jvp_analytic`, `dm_dz`, `lap` and
    `_zero_pad_interior`.
- Tests:
  - A new `tests/test_joint_odil.py` (objective items 1–3).
  - A 2-band test in `tests/test_integration.py` (item 4).
  - A new `tests/test_e2e.py` (item 5).
  - Default-centre tests and a fixture guard (item 7).
- Register an `mps` pytest marker in `pyproject.toml`.

Non-goals:
- CUDA tests.
- Running the Helmholtz solve on MPS.
- Changing the `"grid_center"` config option or `_resolve_center`.
- The overdensity default `radius` (0.3 m).
- Coverage tooling, plot tests, failure-path tests and numerical validation.

## Proposed approach
**Centre defaults (`fix:`)**
- `Grid.interior_center` returns `((x0 + x1) / 2, (y0 + y1) / 2)` from
  `interior_extent`.
- `AcquisitionGeometry`: change the signature to
  `ring_center: Optional[Tuple[float, float]] = None`, resolve `None` to
  `grid.interior_center` in `__init__`, and update the docstring.
- `VelocityModel` overdensity: when `center` is absent, use
  `g.interior_center`.
- Explicit centres keep working as before. The test call sites need no
  edits, because the defaults now produce the centred setup the probe
  verified.
- New tests:

  | File | Test |
  |---|---|
  | `test_grid.py` | `interior_center` on an asymmetric extent |
  | `test_geometry.py` | The default ring centre equals the interior centre |
  | `test_geometry.py` | Every `recv_ij` and `src_ij` of the `geometry` fixture lies inside `grid.interior_slice` (guard) |
  | `test_models.py` | The default overdensity anomaly is centred on the interior |

**Removal of the analytic JVP (`refactor:`)**
- Delete `residual_jvp_analytic`, `SlownessLatent.dm_dz`,
  `JointFreqODIL.lap` and `JointFreqODIL._zero_pad_interior`.
- Drop module-docstring text that refers to them, if any.
- Behaviour is unchanged: the existing tiny run gives the same result.

**`tests/test_joint_odil.py`**

A module-scoped float64 fixture builds a tiny problem, which is centred by
the new defaults:
- An overdensity truth model.
- Helmholtz observations.
- An `InverseLoss` with the solver's `space_order` and `pml_weight`.
- `JointFreqODIL(wf_truth, loss, u_init=observed, c_min=1400, c_max=1700)`.

| Test | Check | Tolerance |
|---|---|---|
| Latent round trip | Parametrised over `z_scale ∈ {1, 10}`; `z` spans `[-10, 10]/z_scale`, so `|z_scale·z| ≤ 10` stays inside the clip | `atol=1e-10` |
| Latent bounds | `c_of_z(±50)` stays within `[c_min, c_max]` | — |
| Objective gradient | At `(u, z + 0.1·ξ)`, compare the autograd directional derivative `⟨∇L, d⟩` with a central difference at `h=1e-6`. Parametrise the direction `d` over u-only, z-only and joint. | rel. error `< 1e-6` |
| Objective gradient, regulariser path | One extra case with `reg_weight > 0` and a Tikhonov regulariser | rel. error `< 1e-6` |
| Objective at truth | `objective(*init_params())`, plus `_last["pde_term"]` and `_last["data_term"]` | each `< 1e-20` |

A directional finite-difference check is used instead of `gradcheck`, which
would need about 3.5k evaluations.

**2-band test (`tests/test_integration.py`)**
- Factor the existing tiny config into a helper.
- Bands are `[30e3]` and `[40e3]`, with a small `n_iter`.
- `monkeypatch.setattr(runner, "run_frequency_band", spy)`. The spy records
  each input `velocity_model` and delegates to the original.
- Assertions:
  - `bands/` contains exactly `band_00_*` and `band_01_*`, each with the
    four artifacts and the matching `frequencies_hz`.
  - Band 0's input `c` is the init model's.
  - Band 1's input is the same object as `band_results[0].velocity_model`,
    and its `c` equals `bands/band_00_*/c_final.npy` (`rtol=1e-6`, because
    the file is float32).
  - `final/c_final.npy` equals band 1's.

This also covers the band-2 warm start, which uses the same model.

**E2E (`tests/test_e2e.py`)**
- Write the tiny 2-band config as YAML in `tmp_path`, then call
  `main.main(["--config", path, "-o", "runtime.dtype=float32",
  "-o", f"runtime.device={dev}"])`.
- `dev` is parametrised over:
  - `cpu`
  - `mps`, with `pytest.mark.mps` and
    `skipif(not torch.backends.mps.is_available())`
- Assertions:
  - Exit code 0.
  - `metadata.json` has `device_resolved == dev` (this catches a silent CPU
    fallback) and `dtype == "float32"`.
  - `run_summary.json` has status `completed` and a finite
    `final_rel_c_error` below the starting model's error.
  - `final/c_final.npy` has the grid shape.
- Register the `mps` marker in `[tool.pytest.ini_options].markers`.

## Implementation steps
1. `docs: plan for 5-optimiser-e2e-tests`.
2. `fix: default ring and overdensity centre to the interior centre`
   (library change, plus the grid, geometry and model tests).
3. `refactor: remove unused analytic residual jvp`.
4. `test: add joint optimiser latent, gradient and truth tests`.
5. `test: add two-band continuation integration test`.
6. `test: add float32 cpu and mps end-to-end cli runs` (with the marker in
   `pyproject.toml`).
7. `docs: report for 5-optimiser-e2e-tests`, with the feature-register
   updates.

## Acceptance criteria
- The tests for objective items 1–5 and 7 exist as described.
- The four removed names no longer appear in the repository.
- Existing tests pass unchanged; no assertion is weakened or removed.
- `python -m pytest tests/ -q` passes locally, with the MPS case running
  (not skipped). In CI it passes with the MPS case skipped.
- Full-suite wall time stays under about 10 s locally. It is currently
  1.2 s.

## Verification
- `python -m pytest tests/ -q -rs --durations=10`
- `python -m pytest -m mps -q -rs` (should report a run, not a skip).
- `rg -n "residual_jvp_analytic|dm_dz|_zero_pad_interior|\.lap\(" odil_wave tests notebooks`
  should return no matches.
- `pre-commit run --all-files`

## Knowledge updates
- `docs/features.md`:
  - F-003: the ring now defaults to the interior centre.
  - F-004: the overdensity anomaly now defaults to the interior centre.
  - F-008: add the new unit tests and update the Note; record the removal of
    the analytic JVP.
  - F-009: add the 2-band test and remove "no test runs more than one band".
  - F-012: add the E2E CLI run.
  - Notes: the MPS E2E case is skipped in CI.
- No decision record:
  - The new default is a bug fix: the old default placed the setup off the
    domain for any extent not centred at the origin, and no in-repo caller
    relied on it.
  - The JVP removal deletes dead code.

## Risks and open questions
- **Default-centre behaviour change.** External code that relied on the
  `(0, 0)` default without passing a centre will get a different geometry.
  Only extents centred at the origin are unaffected.
- **Centring changes what existing tests cover.** For example, `wave_stack`
  now has two interior sources. The report will note this change in
  coverage, not just the pass status.
- **Fragility of "improves on the start".** For float32 and MPS runs, the
  check that the run improves on the starting model depends on optimiser
  behaviour. If it proves flaky, fall back to a finite error plus a loss
  decrease, and ask before changing it.
