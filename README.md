# odil-wave

Frequency-domain acoustic **full-waveform inversion (FWI)** built on the **ODIL**
(*Optimizing a DIscrete Loss*) framework.

Instead of eliminating the wavefield with a forward PDE solve and an adjoint (as
in classical FWI), ODIL keeps the **discretized wavefield `u` as an explicit
optimization variable** alongside the **velocity model `c`**, and minimizes a
single discrete loss

```
L(u, c) = w_pde · mean|r_pde|²  +  w_data · mean|r_data|²  +  w_reg · R(c)
```

where `r_pde` is the discrete wave-equation residual and `r_data` is the misfit
at the receivers. The PDE is a *soft* penalty, so `u` and `c` are recovered
jointly (or block-by-block).

---

## Library structure

Everything importable lives under the `odil_wave` package:

| Subpackage        | Responsibility |
|-------------------|----------------|
| `grid`            | `Grid` — 2D space+time discretization with a PML sponge, non-dimensionalisation, and frequency-bin selection (`FrequencySelection`, usable-frequency limits). |
| `source`          | `SourceSignal` — tone-burst / Ricker source wavelets. |
| `geometry`        | `AcquisitionGeometry` — receiver ring + rotated source octets ("offsets"). |
| `models`          | `VelocityModel` — the `c` field and built-in profiles (`shepp_logan`, `…_skull`, …). |
| `wavefield`       | `Wavefield` — the complex field `u(shot, freq, x, y)` optimization variable. |
| `operator`        | Discrete physics: `WaveEquation` (frequency residual), spatial Laplacians (orders 2–10), PML/Neumann conditions, plus `LeapfrogSolver` (time) and `HelmholtzSolver` (frequency) forward solvers. |
| `loss`            | `InverseLoss` / `ForwardLoss` — assemble the PDE + data (+ regulariser) terms; `Regulariser` (Tikhonov / TV). |
| `optimisation`    | Inversion drivers: `LBFGSB` (block-coordinate dual L-BFGS), `JointFreqODIL` (full-space joint), frequency-continuation loop, u-block Hessian / preconditioners. |
| `metrics`         | `ssim`, `mse`, `mae` for scoring a recovered `c` against truth. |
| `experiment`      | The reproducible run layer: `config` (schema, resolution, validation — **torch-free**), `problem` (builds grid/source/data once), `runner` (drives the bands, writes artifacts), `recorder`, `plots`, `diagnostics`. |

Top-level `main.py` is the CLI entry point; `configs/default_inverse.yaml` is the
reference configuration.

---

## Running an inversion

From the `odil/` directory:

```bash
# Reference single-band Shepp-Logan run
python main.py --config configs/default_inverse.yaml

# Validate + print a summary, write nothing
python main.py --config configs/default_inverse.yaml --dry-run

# Override any dotted config key (repeatable)
python main.py --config configs/default_inverse.yaml \
    --override optimiser.n_iter=40 \
    --override runtime.device=mps

# Built-in defaults (no config file)
python main.py
```

Config resolution is lazy and torch-free, so `--help` and `--dry-run` are
instant; the numerical stack is only imported once a real run starts.

### Programmatic use

```python
from odil_wave.experiment.config import resolve_config, load_config_file
from odil_wave.experiment.runner import run_inverse

cfg = resolve_config(load_config_file("configs/default_inverse.yaml"),
                     overrides=["optimiser.n_iter=40"])
result = run_inverse(cfg)          # -> RunResult with per-band metrics
print(result.metrics_summary)
```

### Outputs

Each run writes a self-contained directory under `outputs/<run_id>/`:

- `config_resolved.yaml` — every effective value (single source of truth).
- `metadata.json` — env, seed, git, grid summary.
- `bands/band_XX_*/` — recovered `c_final.npy` + per-band summary/metrics.
- `final/c_final.npy`, `final/run_summary.json` — final model + SSIM / rel-c-error.
- `diagnostics/…` — optional block-coordinate diagnostics (when enabled).

---

## Main configuration options

Grouped as in `configs/default_inverse.yaml`; omitted keys fall back to the code
defaults in `odil_wave/experiment/config.py`.

| Group | Key options | Notes |
|-------|-------------|-------|
| `runtime`      | `device` (`cpu`/`cuda`/`mps`), `dtype` (`float32`/`float64`) | f64 reaches far lower loss floors; f32 is faster on MPS/GPU. |
| `grid`         | `interior_shape`, `interior_extent`, `c_min`/`c_max`, `pml_width`, `t_max`, `init_nt` | Optimization variable lives on the PML-extended grid. |
| `source`       | `kind` (`tone_burst`), `f0`, `n_cycles`, `envelope` | Center frequency `f0` drives resolution and cycle-skipping. |
| `acquisition`  | `n_receivers`, `n_sources`, `a_frac`/`b_frac`, `ring_center` | Ring transmission geometry (ultrasound-style). |
| `physics`      | `space_order`, `time_order`, `pml_weight` | Higher `space_order` reduces numerical dispersion. |
| `truth` / `init` | `profile`, `scale` | Ground truth vs. starting model (e.g. `shepp_logan` vs `…_skull`). |
| `observation`  | `method` (`leapfrog_fft`/`helmholtz`), `normalize_data` | `leapfrog_fft` avoids the inverse crime; per-receiver normalisation stabilises the data term. |
| `continuation` | `warm_start` (`helmholtz`/`none`), `bands[]` | Low→high multiscale: each band's recovered `c` warm-starts the next. |
| `loss`         | `weights.{pde,data,reg}`, `regulariser` | Set `reg > 0` to activate Tikhonov / TV. |
| `optimiser`    | `name` (`lbfgsb`/`joint`), `n_iter`, `u_steps`, `c_steps` | See below. |
| `metrics`      | `ssim.mask` (`head_roi`/`interior`), `ssim.win_size` | Scoring region and window. |
| `diagnostics`  | `enabled`, `per_outer_field_maps`, `verify_hessian` | Extra per-outer maps (LBFGSB only; expensive). |

### Optimisers

- **`lbfgsb`** (default) — block-coordinate dual L-BFGS. Each outer iteration
  updates the wavefield `u`-block, then the `c`-block. Key knobs under
  `optimiser.lbfgsb`: `u_solve` (`optim` = L-BFGS u-step, or `exact` = direct
  sparse `u* = u₀ − H⁻¹g`), `u_precond` (`z` reparameterises `u = A(c)⁻¹z`),
  `c_lr` / `c_max_iter`, and `c_grad_smooth_sigma` (the sole c-gradient
  preconditioner).
- **`joint`** — full-space ODIL: one L-BFGS over `(u, c)` together, with optional
  model/u-block preconditioning (`optimiser.joint.*`).

### Frequency continuation (`bands`)

List several single- or multi-frequency stages to run low→high; each recovered
model seeds the next. `source_offsets` picks rotated source octets on the ring,
and `source_schedule` controls them: `joint` (all at once), `sequential` (one
octet after another), or `cyclic` (alternate one octet per outer step).

---

## Important points

- **No inverse crime.** Observed data is a broadband *leapfrog time* solve FFT'd
  onto the band bins (`observation.method: leapfrog_fft`), so the inversion's
  frequency operator never regenerates its own data.
- **Non-dimensional physics.** The grid rescales length/velocity/time so the
  discrete residual is dimensionless; physical `dx, dt, …` are retained only for
  plotting/indexing.
- **Reproducible by construction.** Seeds, device, git state, and the fully
  resolved config are recorded per run; a crash still leaves a reproducible
  `config_resolved.yaml` and a `failure.json`.
- **dtype matters.** `float64` reaches much lower loss floors (useful for
  verification); `float32` is the fast path on GPU/MPS.

---

## Tests

```bash
pytest                     # full suite
pytest tests/test_experiment_config.py   # a single module
```

## Install

```bash
pip install -e .           # editable; core deps: numpy, scipy, torch, jax, scikit-image, pyyaml
pip install -e ".[dev]"    # + black, flake8, pre-commit
```

Requires Python ≥ 3.12.
