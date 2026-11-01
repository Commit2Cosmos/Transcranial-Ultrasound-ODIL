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
jointly.

---

## Library structure

Everything importable lives under the `odil_wave` package:

| Subpackage        | Responsibility |
|-------------------|----------------|
| `grid`            | `Grid` — 2D space+time discretization with a PML sponge and non-dimensionalisation; `FrequencySelection` maps requested Hz to the nearest FFT bins. |
| `source`          | `SourceSignal` — tone-burst / Ricker source wavelets. |
| `geometry`        | `AcquisitionGeometry` — transducer ring; shot `s` transmits from transducer `s` and is recorded by the other `n_transducers − 1`. |
| `models`          | `VelocityModel` — the `c` field and built-in profiles (`shepp_logan`, `…_skull`, …). |
| `wavefield`       | `Wavefield` — the complex field `u(freq, x, y)` for one shot; multi-shot batches used by losses/optimisers stack these to `(n_shots, nf, nx, ny)`. |
| `operator`        | Discrete physics: `WaveEquation` (frequency residual), spatial Laplacians (orders 2–10), PML sponge/Neumann conditions, plus the `HelmholtzSolver` (frequency) forward solver. |
| `loss`            | `InverseLoss` / `ForwardLoss` — assemble the PDE + data (+ regulariser) terms; `Regulariser` (Tikhonov / TV). |
| `optimisation`    | Inversion driver: `JointFreqODIL` (full-space joint L-BFGS over `u` and `c`). |
| `metrics`         | `ssim` / `ssim_map`, `mse`, `mae` for scoring a recovered `c` against truth. |
| `experiment`      | The reproducible run layer: `config` (schema, resolution, validation — **torch-free**), `problem` (builds grid/source/data once), `runner` (drives the bands, writes artifacts), `recorder`, `plots`. |

Top-level `main.py` is the CLI entry point. The **default config** is defined in
code — the `RunConfig` dataclass defaults in `odil_wave/experiment/config.py`,
whose docstrings document every field — and `configs/` holds **run
configs** that name only the values they change (see
[ADR 0001](docs/adr/0001-config-defaults-single-owner.md)).

---

## Running an inversion

From the `odil/` directory:

```bash
# The default config: single-band (60 kHz) Shepp-Logan run
python main.py

# Print the full default config as YAML
python main.py --print-defaults

# A run config (only the values that differ from the defaults)
python main.py --config configs/example_multiband.yaml

# Validate + print a summary, write nothing
python main.py --config configs/example_multiband.yaml --dry-run

# Override any dotted config key (repeatable)
python main.py --config configs/example_multiband.yaml \
    --override optimiser.n_iter=40 \
    --override runtime.device=mps

# A run needs a clean git tree; --scratch allows uncommitted changes
python main.py --scratch --config configs/example_multiband.yaml
```

Config resolution is lazy and torch-free, so `--help` and `--dry-run` are
instant; the numerical stack is only imported once a real run starts.

### Programmatic use

```python
from odil_wave.experiment import (
    load_config_file, new_run_directory, resolve_config, run_inverse,
)

cfg = resolve_config(load_config_file("configs/example_multiband.yaml"),
                     overrides=["optimiser.n_iter=40"])
result = run_inverse(cfg, new_run_directory())  # -> RunResult with per-band metrics
print(result.metrics_summary)
```

### Outputs

Each run writes a self-contained directory
`outputs/runs/<YYMMDD-HHMMSS>_<sha7>[_scratch]/` (layout in
`odil_wave/experiment/README.md`):

- `config_resolved.yaml` — every effective value (single source of truth).
- `metadata.json` — schema version, commit, scratch flag, `uv.lock` hash, env,
  seed, grid summary.
- `tape.csv` — the per-iteration metric tape.
- `problem.npz` — truth and starting models, masks, transducer angles.
- `bands/band_XX_*/` — `arrays.npz` (model, data, saved wavefields) +
  `band_summary.json` (metrics, costs, health warnings), per completed band.
- `run_summary.json` — status (`completed | failed | interrupted`), final
  metrics, and the traceback on failure.

---

## Main configuration options

Omitted keys keep the default config; `python main.py --print-defaults` prints
it, and the dataclass docstrings in `odil_wave/experiment/config.py` are the
single place each field is documented (units, allowed values, meaning). Unknown
or removed keys are rejected.

| Group | Key options | Notes |
|-------|-------------|-------|
| `runtime`      | `device` (`cpu`/`cuda`/`mps`), `dtype` (`float32`/`float64`), `verbose` | f64 reaches far lower loss floors; f32 is faster on MPS/GPU. |
| `grid`         | `interior_shape` (`[nx, ny]` nodes), `interior_extent` (`[[x0, x1], [y0, y1]]` m), `c_min`/`c_max`, `space_order` (2/4/6/8/10) | Optimization variable lives on the PML-extended grid; higher `space_order` reduces numerical dispersion. |
| `pml`          | `width`, `power`, `R0`, `damping_scale`, `observation_width` | `observation_width: null` generates the data on the inversion grid (inverse crime); a different width is a deliberate modelling error. |
| `time`         | `t_max`, `init_nt`, `cfl_safety` | Sets `dt` and the FFT bins the band frequencies snap to. |
| `source`       | `kind` (`tone_burst`/`ricker`), `f0`, `n_cycles`, `envelope` | Center frequency `f0` drives resolution and cycle-skipping. |
| `acquisition`  | `n_transducers`, `a_frac`/`b_frac`, `ring_center`, `sigma_s`, `source_spatial` | Ring of transducers (ultrasound-style): each transmits one shot, recorded by all the others. The source is a density that integrates to 1 on any grid: a Gaussian of width `sigma_s` (metres, default 3.4 mm) or a point `1/(dx·dy)`; see below. |
| `truth` / `init` | `profile`, `scale`, `pml_c` | Ground truth vs. starting model (e.g. `shepp_logan` vs `…_skull`); the PML ring is always padded with the constant `pml_c`. |
| `continuation` | `bands[]` | Low→high multiscale: each band's recovered `c` warm-starts the next; every band's wavefield is warm-started by a Helmholtz solve on that model. |
| `loss`         | `weights.{pde,data,reg}`, `regulariser.{name,eps}` | The only objective weights (defaults 1 / 1 / 0); `reg > 0` activates the regulariser (`tv_iso`/`tikhonov`/`tv_aniso`). |
| `optimiser`    | `n_iter`, `log_every`, `inner_max_iter`, `history_size`, … | Joint L-BFGS settings; see below. |
| `metrics`      | `ssim.sigma` | Gaussian window of the tape's `ssim_brain`; the truth must have brain and skull regions (`shepp_logan`/`shepp_logan_skull`). |

### Optimiser

- **`joint`** — full-space ODIL (`JointFreqODIL`): a single persistent L-BFGS
  over the wavefield `u` and a bounded squared-slowness latent for `c` together,
  no block alternation. `u` is scaled by a fixed per-(shot, frequency) reference;
  `c` is reparameterised through a sigmoid so its bounds hold structurally. It
  minimises `Phi = pde * pde_term + data * data_term + reg * R(c)` with the
  weights from `loss.weights`. The L-BFGS
  settings (`inner_max_iter`, `history_size`, …) live under `optimiser`; the
  latent velocity bounds are `grid.c_min` / `grid.c_max`.

### Objective balance

The PDE/data balance does not depend on grid spacing, PML width or the
source's normalisation (ADR-0002), so `loss.weights` mean the same at any
resolution:

- **PDE term**: the residual of each (shot, frequency) divided by
  `|λ_tt|·u_ref`, the size of the wave term (`u_ref` is the interior RMS of
  the band's Helmholtz warm start, fixed for the band), summed over interior
  **and** PML nodes and normalised by the interior node count.
- **Data term**: the receiver misfit of each (shot, frequency) divided by the
  RMS of its observed traces, then a **mean** over traces, so it does not grow
  with `n_transducers`. There is no noise model: the data are noise-free
  (the inverse crime).
- `python -m odil_wave.experiment.balance -c configs/checks/balance.yaml
  --cases 1 2 3 1:2` checks it: the PDE term at `(H⁻¹(c_true), c_init)` over
  the data term at `(H⁻¹(c_init), c_init)` on 125²/PML 40, 250²/80, 375²/120
  and 125²/80 (case `k` scales interior and PML; `k:p` scales them apart).

**Source width.** `acquisition.sigma_s` is a length in metres. Keep
`1.5 h ≤ sigma_s ≤ 1/k_max` with `k_max = 2π f_max / grid.c_min`: below 1.5 h
the Gaussian is under-resolved, above `1/k_max` it low-pass filters the top
frequency. That window needs ≥ 3π ≈ 9.4 points per wavelength at `f_max`
(the default grid has 10.7 at 60 kHz, 8.1 at 80 kHz). The config warns, printing
`sigma_s/h`, `k_max·sigma_s` and the ring's clearance, when either bound is
crossed or the ring is within 3 `sigma_s` of the PML.

### Frequency continuation (`bands`)

List several single- or multi-frequency stages under `continuation.bands` to run
low→high; each stage's recovered `c` warm-starts the next, and its wavefield is
warm-started by a Helmholtz solve on that model.
Frequencies within one band are inverted jointly; every transducer
(`acquisition.n_transducers`) fires one shot in every band — there is no
per-band source rotation or scheduling.

---

## Important points

- **Inverse crime.** By default (`pml.observation_width: null`) observed data
  are a Helmholtz solve on the truth model with the same grid and
  frequency-domain operator the inversion uses, so the true model fits the data
  exactly (a deliberate inverse-crime baseline). Setting a different
  `pml.observation_width` synthesises the data with that PML width instead.
- **Non-dimensional physics.** The grid rescales length/velocity/time so the
  discrete residual is dimensionless; physical `dx, dt, …` are retained only for
  plotting/indexing.
- **Reproducible by construction.** Seeds, device, commit, `uv.lock` hash and
  the fully resolved config are recorded per run; a crash still leaves a
  reproducible `config_resolved.yaml` and a `run_summary.json` saying why.
- **dtype matters.** `float64` reaches much lower loss floors (useful for
  verification); `float32` is the fast path on GPU/MPS.

---

## Example notebook

`notebooks/odil_wave_inverse_example.ipynb` walks through a full inversion using
the library building blocks directly (`grid`, `source`, `geometry`, `models`,
`loss`, `operator`, `optimisation`), then shows the batteries-included
`odil_wave.experiment.run_inverse` wrapper.

## Tests

```bash
pytest                              # full suite
pytest tests/test_optimisation.py   # a single module
```

`tests/` covers the CLI (`test_main.py`), the optimisation drivers
(`test_optimisation.py`), scoring metrics (`test_metrics.py`), the discrete
physics operators (`test_operator.py`), the `Wavefield` container
(`test_wavefield.py`), the grid/PML/frequency-selection machinery
(`test_grid.py`), source wavelets (`test_source.py`), velocity model profiles
(`test_models.py`), acquisition geometry and trace extraction
(`test_geometry.py`), the loss/regulariser/loss-tape assembly (`test_loss.py`),
the run-config/problem-building layer (`test_experiments.py`), the
one-owner-per-parameter rule for library constructors
(`test_parameter_ownership.py`), and end-to-end forward-to-loss and tiny-inversion runs (`test_integration.py`).

## Install

```bash
pip install -e .           # editable install; see pyproject.toml for the full dependency list
pip install -e ".[dev]"    # + ruff, pre-commit, ipykernel, torchinfo
```

Requires Python ≥ 3.12.
