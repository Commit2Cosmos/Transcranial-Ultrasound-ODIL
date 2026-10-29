# `odil_wave.experiment` — reproducible FWI runs

A thin, reproducible run layer over `odil_wave`. Every inversion is fully
described by a saved, resolved configuration plus a self-contained artifact
directory, and can be reproduced from them. It runs the **joint** optimiser
(`JointFreqODIL`) behind one config-driven entry point.
No notebook runtime is required; there are no interactive prompts.

The numerical behaviour (physics, defaults, frequency handling, solver update
rules) is unchanged from the notebook `sandbox/odil_2d_freq_domain.ipynb`; this
layer only resolves configuration, orchestrates the run, logs, and saves.

---

## Start a run from a config

```bash
python main.py                                         # the default config
python main.py --print-defaults                        # print it as YAML
python main.py --config configs/example_multiband.yaml # an experiment config
```

The **default config** (a single-band joint Shepp-Logan run) is defined only in
code: the `RunConfig` dataclass defaults in `config.py`, whose docstrings
document every field. An **experiment config** names only the values that
differ from it; `resolve_config` applies it, then any CLI overrides, to the
defaults. YAML and JSON are both accepted; scientific notation such as `40e3`
is parsed as a float, matching the notebook's Python literals.

Programmatic use:

```python
from odil_wave.experiment import resolve_config, run_inverse, load_config_file

cfg = resolve_config(load_config_file("configs/example_multiband.yaml"))
result = run_inverse(cfg)                 # -> RunResult
print(result.metrics_summary["final_ssim_head_roi"])
```

## Plotting & analysis

All figures come from the library, so a saved run reproduces without
re-solving. These helpers read the artifacts below (`metrics.jsonl`,
`bands/*/c_final.npy`, `final/c_final.npy`):

```python
from odil_wave.experiment import (
    load_metrics, load_loss_tape, plot_run_history,
    plot_velocity_recovery, animate_bands,
)

rd = result.run_dir
plot_run_history(rd)                       # per-band loss & head-ROI SSIM
load_loss_tape(rd).show()                  # rebuilt LossTape: loss / PDE / data RMS
plot_velocity_recovery(rd, problem.truth_velocity, band0.geom)  # truth|rec|diff|error
animate_bands(rd, problem.grid, filename="c_bands.gif")         # one frame per band
```

`Wavefield.show(idx)` draws a 2x2 of `|u|`, `Re(u)`, `Im(u)` and phase;
`SourceSignal.plot_spectrum(frequencies=band.freq)` marks the inversion bins on
the source spectrum; `odil_wave.metrics.ssim_map(...)` shows the per-pixel SSIM.

### Watch a run live and stop it early

`run_inverse(..., on_band_end=cb)` calls `cb(band_result, run_result)` after
every completed frequency band. `LiveVelocityView` is a ready-made callback that
redraws the recovered velocity and the SSIM / relative-error trend after each
band, so a stalled run is obvious and can be interrupted:

```python
from odil_wave.experiment import LiveVelocityView

live = LiveVelocityView(problem.truth_velocity)
result = run_inverse(cfg, on_band_end=live)
```

A callback exception is caught and reported, never aborting the run.

## Override settings from the CLI

Dotted-key overrides are applied after the config file and before validation.
Values are parsed as JSON (so numbers, booleans, `null`, and lists are typed),
falling back to a bare string.

```bash
python main.py --override optimiser.n_iter=40

# switch to a low->high continuation schedule
python main.py \
    -o 'continuation.bands=[{"frequencies_hz":[50e3]},{"frequencies_hz":[60e3]}]'
```

## Dry run

```bash
python main.py --config configs/example_multiband.yaml --dry-run
```

`--dry-run` resolves + validates the configuration, prints a concise summary,
and writes **no** artifacts.

---

## Where each artifact is written

> **Spec ahead of code.** This layout is the one decided in "Contents of a
> saved run" (#54). The implementation issue lands the code and removes this
> note.

One self-contained directory per run at `<output_root>/<run_id>/`:

```
outputs/<run_id>/
  config_resolved.yaml     # every effective value used by the run
  metadata.json            # written once at start: code/env/host/git/uv.lock/seed/grid
  git.diff                 # scratch runs only: tracked + staged diff
  tape.csv                 # per-row metric tape (see below); empty cell = null
  problem.npz              # c*, c0 (interior), pml_c, brain/skull masks,
                           #   transducer angles, interior slice
  bands/
    band_00_40khz/
      arrays.npz               # freqs, c (interior), d_obs, d_syn_start,
                               #   d_syn_final (all shots), shots, u, u_star
      band_summary.json        # frequencies, times, iters, termination, final
                               #   metrics, costs, warnings, worst-fit shot
    band_01_50khz/ ...
  run_summary.json         # always written: status, wall time, warnings,
                           #   per-band + final metrics, traceback on failure
  figures/                 # post-processing output (PNG + CSV)
```

- **YAML** for configs, **JSON** for metadata and summaries, **CSV** for the
  tape, uncompressed **`.npz`** for arrays. Writes are atomic (tmp + rename) for
  summaries, configs and arrays; the tape is flushed per row so a crash still
  leaves a valid, truncated log.
- Arrays are always **float32 / complex64**, whatever `runtime.dtype`; masks
  are bool.
- `c`, `c*` and `c0` cover the **interior** only: the PML is frozen at the
  scalar `pml_c`, so the full-grid `c` is the interior padded with `pml_c`.
- **Wavefields** are saved at each band's final iterate, all band frequencies,
  for the transducers nearest 0°, 90°, 180° and 270° plus the worst-fit shot
  (highest per-shot `rel_data_residual`) unless it is one of those four;
  `shots` holds the indices. `u` is on the full inversion grid (the PDE
  residual needs the PML); `u_star` is on the interior. Shapes are
  `(n_saved_shots, n_freq, nx, ny)`.
- `metadata.json` records the git sha and `dirty` flag (false unless the run is
  scratch), `scratch`, and `uv_lock_sha256`.
- `run_summary.json` has `status: completed | failed | interrupted`. A failed
  or interrupted band keeps its tape rows but writes no `arrays.npz` or
  `band_summary.json`.
- **Size**: about 2.3 MB per frequency at the default grid (5 shots ×
  (0.34 MB `u` + 0.125 MB `u_star`)), so ~15 MB for a 3-band, 6-frequency run.
  Not enforced.

### Fully-resolved config

`config_resolved.yaml` contains every effective value, including the default
config values the input did not change, so the exact run is reconstructable
even when the input omitted fields. Derived quantities that are *outputs* rather
than inputs (grid `dt`/`dx`/`nt`, FFT bins, head-mask cell count) live in
`metadata.json` and the per-band summaries, not in the config.

---

## Iteration-level log record — exact semantics

One row is appended to `tape.csv` at every
`optimiser.log_every` **logical outer iterations** (and always on the last
iteration). A record corresponds to **one accepted outer iteration `i`** of the
solver — *not* an L-BFGS closure / function evaluation.

Implementation: `RecordingTape` subclasses `LossTape` and is passed both as the
loss `callback` and as the optimiser's `on_iteration` hook. `on_iteration(i, c)`
fires every outer iteration and stamps the outer index + current velocity field;
the inherited `log(...)` fires only at `log_every` and carries the exact loss the
solver reports. This needs **zero changes to the optimiser**.

Fields (`tape.csv` columns), see `recorder.FIELDNAMES`.

> **Spec ahead of code.** This table is the column set decided in
> "Review reconstruction metrics" (#42), amended by "Standard post-processing
> and plot set" (#53). The implementation issue lands the
> code and removes this note. Regions follow `GLOSSARY.md` (**Brain region**,
> **Skull region**, **Interior**); `c*`, `u*`, `d_obs` are the truth model, the
> true wavefield and the observed traces.

Rows: one per `optimiser.log_every` outer iterations, plus one at the **final
iterate of every band** (always), plus one **start row** per band (`iter = 0`:
after the band's warm start, before any update). The warm start is a Helmholtz
solve on the current model, so a start row's data residual is the forward-model
misfit at the band's frequencies.
Band and run summaries report exactly these columns at the final iterate.

#### Identity and cost

| column | meaning / calculation |
|---|---|
| `run_id` | run identity; lets rows of several runs be concatenated |
| `band_index` | continuation stage; join to `band_summary.json` for its frequencies |
| `iter` | outer iteration within the band (`0` = start row) |
| `iter_total` | cumulative outer iterations across all bands, up to this row |
| `wall_s` | wall-clock seconds since the run started |
| `n_evaluations` | cumulative `InverseLoss.evaluate` calls (L-BFGS closures) across the **whole run**; the cost axis for equal-cost comparisons |

#### Objective (what the optimiser minimises)

| column | meaning / calculation |
|---|---|
| `loss_total` | total objective exactly as the solver reports it |
| `pde_term` | `w_pde · mean(\|r_pde\|²)`, the weighted **PDE term** |
| `data_term` | `w_data · mean(\|r_data\|²)`, the weighted **Data term** |
| `reg_term` | `w_reg · R(c_interior)` evaluated at the logged point; `0.0` without a regulariser |

#### Normalised residuals (comparable across bands and configs)

| column | meaning / calculation |
|---|---|
| `rel_data_residual` | `‖d_syn − d_obs‖₂ / ‖d_obs‖₂` over the band's shots × frequencies × receivers; `d_syn` is the wavefield sampled at the receivers |
| `pde_src_ratio` | `RMS(r_pde) / RMS(source)`, PDE residual over interior + PML relative to the source |
| `rel_wavefield_error` | `‖u − u*‖₂ / ‖u*‖₂` over the interior, all shots × band frequencies; `u*` is the true wavefield, cropped to the interior |
| `rel_model_update` | `‖c − c_prev‖₂ / ‖c_prev‖₂` over the interior, `c_prev` at the previous row of the same band (`null` on a start row); truth-free plateau signal |

#### Model error (m/s, against `c*`)

| column | meaning / calculation |
|---|---|
| `rmse_brain` | `sqrt(mean((c − c*)²))` over the brain region |
| `bias_brain` | `mean(c − c*)` over the brain region; sign shows too slow / too fast |
| `rmse_skull` | as `rmse_brain`, over the skull region |
| `bias_skull` | as `bias_brain`, over the skull region; catches a skull left at its starting value |
| `ssim_brain` | SSIM of `c` vs `c*`: per-pixel map over the interior (`gaussian_weights=True`, `sigma=1.5`, `use_sample_covariance=False`, `data_range` = peak-to-peak of `c*` in the brain region), averaged over the brain region **eroded by the window radius** so windows straddling the skull edge are excluded |

#### Optimiser health

| column | meaning / calculation |
|---|---|
| `c_min`, `c_max` | extremes of `c` over the interior; blow-up check |
| `sat_frac` | fraction of cells whose bounded-`c` sigmoid is within `1e-3` of 0 or 1, i.e. pinned at a speed bound |
| `grad_u_norm` | `‖∇_u L‖₂` over real and imaginary wavefield parts |
| `grad_z_norm` | `‖∇_z L‖₂`, with `z` the unconstrained model variable |

Missing truth, brain mask or an SSIM window too large for the grid is a
**config error at startup**, not an empty cell.

Derived in post-processing, not on the tape: `rel_data_residual` per frequency
and per **Offset** (ring angle 0–180°, one point per distinct offset) at each
band's final iterate, from the saved receiver wavefield and observed data.

### Iteration count vs. closure/function-evaluation count

- `iter` is the **outer** iteration of the joint solver (one L-BFGS
  `.step` of up to `optimiser.inner_max_iter` iterations over `u` and `c`
  together per `i`). This is what the record is keyed on.
- `n_evaluations` is the cumulative number of `InverseLoss.evaluate` calls, i.e.
  L-BFGS **closures / function evaluations** (the line search calls the closure
  several times per accepted step). It is logged only for diagnostics and is
  **not** an iteration count.
- Per-band totals — `n_outer_iter`, `n_closure`, `n_evaluations` — are recorded
  once in `band_summary.json` under `optimiser_result`.

Note: `loss_total` (the solver's returned loss for the step) and
`pde_term + data_term` are evaluated at slightly different points within an
L-BFGS line search, so they need not sum exactly — this is inherent to the
solver's own logging, not introduced here. `reg_term` is therefore taken from
the configured regulariser, never by subtraction.

---

## Brain-region SSIM

Settings live under `metrics.ssim` in the config and are recorded in the
resolved config; the scoring rule is in the `ssim_brain` row above.

---

## Add a config parameter

Each parameter has exactly one owner (ADR 0001):

1. Add the field with its default and a docstring entry to the matching frozen
   dataclass in `config.py`; that default is part of the default config and is
   serialised into `config_resolved.yaml` for every run automatically.
2. Make the library constructor it feeds take it as a required keyword-only
   argument (no library default), and pass it in `problem.py` / `runner.py`.

A numerical-method detail that is not in the config keeps its library default.
