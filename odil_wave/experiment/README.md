# `odil_wave.experiment` — reproducible FWI runs

A thin, reproducible run layer over `odil_wave`. Every inversion is fully
described by a saved, resolved configuration plus a self-contained artifact
directory, and can be reproduced from them. It supports all implemented
optimisers — **LBFGSB**, **closed-form (`cf`)**, and **MODIL** — behind one
config-driven entry point. No notebook runtime is required; there are no
interactive prompts.

The numerical behaviour (physics, defaults, frequency handling, solver update
rules) is unchanged from the notebook `sandbox/odil_2d_freq_domain.ipynb`; this
layer only resolves configuration, orchestrates the run, logs, and saves.

---

## Start a run from a config

```bash
python main.py --config configs/default_inverse.yaml
```

`configs/default_inverse.yaml` reproduces the notebook's single-band 40 kHz
LBFGSB Shepp-Logan run. Running with **no** `--config` uses the built-in
defaults (identical to the shipped file). YAML and JSON are both accepted;
scientific notation such as `40e3` is parsed as a float, matching the notebook's
Python literals.

Programmatic use:

```python
from odil_wave.experiment import resolve_config, run_inverse, load_config_file

cfg = resolve_config(load_config_file("configs/default_inverse.yaml"))
result = run_inverse(cfg)                 # -> RunResult
print(result.metrics_summary["final_ssim_head_roi"])
```

Force a specific solver (each returns a `RunResult`):

```python
from odil_wave.experiment import (
    run_inverse_lbfgsb, run_inverse_closed_form, run_inverse_modil,
)
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
python main.py --config configs/default_inverse.yaml \
    --override optimiser.name=cf \
    --override optimiser.n_iter=40

# switch to a low->high continuation schedule
python main.py -c configs/default_inverse.yaml \
    -o 'continuation.bands=[{"frequencies_hz":[50e3]},{"frequencies_hz":[60e3]}]'
```

## Dry run

```bash
python main.py --config configs/default_inverse.yaml --dry-run
```

`--dry-run` resolves + validates the configuration, prints a concise summary,
and writes **no** artifacts.

---

## Where each artifact is written

One self-contained directory per run at `<output_root>/<run_id>/`:

```
outputs/<run_id>/
  config_input.yaml        # the raw user config (only if one was supplied)
  config_resolved.yaml     # every effective value used by the run
  metadata.json            # code/env/host/git/seed/grid tying artifacts to this run
  metrics.csv              # per-iteration metric tape (plotting-friendly)
  metrics.jsonl            # per-iteration metric tape (append-friendly)
  bands/
    band_00_40khz/
      band_config.yaml         # resolved per-band configuration
      c_final.npy              # full-grid c after this band (float32)
      c_final_metadata.json    # shape/dtype/bins/grid summary/stats
      band_summary.json        # times, iters, termination, final metrics, closures
    band_01_50khz/ ...
  final/
    c_final.npy            # final run-level c (float32)
    run_summary.json       # status, wall time, per-band + final metrics
  failure.json             # only if the run stopped/failed mid-band
```

- **YAML** for configs, **JSON/JSONL** for metadata + logs, **CSV** for plotting,
  `.npy` for velocity fields. Writes are atomic (tmp + rename) for summaries and
  configs; the CSV/JSONL tapes are flushed per record so a crash still leaves a
  valid, truncated log.
- Wavefield snapshots are **not** written (no option requests them).
- On failure mid-band, `failure.json` records the exception type/message,
  traceback, last band index, and last completed logical iteration; the metric
  tape and already-completed band artifacts are preserved.

### Fully-resolved config

`config_resolved.yaml` contains every effective value, including code-derived
defaults and the default blocks for **all** supported optimisers (`lbfgsb`,
`cf`, `modil`) — not just the selected one — so the exact run is reconstructable
even when the input omitted fields. Derived quantities that are *outputs* rather
than inputs (grid `dt`/`dx`/`nt`, FFT bins, head-mask cell count) live in
`metadata.json` and the per-band summaries, not in the config.

---

## Iteration-level log record — exact semantics

One record is appended to `metrics.csv` and `metrics.jsonl` at every
`optimiser.log_every` **logical outer iterations** (and always on the last
iteration). A record corresponds to **one accepted outer iteration `i`** of the
block-coordinate solver — *not* an L-BFGS closure / function evaluation.

Implementation: `RecordingTape` subclasses `LossTape` and is passed both as the
loss `callback` and as the optimiser's `on_iteration` hook. `on_iteration(i, c)`
fires every outer iteration and stamps the outer index + current velocity field;
the inherited `log(...)` fires only at `log_every` and carries the exact loss the
solver reports. This needs **zero changes to any optimiser** and behaves
identically for LBFGSB, CF and MODIL.

Fields (CSV columns == JSONL keys), see `recorder.FIELDNAMES`:

| field | meaning |
|---|---|
| `run_id`, `solver` | run identity + solver name (`lbfgsb`/`cf`/`modil`) |
| `band_index`, `band_frequencies_hz` | which continuation stage, and its Hz |
| `solver_iter` | the **logical outer iteration** index `i` (per band) |
| `global_iter` | cumulative logged-record index across all bands |
| `wall_s` | wall-clock seconds since the run started |
| `loss_total` | total objective (exactly what the solver reports/prints) |
| `pde_term`, `data_term`, `reg_term` | weighted contributions `w·term`; `reg_term` is `0.0` with no regulariser configured (and `null` if a regulariser is wired but not decomposable) |
| `pde_loss`, `data_loss` | unweighted mean‑squared residual terms |
| `pde_rms`, `data_rms`, `pde_src_ratio` | residual RMS diagnostics |
| `c_min`, `c_max`, `c_mean`, `c_std` | velocity summary over the non-PML interior |
| `rel_c_error` | relative L2 `‖c−c*‖/‖c*‖` over the interior (`null` without truth) |
| `ssim_head_roi` | masked head-ROI SSIM (see below); `null` if unavailable |
| `ssim_reason` | why SSIM is `null` (e.g. `no ground truth`), else `null` |
| `n_evaluations` | cumulative loss **function-evaluation** count so far |

### Iteration count vs. closure/function-evaluation count

- `solver_iter` is the accepted **outer** iteration of the block-coordinate loop
  (one `u`-block + one `c`-block per `i`). This is what the record is keyed on.
- `n_evaluations` is the cumulative number of `InverseLoss.evaluate` calls, i.e.
  L-BFGS **closures / function evaluations** (the line search calls the closure
  several times per accepted step). It is logged only for diagnostics and is
  **not** an iteration count.
- Per-band totals — `n_outer_iter`, `n_u_closure`, `n_c_closure`, `n_closure` —
  are recorded once in `band_summary.json` under `optimiser_result`.

Note: `loss_total` (the solver's returned loss for the step) and
`pde_term + data_term` are evaluated at slightly different points within an
L-BFGS line search, so they need not sum exactly — this is inherent to the
solver's own logging, not introduced here. `reg_term` is therefore taken from
the configured regulariser, never by subtraction.

---

## Masked head-ROI SSIM

`ssim_head_roi` scores structural similarity **only inside the intracranial head
ROI**, using `odil_wave.metrics.ssim(pred, true, grid, mask=...)`:

1. Both fields are sliced to the **non-PML interior** (PML pixels never enter the
   score).
2. The mask is `VelocityModel.head_mask` (the region inside the skull, for
   `shepp_logan` / `shepp_logan_skull`). The per-pixel SSIM map is computed on
   the interior and averaged **only over the mask**, so outside-head pixels are
   excluded from the mean.
3. `data_range` defaults (`"auto"`) to the peak-to-peak of the *true* field
   within the mask. `win_size`, `gaussian_weights`, `sigma` are configurable
   under `metrics.ssim`.
4. If ground truth is unavailable, or the truth profile has no head mask, or the
   window is too large for the grid, `ssim_head_roi` is `null` and `ssim_reason`
   records why — the run never fails on SSIM.

Configure via:

```yaml
metrics:
  ssim:
    mask: head_roi        # head_roi | interior | none
    data_range: auto      # "auto" or a float
    win_size: 7
    gaussian_weights: false
    sigma: 1.5
```

The existing metrics are preserved unchanged; `ssim_head_roi` is an addition.

---

## Add a new optimiser-specific config section

1. Add a frozen dataclass in `config.py` (e.g. `class MyOptCfg`) with every
   tunable and its default, and reference it from `OptimiserCfg`:

   ```python
   @dataclass(frozen=True)
   class OptimiserCfg:
       ...
       myopt: MyOptCfg = field(default_factory=MyOptCfg)
   ```

   It is now serialised into `config_resolved.yaml` for every run automatically.

2. Register the name/alias in `_OPTIMISER_ALIASES` (`canonical_optimiser_name`).

3. Add a branch in `runner._build_optimiser` mapping `cfg.optimiser.myopt.*`
   (plus the shared block-coordinate fields) to your optimiser's constructor.
   The recorder, artifact layout, and CLI need no changes — they are
   solver-agnostic.

Do the same for regularisers by extending `runner._build_regulariser`.
