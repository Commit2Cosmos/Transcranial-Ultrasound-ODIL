# `odil_wave.experiment`: reproducible runs

The run layer over `odil_wave`. A run is fully described by its resolved
config and a self-contained run directory, and can be regenerated from them.
`main.py` is the CLI over it; terms (Run config, Experiment, Baseline, Probe
run, …) are in `GLOSSARY.md`.

## Where things live

| module | owns |
|---|---|
| `config.py` | The run config schema; its dataclass defaults are the **default config** and their docstrings document every field (ADR 0001) |
| `problem.py` | Builds the grid, source, acquisition and models once per run, and each band |
| `observation.py` | Each band's observed data: the inverse crime or independent observations, and their noise |
| `runner.py` | `run_inverse`: drives the bands, writes `band_summary.json` and `run_summary.json`; `on_log` / `on_band_end` callbacks |
| `rundir.py` | Where a run is written, the clean-tree / scratch policy, `metadata.json` provenance, schema versions |
| `saved.py` | Every array and summary a run saves, and `SavedRun` to read them back |
| `tape.py` | `tape.csv`: `COLUMNS` lists each column and what it measures |
| `constraints.py` | Constraint checks on a config and their thresholds |
| `health.py` | Optimiser-health warnings and their thresholds |
| `postprocess.py` | A run's figures (`RUN_PLOTS`), drawn from the run directory only |
| `on_demand.py` | Notebook plots of a saved run, not saved |
| `plots.py` | Notebook loaders of a saved run: its config, problem, velocities and `LossTape` |
| `live.py` | `LiveVelocityView`, the `--live` window |
| `spec.py` | Experiment specs: parsing, the sweep, run labels |
| `launch.py` | `--experiment`: runs a spec, writes `experiment.json` and `summary.csv`, resumes |
| `experiment_figures.py` | An experiment's figures (`EXPERIMENT_PLOTS`) |
| `probe.py` | `--probe`: runtime estimate before a launch |
| `reproduce.py` | `--reproduce`: regenerate an experiment at its recorded commit |
| `compare.py` | Does a code change alter a run's results? |
| `balance.py` | Is the PDE/data balance grid independent? (ADR-0002) |
| `bank.py` | The experiment bank's index and checks (`docs/experiments/`) |

## Runs

```bash
python main.py --print-defaults                         # the default config
python main.py -c configs/example_multiband.yaml        # a run config
python main.py -c configs/example_multiband.yaml --dry-run
python main.py -c configs/example_multiband.yaml -o optimiser.n_iter=40
python main.py --scratch -c configs/example_multiband.yaml  # dirty tree allowed
python main.py --live -c configs/example_multiband.yaml     # watch it
python main.py --postprocess outputs/runs/<run>             # redraw figures
```

A run config names only the values it changes from the default config.
Overrides are dotted keys, parsed as JSON with a bare-string fallback. A run
needs a clean git tree; `--scratch` allows a dirty one and saves `git.diff`.
No experiment record may cite a scratch run. Ctrl-C ends a run as
`interrupted`, keeping its tape and completed bands.

`--dry-run` and every run's log print the constraint checks as a table, and
`metadata.json` records them under `constraint_report`. A failed `error`
check refuses the run; a failed `warn` check is printed and also recorded in
`config_warnings`.

From Python:

```python
from odil_wave.experiment import (
    load_config_file, new_run_directory, resolve_config, run_inverse,
)

cfg = resolve_config(load_config_file("configs/example_multiband.yaml"))
result = run_inverse(cfg, new_run_directory())
```

## Experiments

An experiment sweeps config keys of a Baseline, at one commit, into one
directory. The workflow (protocol, launch, record) is the `experiment` skill;
the bank is `docs/experiments/`.

```bash
python main.py --experiment configs/experiments/NNNN-slug.yaml            # launch / resume
python main.py --experiment configs/experiments/NNNN-slug.yaml --dry-run  # validate every run
python main.py --experiment configs/experiments/NNNN-slug.yaml --probe    # runtime estimate
python main.py --reproduce NNNN-slug [-o runtime.device=cuda]             # regenerate
```

A spec, committed before launch:

```yaml
schema_version: 2
id: 0007-tv-weight
issue: 61                      # the protocol issue
baseline:                      # an inline run config
  continuation:
    bands:
      - {frequencies_hz: [40e3], n_iter: 30}
sweep:                         # every combination of the axes is a run
  - key: loss.weights.reg
    values: [0.0, 1.0e-7, 1.0e-6]
  - key: loss.regulariser.eps
    values: [1.0e-4, 1.0e-3]
    # labels: [...]            # required when a value is a list or dict
```

- Each value is applied at its key like a run config: dicts deep-merge, lists
  replace. A feature touching several keys is swept at their common parent.
- Exactly one run must resolve to the Baseline.
- **Everything compared must be selectable by config at one commit.** New code
  lands behind a config key whose default keeps current behaviour (ADR 0001);
  comparing two commits is not an experiment.

`--experiment` takes no `--config`, `--override` or `--scratch`, except
`-o runtime.device=<device>`. A resume needs the same commit, spec and device
type.

## Outputs

```
outputs/runs/<YYMMDD-HHMMSS>_<sha7>[_scratch]/   # a run directory (saved.py)
  config_resolved.yaml  metadata.json  tape.csv  problem.npz  run_summary.json
  bands/band_<NN>_<label>/{arrays.npz, band_summary.json}
  figures/
outputs/experiments/NNNN-slug[_cuda|_mps]/       # an experiment (launch.py)
  spec.yaml  experiment.json  summary.csv  figures/
  runs/<idx>_<label>/                            # a run directory each
outputs/probes/<YYMMDD-HHMMSS>/                  # --probe
```

Post-processing reads only these directories and never solves or optimises.

## Stable contract

`--reproduce` runs the **recorded** commit's `main.py`, so later commits may
add flags but never rename, remove or change the meaning of `--experiment`,
`--out`, `--postprocess`, `--reproduce`, `--probe` and
`-o runtime.device=<device>` with `--experiment`. Exit codes: `0` completed,
`1` a run failed or was interrupted, `2` could not start.

The spec, `metadata.json` and `experiment.json` carry an integer
`schema_version`. A format change bumps it and adds a reader path; old
directories are never rewritten.
