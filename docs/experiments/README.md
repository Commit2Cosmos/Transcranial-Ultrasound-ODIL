# Experiment bank

One **experiment record** per launched experiment (see `GLOSSARY.md`), including
aborted and inconclusive ones, so negative results are banked; only a voided
launch (below) gets none. Read the index below first; take context primarily
from `current` records.

- **Record**: `NNNN-slug.md`, copied from [`TEMPLATE.md`](TEMPLATE.md). The
  **experiment id** `NNNN-slug` is 4-digit and unique, reserved at launch,
  and shared by the experiment spec (`configs/experiments/NNNN-slug.yaml`), the
  experiment directory and `--reproduce NNNN-slug`. Its number is the lowest
  one no spec, `exp/*` branch name or `launch/*` tag holds, so a voided
  launch's number is taken again; the slug tells the two apart. A branch holds
  only its own records, so the bank has gaps. An aborted record can be brief:
  the question, what happened and why it stopped.
- **Committed results**: `NNNN-slug/` holds `experiment.json`, `summary.csv` and
  up to 3 key PNGs under their standard filenames. Everything else is
  regenerated with `--reproduce` / `--postprocess`.
- **Lifecycle**: the body is frozen once the record's PR merges (typo fixes
  allowed); a wrong analysis is fixed by a new record whose Context names the
  old one. After merge only `status` and new Validity entries change.
  `invalidated`: a later code change broke one of its supported conclusions; an
  invalidated record is never checked or edited again and remains as context.
- **Index**: generated from the frontmatter by
  `uv run python -m odil_wave.experiment.bank`. `--check` validates only
  (required fields, unique id numbers) and runs in CI through
  `tests/test_experiment_bank.py`.

## Voiding a launch

A launch is **voided** when its runs no longer test its Protocol before any
record of it reaches `main`, e.g. because the default config changed under its
Baseline. Runs that test the Protocol but stopped are recorded as `aborted`
instead; a banked record is invalidated, never voided.

1. Delete its spec, any drafted record and results, and its launch tag, locally
   and on `origin`. Delete its `exp/*` branch too unless another experiment
   shares it; keep a feature under test on a `feat/*` branch first.
2. Comment on the Protocol issue:

   ````markdown
   Launch `NNNN-slug` voided (YYYY-MM-DD)

   <why the runs no longer test this Protocol>. They were never banked.

   - <what was deleted>, and the id `NNNN` released. The relaunch reserves a fresh id.
   - <where a feature under test is kept, if any>
   - **The Protocol is unfrozen.** <which parts are stale>

   Spec at launch, for reference:

   ```yaml
   <the spec>
   ```
   ````

The comment unfreezes the Protocol: revise it, and relaunch it as a new launch
with its own id. The number is free once no spec, `exp/*` branch name or
`launch/*` tag holds it.

## Maintaining the bank

1. **Invalidation note.** A PR that touches `odil_wave/`, `configs/` or
   `main.py` adds one Validity line to each `current` record whose conclusions
   it may change: `YYYY-MM-DD · note · <PR> · <what may no longer hold, and
   why>`. `status` is unchanged; no line when nothing is affected. To settle
   whether a change alters results at all,
   `python -m odil_wave.experiment.compare main` runs the tiny config at `main`
   and on your tree and compares the results exactly.
2. **Only the user writes `invalidated`**:
   `YYYY-MM-DD · invalidated by <PR or commit> · <why>`, flipping `status` to
   `invalidated`. Doubt that needs data is settled by a new experiment.
3. **Index.** After writing or updating a record, run
   `uv run python -m odil_wave.experiment.bank` to regenerate the table below.

## Index

<!-- index:start -->
| id | question | variable | status | answer |
|---|---|---|---|---|
| [0002-wavefield-preconditioner](0002-wavefield-preconditioner.md) | Does the wavefield preconditioner pay off in the joint solve? | optimiser.wavefield_preconditioner | current | Yes at equal outer budget (final rel_wavefield_error 0.23×, rmse_skull −25%, rmse_brain −30%); at equal wall time it misses the 0.5× threshold narrowly (0.54×), so the cost criterion is not met |
<!-- index:end -->
