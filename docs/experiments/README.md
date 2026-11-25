# Experiment bank

One **experiment record** per launched experiment (see `GLOSSARY.md`), including
aborted and inconclusive ones, so negative results are banked; only a voided
launch (below) gets none. The bank holds only records the user still
considers current; a record is a snapshot pinned to its `commit`. Read the
index below first.

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
  old one. Only the user deletes a record, when they judge it stale; git
  history keeps it. Agents never delete or annotate records.
- **Index**: generated from the frontmatter by
  `uv run python -m odil_wave.experiment.bank`. `--check` validates only
  (required fields, unique id numbers) and runs in CI through
  `tests/test_experiment_bank.py`.

## Voiding a launch

A launch is **voided** when its runs no longer test its Protocol before any
record of it reaches `main`, e.g. because the default config changed under its
Baseline. Runs that test the Protocol but stopped are recorded as `aborted`
instead.

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

After writing or updating a record, run
`uv run python -m odil_wave.experiment.bank` to regenerate the index below.

## Index

<!-- index:start -->
| id | question | variables | answer |
|---|---|---|---|
<!-- index:end -->
