# Experiment bank

One **experiment record** per launched experiment (see `GLOSSARY.md`), including
aborted and inconclusive ones, so negative results are banked and numbering has
no gaps. Read the index below first; take context primarily from `current`
records.

- **Record**: `NNNN-slug.md`, copied from [`TEMPLATE.md`](TEMPLATE.md). The
  **experiment id** `NNNN-slug` is 4-digit and sequential, reserved at launch,
  and shared by the experiment spec (`configs/experiments/NNNN-slug.yaml`), the
  experiment directory and `--reproduce NNNN-slug`. An aborted record can be
  brief: the question, what happened and why it stopped.
- **Committed results**: `NNNN-slug/` holds `experiment.json`, `summary.csv` and
  up to 3 key PNGs under their standard filenames. Everything else is
  regenerated with `--reproduce` / `--postprocess`.
- **Lifecycle**: the body is frozen once the record's PR merges (typo fixes
  allowed); a wrong analysis is fixed by a new record that supersedes it. After
  merge only `status` and new Validity entries change, and only while the record
  is `current`, so the entry that took it out of `current` is always the last.
  `superseded`: a newer record answers the same question. `invalidated`: a later
  code change broke one of its supported conclusions. Superseded and invalidated
  records are never checked or edited again; they remain as context.
- **Index**: generated from the frontmatter by
  `uv run python -m odil_wave.experiment.bank`. `--check` validates only (required
  fields, sequential ids, maintained `supersedes` targets) and runs in CI through
  `tests/test_experiment_bank.py`.

## Maintaining the bank

The invalidation check is a helper for the user, not a gate: the agent flags,
the user decides.

1. **Invalidation notes.** In every PR that touches `odil_wave/`, `configs/` or
   `main.py`, read the diff against each `current` record's `variable`,
   `baseline`, `answer` and Conclusions supported.
   - If its conclusions hold, write nothing.
   - If you can name a conclusion and a reason the change could break it, add a
     Validity entry in that PR:
     `YYYY-MM-DD · note · <PR> · affects <conclusion bullets> · <why>`.
     `status` is unchanged.
   - Fill the PR template's "Experiment bank" line. PRs touching only docs,
     tests, notebooks or dependencies are skipped.
   - **Only the user writes `invalidated`** (usually in review of a PR that
     added a note): `YYYY-MM-DD · invalidated by <PR or commit> · <what changed> ·
     affects <conclusion bullets | all> · <why this breaks them>`, flipping
     `status` to `invalidated`. Doubt that needs data is settled by a new
     experiment that, if it answers the same question, supersedes the record.
   - **Records written after a code change**: when writing a record, check
     `git diff <commit>...main -- odil_wave/ configs/ main.py` (from the
     merge-base, so a feature left on an unmerged branch isn't read as removed)
     against it, and put any notes in the record's PR. Before an experiment
     branch's PR is marked ready, re-run this for every record on the branch
     against current `main`.
2. **Supersession.** When a record becomes `current`, check whether it answers
   the same question as any `current` record. If it does, list in its
   `supersedes` **every** record it replaces, transitively (in a chain
   0001→0002→0003, 0003 has `supersedes: [0001-…, 0002-…]`), flip each `current`
   one to `superseded` and give it the entry
   `YYYY-MM-DD · superseded by [NNNN-slug](NNNN-slug.md) · <why>`. The link lives
   only on the newer record; there is no `superseded_by` field.
3. **Index.** After writing or updating a record, run
   `uv run python -m odil_wave.experiment.bank` to regenerate the table below.

## Index

<!-- index:start -->
| id | question | variable | status | answer |
|---|---|---|---|---|
<!-- index:end -->
