# Finishing a branch

Bring every record on an `exp/*` branch up to date with `main` and open the PR that lands them. Records can sit unmerged while `main` moves, so the check from the Record stage runs again here, for all of them.

## 1. Re-check every record

```bash
git fetch origin
git diff --name-only origin/main...HEAD -- 'docs/experiments/[0-9]*.md'
```

For **every** record listed, re-run the invalidation check against current `main`: `git diff <its commit>...origin/main -- odil_wave/ configs/ main.py`, read against its Conclusions supported, adding `note` Validity entries where a change could break one. Then run `uv run python -m odil_wave.experiment.bank` and `--check` until it exits 0, and commit.

Done when each listed record has been read against its diff.

## 2. Feature kept or dropped

Ask the user. A branch with no feature under test takes the kept path.

- **Feature kept**: one PR from the `exp/*` branch, carrying code and records, reviewed as a normal code PR. The bank duty in `docs/experiments/README.md`, "Maintaining the bank", applies to its code against the `current` records on `main`; fill the PR template's Experiment bank line. Any merge strategy.
- **Feature dropped**: a docs-only PR from a fresh branch off `origin/main`, carrying only the branch's changes under `configs/experiments/` and `docs/experiments/`:

  ```bash
  git switch -c exp/NNNN-slug-records origin/main
  git diff origin/main...exp/NNNN-slug -- configs/experiments/ docs/experiments/ | git apply -3
  uv run python -m odil_wave.experiment.bank
  ```

  Applying the branch's diff (rather than checking its files out) keeps anything `main` changed in the bank meanwhile; regenerating the index settles the README. Each record still cites its tagged launch commit, so it stays reproducible. The spec on `main` names a config key `main` lacks; that is fine, since a spec only runs at its own commit.

## 3. Open the PR

The PR body has one `Closes #<issue>` line per Protocol issue on the branch. Mark it ready only after step 1 ran against the `main` it will merge into.

Then comment the record path (`docs/experiments/NNNN-slug.md`) and the PR on each Protocol issue.

Done when the PR is open with every `Closes` line and every issue has its comment.
