# Finishing a branch

Open the PR that lands an `exp/*` branch's records.

## 1. Feature kept or dropped

Ask the user. A branch with no feature under test takes the kept path.

- **Feature kept**: one PR from the `exp/*` branch, carrying code and records, reviewed as a normal code PR. The bank duty in `docs/experiments/README.md`, "Maintaining the bank", applies to its code against the `current` records on `main`. Any merge strategy.
- **Feature dropped**: a docs-only PR from a fresh branch off `origin/main`, carrying only the branch's changes under `configs/experiments/` and `docs/experiments/`:

  ```bash
  git switch -c exp/NNNN-slug-records origin/main
  git diff origin/main...exp/NNNN-slug -- configs/experiments/ docs/experiments/ | git apply -3
  uv run python -m odil_wave.experiment.bank
  ```

  Applying the branch's diff (rather than checking its files out) keeps anything `main` changed in the bank meanwhile; regenerating the index settles the README. Each record still cites its tagged launch commit, so it stays reproducible. The spec on `main` names a config key `main` lacks; that is fine, since a spec only runs at its own commit.

## 2. Open the PR

The PR body has one `Closes #<issue>` line per Protocol issue on the branch.

Then comment the record path (`docs/experiments/NNNN-slug.md`) and the PR on each Protocol issue.

Done when the PR is open with every `Closes` line and every issue has its comment.
