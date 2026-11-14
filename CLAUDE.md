## Agent skills

### Issue tracker

Issues live in GitHub Issues on `Commit2Cosmos/Transcranial-Ultrasound-ODIL` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Merge-ready PR

**Merge-ready PR** (also "mergeable") is defined in `CODING_STANDARDS.md`. Implement to it, and use it to answer whether a PR can merge.

### Experiments

A PR touching `odil_wave/`, `configs/` or `main.py` adds a one-line note to each `current` record in `docs/experiments/` it may affect: see "Maintaining the bank" in its README.
