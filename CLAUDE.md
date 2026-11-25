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

- Before/after metrics of a code change: `python -m odil_wave.experiment.compare <ref> --keep DIR`.
- Run `main.py` and `compare` one at a time, never in parallel (parallel runs: #185).
