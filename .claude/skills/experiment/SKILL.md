---
name: experiment
description: Walk an experiment from Protocol issue through launch to experiment record, or reproduce one. Use when the user wants to plan, launch, record or reproduce an experiment, or names an experiment issue or id.
argument-hint: "[<issue#> | <experiment-id> | <loose idea> | reproduce <id>]"
---

# /experiment

Drives one **experiment** (`GLOSSARY.md`) through its lifecycle: **Protocol** issue → spec, probe, launch → experiment record → finished branch. The stage comes from the artifacts that exist; there are no stage labels. Conventions for records and the bank live in `docs/experiments/README.md`; the CLI contract lives in `odil_wave/experiment/README.md`, "Experiments". Both are the source of truth: read them, don't restate them.

## Ground rules

- `gh` operations follow `docs/agents/issue-tracker.md`.
- The `experiment` label means **this issue's body is a Protocol**. Experiment issues carry no triage labels.
- **Claim** the issue by assigning it to `@me` as the session's first write on it.
- Speak in glossary terms: Protocol, Baseline, Experiment spec, Experiment id, Probe run, Experiment chain, Experiment record.
- Only the user writes `invalidated` on a record. You flag with `note` entries.

## Work out the stage

Resolve the argument, then take the first stage whose condition holds:

- `reproduce <id>` → [REPRODUCE.md](REPRODUCE.md).
- A loose idea, or nothing → [PROTOCOL.md](PROTOCOL.md) for a new issue. With no argument, first list open `experiment` issues and ask which one, or whether to start a new one.
- An issue number → read it with its comments.
- An experiment id → find its spec (`git show launch/<id>:configs/experiments/<id>.yaml`, after `git fetch origin --tags`); its `issue` field names the issue.

For an issue:

| What exists | Stage |
|---|---|
| No `experiment` label, or a body missing a Protocol section, or a gate unmet | [PROTOCOL.md](PROTOCOL.md) |
| A Protocol, no comment starting `Launched` | [LAUNCH.md](LAUNCH.md) |
| A launch comment, no record `docs/experiments/<id>.md` on the experiment's branch | [RECORD.md](RECORD.md) |
| A record on an `exp/*` branch whose PR isn't ready | [FINISH.md](FINISH.md) |
| A record on `main` | Done: say so and give the record path |

The launch comment's first line names the id; the experiment's branch is the `exp/*` branch containing `launch/<id>` (`git branch -a --contains launch/<id>`).

Read only the stage file you land on. A stage file names the next one when the session may continue into it.
