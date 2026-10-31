---
id: NNNN-slug                # the experiment id, reserved at launch; the filename without .md
title: ""                    # the question, short; the index's question column
date: YYYY-MM-DD             # the day the record was written
status: current              # current | superseded | invalidated
outcome: answered            # answered | inconclusive | aborted
issue: 0                     # the protocol issue number
commit: ""                   # the SHA the runs used (the launch/NNNN-slug tag)
spec: configs/experiments/NNNN-slug.yaml
experiment_dir: outputs/experiments/NNNN-slug   # local, gitignored; regenerate with --reproduce
device: cpu                  # cpu | cuda | mps
variable: ""                 # the one swept key, e.g. loss.weights.reg
baseline: ""                 # the baseline run config, or where it came from
answer: ""                   # the answer, in one line
supersedes: []               # every record this one replaces, transitively, as full ids: [0001-slug, 0002-slug]
---

# NNNN-slug: Title

## Question

One sentence, from the protocol.

## Hypothesis

The predicted outcome, the primary metric that decides it, and the effect size that counts, all from the protocol.

## Design

The variable, its values and why they were chosen, the baseline, and what stays fixed (phantom, grid, bands, budgets).

## Deviations from protocol

Anything done differently from the protocol issue, with the reason; "None." otherwise.

## Assumptions and limitations

What the setup takes for granted (inverse crime, a single phantom, no seeds), and so what it can't speak to.

## Results

The table from `summary.csv` (one row per run: swept value, final-iterate columns, start-row reference), then up to 3 key PNGs from `docs/experiments/NNNN-slug/`.

## Conclusions supported

Bullets, each citing a row, column or figure above; a difference smaller than the protocol's effect size for that metric is not a conclusion.

## Conclusions not supported

Claims a reader might be tempted to draw that this data doesn't back.

## Surprises

Departures from the hypothesis, with the likely cause and a possible fix.

## Follow-ups

Candidate next experiments, each with a proposed baseline (ids are given only at launch).

## Validity

Current.

<!--
Replace "Current." with the first dated entry; later entries go below it. Formats:
- YYYY-MM-DD · superseded by [NNNN-slug](NNNN-slug.md) · <why the newer record replaces this one>
- YYYY-MM-DD · note · <PR> · affects <conclusion bullets> · <why the change could break them>
- YYYY-MM-DD · invalidated by <PR or commit> · <what changed> · affects <conclusion bullets | all> · <why this breaks them>
Only the user writes `invalidated`. A superseded or invalidated entry flips `status` and is the last entry.
-->
