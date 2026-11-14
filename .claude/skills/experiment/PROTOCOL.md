# Stage 1: Protocol

Grill the user, one question at a time, into a pre-registered Protocol on an `experiment` issue. The body stays editable until launch.

## 1. Gather context

Read the index in `docs/experiments/README.md`, then the `current` records that bear on the idea. Weigh each one's `note` Validity entries: a noted conclusion may not hold at HEAD. Read `invalidated` records only when a current one points you at them. If the idea follows up a record, that record's Follow-ups and baseline are the starting point.

Done when you can name every record the Context section will list, with the notes you weighed.

## 2. Grill into the body

The body has these sections, in this order, mirroring the record's first sections so the record can copy them:

```markdown
## Question
One sentence.

## Hypothesis
The prediction, and any expected effect sizes.

## Design
The swept keys (dotted config keys), their values and why, the Baseline, what stays fixed (phantom, grid, bands, budgets), and per-band `n_iter` when bands or `n_iter` are swept. Several keys are a factorial sweep: every combination is a run.

## Assumptions and limitations
What the setup takes for granted (inverse crime, a single phantom, no seeds) and so cannot speak to.

## Context
The records consulted, with any notes weighed; the record this follows up on.

## Chain
Only for an Experiment chain: the planned next questions.
```

Metrics are tape columns (`odil_wave/experiment/README.md`, "Iteration-level log record"), read at the final iterate from `summary.csv`. The record judges the outcome over all of them. An expected effect size (e.g. "`rmse_brain` drops by ≥ 5 m/s") sharpens that judgement but is optional.

## 3. Pass the gate

- **Config-selectable.** Everything compared is selectable by config at one commit. A comparison that needs new code becomes a **feature under test**: code on the experiment branch behind a config key whose default keeps current behaviour (ADR 0001). Name that key in the Design. Comparing two commits is not an experiment.

When per-band `n_iter` or the bands are swept, the Design settles every run's per-band `n_iter` (there is no budget field).

If a later question's Baseline depends on this one's findings, make it an **Experiment chain**: this issue is the first link and its Chain section lists the planned next questions. Later links get issues only when the previous record lands.

## 4. Write the issue

Show the user the full body and title (the Question, short). On approval, create the issue labelled `experiment`, or edit the existing one's body and add the label. Assign it to `@me`.

Done when the issue carries the label, the assignment, and a body with every section filled and the gate met. Then ask whether to launch now; if yes, continue with [LAUNCH.md](LAUNCH.md).
