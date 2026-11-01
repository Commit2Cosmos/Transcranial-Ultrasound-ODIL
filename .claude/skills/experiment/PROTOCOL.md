# Stage 1: Protocol

Grill the user, one question at a time, into a pre-registered Protocol on an `experiment` issue. The body stays editable until launch.

## 1. Gather context

Read the index in `docs/experiments/README.md`, then the `current` records that bear on the idea. Weigh each one's `note` Validity entries: a noted conclusion may not hold at HEAD. Read `superseded` or `invalidated` records only when a current one points you at them. If the idea follows up a record, that record's Follow-ups and baseline are the starting point.

Done when you can name every record the Context section will list, with the notes you weighed.

## 2. Grill into the body

The body has these sections, in this order, mirroring the record's first sections so the record can copy them:

```markdown
## Question
One sentence.

## Hypothesis
The prediction; the primary metric that decides it and its required effect size; optional effect sizes for secondary metrics.

## Design
The variable (one dotted config key), its values and why, the Baseline, what stays fixed (phantom, grid, bands, budgets), and per-band `n_iter` when bands or `n_iter` are swept.

## Assumptions and limitations
What the setup takes for granted (inverse crime, a single phantom, no seeds) and so cannot speak to.

## Context
The records consulted, with any notes weighed; the record this follows up on.

## Chain
Only for an Experiment chain: the planned next questions.
```

Metrics are tape columns (`odil_wave/experiment/README.md`, "Iteration-level log record"), read at the final iterate from `summary.csv`. An effect size is a threshold on one column (e.g. "`rmse_brain` drops by ≥ 5 m/s"); without one on the primary metric the Hypothesis is unfinished.

## 3. Pass the gates

Both must hold before the Protocol is done:

- **One variable.** Exactly one thing varies. A feature touching several keys is one variable swept at their common parent key. When per-band `n_iter` or the bands are what is swept, the Design settles every run's per-band `n_iter` (there is no budget field).
- **Config-selectable.** Everything compared is selectable by config at one commit. A comparison that needs new code becomes a **feature under test**: code on the experiment branch behind a config key whose default keeps current behaviour (ADR 0001). Name that key in the Design. Comparing two commits is not an experiment.

If more than one thing varies, split the idea into an **Experiment chain**: this issue becomes the first link, varying one thing, and its Chain section lists the planned next questions. Later links get issues only when the previous record lands, because their Baseline depends on its findings.

## 4. Write the issue

Show the user the full body and title (the Question, short). On approval, create the issue labelled `experiment`, or edit the existing one's body and add the label. Assign it to `@me`.

Done when the issue carries the label, the assignment, and a body with every section filled and both gates met. Then ask whether to launch now; if yes, continue with [LAUNCH.md](LAUNCH.md).
