---
id: EXP-000                # unique, zero-padded, never reused
name: short-kebab-case-name
status: planned            # planned | running | completed | failed | abandoned
created: YYYY-MM-DD
git_commit: <hash>         # commit of code used to produce results
parent: EXP-000            # experiment this one builds on, or "none"
---

# EXP-000: <Human-readable title>

## Summary
One to three sentences: what was tested and the headline outcome.
(Fill in the outcome after the run; leave "TBD" while planned.)

## Motivation and hypothesis
- **Question:** What are we trying to find out?
- **Hypothesis:** Falsifiable statement, e.g. "Increasing the TV weight from 1e-4 to 1e-3 reduces artefacts without raising the data misfit above X."
- **Success criteria:** Quantitative thresholds decided *before* running (e.g. relative L2 error < 5%, SSIM > 0.9).

## Description
What is being done, in enough detail to repeat it. Include any mathematical explanations, where appropriate.

## Parameters

### Fixed parameters
| Parameter | Value |
|---|---|
| grid (nx, ny) | |
| frequency / time step | |
| optimiser | |

### Parameters to vary
| Parameter | Baseline | Values tested | Why this range |
|---|---|---|---|
| | | | |

## Results table

| Run ID | <varied param 1> | <varied param 2> | Final loss | Data misfit | Rel. L2 error | SSIM / PSNR | Iterations | Wall time (s) |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |


## Interpretation and limitations
**What the results establish:**
- 

**What they do not establish:**
- 

## Knowledge updates
Feature IDs and decision records added or updated.

## Follow-up
Proposed further experiments.
Change from this experiment | Expected outcome | Priority (H/M/L) |
|---|---|---|
| | | |

## Open questions and issues
- 