# D-004: Remove the leapfrog time-domain solver

Status: Proposed
Features: F-016
Source: 2-feature-inventory (retrospective; change made in `7c130ad`, PR #2)
Supersedes: —
Superseded by: —

## Context
Before `7c130ad` the package contained a leapfrog time-domain solver
(`odil_wave/operator/leapfrog.py`, with temporal operators in
`odil_wave/operator/temporal.py`) and a `leapfrog_fft` observation method
that generated observed data in the time domain and transformed them to the
frequency domain. Commit `7c130ad` (PR #2) removed both. Neither the commit
nor the PR records a reason.

## Decision
The leapfrog solver, its temporal operators and `leapfrog_fft` observations
stay removed; the config rejects `observation.method: leapfrog_fft` and
`physics.time_order`.

## Alternatives
Keeping the leapfrog solver. A "more sophisticated" replacement is intended
(see Rationale) but not specified or planned.

## Rationale
Observed: the removal in `7c130ad`; the config-rejection test
`tests/test_experiments.py::test_removed_settings_are_rejected`.

Maintainer statement (local session, 2026-10-22, given after the change):
"not good enough, need something more sophisticated".

Unrecorded: in what respect the solver was not good enough (accuracy,
dispersion, stability, cost or otherwise), and on what evidence.

## Consequences
- Lost: the only time-domain forward solver, and with it the only
  observation path that did not use the inversion's own operator.
  Observations are now Helmholtz-only (see D-005).

## Scope and limitations
No report or test result documenting the leapfrog solver's shortcomings
exists in the repository.

## Revisit when
A replacement time-domain (or otherwise independent) forward solver is
planned, or non-inverse-crime observations are needed (see D-005).
