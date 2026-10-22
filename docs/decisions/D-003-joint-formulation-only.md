# D-003: Focus solely on the joint ODIL formulation

Status: Proposed
Features: F-008, F-015, F-017
Source: 2-feature-inventory (retrospective; change made in `7c130ad`, PR #2)
Supersedes: —
Superseded by: —

## Context
Before `7c130ad` the package offered two inversion formulations: the joint
full-space optimiser `JointFreqODIL`, and a block-coordinate dual L-BFGS
optimiser (`LBFGSB` with `StepScheduler`, `UBlockHessian`,
`HelmholtzUTransform` and a frequency-continuation helper), with
block-coordinate diagnostics in `odil_wave/experiment/diagnostics.py`.
Commit `7c130ad` ("clean: remove 2 optimiser odil", PR #2) removed the
block-coordinate path. Neither the commit nor the PR records a reason.

## Decision
Retain only the joint formulation (`JointFreqODIL`) as the project's
inversion method. The block-coordinate optimiser and its diagnostics stay
removed; the config rejects their settings (`lbfgsb`, `u_steps`, `clamp`,
`diagnostics`).

## Alternatives
Keeping both formulations, as before `7c130ad`. No other alternative is
recorded.

## Rationale
Observed: the removal in `7c130ad`; the config-rejection test
`tests/test_experiments.py::test_removed_settings_are_rejected`.

Maintainer statement (local session, 2026-10-22, given after the change):
"i want to focus solely on the joint formulation in this project".

No comparison of the two formulations (accuracy, cost, robustness) is
recorded in the repository; the statement is a scope choice, not an
evidence-based ranking.

## Consequences
- One optimiser to maintain, configure and test.
- Lost: block-coordinate inversion, the exact u-solve / u-block Hessian path,
  and the diagnostics that went with it.
- The optimiser unit tests (`tests/test_optimisation.py`) were deleted with
  the removal; `JointFreqODIL` is now covered only by the tiny integration run
  in `tests/test_integration.py`.

## Scope and limitations
Applies to this project's current research scope. Whether the joint
formulation performs better than the block-coordinate one is untested here.

## Revisit when
The project scope changes, or the joint formulation proves unable to reach
the required reconstruction quality.
