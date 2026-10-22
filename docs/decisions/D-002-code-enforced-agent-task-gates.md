# D-002: Enforce agent task gates in trusted workflow code

Status: Proposed
Features: Not applicable
Source: pending (local change; plan docs/plans/agent-workflow-stabilization.md)
Decided by: pending
Supersedes: —
Superseded by: —

## Context
D-001 adopted plan-first development and relied on human checks and agent
instructions for enforcement. An audit of the initial workflows (recorded in
docs/reports/agent-workflow-stabilization.md) found:
- duplicate `@claude` handlers with contradictory approval rules;
- planning instructions that conflicted with the action's tag mode;
- no duplicate-plan prevention;
- approval and plan-only rules expressed only in prompts, while the agent had
  unrestricted Bash and GitHub write access.

D-001 names "unclear approvals" as a reason to revisit it.

## Decision
Keep D-001's process. Add a small controller (`.github/scripts/claude-task.js`),
always loaded from the default branch, that:
- checks requester access;
- rejects fork PRs;
- discovers and reuses task branches and PRs;
- validates SHA-bound `approve-plan`/`routine` authorizations, activation and
  revocation with local git;
- selects a least-privileged Claude tool profile for each command.

A `plan-guard` check, required in branch protection, blocks merging of
unauthorized changes. Remaining rules stay prompt-governed and are listed in
docs/agent-workflow.md.

## Alternatives
- Prompt-only enforcement, as before: rejected because the audit found
  contradictory and truncated prompts, and those rules cannot be tested.
- The action's tag mode with `track_progress`: rejected because the action
  creates its own branch per run and instructs "Create PR" links. This
  conflicts with one-task-one-PR and with opening draft PRs.
- `pull_request_target` for privileged runs: rejected (fork code with
  secrets).
- A larger external controller or GitHub App: not needed for the stated
  requirements.

## Rationale
Observed:
- the pinned action's mode, branch and credential behaviour (source at
  6b6870e);
- the Claude Code permission flags blocking prohibited calls in local tests;
- `sh -c` bypassing Bash prefix rules.

Details in docs/agent-workflow.md and the report.

Judgement: the trust boundaries that matter most are authorization, forks,
approval validity and merge gating. They are cheap to enforce and test in
code. Scope adherence is not.

## Consequences
- More workflow machinery to maintain.
- The pinned action must be re-audited before upgrading.
- Maintainers use exact commands with full SHAs.
- Activation markers depend on no other workflow granting `issues: write` to
  jobs that run repository code.

## Scope and limitations
Live GitHub behaviour has not been verified; see the smoke tests in
docs/agent-workflow.md. Running tests in the implement profile still executes
repository code with a GitHub token present.

## Revisit when
- Smoke tests fail.
- The action or Claude Code permission semantics change.
- Maintainers find the command protocol too burdensome.
