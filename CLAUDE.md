# Project instructions

## Working principles

- Inspect relevant code, tests, and repository conventions before proposing changes.
- Infer the minimum sufficient scope; do not expand the task unnecessarily.
- Preserve unrelated work.
- Ask for clarification when a missing decision materially changes the task.
- Distinguish observed facts, assumptions, and unverified claims.
- Do not weaken tests or acceptance criteria merely to obtain a pass.

## Plan-first workflow

For a new task:
1. Inspect the task and relevant repository context.
2. Create docs/plans/<task-id>.md using the plan template.
3. Prepare a draft PR containing only the plan.
4. Do not modify implementation, tests, dependencies, or configuration.
5. Stop and wait for explicit implementation approval.

During plan review:
- Incorporate requested corrections into the same plan and branch.
- Do not interpret comments or positive feedback as implementation approval.

After explicit approval:
- Implement on the existing task branch and PR.
- Follow the approved plan.
- Stop for approval if a material change to scope, approach, or acceptance
  criteria is required.
- Write docs/reports/<task-id>.md using the report template.
- Update the feature register and decision records.
- Never merge the PR.

## Project knowledge

- docs/features.md records capabilities and lifecycle status.
- docs/decisions/ records consequential decisions and rationale.
- docs/reports/ records implementation and verification evidence.
- docs/plans/ records proposals; a plan is not proof of implementation.

Before proposing a capability or reversing an approach:
- Check the feature register and relevant decisions.
- Read any superseding decisions.
- Inspect current code and tests.

For historical or onboarding answers:
- Cite repository paths, decision IDs, and source PRs.
- Distinguish historical rationale from current implementation.
- Treat unmerged changes as provisional.
- Say when rationale or evidence is missing; do not invent it.

Local agent memory is a convenience, not the authoritative project record.

## Verification

- Run the checks agreed in the plan where feasible.
- For numerical work, distinguish software correctness from numerical validity.
- Do not advance to expensive experiments when prerequisite checks fail.
- Report commands, results, relevant configuration, and unverified items.
- A negative finding is a valid outcome.