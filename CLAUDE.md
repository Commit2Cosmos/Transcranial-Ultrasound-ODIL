# Project instructions

## Working principles

- Inspect relevant code, tests, and repository conventions before proposing changes.
- Infer the minimum sufficient scope; do not expand the task unnecessarily.
- Preserve unrelated work.
- Ask for clarification when a missing decision materially changes the task.
- Distinguish observed facts, assumptions, and unverified claims.
- Do not weaken tests or acceptance criteria merely to obtain a pass.

## Plan-first workflow

Tasks are requested, revised, and approved in the local agent session.
GitHub issues and @claude mentions are not used.

Task IDs:
- Task ID: `<n>-<short-description>`, e.g. `2-tv-regulariser`.
- Task branch: `<n>/<short-description>`, e.g. `2/tv-regulariser`.
- `<n>` is one greater than the highest task number found in docs/plans/,
  docs/reports/, and local and remote branch names.
- A retried or resumed task reuses its existing ID, branch, plan, and PR.

| Request type   | Claude's action                                          |
|----------------|----------------------------------------------------------|
| Question       | Answers only, no commits                                 |
| New task       | Writes a plan                                            |
| Plan revision  | Updates the uncommitted plan in place                    |
| Implement      | Implements the approved plan                             |
| Review fix     | Fixes on the same branch and PR                          |

For a new task:
1. Inspect the task and relevant repository context. Ask clarifying questions if the task is ambigous, unclear or there are multiple ways of performing it.
2. Create the task branch from the up-to-date default branch. Write
   docs/plans/<task-id>.md using the plan template and, where a consequential
   decision is involved, a proposed decision record.
3. Do not commit or push the plan or decision record.
4. Do not modify implementation, tests, dependencies, or configuration.
5. Stop and wait for explicit implementation approval.

During plan review:
- Incorporate requested corrections into the same uncommitted plan.
- Do not interpret questions, comments, or positive feedback as implementation approval.

Approval is an explicit instruction from the user in the local session to
implement the plan (e.g. "the plan is sound, implement the features"). It
applies to the plan as it stands at that moment. On approval, set the plan
status to `Approved <date>`; do not otherwise edit the plan without renewed approval.

After explicit approval:
- Implement on the existing task branch.
- Follow the approved plan.
- Stop for approval if a material change to scope, approach, or acceptance
  criteria is required. Ask clarifying questions, record the agreed change in
  the plan, and wait for renewed approval.
- Write docs/reports/<task-id>.md using the report template.
- Update the feature register and decision records where the change warrants
  it; otherwise state in the report why no update is needed.
- Commit the plan together with the implementation, report, and knowledge updates.
- Push the task branch and open a PR using the PR template. Never merge.

After the PR is opened, make further changes only when the user asks.
In-scope fixes on the same branch need no new approval, but the report, feature register and decision records should be updated and always kept up-to-date.
Commit and push them to the same branch and PR.

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