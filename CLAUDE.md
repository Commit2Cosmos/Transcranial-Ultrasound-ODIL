# Project instructions

## Working principles

- Inspect relevant code, tests, and repository conventions before proposing changes.
- Infer the minimum sufficient scope; do not expand the task unnecessarily.
- Preserve unrelated work.
- Ask for clarification when a missing decision materially changes the task.
- Distinguish observed facts, assumptions, and unverified claims.
- Do not weaken tests or acceptance criteria merely to obtain a pass.

## Plan-first workflow

Applies to feature changes, consequential refactors, and other changes to
code, tests, dependencies, configuration, or workflows (D-001).

Does not apply to:
- Read-only requests (questions, reviews, audits): answer without creating
  plans, branches, or PRs.
- Routine changes a maintainer explicitly authorizes with the lighter process
  (`@claude routine`, below).

For a new task:
1. Inspect the task and relevant repository context.
2. Create docs/plans/<task-id>.md using the plan template.
3. Open a draft PR containing only the plan.
4. Do not modify implementation, tests, dependencies, or configuration.
5. Stop and wait for explicit implementation approval.

During plan review:
- Incorporate requested corrections into the same plan and branch.
- Do not interpret comments, positive feedback, or PR review approvals as
  implementation approval.

After approval:
- Implement on the existing task branch and PR, following the approved plan.
- Do not modify the approved plan; stop and ask for renewed approval if a
  material change to scope, approach, or acceptance criteria is required.
- Write docs/reports/<task-id>.md using the report template.
- Update the feature register and decision records where the change warrants
  it; otherwise state in the report why no update is needed.
- Never merge the PR, mark it ready for review, or approve it.

## Agent task protocol

This section is canonical. The GitHub workflows enforce the parts marked
**[code]**; parts marked **[prompt]** rely on the agent following them.
Details and residual risks: docs/agent-workflow.md.

### Task identity
- task-id = the originating issue number N.
- Branch `agent/issue-N`, plan `docs/plans/N.md`, report `docs/reports/N.md`.
- One task, one branch, one PR. **[code]** A request on an issue whose task
  already has an open, merged, or closed PR is redirected, not duplicated. An
  existing task branch without a PR is reused only if it changes nothing but
  the plan.

### Commands
A command is the first non-empty line of a comment, from a user with write,
maintain, or admin access. **[code]** checks access for the requester and for
anyone re-running the workflow. Other `@claude` mentions are read-only requests.

| Command | Where | Effect |
|---|---|---|
| (new issue) | issue opened | Plan: create the plan and draft PR |
| `@claude plan [notes]` | issue or task PR | Plan: create or revise only the plan |
| `@claude approve-plan <full-SHA>` | top-level PR comment | Approve the plan at that commit and start implementation |
| `@claude routine <full-SHA>` | top-level PR comment | Lighter process: implement without plan approval |
| `@claude fix [notes]` / `@claude continue [notes]` | task PR, reviews, inline comments | In-scope fixes or resumption under a valid authorization |
| `@claude revoke [reason]` | top-level PR comment | Cancel all earlier authorizations on the PR |
| any other `@claude …` | anywhere | Read-only answer or review |

**[code]** Each request runs with the least-privileged tool profile for its
command, regardless of authorization state:
- read-only: no file edits, commits or pushes;
- plan: edit only `docs/plans/N.md`, commit, push the task branch, open the
  draft PR;
- implement: edit files except the plan, run commands, commit, push.

### Approval and routine authorization
- An authorization is an unedited top-level PR comment whose first line is
  `@claude approve-plan <SHA>` or `@claude routine <SHA>`, with the full
  40-character SHA, from a user who currently has write access. Review
  approvals, review bodies, inline comments, edited comments, and short SHAs
  never authorize. **[code]**
- Activation: the run triggered by the comment accepts it only if `<SHA>` is
  the current PR head, the commit resolves locally, there is no later
  revocation, and, for approve-plan, `docs/plans/N.md` exists at `<SHA>`.
  The workflow then records an activation marker (posted by
  github-actions[bot]). **[code]**
- Later fixes and resumes (`fix`, `continue`, or a re-run) require the most
  recent activated authorization to satisfy all of the following **[code]**:
  - it is after the latest revocation;
  - its SHA is an ancestor of the current head (local git);
  - `docs/plans/N.md` is unchanged between that SHA and the head.

  There is no fallback to an older authorization. If the approved commit cannot
  be resolved, the check fails closed.
- Renewed approval is required after any plan change, a history rewrite, or a
  revocation. Post a new command with the current head SHA. **[code]**
- If work needs a material change of scope, approach, or acceptance criteria,
  stop and explain; the plan is revised and re-approved. **[prompt]**
- `@claude revoke` takes effect for every later check, but does not
  interrupt a job that is already running. Cancel that run in the Actions tab
  if needed.
- Routine authorization is for small, low-risk changes the maintainer names in
  the comment. It is bound to its SHA and subject to the same resume and
  invalidation rules; it is not task-wide permission. The report may be brief.

### Pull requests and review
- Planning PRs are drafts with Stage "Plan review", say `Related to #N`, and
  use no closing keywords. **[prompt]** (the workflow fails the run if a pushed
  task branch has no PR **[code]**)
- Before authorization, a task PR may change only `docs/plans/N.md`; after
  it, the plan must stay unchanged. **[code]** via the required `plan-guard`
  check, which blocks merging, not pushing.
- Automated advisory review runs once when a maintainer marks the PR ready for
  review. Agents never mark PRs ready.
- Do not edit `.github/`, `CLAUDE.md` policy, or permission settings unless
  the approved plan is specifically about them. **[prompt]**

### Rejected proposals
When a task PR is closed without merging, a maintainer may record the
rejection through a new issue and `@claude routine`. The resulting PR is
reviewed like any other. It adds a decision record with Status: Rejected, a
Rejected feature-register entry, or both. Rationale comes only from the PR and
issue discussion; if none was stated, write "rationale not recorded".

## Project knowledge

- docs/features.md records capabilities and lifecycle status.
- docs/decisions/ records consequential decisions and rationale.
- docs/reports/ records implementation and verification evidence.
- docs/plans/ records proposals; a plan is not proof of implementation.
- Knowledge becomes authoritative when merged to the default branch; content on
  unmerged branches is provisional.

Plan lifecycle:
- Draft: open plan PR with no valid authorization.
- Approved: a valid activated authorization, recorded in PR comments, not in
  the plan file.
- Implemented: merged together with its report.
- Rejected or withdrawn: PR closed unmerged.

Do not edit a plan's content after approval to record status.

Before proposing a capability or reversing an approach:
- Check the feature register and relevant decisions.
- Read any superseding decisions.
- Inspect current code and tests.

For historical or onboarding answers:
- Cite repository paths, feature IDs, decision IDs, and source PRs.
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
