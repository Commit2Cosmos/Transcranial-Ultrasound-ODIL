# Agent workflow: operations and security

The policy is defined in CLAUDE.md ("Agent task protocol"). This document
describes how the GitHub workflows implement it, what is and is not enforced
by code, the remaining risks, required repository settings, and the manual
smoke tests.

## Components

| File | Role |
|---|---|
| `.github/workflows/claude.yml` | Single handler for new issues and `@claude` in issue comments, PR comments, review bodies and inline review comments |
| `.github/workflows/plan-guard.yml` | Required check on PRs: plan-only before authorization, plan unchanged after it |
| `.github/workflows/claude-code-review.yml` | One advisory review when a same-repo PR becomes ready for review (or is opened non-draft) |
| `.github/scripts/claude-task.js` | Controller logic: authorization, discovery, approval, profiles, prompt, plan guard |
| `.github/scripts/claude-task.test.js` | Unit tests (`node --test .github/scripts/claude-task.test.js`, run by `tests.yml`) |

### `claude.yml` jobs

1. **gate** (read-only token)
   - Checks out the default branch into `trusted/` and loads the controller
     from there.
   - Classifies the request and checks that the requester, and on re-runs
     `GITHUB_TRIGGERING_ACTOR`, has write access.
   - Rejects fork PRs using the API, before any PR content is fetched.
   - Discovers the task: existing PR or branch.
   - Fetches the task history into `task/`, as data only. Checks the expected
     head and validates approvals with local git.
2. **activate** (`issues`/`pull-requests: write`)
   - Posts the activation marker for a validated `approve-plan`/`routine`
     command.
3. **claude**
   - Holds a concurrency slot per task branch.
   - Loads the controller from the gate's trusted revision, moves it out of
     the workspace, and checks out the task.
   - Re-resolves the decision. It aborts if the result differs from the
     gate's. It records the run attempt and the invocation time as job
     outputs.
   - Runs `anthropics/claude-code-action` in agent mode with the
     controller-built prompt and tool profile, plus `--model opus`.
4. **report** (`issues`/`pull-requests: write`)
   - Posts refusals, revocation acknowledgements and failure notes.
   - Fails a new planning run that left no verifiable deliverable
     (`PLAN_NO_DELIVERABLE`; see [Completion checks](#completion-checks)).

### Action mode and branches

We always pass `prompt` and never `track_progress`. The pinned action then
uses agent mode (`src/modes/detector.ts`), so no tag-mode behaviour applies:
- no automatic `claude/…` branch;
- no tracking comment;
- no "Create PR" link instructions.

This means `branch_prefix` and `branch_name_template` are not used. The
controller decides the branch:
- `agent/issue-N` (created by the workflow from the default branch if
  absent);
- or the existing task branch;
- or the PR head.

Claude opens draft PRs with `gh pr create --draft`.

## Tool profiles

Claude Code receives:
- `--tools <list>`: which built-in tools exist;
- `--permission-mode dontAsk`: every call not pre-approved is denied;
- `--allowedTools`: the pre-approved calls;
- `--disallowedTools`: explicit denials;
- `--setting-sources user`: ignore `.claude/` settings and hooks from the
  task branch;
- `--strict-mcp-config`: only the action's own MCP servers.

In agent mode the action installs a GitHub MCP server only if one of its
tools is listed, and no profile lists one (the review workflow lists only the
inline-comment tool).

| Profile | Built-in tools | Approved calls |
|---|---|---|
| read | Read, Glob, Grep, Bash | Read/Glob/Grep; `git status/log/show/diff`; `gh issue/pr view`, `gh pr diff`, `gh issue/pr comment` |
| plan | + Edit, Write | read rules + `Edit`/`Write` of `docs/plans/N.md` only; `git add docs/plans/N.md`; `git commit`; `git push origin agent/issue-N`; `gh pr create` (when no PR exists) |
| implement | + NotebookEdit | all Read/Edit/Write/Bash, except denied: edits to the plan file, `gh pr merge/ready/review/close`, `gh api/workflow/run/secret`, force or delete pushes |

The command chooses the profile, and authorization only caps it. A read-only
request on an approved task still gets the read profile.

### Evidence (2026-10-22)

- **Parser:** the generated arguments were parsed by the pinned action's own
  `parse-sdk-options.ts` (with `shell-quote` 1.8.4 as locked). They produced
  the expected `tools`, `permission-mode`, allow and deny lists.
- **SDK:** Agent SDK 0.3.308 (locked by the action) passes these as CLI
  flags. Neither the action nor the SDK sets its own permission mode or
  permission callback.
- **Local runs:** the same flags were tested with Claude Code 2.1.304 (the
  action installs 2.1.308) in a scratch repository, with a fake `gh`:

| Profile | Allowed | Denied or unavailable |
|---|---|---|
| read | `git log`, `gh pr comment` | Write and Edit unavailable; `touch` and `gh pr merge` denied |
| plan | editing `docs/plans/42.md` | editing `src/app.py`, writing other files, `git push origin main`, `python3` |
| implement | editing source files | editing the plan, `gh pr merge`, `gh pr ready` |

- **Bypass:** in the implement profile, `sh -c 'gh pr merge 9'` **was
  executed**. Bash prefix rules are not isolation.
- **Model:** `claude.yml` appends `--model opus` once to the generated
  arguments (`profileArgs` never emits `--model`). For every profile, the
  pinned parser turns the final `claude_args` into `sdkOptions.model =
  "opus"`. Run 39773385519 (before this flag) passed no model and initialized
  `claude-sonnet-5-5`. Which model the `opus` alias resolves to has not been
  observed in a live run.

## What is enforced by code, and what is not

Enforced by trusted controller code (tested in `claude-task.test.js`):
- Requester and re-runner write access; bot senders ignored.
- Fork PRs rejected before checkout or invocation, including PRs reached
  through `issue_comment`.
- Command grammar:
  - full SHAs only;
  - grants and revocations only as top-level PR comments;
  - edited grant comments are ignored.
- Task discovery and duplicate prevention. An existing branch without a PR is
  reused only if it is plan-only.
- Approval activation:
  - SHA equals the PR head, both via the API and locally;
  - the commit resolves;
  - the plan exists at that SHA;
  - there is no later revocation.
- Resume checks:
  - the latest activated grant after the latest revocation;
  - its SHA is an ancestor of the head (local git);
  - the plan is unchanged since that SHA;
  - no fallback to an older grant; fails closed if the SHA cannot be resolved.
- Tool profile selection and the per-profile permission rules.
- Completion of new planning runs (see below).
- Merge gating by `plan-guard` (only once it is a required check).

Prompt-only rules (the agent is instructed, but nothing stops a determined or
prompt-injected agent with the relevant tools):
- Staying within the approved scope, and recognising material scope changes.
- In the implement profile: not merging, not marking ready, not editing
  `.github/`. Denials can be bypassed via Bash (see above). The real controls
  are branch protection and the app's missing `workflows` scope.
- Treating issue, PR and repository text as data.
- Truthful reports, conditional knowledge updates, PR body conventions.
- One final comment per run.

## Completion checks

After a successful Claude job, the report job checks the deliverable only for
new planning runs. Other runs are judged by the Claude job result.

| Run | Passes when |
|---|---|
| New planning run (no task PR) | (a) the plan is published, or (b) a correlated clarification exists; otherwise `PLAN_NO_DELIVERABLE` |
| Plan revision (`@claude plan` on a task PR) | Claude job succeeded |
| Read-only request | Claude job succeeded |
| Implementation (`approve-plan`, `routine`, `fix`, `continue`) | Claude job succeeded |

(a) **Published plan**, read from the repository:
- `agent/issue-N` exists;
- `docs/plans/N.md` is a file at its head commit;
- an open PR from it is a draft.

No invocation metadata is needed for (a): the gate admits a new planning run
only when the task has no PR.

(b) **Clarification.** The planning prompt asks the agent to end a
clarification comment with
`<!-- claude-task:clarification run=<run id> attempt=<attempt> -->`. A comment
on issue N counts only if:
- its author is `claude[bot]` with type `Bot`;
- it was created at or after the claude job's recorded invocation time;
- it contains the marker for this run ID and attempt;
- after removing the marker, it has a line of at least 15 characters that
  contains a letter and a `?`.

The marker is not a secret: anyone could copy it, but only the `claude[bot]`
author is accepted.

The clarification check needs the claude job's `run_attempt` and
`invoked_at` outputs. It reports **unverifiable**, with the reason, and
never accepts a comment, if:
- either output is missing or malformed;
- `invoked_at` is in the future;
- `run_attempt` differs from the report job's attempt. This happens when
  only the failed report job is re-run, because GitHub then reuses the
  earlier attempt's outputs.

For a new planning run, a bad run identity fails the claude job before Claude
starts.

**Notice.** When neither (a) nor (b) holds, the report job posts a fixed
comment starting `PLAN_NO_DELIVERABLE` and fails with the same code. The
comment lists:
- the run link and attempt;
- whether the branch was found, with its SHA;
- whether the plan file is present;
- the draft PR state;
- the clarification result ("none found" or "unverifiable: <reason>").

It says that the workflow does not know why the agent stopped. It contains no
agent output.

**Recovery.** The supported path is a fresh `@claude plan` comment on the
issue, after inspecting any branch or PR the run left. If the branch exists
without a PR, the gate reuses it only if it is plan-only. Re-running jobs is
not a supported recovery for this check.

**Remaining gaps:**
- Plan revisions, read-only requests and implementation runs pass on the
  Claude job result alone. A revision that pushes nothing, a read-only run
  that posts no answer, or an implementation run that stops early is not
  detected.
- Partial outcomes are not inspected beyond (a): extra commits, a PR body
  that ignores the template, or a missing issue comment after a published
  plan.
- Question quality is not judged; the check only requires a question line.
- The agent's final response is not recorded: `show_full_output` stays at
  the action's default (false) and the execution file is not uploaded.
  Diagnostic run metadata is a separate, deferred task.

## Remaining risks

- **Repository code runs with credentials.** The implement profile runs
  tests (repository code) with:
  - the action's GitHub App installation token, which is in `GH_TOKEN` and
    embedded in the `origin` URL in `.git/config`;
  - the read-only `DEFAULT_WORKFLOW_TOKEN`.

  `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` removes Anthropic credentials (such as
  `CLAUDE_CODE_OAUTH_TOKEN`) from subprocesses, but by design **not GitHub
  tokens**. `persist-credentials: false` does not stop the action from
  configuring its own credentials. The token is masked in logs and revoked
  when the step ends. No allowlist makes running tests safe.
- **Credential options considered and rejected:**
  - `use_commit_signing` still writes the token into the remote URL;
  - `allowed_non_write_users` switches to a credential helper but weakens
    authorization;
  - `pull_request_target` is not used.
- **Workflow definitions for review events come from the PR.** For
  `pull_request_review`, `pull_request_review_comment` and `pull_request`
  events (including plan-guard and the advisory review), GitHub uses the
  workflow file from the PR's merge ref.
  - A PR that edits these workflows can change what runs. The action's
    token-exchange service is expected to refuse an app token when the
    workflow differs from the default branch: the client code handles a
    "workflow validation" refusal in `src/github/token.ts`, but this is a
    server-side check and not verified here. Repository secrets are still
    available to same-repo PRs.
  - This matters only if someone who can push workflow changes is
    compromised. The Claude app must not hold the `workflows` permission.
  - The `issues` and `issue_comment` triggers always run the default-branch
    workflow.
- **Activation markers** are trusted only from `github-actions[bot]`. Any
  same-repo workflow with `issues: write` could forge one. All workflows here
  declare explicit, minimal permissions. New workflows must not grant
  `issues`/`pull-requests: write` to jobs that run repository code.
- **Permission at check time.** Approver access is checked when a request is
  evaluated, not when the comment was written. Removing someone's access
  invalidates their grants.
- **Revocation does not interrupt a running job**; cancel the run manually.
  plan-guard re-evaluates only on PR pushes; re-run it after a revocation.
- **Floating references.** The Claude action is pinned to commit `6b6870e`.
  Its install script fetches Claude Code 2.1.308 from claude.ai at run time,
  and the action resolves its own dependencies via `bun install`.

## Concurrency

- Per-task concurrency group `claude-task-<branch>` (shared with the advisory
  review) with `queue: max` and `cancel-in-progress: false`.
- This is not a durable task queue:
  - GitHub keeps at most 100 pending runs and cancels the rest;
  - pending runs are roughly first-in, first-out, but GitHub states that
    ordering is not guaranteed;
  - failed or cancelled runs are not retried;
  - the gate runs before the slot is acquired, so the claude job re-resolves
    and aborts on any change.

  Re-issue a request whose run was cancelled or aborted.
- Issue and PR events for the same task share a group because both resolve to
  `agent/issue-N`. Read-only requests on non-task PRs use the PR head branch.
- The advisory review reviews the fixed `head.sha` from the triggering event
  and names it, even if the PR moved while it waited.

## Upgrading the pinned action

Do not move to a newer action or a floating `@v1` without re-checking:
- mode detection (`src/modes/detector.ts`);
- agent mode (`src/modes/agent/index.ts`);
- the token scopes for `additional_permissions` (`src/github/token.ts`);
- git credential setup (`src/github/operations/git-config.ts`);
- argument parsing (`base-action/src/parse-sdk-options.ts`);
- the Claude Code version installed.

Then repeat the profile checks above.

## Required manual settings

1. Branch protection or ruleset on the default branch:
   - require a PR and one approving review from a maintainer;
   - required status checks `test`, `controller` and `plan-guard`;
   - no bypass for the Claude GitHub App;
   - block force pushes and deletions.
2. Claude GitHub App:
   - installed on this repository;
   - confirm its permissions, and that it does **not** have Workflows: write;
   - confirm whether the token-exchange service honours `additional_permissions`.
3. Secret `CLAUDE_CODE_OAUTH_TOKEN`: owner, scope and rotation.
4. Actions settings:
   - default workflow token read-only;
   - fork PR workflows require approval;
   - allowed actions limited to `actions/*` and `anthropics/*`, or to the
     pinned SHAs.
5. Optionally, CODEOWNERS review for `.github/**` and `CLAUDE.md`.

## Manual smoke tests (after merge to the default branch)

Not run; they need live GitHub. Record results in the report for the change
that introduced this workflow.

| # | Scenario | Steps | Expected |
|---|---|---|---|
| 1 | New issue | A maintainer opens issue N | One run; branch `agent/issue-N` with only `docs/plans/N.md`; one draft PR "Related to #N"; issue comment with link |
| 2 | Repeated planning | Comment `@claude plan` on issue N | No agent run; comment pointing to the existing PR |
| 3 | Unauthorized request | A read-only user comments `@claude plan`, or opens an issue | gate ends with notice "does not have write access"; no claude job |
| 4 | Fork PR | A maintainer comments `@claude why?` on a fork PR | Refusal comment; no checkout of PR content; no claude job |
| 5 | Missing or stale approval | `@claude fix` before approval; then `@claude approve-plan <older SHA>` | Both refused with reasons; no implement run |
| 6 | Valid approval | `@claude approve-plan <current head SHA>` | Activation marker by github-actions[bot]; implementation commits on the same branch and PR; report written |
| 7 | Partial failure and resume | Cancel run 6 midway, then re-run it or comment `@claude continue` | Resumes under the original approval (SHA is an ancestor, plan unchanged) |
| 8 | Changed plan | `@claude plan` revision after approval, then `@claude fix` | Fix refused ("changed after the authorized commit"); plan-guard fails until `@claude approve-plan <new head>` |
| 9 | Revocation | `@claude revoke`, then `@claude continue` | Continue refused; no fallback to the earlier approval |
| 10 | Ready PR | A maintainer marks the PR ready for review | Exactly one advisory review naming the reviewed SHA; none while it was a draft |
| 11 | Read-only on approved task | `@claude explain the change` | Read profile: answer only, no commits |
| 12 | Silent planning run | `@claude plan` on an issue where the agent ends without publishing or asking (as in run 39773385519) | report job fails with `PLAN_NO_DELIVERABLE`; issue comment with run link and branch/plan/PR state |
| 13 | Clarification | `@claude plan` on an issue too vague to plan | `claude[bot]` comment ending with the run's clarification marker; report job succeeds |
| 14 | Report-only re-run | After 12, re-run only the failed report job | Fails again with "clarification unverifiable: claude job attempt 1, report attempt 2" |
| 15 | Model | Any run | Claude init log shows the model the `opus` alias resolved to; record it |
