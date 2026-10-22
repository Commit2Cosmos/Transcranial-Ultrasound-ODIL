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
| `.github/scripts/claude-task.js` | Controller logic: authorization, discovery, approval, profiles, prompt, plan guard, redacted session summary |
| `.github/scripts/claude-sandbox.sh` | Prepares Bubblewrap for Claude Code's subprocess isolation and proves that a command runs inside it (see [Subprocess sandbox](#subprocess-sandbox)) |
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
   - Prepares and verifies the subprocess sandbox
     (`claude-sandbox.sh`). If that fails, the job fails before Claude
     starts.
   - Runs `anthropics/claude-code-action` in agent mode with the
     controller-built prompt and tool profile, plus `--model opus`.
   - For failed jobs and new planning runs, uploads a redacted session
     summary (see [Session diagnostics](#session-diagnostics)).
4. **report** (`issues`/`pull-requests: write`)
   - Posts refusals, revocation acknowledgements and failure notes. A failed
     sandbox preflight gets its own note and fails with
     `SANDBOX_PREFLIGHT_FAILED`.
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

## Subprocess sandbox

The claude job sets `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`, so Claude Code runs
every Bash subprocess in Bubblewrap with Anthropic credentials removed.
Without `bwrap`, Claude Code refuses to start (run 39772040277). At startup
it checks only that a `bwrap` binary exists (Claude Code 2.1.308, observed in
the binary). An installed `bwrap` that cannot create its namespaces therefore
passes startup and fails each command at run time. Such a tool error is not
a permission denial.

`claude-sandbox.sh` runs before `Run Claude`, from the trusted controller
copy:
1. Installs `bubblewrap` and `socat` if missing. It logs the `bwrap` path and
   version, the kernel, and `kernel.apparmor_restrict_unprivileged_userns`.
2. **Probe.** As the runner user, runs `/bin/sh` in `bwrap` with new user,
   pid, ipc, uts and net namespaces, a read-only root, `/proc`, `/dev` and a
   writable bind mount. It passes only if the marker the shell writes is
   visible outside.
3. If the probe passes, nothing on the system is changed.
4. If it fails while the restriction is `1`, the script installs
   `/etc/apparmor.d/claude-code-bwrap`. That profile grants `userns` to the
   resolved `bwrap` path (`flags=(unconfined)`), the form Ubuntu uses for
   applications that need user namespaces. The script loads it with
   `apparmor_parser -r` and probes again.
5. On final failure it emits an `::error::` annotation and the probe output,
   sets output `result=failed`, and exits non-zero. Claude is not invoked.

**Security scope.** On Ubuntu 24.04 with the restriction set to `1`, an
unprivileged process may create a user namespace but gets no capabilities in
it, unless its executable has an AppArmor profile granting `userns`.
- The pinned action's own setup (used only with `allowed_non_write_users`)
  sets the sysctl to `0`, lifting the restriction for every process on the
  VM.
- The profile keeps the default for every other executable. It is not a
  containment boundary, though: any process, including repository code in
  the implement profile, can execute `bwrap` to obtain a user namespace.
  Nested namespaces inside Claude's sandbox are limited by Claude Code's own
  `--disable-userns` handling, not by this profile.
- Neither option widens what the job can do. On GitHub-hosted runners the
  runner user has passwordless `sudo`, and the VM is discarded after the job.

**Evidence and gaps.**
- Unverified on GitHub-hosted runners: whether the first probe fails there,
  and whether the profile loads and is enough. The "Prepare and verify
  sandbox" log of the first hosted run answers both.
- The probe approximates Claude Code's `bwrap` arguments; it does not
  reproduce them exactly.
- A working sandbox does not prove that every scrub-mode restriction allows
  publication: `.git` and `gh` paths are protected, and network and script
  calls are mediated. Session diagnostics show such failures.
- `claude-code-review.yml` sets the same variable but does not run this
  script, so it is expected to fail at Claude Code startup until it does.

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

It says that the workflow does not know why the agent stopped. It names the
diagnostics artifact for the claude job's attempt. The comment itself
contains no agent output.

A failed sandbox preflight is reported instead as `SANDBOX_PREFLIGHT_FAILED`.
Claude was not invoked, so there is no deliverable to check.

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
- The claude job stays green when the agent ends without a deliverable. The
  report job and the run are red.

## Session diagnostics

`show_full_output` stays false. The action's execution file
(`$RUNNER_TEMP/claude-execution-output.json`) is the full transcript: the
prompt, file contents and command output. It is never uploaded.

After `Run Claude`, two steps run when the agent step ran and either an
earlier step failed or the run is a new planning run (the report job decides
afterwards whether its deliverable is missing):
- `writeDiagnostics` (trusted controller) writes
  `claude-session-summary.md`. It contains:
  - the init block (model, permission mode, Claude Code version, tools);
  - each tool call's name and target: the Bash command, or the file path or
    pattern; never file contents, edit bodies or other inputs (only their
    keys);
  - for failed tool results, the first 1000 characters; successful results
    appear only as a character count;
  - assistant text truncated to 500 characters (thinking blocks omitted);
  - the result block: subtype, turns, denials, errors, and the final
    response truncated to 4000 characters.

  The prompt is omitted. Every string is redacted before truncation (GitHub
  tokens, Anthropic keys, JWTs, URL credentials, authorization headers), and
  the file is capped at 200,000 characters.
- `actions/upload-artifact` (v7.0.1, pinned) uploads it as
  `claude-diagnostics-attempt-<attempt>`, kept for 5 days.

Both steps are `continue-on-error` and cannot change the job result. The
action revokes its GitHub App token before they run.

**Exposure.** The repository is public, so any signed-in GitHub user can
download the artifact while it exists. Redaction is pattern-based: a secret
in an unrecognised format, inside a failed command's output or in the
agent's text, would be published. Delete the artifact from the run page if
that happens.

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
- the Claude Code version installed, and its subprocess-scrub sandbox: the
  `bwrap` arguments it builds and the paths it protects. Keep the probe in
  `claude-sandbox.sh` representative.

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
| 16 | Sandbox preflight | Any claude job | "Prepare and verify sandbox" logs the restriction value and the first probe's result, and whether the AppArmor profile was needed; the step passes before `Run Claude` |
| 17 | Sandbox failure | Not reproducible on demand; observe if it occurs | The step fails with an `::error::` annotation; `Run Claude` is skipped; the report posts the sandbox note and fails with `SANDBOX_PREFLIGHT_FAILED` |
| 18 | Session diagnostics | After 1 (or 12, 13) | Run has artifact `claude-diagnostics-attempt-1` with 5-day retention, containing only `claude-session-summary.md`; check it for unredacted secrets |
