# Plan: agent-sandbox-repair — Make the Claude subprocess sandbox work, verify it before invoking Claude, and keep redacted diagnostics

Task: maintainer request in a local Claude Code session, 2026-10-22 (no
issue: opening one would start the planning workflow this task repairs). It
follows the read-only investigation of run 39782086293 (issue #9).
Branch: `chore/agent-sandbox-repair` (not an agent task branch, so plan-guard
does not apply)

The maintainer asked for implementation in the same session. This plan is
committed before any implementation change so that it can be reviewed on its
own.

## Objective
Opening an eligible issue should produce a committed `docs/plans/N.md` on
`agent/issue-N` and an open draft PR. Requirements:
1. Bubblewrap can run on the GitHub-hosted Ubuntu runner. A scoped AppArmor
   permission is preferred over the pinned action's global sysctl.
2. A preflight runs a real command inside the sandbox before Claude is
   invoked (a version check is not enough). If it fails, the job stops with a
   clear error.
3. Agent execution diagnostics are kept when the job fails or a planning
   deliverable may be missing. Retention is short, secrets are redacted, and
   the raw transcript is never uploaded.
4. A missing planning deliverable stays visibly unsuccessful. A verified
   clarification still passes.

## Repository findings
Observed:
- Every agent run since `b1518eb` ended with a successful Claude step, 0
  permission denials and nothing published: 39773385519 (#6), 39777054550
  (#7), 39780089576 (`approve-plan` on PR #8, 4 turns), 39782086293 and
  39782850456 (#9). All deliverables on `main` were committed by the
  maintainer.
- Before `b1518eb`, run 39772040277 failed with: "bubblewrap is required for
  subprocess env scrubbing and isolation … or set
  CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=0". The claude job sets
  `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB: "1"`, so Claude Code runs Bash
  subprocesses through Bubblewrap.
- `claude.yml` installs `bubblewrap` and `socat` and only checks
  `bwrap --version`. The pinned action has its own isolation step
  (`action.yml:225–247` at `6b6870e`). It is skipped here because
  `allowed_non_write_users` is empty. That step also runs
  `sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`; its comment says
  Ubuntu 24.04+ restricts unprivileged user namespaces through AppArmor. The
  runner image is `ubuntu-24.04`.
- Claude Code 2.1.308, the version the action installs, was inspected with
  `strings` only (linux-x64 binary, checksum matches the release manifest):
  - In scrub mode, startup records sandbox availability from whether a
    Bubblewrap path is found (`setScrubSandboxAvailableLatched(L()!==null)`).
    It does not run the binary. An installed but unusable `bwrap` therefore
    passes startup and fails only when a command runs.
  - The Linux sandbox code builds `bwrap` arguments with `--unshare-user`,
    `--unshare-pid`, `--proc /proc`, `--dev /dev` and bind mounts, plus a
    variant that also unshares net, ipc and uts.
- The agent transcript is not available: `show_full_output` and
  `display_report` are false, and `$RUNNER_TEMP/claude-execution-output.json`
  is not uploaded (`docs/agent-workflow.md`, "Remaining gaps").
- The repository is public. Logs are world-readable, and artifacts can be
  downloaded by any signed-in GitHub user.
- The `PLAN_NO_DELIVERABLE` check and the clarification check (task #7)
  failed both #9 runs as designed. When the claude job fails, the report
  posts a generic "did not complete … Partial changes may have been pushed".
- `claude-code-review.yml` also sets `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB: "1"`
  but installs no Bubblewrap, so it would fail like 39772040277.

Not established:
- That `bwrap` actually failed in those runs. No stderr was captured.
- Whether other parts of scrub mode block publication even once `bwrap`
  works. The binary's protected paths include `/.git/config`, `/.git/hooks`
  and `/.config/gh`; scrub mode also uses a `socat` proxy and per-script
  call caps.

## Scope and non-goals
In scope:
- the claude job in `.github/workflows/claude.yml`;
- a new `.github/scripts/claude-sandbox.sh`;
- `.github/scripts/claude-task.js` (diagnostics summary, report messages) and
  its tests;
- `docs/agent-workflow.md`;
- the report for this task.

Non-goals:
- `claude-code-review.yml`: same defect, separate follow-up;
- disabling `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`;
- tool profiles, prompts, the authorization protocol, plan-guard;
- repository, app or Actions settings;
- upgrading the pinned action or Claude Code;
- triggering workflow or agent runs;
- feature-register entries.

## Proposed approach
1. **Sandbox setup and preflight** (`.github/scripts/claude-sandbox.sh`,
   loaded from the trusted controller copy in `$RUNNER_TEMP`). It replaces
   the inline install step and runs before `Run Claude`:
   - install `bubblewrap` and `socat` if they are missing;
   - log the `bwrap` path and version, the kernel version, and the AppArmor
     restriction value;
   - **probe**: as the runner user, run `/bin/sh` inside `bwrap` with new
     user, pid, ipc, uts and net namespaces, a read-only root, `/proc`,
     `/dev` and a writable bind mount. It must write a marker through the
     bind mount, and the marker must be read back outside the sandbox;
   - if the probe passes, make no system change. This is also evidence
     against the AppArmor diagnosis;
   - if it fails while `kernel.apparmor_restrict_unprivileged_userns` is
     `1`, install an AppArmor profile that grants `userns` to the resolved
     `bwrap` path (`flags=(unconfined)`), load it with `apparmor_parser -r`,
     and probe again. Ubuntu 24.04 uses this same form for applications that
     need user namespaces;
   - on final failure, emit an `::error::` annotation with the probe's first
     error line, print the full probe output, write `result=failed`, and exit
     non-zero, so `Run Claude` is skipped.

   *Security scope.* The pinned action's sysctl turns the restriction off for
   every process on the VM. The profile leaves the default in place and
   grants user namespaces only to processes executing that `bwrap` binary.
   Any process can execute `bwrap`, though, so the profile is not a
   containment boundary against code that wants a user namespace; Claude
   Code's own `--disable-userns` handling is what limits nested namespaces.
   Neither option widens the job's privileges: on a GitHub-hosted runner, the
   runner user has passwordless `sudo` and the VM is discarded after the job.
   The profile is chosen because it is the narrower change.
2. **Redacted diagnostics.**
   - A new `writeDiagnostics` / `summarizeExecution` in `claude-task.js`
     turns the action's execution file into a Markdown summary. It runs in a
     github-script step from the trusted controller after `Run Claude` (`if:`
     always, when the agent step ran and either an earlier step failed or this
     is a new planning run). That covers every run whose deliverable the
     report job will judge.
   - The summary contains:
     - the init block (model, permission mode, tools);
     - one line per tool call: the tool name and its target (Bash command,
       file path, search pattern), never file contents or edit bodies;
     - the first 1000 characters of failed tool results; successful results
       appear only as a character count;
     - assistant text, truncated;
     - the result block (subtype, turns, denials, errors, final text,
       truncated).
   - Every string is redacted (GitHub tokens, Anthropic keys, JWTs, URL
     credentials, authorization headers) before truncation, and the total
     size is capped. The prompt is omitted.
   - Upload uses `actions/upload-artifact` pinned to v7.0.1
     (`043fb46d1a93c77aae656e7c1c64a875d1fc6a0a`), name
     `claude-diagnostics-attempt-<attempt>`, retention 5 days.
   - Both steps use `continue-on-error`, so diagnostics can never change the
     job result. The action revokes its app token before these steps run.
3. **Report.**
   - The claude job exports the sandbox result.
   - A failed preflight gets its own issue/PR comment ("sandbox preflight
     failed; Claude was not invoked") and fails the job with
     `SANDBOX_PREFLIGHT_FAILED`, instead of the generic "Partial changes may
     have been pushed".
   - The `PLAN_NO_DELIVERABLE` notice names the diagnostics artifact.
   - A published plan and a verified clarification still pass. A missing
     deliverable still fails the report job and the run, and posts the notice
     (requirement 4). Making the claude job itself red would duplicate the
     publication check there, so it is not proposed.

## Implementation steps
1. `claude-sandbox.sh` and the workflow step that calls it.
2. `summarizeExecution` / `writeDiagnostics`, with tests.
3. Report messages, with tests.
4. Workflow diagnostics steps and the job output.
5. `docs/agent-workflow.md`; report `docs/reports/agent-sandbox-repair.md`.

## Acceptance criteria
- In the claude job, the sandbox step runs before `Run Claude`. When the
  probe cannot run a command inside Bubblewrap (after remediation, if it
  applies), the step exits non-zero with an `::error::` annotation, and
  `Run Claude` does not run.
- The probe executes a command inside new namespaces and checks its effect.
- The AppArmor profile is installed only when the first probe fails and the
  restriction is active.
- Diagnostics:
  - only the redacted summary is uploaded, for failures and new planning
    runs, with 5-day retention;
  - tests show that tokens are redacted, and that successful tool output,
    file contents and the prompt are excluded.
- Report:
  - a sandbox failure gets a specific notice and failure;
  - `PLAN_NO_DELIVERABLE` still fails and names the artifact;
  - a published plan or a verified clarification still passes;
  - existing tests are not weakened.
- `node --test`, actionlint and shellcheck pass. The script's pass, fail and
  AppArmor-remediation paths are exercised in containers.

## Verification
Local:
- `node --test .github/scripts/claude-task.test.js`;
- actionlint and shellcheck (pinned container images);
- `claude-sandbox.sh` in `ubuntu:24.04` containers:
  - default seccomp: the probe fails and no AppArmor restriction is present,
    so the job must fail clearly;
  - seccomp unconfined: the probe passes and nothing changes;
  - AppArmor branch with a stubbed restriction file, `bwrap`, `sudo` and
    `apparmor_parser`;
- compile the generated profile with the real `apparmor_parser -Q`.

Requires a GitHub-hosted run (not done here):
- the first probe's result on `ubuntu-24.04` (evidence for or against the
  diagnosis);
- the profile loading;
- Claude's Bash working inside its sandbox, including `git push` and `gh`;
- the published plan and draft PR;
- the artifact's contents and retention.

## Knowledge updates
- No feature-register entry (workflow infrastructure, as in
  `agent-workflow-stabilization`).
- No new decision record: this repairs the D-002 implementation. The
  AppArmor-versus-sysctl choice and the diagnostics exposure are documented
  in `docs/agent-workflow.md`.

## Risks and open questions
- The diagnosis is unproven. If `bwrap` already works on the runner, the
  probe passes unchanged and the diagnostics expose the real cause.
- Other scrub-mode blockers (writes under `.git`, the `gh` stand-in, the
  network proxy, script caps) would still stop publication. The diagnostics
  are meant to reveal them.
- The probe's flags approximate Claude Code's; a failure specific to Claude
  Code would surface only in the diagnostics.
- Redaction is pattern-based. The artifact is readable by any signed-in
  GitHub user for 5 days.
- AppArmor profile loading cannot be tested locally (Docker Desktop's VM has
  no AppArmor).
- The advisory review workflow remains broken until a follow-up.
