# Report: agent-sandbox-repair — Make the Claude subprocess sandbox work, verify it before invoking Claude, and keep redacted diagnostics

Task: maintainer request in a local Claude Code session, 2026-10-22 (no
issue; see the plan)
PR: the draft PR from `chore/agent-sandbox-repair`
Authorization: in-session maintainer request to implement. The plan was
committed at `ace912adf8004f23f119e3a773487459781edc83` before any
implementation change and was not separately re-approved. No `@claude`
command was used: the hosted agent is the component being repaired.
Reported head: `b1b2f32336b4634ef3f008f8e9f73d91a86cef05` (implementation);
this report is the next commit.

## Outcome
Completed locally. Unit tests, mutation checks, shellcheck and actionlint
pass. The setup script's six paths behave as intended in `ubuntu:24.04`
containers. Behaviour on GitHub-hosted runners is **not verified**: nothing
was triggered. The diagnosis (AppArmor blocking Bubblewrap) is still
**unproven**. The new step logs the evidence on the first hosted run.

## Changes
### `.github/scripts/claude-sandbox.sh` (new)
- Installs `bubblewrap` and `socat` if missing. Logs the `bwrap` path and
  version, the kernel, and `kernel.apparmor_restrict_unprivileged_userns`.
- **Probe:** runs `/bin/sh` inside `bwrap` with new user, pid, ipc, uts and
  net namespaces, a read-only root, `/proc`, `/dev` and a writable bind
  mount. It passes only if the marker written inside is visible outside.
- If the probe fails while the restriction is `1`, installs and loads the
  profile `/etc/apparmor.d/claude-code-bwrap` (`userns,` for the resolved
  `bwrap` path, `flags=(unconfined)`), then probes again.
- On failure: `::error::` annotation, probe output, `result=failed`,
  non-zero exit.

### `.github/workflows/claude.yml` (claude job only)
- "Prepare and verify sandbox" (`id: sandbox`) runs the script from the
  trusted controller copy. It replaces the inline install step that only ran
  `bwrap --version`.
- `Run Claude` gets `id: agent`. The job exports `sandbox`.
- "Summarize agent session (redacted)" (github-script, trusted controller)
  and "Upload agent session summary" (`actions/upload-artifact` v7.0.1,
  pinned SHA). They run when the agent step ran and either a step failed or
  the run is a new planning run. Artifact `claude-diagnostics-attempt-<n>`,
  retention 5 days. Both steps are `continue-on-error`.

### `.github/scripts/claude-task.js`
- `summarizeExecution`, `writeDiagnostics`, `redact`, `diagnosticsArtifact`.
  The summary keeps tool names and targets, failed results (1000 chars),
  assistant text (500 chars) and the result block (final response, 4000
  chars). It omits the prompt, thinking, file contents, edit bodies and
  successful output. Everything is redacted before truncation; total size is
  capped at 200,000 chars.
- `report`: when the claude job failed and `sandbox` is `failed`, it posts a
  sandbox-specific note and fails with `SANDBOX_PREFLIGHT_FAILED`.
- The `PLAN_NO_DELIVERABLE` notice names the artifact for the claude job's
  attempt.
- Published-plan and clarification semantics are unchanged.

### `.github/scripts/claude-task.test.js`
Six tests added; existing tests unchanged except one added assertion.

### `docs/agent-workflow.md`
- New sections "Subprocess sandbox" (including security scope) and "Session
  diagnostics".
- Updated job descriptions, notice text, remaining gaps and the upgrade
  checklist.
- Smoke tests 16–18 added.

## Deviations from the plan
None in scope or approach. Details settled during implementation:
- the artifact file name is `claude-session-summary.md`;
- the sandbox output name is `result`;
- the failure code is `SANDBOX_PREFLIGHT_FAILED: Claude was not invoked`.

## Verification
| Check/command | Configuration | Result | Evidence |
|---------------|---------------|--------|----------|
| `node --test .github/scripts/claude-task.test.js` | Node (local) | 47/47 pass (41 before) | — |
| Mutation checks on a copy of `claude-task.js` | No redaction; no sandbox branch; leak successful output; leak prompt; clip without redaction; no size cap | Each caught (1–3 failures) | — |
| shellcheck | `koalaman/shellcheck:v0.10.0`, `--network none` | Clean (one intentional SC2016, annotated) | — |
| actionlint | `rhysd/actionlint:1.7.12`, read-only mount, `--network none` | Only the known `queue` key gap, identical on the base commit | — |
| Script: default Docker seccomp | `ubuntu:24.04`, bubblewrap 0.9.0, `runner` user with sudo | Probe fails ("No permissions to create new namespace"); restriction `unavailable` → `::error::`, `result=failed`, exit 1, no profile | — |
| Script: unconfined container | seccomp, apparmor, systempaths unconfined | Probe passes, no system change, exit 0 | — |
| Script: restricted, profile fixes it | Restriction stub `1`; `bwrap` shim fails until a profile is "loaded"; `apparmor_parser` shim compiles with the real parser (`-Q -K`) | Profile installed (0644), compiled, re-probe passes, exit 0 | — |
| Script: restricted, still failing | As above, default seccomp | "even with the AppArmor profile" error, exit 1 | — |
| Script: profile load fails | Parser shim fails | "could not load AppArmor profile", exit 1 | — |
| Script: restriction `0` | Stub `0`, default seccomp | Fails without installing a profile | — |
| Script: packages missing | bubblewrap and socat removed first, with network | apt installs them; probe passes | — |
| Claude Code 2.1.308 inspection | linux-x64 binary from `downloads.claude.ai`, SHA-256 matches manifest; `strings` only, never executed | Scrub startup checks only that `bwrap` exists; protected-path list and `socat` proxy observed | Plan, "Repository findings" |

Not run:
- any GitHub-hosted run or smoke test (by instruction: no agent runs);
- AppArmor profile loading into a real kernel (Docker Desktop's VM has no
  AppArmor);
- Claude Code's own sandboxed Bash with this setup (it needs a live model
  session).

## Interpretation and limitations
- Software correctness: the controller changes are covered by unit tests,
  and the script's control flow by the container cases.
- Not established:
  - that the hosted runner's restriction is what broke the earlier runs;
  - that the profile alone is sufficient there;
  - that Claude Code's sandboxed `git push`, `gh pr create` and
    `gh issue comment` succeed under scrub mode. Its protected paths, `gh`
    stand-in, network proxy and script caps were seen only as strings.
- If publication still fails after this change, the session summary should
  show the failing tool result.
- The probe approximates Claude Code's `bwrap` arguments.
- The diagnostics artifact is downloadable by any signed-in GitHub user for
  5 days; redaction is pattern-based.
- `claude-code-review.yml` still sets `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`
  without installing Bubblewrap. It will fail at Claude Code startup when a
  PR is marked ready for review, including this one.

## Knowledge updates
None:
- feature register: workflow infrastructure, not a project capability;
- decision records: this repairs the D-002 implementation. The security
  choice is documented in `docs/agent-workflow.md`.

## Follow-up
Required:
- After merge, open an eligible issue and check smoke tests 1, 16 and 18:
  - the first probe's result and whether the profile was needed;
  - the published plan and draft PR;
  - the artifact's contents (check it for secrets).

  If publication still fails, use the summary to find the blocking tool
  error.
- Apply the same sandbox step to `claude-code-review.yml`, or decide on
  another approach for it.

Optional:
- Once the profile is confirmed sufficient on hosted runners, record the
  observed restriction value in `docs/agent-workflow.md`.
