# Report: agent-workflow-stabilization — Stabilize the agent-assisted development setup

Task: maintainer request in a local session, 2026-10-22 (audit findings F2–F14)
PR: none (local working tree only, by instruction)
Authorization: in-session approval of the plan with amendments 1–9 (docs/plans/agent-workflow-stabilization.md)
Reported head: uncommitted working tree on `main` at fe39165

## Outcome
Completed locally. Static and local checks pass. Live GitHub behaviour is
**not verified**; nothing was pushed or triggered.

## Changes
- **One handler.** `.github/workflows/claude.yml` was rewritten as the single
  `@claude` and new-issue handler, with jobs gate → activate → claude →
  report. The duplicate `.github/workflows/claude-plan.yml` (untracked) was
  removed.
  - Retained from it: automatic planning on `issues: opened`, the rerun-actor
    check, fork- and bot-aware gating.
  - Retained from the old `claude.yml`: the review-comment and review
    triggers, and `actions: read`.
- **Controller.** `.github/scripts/claude-task.js` is new. It is loaded only
  from the default-branch revision and covers command parsing, authorization,
  fork rejection, task discovery, SHA-bound grants, activation, revocation,
  local-git resume checks, tool profiles, the prompt, plan-guard and report.
  Tests are in `.github/scripts/claude-task.test.js`.
- **Plan guard.** `.github/workflows/plan-guard.yml` is new: a required-check
  candidate that runs controller code from the default branch.
- **Advisory review.** `.github/workflows/claude-code-review.yml` was
  restored with a new configuration:
  - triggers: `ready_for_review`, or `opened` when not a draft; same-repo PRs
    and human senders only;
  - fixed SHA and per-branch concurrency shared with `claude.yml`;
  - read-only tools; app token requested with `contents: read`;
  - its own prompt, because the `code-review` plugin skips PRs where Claude
    has already commented.
- **Tests workflow.** `.github/workflows/tests.yml` gets a workflow-level
  read-only token and a `controller` job (`node --test .github/scripts/`).
- **Pinned actions.** `anthropics/claude-code-action@6b6870e…` (v1 as
  audited), `actions/checkout@d23441a…` (v6) and
  `actions/github-script@ed59741…` (v8).
- **Policy.**
  - `CLAUDE.md`: plan-first scope, and the canonical "Agent task protocol"
    with [code]/[prompt] markers, commands, approval, routine, revocation,
    resume, rejected proposals, plan lifecycle, and conditional knowledge
    updates.
  - `docs/features.md`: F-IDs, Source column, update rules. The inventory was
    not populated.
  - `docs/decisions/README.md` and `TEMPLATE.md`: Superseded status,
    acceptance authority, a `Decided by` field.
  - `docs/plans/TEMPLATE.md`: no status in the file.
  - `docs/reports/TEMPLATE.md`: authorization and reported head.
  - `.github/pull_request_template.md`: stages and authorization field.
- **Knowledge.** `docs/agent-workflow.md` (operations, enforcement split,
  risks, settings, smoke tests) and `docs/decisions/D-002-…` (Proposed).

## Deviations from the plan
- The action's own scope list was not used to restrict the review job's app
  token beyond `contents: read`. Issue and PR write remain, because posting
  comments needs them.
- Job-level `concurrency.queue: max` is used as described in the GitHub
  docs. actionlint 1.7.12 does not recognise the key (see below).

## Verification
| Check/command | Configuration | Result | Evidence |
|---|---|---|---|
| `node --test .github/scripts/` | Node 26.8.1 locally | 30/30 pass | authorization, forks, re-run actor, discovery and duplicates, activation, resume, revocation (no fallback), edited and forged markers, unresolvable SHA, plan guard, profiles, prompt |
| actionlint | `rhysd/actionlint@sha256:b1934ee5…` (1.7.12); `--network none --read-only`, repo mounted `:ro`, no secrets, no Docker socket | Pass, apart from `unexpected key "queue"` (2×) | that one message filtered with `-ignore`; exit 0 |
| YAML parse | PyYAML | 4 workflows parse | — |
| Action argument parsing | pinned `base-action/src/parse-sdk-options.ts` and `src/modes/agent/parse-tools.ts` with `shell-quote` 1.8.4 | `tools`, `permission-mode`, allow and deny lists, `max-turns`, `settingSources=["user"]` as intended; no MCP tools in profiles | scratchpad harness |
| SDK flag passing | `@anthropic-ai/claude-agent-sdk` 0.3.308 (locked by the action) | `extraArgs` → `--flag value` / `--flag`; no SDK-set permission mode or `canUseTool` in the action | source inspection |
| Profile enforcement | `claude -p` 2.1.304 (action installs 2.1.308), Haiku, scratch repo, fake `gh`, `--setting-sources project` (no project settings present) | read: Write/Edit unavailable; `touch` and `gh pr merge` denied; `git log` and `gh pr comment` allowed. plan: only `docs/plans/N.md` editable or creatable; `git add` other files, `git push origin main`, `python3` denied; commit, task push and `gh pr create` allowed. implement: plan edit, `gh pr merge`, `gh pr ready` denied | JSON `permission_denials` |
| Bash prefix bypass | implement profile | `sh -c 'gh pr merge 9'` **executed** | fake `gh` log; documented as not isolation |
| Scope | `git status`, `git diff --stat`, `git diff --check` | only `.github/`, `docs/` and `CLAUDE.md` changed; no whitespace errors | — |

Not run: any live GitHub workflow, branch protection, app permission checks,
or the smoke tests in docs/agent-workflow.md.

## Interpretation and limitations
Established:
- The controller logic behaves as specified on the tested inputs.
- The generated Claude arguments are parsed by the pinned action as intended.
- Claude Code 2.1.304 enforces the profiles for direct tool calls.

Not established:
- **End-to-end runtime behaviour on GitHub:**
  - event payload handling against real payloads;
  - `GITHUB_TOKEN` access to the collaborator-permission endpoint;
  - acceptance of job-level `queue`;
  - token downscoping by the exchange service;
  - Claude Code 2.1.308 behaving like 2.1.304;
  - the app token being able to run `gh pr create`.
- **Prompt-only rules:** scope adherence; and, in the implement profile, not
  merging, marking ready or editing workflows (bypassable via Bash).
- **Running tests** executes repository code with a GitHub app token present.
  The env scrub only removes Anthropic credentials.
- **Workflow files for `pull_request*` events come from the PR merge ref.**
  plan-guard and the review workflow can be altered by anyone who can push
  workflow changes.
- **Refusal comments on fork PRs** reached through
  `pull_request_review(_comment)` may fail to post, because GitHub gives those
  runs a read-only token. Invocation is still refused.

## Knowledge updates
- D-002 added as Proposed. It needs a maintainer decision ("Accept D-002").
- No feature-register entries.

## Follow-up
Required before relying on the workflow:
1. Configure the settings in docs/agent-workflow.md ("Required manual
   settings").
2. Merge, then run the smoke tests and record the results.

Optional, kept separate:
- `tests.yml` runs both push and pull_request builds on PR branches.
- The black/ruff TODO.
- README's stale `odil/` path.
- Stale remote branches.
