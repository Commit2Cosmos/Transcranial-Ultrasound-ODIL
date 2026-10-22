# Plan: agent-workflow-stabilization — Stabilize the agent-assisted development setup

Task: maintainer request in a local Claude Code session, 2026-10-22 (no
issue). It addresses audit findings F2–F14; the truncated prompt (F1) is fixed
by the rewrite.
Branch: none (local working tree; no push or PR, by instruction)

Approved by the maintainer in-session, with amendments 1–9 below.

## Objective
- One `@claude` handler with compatible planning behaviour.
- One canonical approval protocol, with code enforcement where feasible.
- Explicit security boundaries.
- A defined knowledge lifecycle.
- Coordinated advisory review.

## Repository findings
Audit of the initial setup: two workflows answered the same `@claude`
comment with contradictory approval rules. The planning prompt was truncated,
and the action's tag mode created its own branches and compare links. There
was no duplicate-plan prevention, enforcement was prompt-only, and the
reviewer had been deleted.

The pinned action (`anthropics/claude-code-action` v1 = 6b6870e) was
inspected:
- mode detection, agent mode, branch setup;
- git credential configuration;
- token scopes;
- argument parsing;
- the human-actor check.

## Scope and non-goals
In scope:
- `.github/workflows/*`, `.github/scripts/*`, PR template;
- `CLAUDE.md`, docs templates and policies;
- `docs/agent-workflow.md`, D-002.

Non-goals:
- application code;
- populating the feature inventory;
- retrospective rationale;
- GitHub settings;
- pushing or opening PRs;
- unrelated cleanup (tests.yml push and PR double run, black/ruff TODO,
  README path, stale branches).

## Proposed approach
- Agent mode everywhere (explicit `prompt`, no `track_progress`).
- Controller-chosen task branch `agent/issue-N`. Draft PR via
  `gh pr create --draft`.
- `claude.yml` jobs: gate → activate → claude (per-task concurrency) → report.
- Controller (`.github/scripts/claude-task.js`) loaded only from the trusted
  default-branch revision (amendment 1).
- Tool profiles: read, plan and implement. Each uses `--tools`,
  `--permission-mode dontAsk`, allow and deny rules, `--setting-sources user`
  and `--strict-mcp-config` (amendment 2). The command chooses the profile,
  and authorization only caps it (amendment 3).
- SHA-bound `approve-plan` and `routine` commands, accepted only as top-level
  PR comments. Activation markers. Resume requires the latest activated grant,
  local-git ancestry and an unchanged plan; fails closed (amendments 4, 5).
- `@claude revoke` with no fallback (amendment 6).
- `plan-guard` required check, run from trusted code (amendment 7).
- Advisory review on ready-for-review: same concurrency group, fixed SHA
  (amendment 8).
- Action pinned to full SHA 6b6870e; checkout and github-script pinned too
  (amendment 9).

## Implementation steps
1. Controller module and unit tests.
2. Workflows: `claude.yml`, `plan-guard.yml`, `claude-code-review.yml`,
   `tests.yml` controller job and read-only permissions. Remove
   `claude-plan.yml`.
3. `CLAUDE.md` protocol. Knowledge-lifecycle rules in features, decisions,
   plan, report and PR templates.
4. `docs/agent-workflow.md` and D-002 (Proposed).
5. Static validation and local permission tests. Write the report.

## Acceptance criteria
- Exactly one workflow responds to each `@claude` event.
- Controller tests cover authorization, forks, discovery, activation,
  resume, revocation, plan guard and profiles.
- The generated arguments parse correctly with the pinned action's parser.
- Prohibited tool calls fail in local CLI tests.
- actionlint passes, except for known schema gaps.
- No application code changes.

## Verification
- `node --test .github/scripts/`
- actionlint (pinned image, read-only mount, no network)
- YAML parse
- The action's parser applied to the generated arguments
- Local `claude -p` permission tests
- `git diff --stat`

Live GitHub smoke tests are listed in docs/agent-workflow.md and are out of
scope here.

## Knowledge updates
- D-002 (Proposed).
- No feature-register entries (workflow infrastructure, not a project
  capability).

## Risks and open questions
- Runtime behaviour on GitHub is unverified.
- Job-level `queue` is not recognised by actionlint 1.7.12.
- Token-exchange downscoping and workflow validation happen on the server
  side.
