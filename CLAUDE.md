# Project instructions

## Working principles

- Inspect relevant code, tests, and repository conventions before proposing changes.
- Infer the minimum sufficient scope; do not expand the task unnecessarily.
- Preserve unrelated work.
- Ask for clarification when a missing decision materially changes the task.
- Distinguish observed facts, assumptions, and unverified claims.
- Do not weaken tests or acceptance criteria merely to obtain a pass.
- Do not change experiment hypotheses, success criteria, or parameters after seeing results (see Experiment workflow).

## Plan-first workflow

Applies to features and code changes. Experiments follow the Experiment workflow below.

Task IDs:
- Task ID: `<n>-<short-description>`, e.g. `2-tv-regulariser`.
- Task branch: `<n>/<short-description>`, e.g. `2/tv-regulariser`.
- `<n>` is one greater than the highest task number found in docs/plans/,
  docs/reports/, and local and remote branch names.
- A retried or resumed task reuses its existing ID, branch, plan, and PR.

| Request type    | Claude's action                                                     |
|-----------------|---------------------------------------------------------------------|
| Question        | Answers only, no commits                                            |
| New task        | Writes a plan                                                       |
| Plan revision   | Updates the uncommitted plan in place                               |
| Implement       | Implements the approved plan                                        |
| Review fix      | Fixes on the same branch and PR                                     |
| New experiment  | Writes the three pre-run sections of an experiment file, then stops |
| Experiment revision | Updates the unapproved experiment file in place                 |
| Run experiment  | Runs the approved experiment on its experiment branch               |

For a new substantial (ask if task is considered substantial if unsure) task:
1. Inspect the task and relevant repository context. Ask clarifying questions if the task is ambiguous, unclear or there are multiple ways of performing it.
2. Create the task branch from the up-to-date default branch. Write
   docs/plans/<task-id>.md using the plan template (only fill in the sections that are appropriate for the task at hand) and, where a consequential
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
- Commit the plan, implementation, report, and knowledge updates on the task
  branch as atomic commits (see Commits).
- Push the task branch and open a PR using the PR template. Never merge.

After the PR is opened, make further changes only when the user asks.
In-scope fixes on the same branch need no new approval, but the report, feature register and decision records should be updated and always kept up-to-date.
Commit and push them to the same branch and PR.

After the user says the PR has been merged:
1. Confirm the PR state is merged (e.g. `gh pr view <number> --json state`);
   if it is not, stop and report.
2. Check out the default branch and pull it.
3. Delete the task branch locally and on the remote.

## Experiment workflow

Experiments test a hypothesis by running existing code. They are tracked in
docs/experiments/ using docs/experiments/TEMPLATE.md. Experiments do not use
the plan template, task IDs, or task branches. Code changes needed for an
experiment (new features, fixes, new options) are a separate task under the
Plan-first workflow and must be merged, or at least committed, before the
experiment is approved.

Experiment IDs:
- Experiment ID: `EXP-<nnn>`, zero-padded to three digits, never reused.
- `<nnn>` is one greater than the highest number found in docs/experiments/
  and in local and remote `exp/` branch names.
- File: `docs/experiments/EXP-<nnn>-<short-kebab-name>.md`, e.g.
  `docs/experiments/EXP-007-tv-weight-sweep.md`.
- Experiment branch: `exp/<nnn>-<short-kebab-name>`, e.g.
  `exp/007-tv-weight-sweep`.
- A retried or resumed experiment reuses its existing ID, file, and branch.
  A rerun with changed hypothesis, description, or parameters is a new
  experiment; set `parent` to the earlier one.

For a new experiment:
1. Inspect the relevant code, configuration, existing experiments, the feature
   register, and decision records. Ask clarifying questions if the question,
   hypothesis, or success criteria are ambiguous or underspecified.
2. Copy docs/experiments/TEMPLATE.md to the new experiment file. Fill in the
   front matter (`id`, `name`, `status: planned`, `created`, `parent`) and
   only these three sections:
   - Motivation and hypothesis (question, falsifiable hypothesis, and
     quantitative success criteria decided before running)
   - Description
   - Parameters (fixed parameters and parameters to vary)
3. Leave every other section at its template content (use "TBD" for Summary).
   Leave `git_commit` unset.
4. Stay on the current branch. Do not create the experiment branch, commit,
   or push the file at this stage.
5. Do not run experiments, benchmarks, sweeps, or exploratory runs. Do not
   modify implementation, tests, dependencies, or configuration.
6. Stop and wait for explicit experiment approval.

During experiment review:
- Incorporate requested corrections into the same uncommitted file.
- Do not interpret questions, comments, or positive feedback as approval.

Experiment approval is an explicit instruction from the user in the local
session to start the experiment (e.g. "approved, run the experiment"). It applies to
the three sections as they stand at that moment. On approval:
1. Create the experiment branch from the up-to-date default branch, carrying
   the uncommitted experiment file onto it.
2. Add `approved: <date>` to the front matter.
3. Commit the file as the first commit on the branch
   (`docs: approve EXP-<nnn> <name>`). This commit is the frozen reference
   for the approved sections.

Locked sections:
- After approval, Motivation and hypothesis (including success criteria),
  Description, and Parameters must not be edited without renewed approval.
  This prevents success criteria or parameters being rewritten after seeing
  results.
- If a change appears necessary (for example a crash, an infeasible
  parameter value, or a flawed criterion), stop. Explain what changed and why,
  state whether any results have been seen, propose the edit, and wait for
  explicit approval.
- On renewed approval, append a dated amendment note under the changed section
  (what changed, why, and whether results had been seen). Keep the original
  text visible, and commit the amendment separately
  (`docs: amend EXP-<nnn> <section>`).
- Adding a run, parameter value, or metric outside the approved sections also
  needs renewed approval. Analyses not pre-specified may be reported only if
  labelled post hoc and exploratory.

After approval, on the experiment branch:
- Set `status: running` and record `git_commit` (the commit of the code
  that produces the results) before the first run.
- Run only what the approved sections specify. Run the agreed verification
  checks first; do not start expensive runs if they fail.
- Add only experiment artefacts (scripts, configs, logs, result data, figures)
  that follow repository conventions. Do not change library code on the
  experiment branch; that needs a separate task.
- Record every run, including failed, crashed, and discarded ones, with the
  reason. Do not omit unfavourable runs.
- Fill in the Results table section as the results land.
- Fill in the remaining sections: Summary, Interpretation and
  limitations, Knowledge updates, Follow-up, and Open questions and issues.
- Judge the outcome against the success criteria exactly as approved (or as
  amended with approval). A negative or inconclusive result is a valid outcome.
- Set `status` to `completed`, `failed`, or `abandoned` (with the reason).
- Proposed follow-up experiments go in the Follow-up table; do not start them
  or edit the approved sections to accommodate them.
- Commit as atomic commits (see Commits), push the experiment branch, and
  open a PR using the PR template. Never merge.

After the PR is opened, make further changes only when the user asks. Fixes to
results, interpretation, and knowledge updates need no new approval; the
locked sections still do.

After the user says the PR has been merged, follow the same merge cleanup as
the Plan-first workflow, deleting the experiment branch locally and on the
remote.

## Commits

- Make each commit one logical, self-contained change; do not bundle unrelated
  changes or split one change across commits. Typically: the plan first, then
  each implementation step with its tests, then the report and knowledge updates.
  For experiments: the approved experiment file first, then scripts/configs,
  then results, then interpretation and knowledge updates.
- Each commit should leave the tests passing where feasible.
- Use Conventional Commit messages: `<type>: <description>`, imperative and
  lower case, e.g. `feat: add TV regulariser` or `docs: plan for 2-tv-regulariser`.
- Types: `feat` (new capability), `fix` (bug fix), `refactor` (no behaviour
  change), `perf`, `test`, `docs` (plans, reports, decisions, register, README,
  experiment files), `build` (dependencies, packaging), `ci` (workflows),
  `chore` (other maintenance).

## Project knowledge

- docs/features.md records capabilities and lifecycle status.
- docs/decisions/ records consequential decisions and rationale.
- docs/reports/ records implementation and verification evidence.
- docs/plans/ records proposals; a plan is not proof of implementation.
- docs/experiments/ records experiments; only experiments with status
  `completed` or `failed` on a merged branch count as evidence. A `planned` or
  `running` experiment is not a result.

Before proposing a capability or reversing an approach:
- Check the feature register and relevant decisions.
- Read any superseding decisions.
- Check relevant experiments, including negative ones.
- Inspect current code and tests.

For historical or onboarding answers:
- Cite repository paths, decision IDs, experiment IDs, and source PRs.
- Distinguish historical rationale from current implementation.
- Treat unmerged changes as provisional.
- Say when rationale or evidence is missing; do not invent it.

Local agent memory is a convenience, not the authoritative project record.

## Verification

- Run the checks agreed in the plan (or approved experiment) where feasible.
- For numerical work, distinguish software correctness from numerical validity.
- Do not advance to expensive experiments when prerequisite checks fail.
- Report commands, results, relevant configuration, and unverified items.
- A negative finding is a valid outcome.