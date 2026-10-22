# D-001: Use plan-first agent-assisted development

Status: Accepted
Features: Not applicable
Source: PR #3 (https://github.com/Commit2Cosmos/Transcranial-Ultrasound-ODIL/pull/3);
prior initial workflow discussion not recorded in the repository
Supersedes: —
Superseded by: —

## Context
We want agents to propose repository-informed changes before implementing
them, with explicit human review of both the plan and resulting code.

We also want shared, reviewable knowledge of project capabilities and the
reasons behind consequential decisions.

## Decision
Use one task branch and PR for planning and implementation.

The initial draft PR contains only the task plan.
Implementation requires explicit maintainer approval of a specified revision.

Implementation PRs include a verification report and relevant feature-register
and decision-record updates.

The default branch contains the accepted project knowledge.
Agent-local memory is not authoritative.

## Alternatives
- Implement immediately and review only the final code.
- Use separate planning and implementation PRs.
- Store decision history only in agent-local memory.

## Rationale
The chosen workflow allows approach-level correction before implementation
and keeps the plan, review discussion, changes, and rationale connected.

## Consequences
Additional planning and documentation effort.
Initially, the workflow depends on human checks and agent instructions,
not an automated enforcement controller.

## Scope and limitations
Applies initially to agent-assisted feature changes and consequential refactors.
Routine changes may use a lighter process with explicit maintainer agreement.

## Revisit when
Experience shows excessive overhead, unclear approvals, or repeated violations
that justify stronger automation.