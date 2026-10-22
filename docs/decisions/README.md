# Decision records

Record consequential choices about capabilities, algorithms, architecture,
APIs, dependencies, platforms, and verification policy.

Do not create a decision record for every routine implementation detail.

Naming: D-<number>-<short-description>.md

Statuses:
- Proposed
- Accepted
- Rejected
- Superseded

Agents create records with status Proposed. A record becomes Accepted only
when the maintainer explicitly approves that decision; the status is then
updated on the task branch before merge. Approving a plan or merging a PR
does not by itself accept the decisions it proposes.

Preserve historical reasoning. When a decision changes, add a new record
and link the old and new records rather than replacing the original rationale:
- The new record names the old one in `Supersedes:` and explains in its
  Context or Rationale what evidence or requirement changed.
- When the new record is accepted, update the old record's header in the
  same PR: set `Status: Superseded` and fill `Superseded by:`. Do not edit
  the rest of the old record.
- If the new record stays Proposed or is rejected, leave the old record
  unchanged.
- For partial supersession, qualify the link, e.g.
  `Supersedes: D-003 (default-choice part only)` and
  `Superseded in part by: D-007`, and keep the old record Accepted.

Evidence must distinguish:
- Observed results.
- Interpretation or judgement.
- Assumptions and unresolved uncertainty.