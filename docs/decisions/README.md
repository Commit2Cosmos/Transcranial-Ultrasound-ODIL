# Decision records

Record consequential choices about capabilities, algorithms, architecture,
APIs, dependencies, platforms, and verification policy.

Do not create a decision record for every routine implementation detail.

Naming: D-<number>-<short-description>.md, numbered sequentially; numbers are
never reused.

Statuses:
- Proposed: awaiting a maintainer's decision.
- Accepted: explicitly accepted by a maintainer.
- Rejected: considered and explicitly not adopted.
- Superseded: replaced by a later Accepted record.

Acceptance authority:
- Only a maintainer (write access or above) accepts or rejects a record, by an
  explicit statement naming the decision ID, for example in a PR comment:
  "Accept D-004".
- Agents create records as Proposed. They change the status only to reflect
  such a statement, and record its link in the `Decided by` field.
- Approving a plan or merging a PR does not by itself accept a record.
  Acceptance is published when the record with its updated status is merged
  to the default branch.

Supersession:
- Preserve historical reasoning. When a decision changes, add a new record
  with `Supersedes: D-x`.
- In the old record, change only the `Status` (to Superseded) and
  `Superseded by` fields, in the same PR.
- Never rewrite the old record's rationale.

Evidence must distinguish:
- Observed results.
- Interpretation or judgement.
- Assumptions and unresolved uncertainty.

Retrospective records must be labelled retrospective. If the original
rationale cannot be established, state that it was not recorded.
