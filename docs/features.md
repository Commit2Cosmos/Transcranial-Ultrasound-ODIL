# Feature register

This register tracks meaningful project capabilities, not individual
functions or every implementation detail.

Statuses:
- Proposed: under consideration.
- Experimental: implemented, but not established as supported.
- Supported: maintained and intended for use.
- Removed: previously implemented and subsequently removed.
- Rejected: considered but not adopted.

Rules:
- IDs are F-<number> (F-001, F-002, …), assigned sequentially and never
  reused or renumbered. Removed and Rejected entries stay in the table.
- Source lists the PRs that added the entry or changed its status, for
  example `#12, #30`. Use the issue number when no PR exists.
- Verification points to evidence, usually docs/reports/<task-id>.md or tests.
- Update an entry only when a change adds, removes, or materially alters a
  capability or its status. Routine changes need no entry.
- Entries become authoritative when merged; entries on unmerged branches are
  provisional.

| ID | Capability | Status | Code/docs | Verification | Decisions | Source |
|----|------------|--------|-----------|--------------|-----------|--------|

## Inventory status

Initial repository inventory has not yet been completed.

Missing entries do not imply that a capability is absent.
