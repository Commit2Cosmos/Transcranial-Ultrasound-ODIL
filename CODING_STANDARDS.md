# Coding Standards

Read during review. The pre-commit hooks enforce formatting and lint; these are the judgement calls it cannot.

## Merge-ready PR

A PR the reviewer has nothing left to ask of: it meets every criterion below. The agent claims it, the user verifies it and merges. "Mergeable" asked of an agent means this, not GitHub's conflict-free sense.

1. **Floor.** CI is green (tests, ruff, private-names) and the branch has no conflicts with `main`. Everything below is what CI cannot check.
2. **Meets its acceptance criteria.** Every Acceptance criterion of the PR's issue holds. A PR with no issue is judged against the intent stated in its PR body.
3. **Out-of-scope changes are listed.** Refactors and fixes the issue did not ask for are allowed, but the PR body lists each with a one-line reason. An unlisted one fails this criterion, whatever its size.
4. **Follows these standards.** Every other section of this file.
5. **Docs are thin and true.** Docstrings and comments the PR adds or changes follow [Thin docs](#thin-docs). Any statement the PR made false, in any doc (`README.md` and `info.md` included), is fixed or deleted. Docs the PR did not affect are not its job.
6. **Glossary and ADRs are updated.** A PR that introduces, renames or changes the meaning of a domain term updates `GLOSSARY.md`. A decision that is hard to reverse, surprising without context, and a real trade-off gets an ADR. A PR contradicting an ADR updates it.
7. **Tests cover the change.** New behaviour has a test through the public interface. A bug fix has a test that failed before the fix. Tests assert behaviour, not implementation: a refactor that keeps behaviour must not break them.
8. **No leftovers.** No commented-out code, debug output, unused config keys or parameters, compatibility shims for callers that no longer exist, or TODOs without an issue number.
9. **One way to do it.** The PR extends the existing mechanism rather than adding a parallel one. If it replaces a mechanism, the old one is removed in the same PR, not left beside the new.
10. **Fails loudly.** Invalid config or input raises where it enters. No silent fallbacks, defaults substituted for bad values, or swallowed exceptions: here they produce a plausible wrong inversion rather than an error.
11. **Results changes are declared.** If the PR changes what the default config computes (any run metric could differ for the same seed), the PR body says so and says what changed.
12. **No later change would have to undo it.** The catch-all for maintainability: a later change in the same area would not first have to undo or work around this PR. To fail a PR on this, name the concrete later change that is obstructed (e.g. "adding a third preconditioner option means editing this `if` chain instead of adding an object"). If none can be named, it passes.

### Answering "is this PR merge-ready?"

Give a yes or no, then each failed criterion by number with `file:line` evidence. Check CI and conflict status live with `gh pr checks` and `gh pr view --json mergeable`; do not assume them. Any failed criterion is a no: there is no "mostly" or "yes, with nits".

## Thin docs

Code is the source of truth for how it works. Docstrings and comments exist to help a reader find the right code, not to repeat it.

**The test:** a statement that would go stale after an implementation change that keeps the interface does not belong.

A docstring is read to decide whether to read the code:

- **Module docstring** (every module): what the module is responsible for, in glossary terms; what a reader might expect to find here that lives elsewhere, and where; which names are its entry points.
- **Public class or function docstring**: one or two lines on its role, meaning what it is for and when a caller picks it over a sibling. Cite the ADR if one governs it.
- **Units and tensor shapes**: state them where the signature cannot (`(n_shots, nx, ny)` complex pressure, Pa). They change only when the interface does.
- **Private helpers**: no docstring, unless the name cannot carry the role.

A comment gives only a reason the code cannot show and that is too small for an ADR ("plain python, not pytest: `conftest.py` imports torch").

Never: Args/Returns sections restating the signature, a walkthrough of the body, a restatement of what a line does, or a definition of a glossary term. Decision-scale "why" belongs in an ADR; what a term means belongs in `GLOSSARY.md`.

## Glossary terms

`GLOSSARY.md` terms, and its _Avoid_ lists, govern prose: docs, docstrings, comments, config keys, log and metadata keys. Code identifiers may name the **role** an element plays rather than the element: `receivers` (the transducers recording a shot), `src_idx` (the transmitting transducer's shot index), `u_recv`. Use the glossary term when the identifier means the element itself (`transducer_ij`, `n_transducers`).
