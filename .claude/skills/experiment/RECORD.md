# Stage 3: Record

Draft the experiment record, get the user's sign-off, and commit it with its results on the experiment's `exp/*` branch. Record conventions (frontmatter, sections, Validity formats) are in `docs/experiments/README.md` and `TEMPLATE.md`.

## 1. Check the runs

Check out the experiment's branch. Find the experiment directory, `outputs/experiments/NNNN-slug[_<device>]/`; if it isn't on this machine, ask the user where it is. Read `experiment.json`:

- `commit` must equal `git rev-parse launch/NNNN-slug^{commit}`. If it doesn't, stop and ask the user: the runs didn't use the launch commit.
- `status: running`, or any run `pending`: the experiment hasn't finished. Hand back the launch command (it resumes) and stop.

## 2. Settle the outcome

- Every run `completed`: the outcome is **answered** if the data decides the Hypothesis, **inconclusive** if it doesn't.
- A run `failed` or `interrupted`: the user picks. **answered** when the question is still decidable (missing runs go under Deviations); **inconclusive** when the runs completed but don't decide it; **aborted** for a brief record (the question, what happened, why it stopped). A fix that needs a code change can't resume, since resume works only at the same commit: the id stays used, and the retry is a new experiment whose Context names this one.

## 3. Draft

Copy `TEMPLATE.md` to `docs/experiments/NNNN-slug.md` and draft every part:

- **Frontmatter** from `experiment.json` (`commit`, `device`, `variable` = `sweep_keys`, comma-separated), the Protocol issue (`issue`, `title`, `baseline`) and the outcome; `experiment_dir` is the directory's repo-relative path.
- **Question, Hypothesis, Design, Assumptions and limitations**: copied from the Protocol issue body.
- **Deviations from protocol**: failed or missing runs, a device other than the spec's, and any edit to the issue body after the launch comment. GitHub's edit history shows those: `gh api graphql -f query='{repository(owner:"Commit2Cosmos",name:"Transcranial-Ultrasound-ODIL"){issue(number:<n>){lastEditedAt}}}'` later than the comment means an edit. "None." otherwise.
- **Results**: the `summary.csv` table (label, the swept-key columns, `is_baseline`, status, the metrics the conclusions weigh, `n_evaluations`, `wall_s`), the baseline run's first start row (first row of `runs/<baseline dir>/tape.csv`) as reference, then up to 3 PNGs from the experiment's `figures/`.
- **Conclusions supported**: a judgement over all reported metrics, each bullet citing a row, column or figure. Where the Protocol names an expected effect size, say whether it was met.
- **Conclusions not supported**, **Surprises**, **Follow-ups** (each with a proposed Baseline; for a chain, the next link comes from the Chain section).

## 4. Sign-off

Show the user **Results, Conclusions supported, Conclusions not supported, Surprises and Follow-ups**, and revise until they approve each one in conversation. PR review is the second look.

Done when the user has approved all five.

## 5. Record attachments

Copy `experiment.json`, `summary.csv` and the PNGs the Results cite (at most 3, standard filenames) into `docs/experiments/NNNN-slug/`, and point the Results' image links there.

## 6. Index and commit

Run `uv run python -m odil_wave.experiment.bank`, then `uv run python -m odil_wave.experiment.bank --check` until it exits 0. Commit the record, its directory and the README as `exp: record NNNN-slug (#<issue>)`, and push the branch.

## 7. Follow-ups and next stage

Ask which Follow-ups, including a chain's next link, to open as `experiment` issues now. Seed each with only its Question and Baseline (under Design), plus a Context line naming this record; its Protocol is finished in a later `/experiment <issue#>` session.

Then ask whether this branch is finished, i.e. no more experiments planned on it. If yes, continue with [FINISH.md](FINISH.md).
