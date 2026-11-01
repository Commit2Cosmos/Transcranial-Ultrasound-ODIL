# Stage 2: Spec, probe, launch

Turn the Protocol into a committed, tagged experiment spec, post the launch comment, and hand the user the command. The spec format and every flag are in `odil_wave/experiment/README.md`, "Experiments".

Before starting, re-check the Protocol's two gates (one variable, config-selectable; see [PROTOCOL.md](PROTOCOL.md) §3). If either fails, go back to the Protocol stage.

## 1. Reserve the id

The experiment id is `NNNN-slug`: the next number not used in `configs/experiments/` on `main` or on any `exp/*` branch, so parallel experiments can't collide. Launch tags count too, which covers a dropped branch since deleted:

```bash
git fetch origin --tags
{ for ref in origin/main $(git for-each-ref --format='%(refname:short)' refs/heads/exp/ refs/remotes/origin/exp/); do
    git ls-tree --name-only "$ref" configs/experiments/
  done; git tag -l 'launch/*'; } \
  | sed -nE 's|.*/([0-9]{4})-.*|\1|p' | sort -n | tail -1 | awk '{n=$1} END {printf "%04d\n", n+1}'
```

The slug is a few lowercase kebab-case words from the Question. The launch's own `id` check only sees the current tree, so this rule is the one that matters.

## 2. Branch

Create `exp/NNNN-slug` from `origin/main`. A later related experiment (a chain link or follow-up) may join an existing `exp/*` branch, named after its first experiment, only when the user says so; an unrelated one gets its own branch.

If the Design needs a **feature under test**, build it here, test-first, behind the config key the Design names, with a default that keeps current behaviour. Commit it before the spec. A feature found ineffective then never has to reach `main`.

## 3. Write the spec

`configs/experiments/NNNN-slug.yaml` with `schema_version: 1`, `id`, `issue` (the Protocol issue), an inline `baseline` holding only values that differ from the default config, and one `sweep` with the Design's key, values (in the Design's order) and labels. Exactly one value resolves to the Baseline.

Validate with `uv run python main.py --experiment configs/experiments/NNNN-slug.yaml --dry-run` until it exits 0.

## 4. Probe

`uv run python main.py --experiment configs/experiments/NNNN-slug.yaml --probe` times a Probe run and prints a per-run and total estimate for this machine and device. Show the user both. The estimate is advisory, with no threshold: if the user finds it too long, revise the Design (edit the issue body; it is not frozen yet), update the spec, and dry-run and probe again.

Done when the user accepts the estimate.

## 5. Commit, tag, push

Commit the spec as `exp: launch NNNN-slug (#<issue>)`; `git status --porcelain` must then be empty. Tag that commit and push both:

```bash
git tag -a launch/NNNN-slug -m "Launch NNNN-slug (#<issue>)"
git push -u origin exp/NNNN-slug launch/NNNN-slug
```

The tag keeps the launch commit fetchable after the branch is deleted, squashed or rebased, so `--reproduce` always finds the record's `commit`.

## 6. Launch comment

Post on the Protocol issue:

```markdown
Launched `NNNN-slug`

- commit: `<full sha>` (tag `launch/NNNN-slug`)
- spec: `configs/experiments/NNNN-slug.yaml`
- command: `uv run python main.py --experiment configs/experiments/NNNN-slug.yaml`
- estimate: <per run>; total <h:mm:ss> (probe on <device>)

This comment freezes the Protocol. Later changes go in the record's Deviations from protocol.
```

From here the Protocol is **frozen**: any change goes only in the record's *Deviations from protocol*, and GitHub's edit history is the audit trail.

## 7. Hand over

Give the user the command (add `-o runtime.device=<device>` if they will run on another device than the spec's) and tell them:

- Run it at the launch commit; re-running the same command resumes failed or interrupted runs, but only at that commit.
- When the runs finish, `/experiment NNNN-slug` writes the record. If they run elsewhere, bring the experiment directory back under `outputs/experiments/` first.

The session ends here.
