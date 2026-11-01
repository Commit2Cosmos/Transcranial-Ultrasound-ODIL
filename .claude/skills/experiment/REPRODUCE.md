# Reproduce

Hand the user the command that regenerates an experiment's runs. The command and its rules are in `odil_wave/experiment/README.md`, "`--reproduce <experiment-id>`".

1. Read `docs/experiments/<id>.md`'s frontmatter for `commit` and `device`. Run `git fetch origin --tags` so the launch commit is local.
2. Ask which device the user will run on. Hand over `uv run python main.py --reproduce <id>`, adding `-o runtime.device=<device>` when it differs from the record's `device`.

Reproduce **only regenerates the runs**: it compares nothing, and you evaluate nothing and comment nowhere. The Protocol, spec and record stay unchanged. Whether a difference from the record matters is judged against the protocol's effect size, by the user.
