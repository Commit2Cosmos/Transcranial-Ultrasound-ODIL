"""Claude Code Stop hook: run pre-commit on the files the session changed."""

import json
import os
import shutil
import subprocess
import sys

event = json.load(sys.stdin)
# stdin cwd follows the session into a worktree; CLAUDE_PROJECT_DIR stays at main.
session_dir = event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")


def git_lines(*args, cwd):
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout
    return out.splitlines()


# The tree's root: git diff prints root-relative paths, the session may be below it.
(root,) = git_lines("rev-parse", "--show-toplevel", cwd=session_dir)


# ls-files -o: new files, which `--all-files` would miss; exists() drops deletions.
changed = git_lines("diff", "--name-only", "HEAD", cwd=root) + git_lines(
    "ls-files", "-o", "--exclude-standard", cwd=root
)
files = [f for f in changed if os.path.exists(os.path.join(root, f))]
if not files:
    sys.exit(0)

if shutil.which("uv") is None:
    print(json.dumps({"systemMessage": "pre-commit check skipped: uv not on PATH"}))
    sys.exit(0)

result = subprocess.run(
    ["uv", "run", "--frozen", "pre-commit", "run", "--files", *files],
    cwd=root,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if result.returncode == 0:
    sys.exit(0)

# stop_hook_active: Claude is already retrying after a block, so hand over to the user.
if event.get("stop_hook_active"):
    print(json.dumps({"systemMessage": f"pre-commit still failing:\n{result.stdout}"}))
    sys.exit(0)

print(result.stdout, file=sys.stderr)
sys.exit(2)
