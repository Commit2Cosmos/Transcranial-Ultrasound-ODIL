"""Claude Code Stop hook: run pre-commit on the files the session changed."""

import json
import os
import shutil
import subprocess
import sys


def git_paths(*args, cwd):
    # -z: plain output quotes non-ASCII names, which exists() then misses.
    out = subprocess.run(
        ["git", *args, "-z"], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\0") if p]


event = json.load(sys.stdin)
# stdin cwd follows the session into a worktree; CLAUDE_PROJECT_DIR stays at main.
session_dir = event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")

if shutil.which("uv") is None:
    print(json.dumps({"systemMessage": "pre-commit check skipped: uv not on PATH"}))
    sys.exit(0)

root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    cwd=session_dir,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
# ls-files -o: `git diff HEAD` omits new files. From root: diff paths are root-relative.
changed = git_paths("diff", "--name-only", "HEAD", cwd=root) + git_paths(
    "ls-files", "-o", "--exclude-standard", cwd=root
)
files = [f for f in changed if os.path.exists(os.path.join(root, f))]
if not files:
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
