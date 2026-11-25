"""Claude Code Stop hook: run pre-commit on every file after each turn."""

import json
import os
import shutil
import subprocess
import sys

if shutil.which("pre-commit") is None:
    sys.exit(0)

# --all-files: at Stop little is staged, so plain `pre-commit run` would check nothing.
result = subprocess.run(
    ["pre-commit", "run", "--all-files"],
    cwd=os.environ.get("CLAUDE_PROJECT_DIR"),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if result.returncode == 0:
    sys.exit(0)

# stop_hook_active: Claude is already retrying after a block, so hand over to the user.
if json.load(sys.stdin).get("stop_hook_active"):
    print(json.dumps({"systemMessage": f"pre-commit still failing:\n{result.stdout}"}))
    sys.exit(0)

print(result.stdout, file=sys.stderr)
sys.exit(2)
