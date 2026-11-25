"""The Claude Code Stop hook in .claude/hooks/, run as Claude Code runs it:
hook input as JSON on stdin, a stand-in ``pre-commit`` on PATH."""

import json
import subprocess
import sys

from conftest import REPO_ROOT

HOOK = REPO_ROOT / ".claude" / "hooks" / "pre_commit_stop.py"


def fake_pre_commit(tmp_path, exit_code):
    """A ``pre-commit`` that echoes its arguments and exits with ``exit_code``."""
    script = tmp_path / "pre-commit"
    script.write_text(
        f'#!/bin/sh\necho "ruff-check failed: pre-commit $*"\nexit {exit_code}\n'
    )
    script.chmod(0o755)
    return tmp_path


def run_hook(path, stop_hook_active=False):
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(
            {"hook_event_name": "Stop", "stop_hook_active": stop_hook_active}
        ),
        capture_output=True,
        text=True,
        env={"PATH": str(path)},
        cwd=REPO_ROOT,
    )


def test_passes_when_pre_commit_is_not_installed(tmp_path):
    assert run_hook(tmp_path).returncode == 0


def test_passes_when_pre_commit_passes(tmp_path):
    result = run_hook(fake_pre_commit(tmp_path, exit_code=0))
    assert result.returncode == 0
    assert result.stdout == ""


def test_blocks_the_stop_when_pre_commit_fails(tmp_path):
    result = run_hook(fake_pre_commit(tmp_path, exit_code=1))
    assert result.returncode == 2
    assert "ruff-check failed: pre-commit run --all-files" in result.stderr


def test_reports_to_the_user_when_the_retry_still_fails(tmp_path):
    result = run_hook(fake_pre_commit(tmp_path, exit_code=1), stop_hook_active=True)
    assert result.returncode == 0
    assert "ruff-check failed" in json.loads(result.stdout)["systemMessage"]
