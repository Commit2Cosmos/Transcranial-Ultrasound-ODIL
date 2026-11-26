"""The Claude Code Stop hook in .claude/hooks/."""

import json
import shutil
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

HOOK = REPO_ROOT / ".claude" / "hooks" / "pre_commit_stop.py"


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for name in ("unchanged.py", "edited.py", "deleted.py"):
        (repo / name).write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def change_files(repo):
    (repo / "edited.py").write_text("x = 2\n")
    (repo / "deleted.py").unlink()
    (repo / "untracked.py").write_text("x = 3\n")
    (repo / "ünïcode.py").write_text("x = 3\n")


def checked_files(result):
    """The files the fake ``uv`` was asked to check, from the hook's stderr."""
    line = next(s for s in result.stderr.splitlines() if "ruff-check failed" in s)
    command, files = line.split(" --files ")
    assert command == "ruff-check failed: uv run --frozen pre-commit run"
    return set(files.split())


def bin_dir(tmp_path, uv_exit_code=None):
    """A PATH directory holding ``git`` and, unless ``uv_exit_code`` is None,
    a ``uv`` that prints a failure line naming its arguments and exits with it."""
    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    (path_dir / "git").symlink_to(shutil.which("git"))
    if uv_exit_code is not None:
        uv = path_dir / "uv"
        uv.write_text(
            f'#!/bin/sh\necho "ruff-check failed: uv $*"\nexit {uv_exit_code}\n'
        )
        uv.chmod(0o755)
    return path_dir


def run_hook(path_dir, session_dir, stop_hook_active=False, project_dir=None):
    payload = {"hook_event_name": "Stop", "stop_hook_active": stop_hook_active}
    if session_dir is not None:
        payload["cwd"] = str(session_dir)
    env = {"PATH": str(path_dir)}
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )


def test_blocks_the_stop_on_the_changed_files_when_pre_commit_fails(tmp_path, repo):
    change_files(repo)
    result = run_hook(bin_dir(tmp_path, uv_exit_code=1), session_dir=repo)
    assert result.returncode == 2
    assert checked_files(result) == {"edited.py", "untracked.py", "ünïcode.py"}


def test_passes_on_a_clean_tree_without_running_pre_commit(tmp_path, repo):
    result = run_hook(bin_dir(tmp_path, uv_exit_code=1), session_dir=repo)
    assert result.returncode == 0
    assert result.stdout == ""


def test_tells_the_user_the_check_was_skipped_when_uv_is_missing(tmp_path, repo):
    result = run_hook(bin_dir(tmp_path), session_dir=repo)
    assert result.returncode == 0
    assert "skipped" in json.loads(result.stdout)["systemMessage"]


def test_checks_the_whole_tree_from_a_subdirectory(tmp_path, repo):
    change_files(repo)
    (repo / "sub").mkdir()
    (repo / "sub" / "new.py").write_text("x = 4\n")
    result = run_hook(bin_dir(tmp_path, uv_exit_code=1), session_dir=repo / "sub")
    assert result.returncode == 2
    assert checked_files(result) == {
        "edited.py",
        "untracked.py",
        "ünïcode.py",
        "sub/new.py",
    }


def test_passes_when_pre_commit_passes(tmp_path, repo):
    change_files(repo)
    result = run_hook(bin_dir(tmp_path, uv_exit_code=0), session_dir=repo)
    assert result.returncode == 0
    assert result.stdout == ""


def test_falls_back_to_the_project_dir_when_stdin_has_no_cwd(tmp_path, repo):
    change_files(repo)
    result = run_hook(
        bin_dir(tmp_path, uv_exit_code=1), session_dir=None, project_dir=repo
    )
    assert result.returncode == 2
    assert "untracked.py" in checked_files(result)


def test_reports_to_the_user_when_the_retry_still_fails(tmp_path, repo):
    change_files(repo)
    result = run_hook(
        bin_dir(tmp_path, uv_exit_code=1), session_dir=repo, stop_hook_active=True
    )
    assert result.returncode == 0
    assert "ruff-check failed" in json.loads(result.stdout)["systemMessage"]
