"""``main.py --reproduce <experiment-id>``: regenerate an experiment at its
recorded commit, driven through ``main.main`` with the code's repository stood
in by a throwaway git repo (spec Seam 4).

The recorded commit carries a stub entry point that records its argv and
working directory and writes the target directory, so no test solves anything;
a stub ``uv`` on ``PATH`` records how it was called and runs that stub.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import main
from conftest import TINY_RUN_CONFIG, git
from odil_wave.experiment import deep_merge

EXP_ID = "0001-reg"
SPEC = f"configs/experiments/{EXP_ID}.yaml"

# The recorded commit's entry point: records how it was called, writes the
# target and exits with $STUB_EXIT.
STUB_MAIN = textwrap.dedent(
    """\
    import json, os, sys
    from pathlib import Path

    args = sys.argv[1:]
    Path(os.environ["STUB_LOG"]).write_text(
        json.dumps({"argv": args, "cwd": os.getcwd(), "commit": "recorded"})
    )
    out = Path(args[args.index("--out") + 1])
    out.mkdir(parents=True)
    (out / "experiment.json").write_text("{}")
    sys.exit(int(os.environ.get("STUB_EXIT", "0")))
    """
)

# HEAD's entry point: --reproduce must never run it.
HEAD_MAIN = 'raise SystemExit("ran HEAD, not the recorded commit")\n'

# ``uv`` stand-in: records its argv, then runs ``python <rest>`` like uv run.
STUB_UV = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import json, os, sys

    args = sys.argv[1:]
    with open(os.environ["UV_LOG"], "w") as f:
        json.dump(args, f)
    assert args[:3] == ["run", "--frozen", "python"], args
    os.execv(sys.executable, [sys.executable, *args[3:]])
    """
)


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        message,
    )
    return git(repo, "rev-parse", "HEAD")


def write_record(repo: Path, **frontmatter) -> Path:
    """``docs/experiments/<EXP_ID>.md`` with ``frontmatter``."""
    path = repo / "docs" / "experiments" / f"{EXP_ID}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n# {EXP_ID}\n"
    )
    return path


@pytest.fixture
def baseline_device() -> str:
    """The spec baseline's ``runtime.device``; parametrise to change it."""
    return "cpu"


@pytest.fixture
def cuda_available(monkeypatch) -> None:
    """Pretend this machine has a CUDA device (nothing runs on it)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def recorded(git_repo, tmp_path, monkeypatch, baseline_device) -> SimpleNamespace:
    """An experiment recorded at a commit that holds the stub entry point and
    the spec; HEAD has since moved on and holds the record."""
    spec = {
        "schema_version": 1,
        "id": EXP_ID,
        "issue": 61,
        "baseline": deep_merge(
            TINY_RUN_CONFIG, {"runtime": {"device": baseline_device}}
        ),
        "sweep": {"key": "loss.weights.reg", "values": [0.0, 1.0e-6]},
    }
    (git_repo / SPEC).parent.mkdir(parents=True)
    (git_repo / SPEC).write_text(yaml.safe_dump(spec, sort_keys=False))
    (git_repo / "main.py").write_text(STUB_MAIN)
    commit = commit_all(git_repo, "launch")

    (git_repo / "main.py").write_text(HEAD_MAIN)
    write_record(git_repo, id=EXP_ID, commit=commit, spec=SPEC, device="cpu")
    commit_all(git_repo, "record")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(STUB_UV)
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("UV_LOG", str(tmp_path / "uv.json"))
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "stub.json"))
    monkeypatch.delenv("STUB_EXIT", raising=False)
    return SimpleNamespace(
        repo=git_repo,
        commit=commit,
        uv_log=tmp_path / "uv.json",
        stub_log=tmp_path / "stub.json",
    )


def reproduce(recorded, *extra: str) -> int:
    return main.main(["--reproduce", EXP_ID, *extra], repo_root=recorded.repo)


def stub_call(recorded) -> dict:
    return json.loads(recorded.stub_log.read_text())


def test_reproduce_runs_the_recorded_commits_entry_point_in_a_worktree(recorded):
    assert reproduce(recorded) == 0

    target = recorded.repo / "outputs" / "experiments" / EXP_ID
    assert json.loads(recorded.uv_log.read_text())[:4] == [
        "run",
        "--frozen",
        "python",
        "main.py",
    ]
    call = stub_call(recorded)
    assert call["commit"] == "recorded"
    assert call["argv"] == ["--experiment", SPEC, "--out", str(target)]
    # A detached worktree at the recorded commit, removed afterwards.
    worktree = Path(call["cwd"])
    assert worktree != recorded.repo
    assert not worktree.exists()
    assert git(recorded.repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert (target / "experiment.json").is_file()


def test_an_existing_target_is_refused(recorded, capsys):
    target = recorded.repo / "outputs" / "experiments" / EXP_ID
    target.mkdir(parents=True)

    assert reproduce(recorded) == 2

    assert "exists" in capsys.readouterr().err
    assert not recorded.stub_log.exists()
    assert list(target.iterdir()) == []


def experiments(recorded) -> Path:
    return recorded.repo / "outputs" / "experiments"


@pytest.mark.usefixtures("cuda_available")
def test_the_device_override_reaches_the_runs_and_names_the_target(recorded):
    assert reproduce(recorded, "-o", "runtime.device=cuda") == 0

    target = experiments(recorded) / f"{EXP_ID}_cuda"
    assert stub_call(recorded)["argv"] == [
        "--experiment",
        SPEC,
        "--out",
        str(target),
        "-o",
        "runtime.device=cuda",
    ]
    assert target.is_dir()
    assert not (experiments(recorded) / EXP_ID).exists()


@pytest.mark.usefixtures("cuda_available")
@pytest.mark.parametrize("baseline_device", ["cuda"])
def test_without_an_override_the_target_follows_the_specs_device(recorded):
    assert reproduce(recorded) == 0

    target = experiments(recorded) / f"{EXP_ID}_cuda"
    assert stub_call(recorded)["argv"] == ["--experiment", SPEC, "--out", str(target)]


@pytest.mark.parametrize("baseline_device", ["cuda"])
def test_the_target_follows_the_device_the_runs_resolve_to(recorded, monkeypatch):
    # No CUDA here: the spec's cuda runs fall back to the CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert reproduce(recorded) == 0

    assert (experiments(recorded) / EXP_ID).is_dir()


@pytest.mark.parametrize(
    "extra",
    [
        ["-o", "optimiser.n_iter=2"],
        ["-o", "runtime.device=cpu", "-o", "loss.weights.reg=0"],
        ["-o", "runtime.device=tpu"],
        ["--config", "run.yaml"],
        ["--experiment", SPEC],
        ["--scratch"],
        ["--dry-run"],
        ["--probe"],
        ["--live"],
        ["--out", "elsewhere"],
    ],
)
def test_only_a_device_override_may_be_given(recorded, extra, capsys):
    assert reproduce(recorded, *extra) == 2

    err = capsys.readouterr().err
    assert "--reproduce" in err or "runtime.device" in err
    assert not recorded.stub_log.exists()
    assert not experiments(recorded).exists()


@pytest.mark.parametrize("code, expected", [(1, 1), (2, 2), (3, 1)])
def test_the_entry_points_exit_code_is_reproduces(
    recorded, monkeypatch, code, expected
):
    # 0 / 1 / 2 pass through; any other exit (a crash) is a run that failed.
    monkeypatch.setenv("STUB_EXIT", str(code))

    assert reproduce(recorded) == expected
    assert git(recorded.repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_without_uv_it_does_not_start(recorded, monkeypatch, tmp_path, capsys):
    only_git = tmp_path / "only_git"
    only_git.mkdir()
    (only_git / "git").symlink_to(shutil.which("git"))
    monkeypatch.setenv("PATH", str(only_git))

    assert reproduce(recorded) == 2

    assert "uv" in capsys.readouterr().err
    assert not recorded.stub_log.exists()
    assert git(recorded.repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_a_missing_record_does_not_start(recorded, capsys):
    assert main.main(["--reproduce", "0002-none"], repo_root=recorded.repo) == 2
    assert "0002-none" in capsys.readouterr().err


@pytest.mark.parametrize(
    "frontmatter, message",
    [
        ({"commit": "", "spec": SPEC}, "commit"),
        ({"commit": None, "spec": SPEC}, "commit"),
        ({"commit": "0" * 40, "spec": SPEC}, "commit"),
        ({"spec": "configs/experiments/0009-other.yaml"}, "spec"),
    ],
)
def test_a_record_that_cannot_be_reproduced_does_not_start(
    recorded, frontmatter, message, capsys
):
    write_record(recorded.repo, **{"commit": recorded.commit, **frontmatter})

    assert reproduce(recorded) == 2

    assert message in capsys.readouterr().err
    assert not recorded.stub_log.exists()
    assert not experiments(recorded).exists()
    assert git(recorded.repo, "worktree", "list", "--porcelain").count("worktree ") == 1
