"""The run directory a standalone run writes and the entry point's policy for it
(location, clean tree, scratch runs, provenance, status), driven through
``main.main`` with the code's repository stood in by a throwaway git repo."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

import main
from conftest import TINY_RUN_CONFIG, git

# One outer iteration keeps each run to about a second.
QUICK = {"optimiser": {"n_iter": 1}}


def write_config(path: Path, user: dict) -> Path:
    path.write_text(yaml.safe_dump(user))
    return path


@pytest.fixture
def quick_config(tmp_path) -> Path:
    """The tiny run config cut to one iteration, outside the repo."""
    from odil_wave.experiment import deep_merge

    return write_config(tmp_path / "quick.yaml", deep_merge(TINY_RUN_CONFIG, QUICK))


def run_dirs(repo: Path) -> list:
    root = repo / "outputs" / "runs"
    return sorted(root.iterdir()) if root.exists() else []


def test_a_standalone_run_lands_in_a_timestamp_and_commit_directory(
    git_repo, quick_config
):
    assert main.main(["--config", str(quick_config)], repo_root=git_repo) == 0

    [run_dir] = run_dirs(git_repo)
    sha7 = git(git_repo, "rev-parse", "HEAD")[:7]
    assert re.fullmatch(rf"\d{{6}}-\d{{6}}_{sha7}", run_dir.name), run_dir.name
    assert (run_dir / "config_resolved.yaml").exists()


def test_a_dirty_tree_refuses_to_run(git_repo, quick_config, capsys):
    """An untracked, non-ignored file makes the tree dirty."""
    (git_repo / "untracked.py").write_text("x = 1\n")

    assert main.main(["--config", str(quick_config)], repo_root=git_repo) == 2
    assert "--scratch" in capsys.readouterr().err
    assert run_dirs(git_repo) == []


def test_a_scratch_run_is_marked_and_saves_its_diff(git_repo, quick_config):
    (git_repo / "tracked.txt").write_text("changed\n")
    argv = ["--scratch", "--config", str(quick_config)]

    assert main.main(argv, repo_root=git_repo) == 0

    [run_dir] = run_dirs(git_repo)
    assert run_dir.name.endswith("_scratch")
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["scratch"] is True
    assert meta["git"]["dirty"] is True
    diff = (run_dir / "git.diff").read_text()
    assert "-committed" in diff and "+changed" in diff


def test_a_scratch_runs_diff_rebuilds_its_tree_with_untracked_files(git_repo, tmp_path):
    """Applied to a clean checkout of the commit, ``git.diff`` gives back the
    tracked changes and the untracked, non-ignored files; ignored ones stay out."""
    from odil_wave.experiment import new_run_directory

    (git_repo / "tracked.txt").write_text("changed\n")
    (git_repo / "pkg").mkdir()
    (git_repo / "pkg" / "new.py").write_text("x = 1\n")
    (git_repo / "outputs").mkdir()
    (git_repo / "outputs" / "ignored.txt").write_text("not code\n")

    run = new_run_directory(scratch=True, out=tmp_path / "run", repo_root=git_repo)

    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(git_repo), str(clone))
    git(clone, "apply", str(run.path / "git.diff"))
    assert (clone / "tracked.txt").read_text() == "changed\n"
    assert (clone / "pkg" / "new.py").read_text() == "x = 1\n"
    assert not (clone / "outputs").exists()


def test_metadata_ties_the_run_to_its_commit_and_lockfile(git_repo, quick_config):
    assert main.main(["--config", str(quick_config)], repo_root=git_repo) == 0

    [run_dir] = run_dirs(git_repo)
    meta = json.loads((run_dir / "metadata.json").read_text())
    lock = hashlib.sha256((git_repo / "uv.lock").read_bytes()).hexdigest()
    assert meta["schema_version"] == 1
    assert meta["git"]["sha"] == git(git_repo, "rev-parse", "HEAD")
    assert meta["git"]["dirty"] is False
    assert meta["scratch"] is False
    assert meta["uv_lock_sha256"] == lock
    assert not (run_dir / "git.diff").exists()


def test_out_sets_the_run_directory(git_repo, quick_config, tmp_path):
    out = tmp_path / "elsewhere" / "run"
    argv = ["--config", str(quick_config), "--out", str(out)]

    assert main.main(argv, repo_root=git_repo) == 0

    assert (out / "config_resolved.yaml").exists()
    assert run_dirs(git_repo) == []


def test_out_refuses_a_directory_that_holds_files(git_repo, quick_config, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("not a run\n")
    argv = ["--config", str(quick_config), "--out", str(out)]

    assert main.main(argv, repo_root=git_repo) == 2
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_a_regenerated_run_has_a_byte_identical_resolved_config(
    git_repo, quick_config, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    argv = ["--config", str(quick_config), "--out", str(first)]
    assert main.main(argv, repo_root=git_repo) == 0
    resolved = first / "config_resolved.yaml"
    argv = ["--config", str(resolved), "--out", str(second)]
    assert main.main(argv, repo_root=git_repo) == 0

    assert (second / "config_resolved.yaml").read_bytes() == resolved.read_bytes()


@pytest.mark.parametrize("key", ["run.run_id=x", "run.output_root=elsewhere"])
def test_the_run_directory_is_not_a_config_value(quick_config, capsys, key):
    argv = ["--config", str(quick_config), "-o", key, "--dry-run"]
    assert main.main(argv) == 2
    assert "config error:" in capsys.readouterr().err


def run_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_summary.json").read_text())


def test_a_completed_run_summarises_its_final_metrics(git_repo, quick_config, tmp_path):
    out = tmp_path / "run"
    argv = ["--config", str(quick_config), "--out", str(out)]
    assert main.main(argv, repo_root=git_repo) == 0

    summary = run_summary(out)
    assert summary["status"] == "completed"
    assert summary["wall_s"] > 0
    assert summary["warnings"] == []
    assert summary["error"] is None
    metrics = summary["metrics_summary"]
    assert set(metrics) >= {"start_metrics", "final_metrics", "per_band"}
    assert metrics["final_metrics"]["iter"] == 1


def test_a_non_finite_loss_fails_the_run(git_repo, quick_config, tmp_path):
    """An infinite data weight makes the objective non-finite."""
    out = tmp_path / "run"
    argv = ["--config", str(quick_config), "-o", "loss.weights.data=1e400"]
    assert main.main(argv + ["--out", str(out)], repo_root=git_repo) == 1

    summary = run_summary(out)
    assert summary["status"] == "failed"
    assert "Traceback" in summary["error"]["traceback"]
    assert "non-finite" in summary["error"]["message"]
    # The tape keeps the rows written before the failure; the band is not saved.
    assert (out / "tape.csv").read_text().count("\n") >= 2
    assert list((out / "bands").iterdir()) == []


def test_ctrl_c_ends_the_run_as_interrupted(git_repo, tmp_path):
    """A KeyboardInterrupt after the first of two bands keeps that band and the
    tape, and the run reports itself interrupted."""
    from odil_wave.experiment import deep_merge, resolve_config, run_inverse
    from odil_wave.experiment.rundir import new_run_directory

    two_bands = {"continuation": {"bands": [{"frequencies_hz": [30e3]}] * 2}}
    cfg = resolve_config(deep_merge(deep_merge(TINY_RUN_CONFIG, QUICK), two_bands))
    out = tmp_path / "run"

    def ctrl_c(band_result, run_result):
        raise KeyboardInterrupt

    result = run_inverse(
        cfg, new_run_directory(out=out, repo_root=git_repo), on_band_end=ctrl_c
    )

    assert result.status == "interrupted"
    summary = run_summary(out)
    assert summary["status"] == "interrupted"
    assert len(summary["metrics_summary"]["per_band"]) == 1
    assert [p.name[:7] for p in (out / "bands").iterdir()] == ["band_00"]
    with open(out / "tape.csv", newline="") as f:
        assert {r["band_index"] for r in csv.DictReader(f)} == {"0"}
