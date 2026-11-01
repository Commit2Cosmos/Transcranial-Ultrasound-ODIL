"""``main.py --experiment <spec>``: an experiment spec run into one experiment
directory, driven through ``main.main`` with the code's repository stood in by a
throwaway git repo (the spec committed in it)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import socket
from pathlib import Path

import pytest
import yaml

import main
from conftest import TINY_RUN_CONFIG, git
from odil_wave.experiment import deep_merge

# One outer iteration keeps each run to about a second.
BASELINE = deep_merge(TINY_RUN_CONFIG, {"optimiser": {"n_iter": 1}})
REG_SWEEP = {"key": "loss.weights.reg", "values": [0.0, 1.0e-6]}


def write_spec(repo: Path, exp_id: str = "0001-reg", *, commit=True, **fields) -> Path:
    """Write an experiment spec into ``repo``'s ``configs/experiments/`` and
    commit it; ``fields`` replace the spec's top-level keys."""
    spec = {
        "schema_version": 1,
        "id": exp_id,
        "issue": 61,
        "baseline": BASELINE,
        "sweep": REG_SWEEP,
        **fields,
    }
    path = repo / "configs" / "experiments" / f"{exp_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    if commit:
        commit_all(repo, f"spec {exp_id}")
    return path


def commit_all(repo: Path, message: str) -> None:
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


def read_csv(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_an_experiment_runs_every_value_into_one_directory(git_repo):
    spec = write_spec(git_repo)

    assert main.main(["--experiment", str(spec)], repo_root=git_repo) == 0

    exp = git_repo / "outputs" / "experiments" / "0001-reg"
    assert (exp / "spec.yaml").read_bytes() == spec.read_bytes()
    runs = sorted(p.name for p in (exp / "runs").iterdir())
    assert runs == ["00_reg-0", "01_reg-1e-6"]
    for name in runs:
        summary = json.loads((exp / "runs" / name / "run_summary.json").read_text())
        assert summary["status"] == "completed"
    record = json.loads((exp / "experiment.json").read_text())
    assert record["schema_version"] == 1
    assert record["commit"] == git(git_repo, "rev-parse", "HEAD")
    assert [r["status"] for r in record["runs"]] == ["completed", "completed"]

    rows = read_csv(exp / "summary.csv")
    assert [r["label"] for r in rows] == ["reg-0", "reg-1e-6"]
    assert [r["is_baseline"] for r in rows] == ["true", "false"]
    resolved = yaml.safe_load(
        (exp / "runs" / "01_reg-1e-6" / "config_resolved.yaml").read_text()
    )
    assert resolved["loss"]["weights"]["reg"] == 1e-6
    assert float(rows[1]["value"]) == 1e-6
    assert rows[1]["iter_total"] == "1"


def test_a_dirty_tree_refuses_to_launch(git_repo, capsys):
    spec = write_spec(git_repo)
    (git_repo / "untracked.py").write_text("x = 1\n")

    assert main.main(["--experiment", str(spec)], repo_root=git_repo) == 2
    assert "uncommitted" in capsys.readouterr().err
    assert not (git_repo / "outputs").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--config", "run.yaml"],
        ["-o", "optimiser.n_iter=2"],
        ["--scratch"],
        ["--live"],
    ],
)
def test_the_spec_alone_defines_the_runs(git_repo, extra, capsys):
    spec = write_spec(git_repo)

    assert main.main(["--experiment", str(spec), *extra], repo_root=git_repo) == 2
    assert "--experiment" in capsys.readouterr().err
    assert not (git_repo / "outputs").exists()


def dry_run(repo: Path, spec: Path) -> int:
    return main.main(["--experiment", str(spec), "--dry-run"], repo_root=repo)


def test_a_dry_run_resolves_every_run_and_writes_nothing(git_repo, capsys):
    """Allowed before the spec is committed, so a spec is checked as it is written."""
    spec = write_spec(git_repo, commit=False)

    assert dry_run(git_repo, spec) == 0

    out = capsys.readouterr().out
    assert "00_reg-0" in out and "(baseline)" in out and "01_reg-1e-6" in out
    assert not (git_repo / "outputs").exists()


SCHEDULES = [
    [{"frequencies_hz": [40e3]}],
    [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}],
]


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": None}, "schema_version"),
        ({"id": "0001-other"}, "filename"),
        ({"sweep": {"values": [0.0]}}, "sweep.key"),
        ({"sweep": {"key": ["a", "b"], "values": [0.0]}}, "sweep.key"),
        ({"sweep": {"key": "loss.weights.reg", "values": []}}, "values"),
        ({"sweep": {"key": "continuation.bands", "values": SCHEDULES}}, "labels"),
        ({"sweep": {**REG_SWEEP, "labels": ["only-one"]}}, "labels"),
        ({"sweep": {"key": "loss.weights.reg", "values": [1e-7, 1e-6]}}, "baseline"),
        (
            {"sweep": {**REG_SWEEP, "values": [0.0, 0], "labels": ["a", "b"]}},
            "baseline",
        ),
        ({"sweep": {**REG_SWEEP, "values": [1e-6, 1.0e-6, 0.0]}}, "unique"),
        ({"sweep": {**REG_SWEEP, "labels": ["a", "b/c"]}}, "b/c"),
        ({"sweep": {"key": "optimiser.n_iter", "values": [1, -1]}}, "01_n-iter--1"),
        ({"question": "?"}, "question"),
        ({"issue": None}, "issue"),
        ({"issue": "sixty-one"}, "issue"),
    ],
)
def test_an_invalid_spec_does_not_start(git_repo, fields, message, capsys):
    spec = write_spec(git_repo, commit=False, **fields)

    assert dry_run(git_repo, spec) == 2
    assert message in capsys.readouterr().err


def test_labels_name_list_and_dict_values(git_repo, capsys):
    sweep = {
        "key": "continuation.bands",
        "values": SCHEDULES,
        "labels": ["one-band", "two-bands"],
    }
    spec = write_spec(git_repo, commit=False, sweep=sweep)

    assert dry_run(git_repo, spec) == 0
    out = capsys.readouterr().out
    assert "00_one-band" in out and "01_two-bands" in out


def test_swept_dicts_deep_merge_into_the_baseline(git_repo, capsys):
    """Sweeping ``loss`` with ``{weights: {reg: 0}}`` keeps the baseline's other
    weights, so that value is the baseline."""
    sweep = {
        "key": "loss",
        "values": [{"weights": {"reg": 0.0}}, {"weights": {"reg": 1e-6}}],
        "labels": ["none", "tv"],
    }
    spec = write_spec(git_repo, commit=False, sweep=sweep)

    assert dry_run(git_repo, spec) == 0
    assert "00_none" in capsys.readouterr().out.split("(baseline)")[0]


@pytest.mark.parametrize(
    "taken", ["configs/experiments/0001-old.yaml", "docs/experiments/0001-old.md"]
)
def test_an_id_number_used_by_another_spec_or_record_does_not_start(
    git_repo, taken, capsys
):
    (git_repo / taken).parent.mkdir(parents=True, exist_ok=True)
    (git_repo / taken).write_text("")
    spec = write_spec(git_repo, commit=False)

    assert dry_run(git_repo, spec) == 2
    assert "0001-old" in capsys.readouterr().err


def test_the_experiments_own_record_does_not_block_it(git_repo):
    """``--reproduce`` re-runs a spec whose record exists."""
    record = git_repo / "docs" / "experiments" / "0001-reg.md"
    record.parent.mkdir(parents=True)
    record.write_text("")
    spec = write_spec(git_repo, commit=False)

    assert dry_run(git_repo, spec) == 0


# An infinite data weight makes the objective non-finite, failing that run.
FAILING_FIRST = {"key": "loss.weights.data", "values": [1e400, 10.0]}


def test_a_failed_run_does_not_stop_the_others(git_repo):
    spec = write_spec(git_repo, sweep=FAILING_FIRST)

    assert main.main(["--experiment", str(spec)], repo_root=git_repo) == 1

    exp = git_repo / "outputs" / "experiments" / "0001-reg"
    record = json.loads((exp / "experiment.json").read_text())
    assert record["status"] == "failed"
    assert [r["status"] for r in record["runs"]] == ["failed", "completed"]
    rows = read_csv(exp / "summary.csv")
    assert [(r["label"], r["status"]) for r in rows] == [
        ("data-inf", "failed"),
        ("data-10", "completed"),
    ]
    assert rows[0]["ssim_brain"] == "" and rows[1]["ssim_brain"] != ""


def test_the_experiment_record_ties_it_to_its_commit_and_spec(git_repo, tmp_path):
    spec = write_spec(git_repo)
    out = tmp_path / "experiment"

    argv = ["--experiment", str(spec), "--out", str(out)]
    assert main.main(argv, repo_root=git_repo) == 0

    text = (out / "experiment.json").read_text()
    record = json.loads(text)
    assert record["id"] == "0001-reg"
    assert record["issue"] == 61
    assert record["spec"] == "configs/experiments/0001-reg.yaml"
    assert record["spec_sha256"] == hashlib.sha256(spec.read_bytes()).hexdigest()
    assert record["device"] == "cpu"
    assert record["started_utc"] <= record["finished_utc"]
    assert [r["dir"] for r in record["runs"]] == ["runs/00_reg-0", "runs/01_reg-1e-6"]
    assert str(tmp_path) not in text and str(git_repo) not in text
    assert socket.gethostname() not in text


def test_a_rerun_resumes_redoing_only_runs_that_did_not_complete(git_repo):
    spec = write_spec(git_repo, sweep=FAILING_FIRST)
    argv = ["--experiment", str(spec)]
    assert main.main(argv, repo_root=git_repo) == 1
    exp = git_repo / "outputs" / "experiments" / "0001-reg"
    for run in ("00_data-inf", "01_data-10"):
        (exp / "runs" / run / "marker").write_text("from the first launch\n")
    started = json.loads((exp / "experiment.json").read_text())["started_utc"]

    assert main.main(argv, repo_root=git_repo) == 1

    assert not (exp / "runs" / "00_data-inf" / "marker").exists()
    assert (exp / "runs" / "01_data-10" / "marker").exists()
    record = json.loads((exp / "experiment.json").read_text())
    assert [r["status"] for r in record["runs"]] == ["failed", "completed"]
    assert record["started_utc"] == started
    assert len(read_csv(exp / "summary.csv")) == 2


def test_a_resume_at_another_commit_is_refused(git_repo, capsys):
    spec = write_spec(git_repo)
    argv = ["--experiment", str(spec)]
    assert main.main(argv, repo_root=git_repo) == 0
    exp = git_repo / "outputs" / "experiments" / "0001-reg"
    before = (exp / "experiment.json").read_bytes()
    (git_repo / "tracked.txt").write_text("changed\n")
    commit_all(git_repo, "change the code")

    assert main.main(argv, repo_root=git_repo) == 2

    assert "commit" in capsys.readouterr().err
    assert (exp / "experiment.json").read_bytes() == before


def test_a_resume_of_a_changed_spec_is_refused(git_repo, tmp_path, capsys):
    """The spec is compared by hash, whatever commit it is read at."""
    spec = write_spec(git_repo)
    out = tmp_path / "experiment"
    argv = ["--experiment", str(spec), "--out", str(out)]
    assert main.main(argv, repo_root=git_repo) == 0
    record = json.loads((out / "experiment.json").read_text())
    record["spec_sha256"] = "0" * 64
    (out / "experiment.json").write_text(json.dumps(record))

    assert main.main(argv, repo_root=git_repo) == 2
    assert "spec" in capsys.readouterr().err


def test_a_directory_that_is_not_this_experiments_is_refused(git_repo, tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("not an experiment\n")
    argv = ["--experiment", str(write_spec(git_repo)), "--out", str(out)]

    assert main.main(argv, repo_root=git_repo) == 2
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_a_spec_outside_the_repository_does_not_launch(git_repo, tmp_path):
    spec = write_spec(tmp_path, commit=False)

    assert main.main(["--experiment", str(spec)], repo_root=git_repo) == 2
    assert not (git_repo / "outputs").exists()


def test_each_run_is_post_processed_when_it_ends(git_repo, tmp_path):
    """Including a failed run: post-processing shows what it saved."""
    from odil_wave.experiment.launch import launch

    seen = []

    def postprocess_run(run_dir: Path) -> None:
        seen.append((run_dir.name, (run_dir / "run_summary.json").exists()))

    spec = write_spec(git_repo, sweep=FAILING_FIRST)
    out = tmp_path / "experiment"
    code = launch(spec, out=out, repo_root=git_repo, postprocess_run=postprocess_run)

    assert code == 1
    assert seen == [("00_data-inf", True), ("01_data-10", True)]


def test_ctrl_c_stops_the_experiment_and_a_rerun_resumes_it(git_repo, tmp_path):
    from odil_wave.experiment.launch import launch

    def ctrl_c(run_dir: Path) -> None:
        raise KeyboardInterrupt

    spec = write_spec(git_repo)
    out = tmp_path / "experiment"
    assert launch(spec, out=out, repo_root=git_repo, postprocess_run=ctrl_c) == 1

    record = json.loads((out / "experiment.json").read_text())
    assert record["status"] == "interrupted"
    assert [r["status"] for r in record["runs"]] == ["interrupted", "pending"]
    assert not (out / "runs" / "01_reg-1e-6").exists()

    assert launch(spec, out=out, repo_root=git_repo) == 0
    record = json.loads((out / "experiment.json").read_text())
    assert [r["status"] for r in record["runs"]] == ["completed", "completed"]


def test_postprocessing_an_experiment_reprocesses_every_run(git_repo, tmp_path):
    from odil_wave.experiment.launch import postprocess_experiment

    out = tmp_path / "experiment"
    argv = ["--experiment", str(write_spec(git_repo)), "--out", str(out)]
    assert main.main(argv, repo_root=git_repo) == 0
    expected = (out / "summary.csv").read_text()
    (out / "summary.csv").unlink()

    seen = []
    postprocess_experiment(out, postprocess_run=lambda d: seen.append(d.name))

    assert seen == ["00_reg-0", "01_reg-1e-6"]
    assert (out / "summary.csv").read_text() == expected


def test_postprocess_flag_on_an_experiment_redraws_every_run(git_repo, tmp_path):
    out = tmp_path / "experiment"
    argv = ["--experiment", str(write_spec(git_repo)), "--out", str(out)]
    assert main.main(argv, repo_root=git_repo) == 0
    runs = sorted((out / "runs").iterdir())
    for run_dir in runs:
        shutil.rmtree(run_dir / "figures")
    (out / "summary.csv").unlink()

    assert main.main(["--postprocess", str(out)]) == 0

    for run_dir in runs:
        assert (run_dir / "figures" / "convergence.png").is_file()
    assert [r["label"] for r in read_csv(out / "summary.csv")] == ["reg-0", "reg-1e-6"]
