"""``main.py --experiment <spec> --probe``: a Probe run of the baseline's first
band, estimating the experiment's runtime, driven through ``main.main`` with
the code's repository stood in by a throwaway git repo."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest
import yaml

import main
from conftest import TINY_RUN_CONFIG, git
from odil_wave.experiment import deep_merge

# Two bands of 4 iterations by default; the sweep sets n_iter to 4 or 10, so
# the runs take 2 x 4 = 8 and 2 x 10 = 20 iterations, 28 in all.
BASELINE = deep_merge(
    TINY_RUN_CONFIG,
    {
        "continuation": {
            "bands": [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]
        },
        "optimiser": {"n_iter": 4, "log_every": 2},
    },
)
SWEEP = {"key": "optimiser.n_iter", "values": [4, 10]}


def write_spec(repo: Path, exp_id: str = "0001-n-iter") -> Path:
    """An experiment spec in ``repo``'s ``configs/experiments/``, not committed."""
    spec = {
        "schema_version": 1,
        "id": exp_id,
        "issue": 61,
        "baseline": BASELINE,
        "sweep": SWEEP,
    }
    path = repo / "configs" / "experiments" / f"{exp_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def probe(repo: Path, spec: Path) -> int:
    return main.main(["--experiment", str(spec), "--probe"], repo_root=repo)


def test_a_probe_runs_three_iterations_of_the_baselines_first_band(git_repo):
    """On a dirty tree (the spec isn't even committed), writing only its probe
    directory."""
    spec = write_spec(git_repo)
    status_before = git(git_repo, "status", "--porcelain")

    assert probe(git_repo, spec) == 0

    assert git(git_repo, "status", "--porcelain") == status_before
    outputs = git_repo / "outputs"
    assert [p.name for p in outputs.iterdir()] == ["probes"]
    (probe_dir,) = (outputs / "probes").iterdir()
    assert re.fullmatch(r"\d{6}-\d{6}", probe_dir.name)

    resolved = yaml.safe_load((probe_dir / "config_resolved.yaml").read_text())
    (band,) = resolved["continuation"]["bands"]
    assert band["frequencies_hz"] == [30e3]
    assert band["n_iter"] == 3
    with open(probe_dir / "tape.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["iter_total"] == "3"


ESTIMATE = re.compile(r"^\s*(\S+)\s*: (\d+) iterations\s+~\s*([\d.]+) s\b", re.M)


def test_a_probe_estimates_each_run_and_the_experiment(git_repo, capsys):
    """s/iter x each run's total iterations, summed for the experiment."""
    spec = write_spec(git_repo)

    assert probe(git_repo, spec) == 0

    out = capsys.readouterr().out
    s_per_iter = float(re.search(r"([\d.]+) s/iter", out).group(1))
    assert s_per_iter > 0
    estimates = {name: (int(n), float(s)) for name, n, s in ESTIMATE.findall(out)}
    assert {name: n for name, (n, _) in estimates.items()} == {
        "00_n-iter-4": 8,
        "01_n-iter-10": 20,
        "experiment": 28,
    }
    for n, seconds in estimates.values():
        assert abs(seconds - s_per_iter * n) <= 0.05 + 1e-3 * seconds


@pytest.mark.parametrize(
    "argv",
    [
        ["--probe"],
        ["--experiment", "SPEC", "--probe", "--dry-run"],
        ["--experiment", "SPEC", "--probe", "--out", "elsewhere"],
    ],
)
def test_a_probe_needs_an_experiment_and_picks_its_own_directory(
    git_repo, argv, capsys
):
    spec = write_spec(git_repo)
    argv = [str(spec) if a == "SPEC" else a for a in argv]

    assert main.main(argv, repo_root=git_repo) == 2
    assert "--probe" in capsys.readouterr().err
    assert not (git_repo / "outputs").exists()
