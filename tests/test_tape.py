"""The metric tape a run writes (``tape.csv``), checked against the run-layer
README's column table and against values recomputed independently of the run."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

import yaml

from odil_wave import (
    HelmholtzSolver,
    Regulariser,
    VelocityModel,
    WaveEquation,
    Wavefield,
)
from odil_wave.experiment import build_problem, deep_merge, resolve_config

import main
from conftest import TINY_RUN_CONFIG

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "odil_wave" / "experiment" / "README.md"


def readme_tape_columns() -> list:
    """Column names, in order, from the README's "Iteration-level log record"
    tables (the first cell of each table row)."""
    text = README.read_text()
    section = text.split("## Iteration-level log record", 1)[1].split("\n## ", 1)[0]
    columns = []
    for line in section.splitlines():
        if line.startswith("| `"):
            first_cell = line.split("|")[1]
            columns += re.findall(r"`([a-z_0-9]+)`", first_cell)
    return columns


def read_tape(run_dir: Path) -> tuple:
    """``(header, rows)``; a row maps column -> float, or None for an empty cell."""
    with open(run_dir / "tape.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                k: (None if v == "" else (v if k == "run_id" else float(v)))
                for k, v in r.items()
            }
            for r in reader
        ]
        return reader.fieldnames, rows


def band_rows(rows: list, band_index: int) -> list:
    return [r for r in rows if r["band_index"] == band_index]


# --------------------------------------------------------------------------- #
# Columns and rows
# --------------------------------------------------------------------------- #
def test_tape_has_exactly_the_readme_columns(tiny_two_band_run):
    header, _ = read_tape(tiny_two_band_run.run_dir)
    expected = readme_tape_columns()
    assert len(expected) == 24
    assert header == expected


def test_only_a_start_rows_model_update_is_empty(tiny_two_band_run):
    """Every cell is written; the one null is rel_model_update on a start row."""
    _, rows = read_tape(tiny_two_band_run.run_dir)
    for r in rows:
        empty = {k for k, v in r.items() if v is None}
        assert empty == ({"rel_model_update"} if r["iter"] == 0 else set()), r


def test_each_band_has_one_start_row_and_one_final_row(tiny_two_band_run):
    """n_iter = 3, log_every = 2: a start row, the cadence row at iter 2 and the
    final iterate at iter 3, per band; iter_total runs on across bands."""
    _, rows = read_tape(tiny_two_band_run.run_dir)
    assert {r["run_id"] for r in rows} == {"two_band"}
    assert [r["band_index"] for r in rows] == [0, 0, 0, 1, 1, 1]
    assert [r["iter"] for r in rows] == [0, 2, 3, 0, 2, 3]
    assert [r["iter_total"] for r in rows] == [0, 2, 3, 3, 5, 6]
    wall = [r["wall_s"] for r in rows]
    assert wall == sorted(wall) and wall[0] > 0


def band_summaries(run_dir: Path) -> list:
    return [
        json.loads((p / "band_summary.json").read_text())
        for p in sorted((run_dir / "bands").iterdir())
        if p.is_dir()
    ]


def test_n_evaluations_is_cumulative_over_the_run(tiny_two_band_run):
    """Each band's closures add on; band 1 starts where band 0 ended."""
    run_dir = tiny_two_band_run.run_dir
    _, rows = read_tape(run_dir)
    closures = [s["optimiser_result"]["n_closure"] for s in band_summaries(run_dir)]
    assert all(n > 0 for n in closures)
    b0, b1 = band_rows(rows, 0), band_rows(rows, 1)
    assert b0[0]["n_evaluations"] == 0
    assert b0[-1]["n_evaluations"] == closures[0]
    assert b1[0]["n_evaluations"] == closures[0] > 0
    assert b1[-1]["n_evaluations"] == closures[0] + closures[1]
    counts = [r["n_evaluations"] for r in rows]
    assert counts == sorted(counts)


def test_summaries_report_the_tape_at_the_final_iterate(tiny_two_band_run):
    run_dir = tiny_two_band_run.run_dir
    _, rows = read_tape(run_dir)
    summaries = band_summaries(run_dir)
    for b, summary in enumerate(summaries):
        assert summary["start_metrics"] == pytest.approx(band_rows(rows, b)[0])
        assert summary["final_metrics"] == pytest.approx(band_rows(rows, b)[-1])
    run_summary = json.loads((run_dir / "run_summary.json").read_text())
    metrics = run_summary["metrics_summary"]
    assert metrics["start_metrics"] == pytest.approx(rows[0])
    assert metrics["final_metrics"] == pytest.approx(rows[-1])
    assert metrics["per_band"] == [s["final_metrics"] for s in summaries]


# --------------------------------------------------------------------------- #
# Independent recomputation
# --------------------------------------------------------------------------- #
def regions(problem) -> tuple:
    """Interior ``(c*, brain, skull)`` of the truth model, as numpy."""
    _, brain, skull = problem.truth_velocity.skull_region_masks()
    sl = problem.grid.interior_slice
    return (
        problem.truth_velocity.c[sl].numpy().astype(np.float64),
        brain[sl].numpy(),
        skull[sl].numpy(),
    )


def expected_model_error(problem, c_full: np.ndarray) -> dict:
    """Region RMSE / bias, the agreed brain-region SSIM and the c extremes of a
    full-grid velocity."""
    truth, brain, skull = regions(problem)
    c = np.asarray(c_full)[problem.grid.interior_slice].astype(np.float64)
    sigma = problem.cfg.metrics.ssim.sigma
    win = 2 * int(3.5 * sigma + 0.5) + 1  # skimage truncates at 3.5 sigma
    _, ssim_map = structural_similarity(
        c,
        truth,
        win_size=win,
        gaussian_weights=True,
        sigma=sigma,
        use_sample_covariance=False,
        data_range=float(truth[brain].max() - truth[brain].min()),
        full=True,
    )
    eroded = ndi.binary_erosion(brain, structure=np.ones((win, win), dtype=bool))
    assert eroded.any()
    diff = c - truth
    return {
        "rmse_brain": np.sqrt(np.mean(diff[brain] ** 2)),
        "bias_brain": np.mean(diff[brain]),
        "rmse_skull": np.sqrt(np.mean(diff[skull] ** 2)),
        "bias_skull": np.mean(diff[skull]),
        "ssim_brain": ssim_map[eroded].mean(),
        "c_min": c.min(),
        "c_max": c.max(),
    }


def saved_final_model(band_dir: Path, problem) -> np.ndarray:
    """A band's saved final model (interior, in ``arrays.npz``) padded to the
    full grid with the frozen PML speed."""
    with np.load(band_dir / "arrays.npz") as saved:
        c = saved["c"]
    return np.pad(
        c, problem.grid.pml_width, constant_values=problem.init_velocity.pml_c
    )


def band_start_models(run_dir: Path, problem) -> list:
    """Full-grid starting model of each band: the init model, then the saved
    final model of the band before."""
    band_dirs = sorted(p for p in (run_dir / "bands").iterdir() if p.is_dir())
    c0 = problem.init_velocity.c.numpy()
    return [c0] + [saved_final_model(d, problem) for d in band_dirs[:-1]]


def final_band_dir(run_dir: Path) -> Path:
    """The last band's directory, holding the run's final model."""
    return sorted(p for p in (run_dir / "bands").iterdir() if p.is_dir())[-1]


def rms(x: torch.Tensor) -> float:
    return float(x.abs().square().mean().sqrt())


def test_start_row_residuals_match_a_helmholtz_solve(tiny_two_band_run):
    """A start row's u is the warm start: a Helmholtz solve on the band's starting
    model. Its data residual, wavefield error and PDE residual are recomputed."""
    run_dir = tiny_two_band_run.run_dir
    problem = build_problem(tiny_two_band_run.config)
    _, rows = read_tape(run_dir)
    bands = tiny_two_band_run.config.continuation.bands
    for b, c_start in enumerate(band_start_models(run_dir, problem)):
        band = problem.make_band(bands[b].frequencies_hz)
        model = VelocityModel.from_field(
            problem.grid,
            torch.as_tensor(c_start, dtype=problem.dtype),
            pml_c=problem.init_velocity.pml_c,
        )
        wf = Wavefield(problem.grid, band.freq, velocity_model=model)
        op = dict(space_order=problem.space_order, damping_scale=problem.damping_scale)
        u = torch.stack(
            [s.amplitude for s in HelmholtzSolver(wf, band.geom, **op).solve()]
        )
        u_star = torch.stack([s.amplitude for s in band.observed_wfs])
        d_obs = band.geom.extract_observations(u_star)
        d_syn = band.geom.extract_observations(u)
        q = torch.stack(
            [band.geom.source_field(i).to(wf.cdtype) for i in range(band.geom.n_shots)]
        ) * (problem.grid.t0**2)
        r_pde = WaveEquation(wf, **op).residual(u, model.c, q)
        sl = (slice(None), slice(None)) + problem.grid.interior_slice

        start = band_rows(rows, b)[0]
        assert start["rel_data_residual"] == pytest.approx(
            float((d_syn - d_obs).norm() / d_obs.norm()), rel=1e-6
        )
        assert start["rel_data_residual"] > 0
        assert start["rel_wavefield_error"] == pytest.approx(
            float((u[sl] - u_star[sl]).norm() / u_star[sl].norm()), rel=1e-6
        )
        # PDE scale |lambda_tt| * u_ref, u_ref the interior RMS of the warm start;
        # the sum over interior + PML is normalised by the interior node count.
        lam_tt = band.freq.symbols_broadcast()[1].abs().real
        u_ref = u[sl].abs().square().mean(dim=(2, 3), keepdim=True).sqrt()
        n_int = u[sl][0, 0].numel()
        expected = float(
            (
                (r_pde / (lam_tt * u_ref)).abs().square().sum()
                / (n_int * u.shape[0] * u.shape[1])
            ).sqrt()
        )
        assert start["pde_rel_residual"] == pytest.approx(expected, rel=1e-4, abs=1e-12)


def test_model_errors_match_the_band_models(tiny_two_band_run):
    """Region RMSE / bias, ssim_brain and the c extremes, recomputed from each
    band's starting model and its saved final model."""
    run_dir = tiny_two_band_run.run_dir
    problem = build_problem(tiny_two_band_run.config)
    _, rows = read_tape(run_dir)
    starts = band_start_models(run_dir, problem)
    finals = starts[1:] + [saved_final_model(final_band_dir(run_dir), problem)]
    for b, (c_start, c_final) in enumerate(zip(starts, finals)):
        rows_b = band_rows(rows, b)
        for row, c in ((rows_b[0], c_start), (rows_b[-1], c_final)):
            for key, value in expected_model_error(problem, c).items():
                assert row[key] == pytest.approx(value, rel=1e-5, abs=1e-6), (b, key)


def run_cli(tmp_path: Path, repo: Path, user: dict, expect: int = 0) -> Path:
    """Run ``user`` (a run config) through the CLI; return its run directory."""
    path = tmp_path / "run_config.yaml"
    path.write_text(yaml.safe_dump(user))
    run_dir = tmp_path / "run"
    argv = ["--config", str(path), "--out", str(run_dir)]
    assert main.main(argv, repo_root=repo) == expect
    return run_dir


def test_reg_term_and_model_update_at_each_row(tmp_path, git_repo):
    """reg_term is w_reg * R(c) at the logged model; rel_model_update compares
    with the previous row (here the start row: log_every > n_iter)."""
    w_reg = 1e-3
    run_dir = run_cli(
        tmp_path,
        git_repo,
        deep_merge(
            TINY_RUN_CONFIG,
            {
                "loss": {
                    "weights": {"reg": w_reg},
                    "regulariser": {"name": "tikhonov"},
                },
                "optimiser": {"n_iter": 2, "log_every": 5},
            },
        ),
    )
    _, rows = read_tape(run_dir)
    assert [r["iter"] for r in rows] == [0, 2]
    problem = build_problem(resolve_config(TINY_RUN_CONFIG))
    sl = problem.grid.interior_slice
    c_start = problem.init_velocity.c[sl].double()
    c_final = torch.as_tensor(saved_final_model(final_band_dir(run_dir), problem))[
        sl
    ].double()
    reg = Regulariser(kind="tikhonov", eps=1e-8)  # eps is unused by tikhonov
    for row, c in zip(rows, (c_start, c_final)):
        # The saved c is float32: its cell-to-cell differences carry ~1e-4 error.
        assert row["reg_term"] == pytest.approx(w_reg * float(reg(c)), rel=1e-3)
        assert row["pde_term"] + row["data_term"] + row["reg_term"] == pytest.approx(
            row["loss_total"], rel=1e-10
        )
    assert rows[-1]["rel_model_update"] == pytest.approx(
        float((c_final - c_start).norm() / c_start.norm()), rel=1e-3
    )
    assert rows[-1]["rel_model_update"] > 0


def test_reg_term_is_zero_without_a_regulariser(tiny_two_band_run):
    _, rows = read_tape(tiny_two_band_run.run_dir)
    assert {r["reg_term"] for r in rows} == {0.0}
    for r in rows:
        assert r["pde_term"] + r["data_term"] == pytest.approx(
            r["loss_total"], rel=1e-10
        )


# --------------------------------------------------------------------------- #
# Nothing to score is a config error at startup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "change",
    [
        # no truth with a brain region
        {"truth": {"profile": "overdensity", "extra": {"radius": 0.012}}},
        {"truth": {"profile": "homogeneous"}},
        # a 23-cell SSIM window on a 20 x 20 interior
        {"metrics": {"ssim": {"sigma": 3.0}}},
        # an 11-cell window erodes the whole tiny brain region
        {"metrics": {"ssim": {"sigma": 1.5}}},
        # a truth constant over the brain region has no SSIM data range
        {"truth": {"profile": "shepp_logan_skull"}},
    ],
)
def test_nothing_to_score_is_a_config_error(tmp_path, git_repo, capsys, change):
    run_dir = run_cli(tmp_path, git_repo, deep_merge(TINY_RUN_CONFIG, change), 2)
    assert "config error:" in capsys.readouterr().err
    assert not (run_dir / "tape.csv").exists()
