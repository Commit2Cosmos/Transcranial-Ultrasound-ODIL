import csv
import json

import numpy as np
import pytest
import torch

from odil_wave import FrequencySelection, HelmholtzSolver, JointFreqODIL, Wavefield
from odil_wave.experiment import (
    build_problem,
    deep_merge,
    resolve_config,
    run_inverse,
    runner,
)

from conftest import (
    make_band,
    make_geometry,
    make_grid,
    make_model,
    make_source,
    rel_interior_error,
)


def _saved_c(band_dir) -> np.ndarray:
    """A band's saved final model over the interior."""
    with np.load(band_dir / "arrays.npz") as saved:
        return saved["c"]


def _rel_c_error(c, truth, grid) -> float:
    """Relative L2 error of a full-grid velocity ``c`` on the interior."""
    c = np.asarray(c)[grid.interior_slice].astype(np.float64)
    t = truth.c[grid.interior_slice].detach().cpu().numpy().astype(np.float64)
    return float(np.linalg.norm(c - t) / np.linalg.norm(t))


def test_forward_to_loss_gradient_flows():
    """A Helmholtz forward solve feeds the joint objective, differentiable in
    the model."""
    grid = make_grid(interior_shape=(16, 16))
    fs = FrequencySelection.from_frequencies(grid, [40e3])
    src = make_source(grid)
    geom = make_geometry(grid, src, fs)

    # Truth data from an overdensity medium.
    truth = make_model(grid, "overdensity", contrast=120.0, radius=0.012)
    wf_truth = Wavefield(grid, fs, velocity_model=truth)
    sols = HelmholtzSolver(wf_truth, geom, space_order=2, damping_scale=1.0).solve(
        verbose=False
    )

    # A homogeneous initial model, warm-started at its own Helmholtz field.
    opt = JointFreqODIL(
        make_band(geom, make_model(grid), sols, keep_factors=False),
        n_iter=1,
        inner_max_iter=1,
        history_size=1,
        line_search_fn=None,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        lbfgs_lr=1.0,
        pde_weight=1.0,
        data_weight=100.0,
        reg_weight=0.0,
        log_every=1,
        verbose=False,
        model_preconditioner="none",
        wavefield_preconditioner="none",
        regulariser=None,
    )

    u_re, u_im, z_m = opt.init_params()
    z_m.requires_grad_(True)
    L = opt.objective(u_re, u_im, z_m)
    L.backward()
    # The data misfit is non-zero (wrong model) and its gradient reaches the
    # model latent z_m (and so c).
    assert opt.objective_terms(u_re, u_im, z_m).data > 0
    assert z_m.grad is not None
    assert float(z_m.grad.abs().max()) > 0


def test_run_inverse_tiny_problem_recovers(tiny_run_config, new_run):
    """A full tiny run_inverse completes, writes artifacts, and improves the model."""
    cfg = resolve_config(tiny_run_config)
    problem = build_problem(cfg)
    result = run_inverse(cfg, new_run())

    # --- run outcome --------------------------------------------------------- #
    assert result.status == "completed"
    # the recovered model is closer to the truth than the starting model. The
    # brain region alone can get worse on the tiny run while the skull improves.
    c_final = _saved_c(result.band_results[-1].band_dir)
    c_final_full = np.pad(c_final, problem.grid.pml_width)
    assert _rel_c_error(
        c_final_full, problem.truth_velocity, problem.grid
    ) < _rel_c_error(problem.init_velocity.c, problem.truth_velocity, problem.grid)

    # --- artifacts ----------------------------------------------------------- #
    run_dir = result.run_dir
    assert (run_dir / "config_resolved.yaml").exists()
    assert c_final.shape == (problem.grid.interior_nx, problem.grid.interior_ny)

    # --- loss decreased over the run ---------------------------------------- #
    with open(run_dir / "tape.csv", newline="") as f:
        losses = [float(r["loss_total"]) for r in csv.DictReader(f)]
    assert len(losses) >= 2
    assert losses[-1] <= losses[0]


def test_band_log_shows_the_snapped_frequencies_and_its_dir_the_requested(
    tiny_run_config, new_run, capsys
):
    """The band's log line shows the FFT-bin frequencies it solved at; its
    directory keeps the requested ones, so paths do not move with ``nt``."""
    requested = [30e3, 40e3]
    user_cfg = deep_merge(
        tiny_run_config,
        {
            "continuation": {"bands": [{"frequencies_hz": requested}]},
            "optimiser": {"n_iter": 1},
        },
    )
    cfg = resolve_config(user_cfg)
    grid = build_problem(cfg).grid
    lo, hi = FrequencySelection.from_frequencies(grid, requested).frequencies / 1e3
    assert (f"{lo:.2f}", f"{hi:.2f}") != ("30.00", "40.00")

    result = run_inverse(cfg, new_run())

    assert result.band_results[0].band_dir.name == "band_00_30-40khz"
    assert f"[band 0 {lo:.2f}-{hi:.2f}khz]" in capsys.readouterr().out


def test_run_inverse_two_bands_chain_models(tiny_run_config, monkeypatch, new_run):
    """Each band writes its artifacts and band 2 starts from band 1's model."""
    user_cfg = deep_merge(
        tiny_run_config,
        {
            "continuation": {
                "bands": [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]
            },
            "optimiser": {"n_iter": 3},
        },
    )
    cfg = resolve_config(user_cfg)
    init_c = build_problem(cfg).init_velocity.c

    # Record the starting model handed to each band.
    band_inputs = []
    original = runner.run_frequency_band

    def spy(problem, velocity_model, *args, **kwargs):
        band_inputs.append(velocity_model)
        return original(problem, velocity_model, *args, **kwargs)

    monkeypatch.setattr(runner, "run_frequency_band", spy)
    result = run_inverse(cfg, new_run())
    assert result.status == "completed"

    # --- one artifact directory per band ------------------------------------ #
    band_dirs = sorted(p for p in (result.run_dir / "bands").iterdir() if p.is_dir())
    assert [p.name[:7] for p in band_dirs] == ["band_00", "band_01"]
    for band_dir, freqs in zip(band_dirs, ([30e3], [40e3])):
        assert sorted(p.name for p in band_dir.iterdir()) == [
            "arrays.npz",
            "band_summary.json",
        ]
        # The outer loop always runs the full budget, so the summary carries
        # no n_iter_run / termination_reason.
        summary = json.loads((band_dir / "band_summary.json").read_text())
        assert summary["frequencies_hz"] == freqs
        assert summary["n_iter_requested"] == 3
        assert not {"n_iter_run", "termination_reason"} & set(summary)
    for b in result.band_results:
        assert not hasattr(b, "n_iter_run") and not hasattr(b, "termination_reason")

    # --- band 1 starts from the init model, band 2 from band 1's result ----- #
    assert len(band_inputs) == 2
    torch.testing.assert_close(band_inputs[0].c, init_c, rtol=0, atol=0)
    assert band_inputs[1] is result.band_results[0].velocity_model
    band0_c = _saved_c(band_dirs[0])
    sl = band_inputs[1].grid.interior_slice
    np.testing.assert_allclose(
        band_inputs[1].c[sl].detach().cpu().numpy(), band0_c, rtol=1e-6
    )
    # the run's final model is the last band's.
    assert result.final_velocity is result.band_results[1].velocity_model


def _built_optimiser(run_config, loss_cfg):
    """The runner's 1-step optimiser for the first band of ``run_config`` with
    ``loss_cfg``."""
    cfg = resolve_config(
        deep_merge(
            run_config,
            {"loss": loss_cfg, "optimiser": {"n_iter": 1, "inner_max_iter": 2}},
        )
    )
    problem = build_problem(cfg)
    first = cfg.continuation.bands[0]
    band = problem.make_band(first.frequencies_hz, problem.init_velocity)
    return runner.build_optimiser(problem, band, cfg.band_n_iter(first))


def _spy_band_solves(monkeypatch) -> list:
    """A list that collects the ``BandSolve`` of every optimiser
    ``runner.build_optimiser`` builds."""
    solves = []
    original = runner.build_optimiser

    def spy(*args, **kwargs):
        opt = original(*args, **kwargs)
        minimise = opt.minimise

        def recorded(*a, **kw):
            solves.append(minimise(*a, **kw))
            return solves[-1]

        opt.minimise = recorded
        return opt

    monkeypatch.setattr(runner, "build_optimiser", spy)
    return solves


def _objective_off_truth(opt):
    """Objective at the warm start with the model latent perturbed (so the PDE
    term is non-zero); returns ``(Phi, unweighted pde_term, unweighted data_term)``."""
    u_re, u_im, z = opt.init_params()
    z = z + 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(0))
    terms = opt.objective_terms(u_re, u_im, z)
    return float(opt.objective(u_re, u_im, z)), terms.pde, terms.data


def test_loss_weights_scale_the_joint_objective(tiny_run_config):
    """loss.weights.pde / .data set through the run config scale the objective."""
    base = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    L1, p, d = _objective_off_truth(_built_optimiser(tiny_run_config, base))
    assert p > 0 and d > 0

    pde3 = {"weights": {"pde": 3.0, "data": 1.0, "reg": 0.0}}
    L_pde, *_ = _objective_off_truth(_built_optimiser(tiny_run_config, pde3))
    np.testing.assert_allclose(L_pde, L1 + 2.0 * p, rtol=1e-10)

    data5 = {"weights": {"pde": 1.0, "data": 5.0, "reg": 0.0}}
    L_dat, *_ = _objective_off_truth(_built_optimiser(tiny_run_config, data5))
    np.testing.assert_allclose(L_dat, L1 + 4.0 * d, rtol=1e-10)


def test_loss_weights_reg_activates_regulariser(tiny_run_config):
    """The regulariser contributes only when loss.weights.reg > 0."""

    def loss_cfg(reg):
        return {
            "weights": {"pde": 1.0, "data": 1.0, "reg": reg},
            "regulariser": {"name": "tikhonov"},
        }

    plain = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    L_plain, *_ = _objective_off_truth(_built_optimiser(tiny_run_config, plain))
    L_off, *_ = _objective_off_truth(_built_optimiser(tiny_run_config, loss_cfg(0.0)))
    L_on, *_ = _objective_off_truth(_built_optimiser(tiny_run_config, loss_cfg(1e-3)))
    assert L_off == L_plain
    assert L_on > L_plain


def test_optimiser_result_loss_is_at_the_final_iterate(tiny_run_config):
    """The result loss is the objective at the returned (u, z), not pre-step."""
    plain = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    opt = _built_optimiser(tiny_run_config, plain)
    solve = opt.minimise()
    with torch.no_grad():
        L = float(opt.objective(*solve.params))
    assert solve.loss == pytest.approx(L, rel=1e-12)


def test_pde_diagonal_preconditioner_runs_end_to_end(tiny_run_config, new_run):
    """The tiny config with optimiser.model_preconditioner = pde_diagonal
    completes with finite tape values and reports its scale (no recovery
    claim)."""
    user_cfg = deep_merge(
        tiny_run_config,
        {"optimiser": {"n_iter": 3, "model_preconditioner": "pde_diagonal"}},
    )
    result = run_inverse(resolve_config(user_cfg), new_run())
    assert result.status == "completed"

    with open(result.run_dir / "tape.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4  # start row + 3 outer iterations
    for key in ("loss_total", "rmse_brain", "grad_z_norm"):
        values = [float(r[key]) for r in rows]
        assert np.all(np.isfinite(values)), key

    (band_dir,) = (p for p in (result.run_dir / "bands").iterdir() if p.is_dir())
    summary = json.loads((band_dir / "band_summary.json").read_text())
    info = summary["optimiser_result"]["scale_info"]
    assert info["model_preconditioner"] == "pde_diagonal"
    assert np.isfinite(info["precond_target"]) and info["precond_target"] > 0


def test_both_preconditioners_run_end_to_end(tiny_run_config, new_run):
    """The tiny config with both preconditioners completes with finite tape
    values and reports the wavefield preconditioner and the time spent in its
    triangular solves (no recovery claim)."""
    user_cfg = deep_merge(
        tiny_run_config,
        {
            "optimiser": {
                "n_iter": 3,
                "model_preconditioner": "pde_diagonal",
                "wavefield_preconditioner": "initial_model",
            }
        },
    )
    result = run_inverse(resolve_config(user_cfg), new_run())
    assert result.status == "completed"

    with open(result.run_dir / "tape.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    for key in ("loss_total", "rmse_brain", "grad_u_norm", "grad_z_norm"):
        values = [float(r[key]) for r in rows]
        assert np.all(np.isfinite(values)), key

    (band_dir,) = (p for p in (result.run_dir / "bands").iterdir() if p.is_dir())
    summary = json.loads((band_dir / "band_summary.json").read_text())
    assert 0.0 < summary["wavefield_solve_s"] < summary["optimise_s"]
    info = summary["optimiser_result"]["scale_info"]
    assert info["wavefield_preconditioner"] == "initial_model"
    assert np.isfinite(info["precond_target"]) and info["precond_target"] > 0


@pytest.mark.parametrize("mode", ["none", "pde_diagonal"])
def test_run_inverse_started_at_the_truth_stays_there(
    tiny_run_config, monkeypatch, new_run, mode
):
    """With init = truth (inverse crime, no regulariser), the warm start seeds
    the true wavefield as the initial wavefield, and the run leaves (c, u) at
    the truth. Measured: objective
    2.0e-29 (threshold 1e-20); c and u errors 6e-17 / 7e-17 (threshold 1e-8)."""
    user_cfg = deep_merge(
        tiny_run_config,
        {
            "init": tiny_run_config["truth"],
            "loss": {"weights": {"reg": 0.0}},
            "optimiser": {"model_preconditioner": mode},
        },
    )
    cfg = resolve_config(user_cfg)
    assert cfg.init == cfg.truth and cfg.runtime.dtype == "float64"
    problem = build_problem(cfg)
    (band,) = cfg.continuation.bands
    u_true = torch.stack(
        [
            w.amplitude
            for w in problem.make_band(
                band.frequencies_hz, problem.init_velocity
            ).true_wavefields
        ]
    )

    solves = _spy_band_solves(monkeypatch)
    assert run_inverse(cfg, new_run()).status == "completed"

    (final,) = solves
    assert final.loss < 1e-20
    grid = problem.grid
    c_true = problem.truth_velocity.c
    assert rel_interior_error(final.velocity_model.c, c_true, grid) < 1e-8
    assert rel_interior_error(final.u, u_true, grid) < 1e-8
