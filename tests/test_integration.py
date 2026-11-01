import csv
import json

import numpy as np
import pytest
import torch

from odil_wave import (
    FrequencySelection,
    HelmholtzSolver,
    InverseLoss,
    LossConfig,
    WaveEquation,
    Wavefield,
)
from odil_wave.experiment import (
    build_problem,
    deep_merge,
    resolve_config,
    run_inverse,
    runner,
)

from conftest import make_geometry, make_grid, make_model, make_source


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
    """A Helmholtz forward solve feeds a differentiable inverse loss in ``c``."""
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

    # A homogeneous starting model, warm-started at its own Helmholtz field.
    init = make_model(grid)
    wf = Wavefield(grid, fs, velocity_model=init)
    u0 = torch.stack(
        [
            w.amplitude
            for w in HelmholtzSolver(wf, geom, space_order=2, damping_scale=1.0).solve(
                verbose=False
            )
        ]
    )

    wave_eq = WaveEquation(wf, space_order=2, damping_scale=1.0)
    cfg = LossConfig(
        wave_eq=wave_eq,
        geometry=geom,
        weights={"pde": 1.0, "data": 100.0},
        regulariser=None,
    )
    loss = InverseLoss(observed_wavefield=sols, config=cfg)

    c = init.c.clone().requires_grad_(True)
    L = loss.evaluate(u0, c)
    L.backward()
    # The data misfit is non-zero (wrong model) and its gradient reaches c.
    assert float(L.detach()) > 0
    assert c.grad is not None
    assert float(c.grad.abs().max()) > 0


def test_run_inverse_tiny_problem_recovers(tiny_run_config, new_run):
    """A full tiny run_inverse completes, writes artifacts, and improves the model."""
    cfg = resolve_config(tiny_run_config)
    problem = build_problem(cfg)
    result = run_inverse(cfg, new_run())

    # --- run outcome --------------------------------------------------------- #
    assert result.status == "completed"
    # the recovered model is closer to the truth than the starting model.
    start = result.metrics_summary["start_metrics"]
    final = result.metrics_summary["final_metrics"]
    assert final["rmse_brain"] < start["rmse_brain"]
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
        # n_iter_run / termination_reason were hard-coded (the outer loop
        # always runs the full budget), so the summary no longer carries them.
        summary = json.loads((band_dir / "band_summary.json").read_text())
        assert summary["frequencies_hz"] == freqs
        assert summary["n_iter_requested"] == 3
        assert not {"n_iter_run", "termination_reason"} & set(summary)
    for b in result.band_results:
        assert not hasattr(b, "n_iter_run") and not hasattr(b, "termination_reason")

    # --- band 1 starts from the init model, band 2 from band 1's result ----- #
    assert len(band_inputs) == 2
    assert torch.equal(band_inputs[0].c, init_c)
    assert band_inputs[1] is result.band_results[0].velocity_model
    band0_c = _saved_c(band_dirs[0])
    sl = band_inputs[1].grid.interior_slice
    np.testing.assert_allclose(
        band_inputs[1].c[sl].detach().cpu().numpy(), band0_c, rtol=1e-6
    )
    # the run's final model is the last band's.
    assert result.final_velocity is result.band_results[1].velocity_model


def _built_optimiser(run_config, monkeypatch, run_directory, loss_cfg):
    """Run a 1-step tiny inversion with ``loss_cfg`` and return the optimiser
    the runner built (captured from ``runner._build_optimiser``)."""
    user_cfg = deep_merge(
        run_config,
        {
            "loss": loss_cfg,
            "optimiser": {"n_iter": 1, "inner_max_iter": 2},
        },
    )
    built = []
    original = runner._build_optimiser

    def spy(*args, **kwargs):
        built.append(original(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(runner, "_build_optimiser", spy)
    assert run_inverse(resolve_config(user_cfg), run_directory).status == "completed"
    return built[0]


def _objective_off_truth(opt):
    """Objective at the warm start with the model latent perturbed (so the PDE
    term is non-zero); returns ``(Phi, unweighted pde_term, unweighted data_term)``."""
    u_re, u_im, z = opt.init_params()
    z = z + 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(0))
    L = float(opt.objective(u_re, u_im, z, record=True))
    return L, opt._last["pde_loss"], opt._last["data_loss"]


def test_loss_weights_scale_the_joint_objective(tiny_run_config, monkeypatch, new_run):
    """loss.weights.pde / .data set through the run config scale the objective."""
    base = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    L1, p, d = _objective_off_truth(
        _built_optimiser(tiny_run_config, monkeypatch, new_run("w1"), base)
    )
    assert p > 0 and d > 0

    pde3 = {"weights": {"pde": 3.0, "data": 1.0, "reg": 0.0}}
    L_pde, *_ = _objective_off_truth(
        _built_optimiser(tiny_run_config, monkeypatch, new_run("w2"), pde3)
    )
    np.testing.assert_allclose(L_pde, L1 + 2.0 * p, rtol=1e-10)

    data5 = {"weights": {"pde": 1.0, "data": 5.0, "reg": 0.0}}
    L_dat, *_ = _objective_off_truth(
        _built_optimiser(tiny_run_config, monkeypatch, new_run("w3"), data5)
    )
    np.testing.assert_allclose(L_dat, L1 + 4.0 * d, rtol=1e-10)


def test_loss_weights_reg_activates_regulariser(tiny_run_config, monkeypatch, new_run):
    """The regulariser contributes only when loss.weights.reg > 0."""

    def loss_cfg(reg):
        return {
            "weights": {"pde": 1.0, "data": 1.0, "reg": reg},
            "regulariser": {"name": "tikhonov"},
        }

    plain = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    L_plain, *_ = _objective_off_truth(
        _built_optimiser(tiny_run_config, monkeypatch, new_run("r0"), plain)
    )
    L_off, *_ = _objective_off_truth(
        _built_optimiser(tiny_run_config, monkeypatch, new_run("r1"), loss_cfg(0.0))
    )
    L_on, *_ = _objective_off_truth(
        _built_optimiser(tiny_run_config, monkeypatch, new_run("r2"), loss_cfg(1e-3))
    )
    assert L_off == L_plain
    assert L_on > L_plain


def test_optimiser_result_loss_is_at_the_final_iterate(
    tiny_run_config, monkeypatch, new_run
):
    """The result loss is the objective at the returned (u, z), not pre-step."""
    plain = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    opt = _built_optimiser(tiny_run_config, monkeypatch, new_run("final_loss"), plain)
    result = opt.loss.callback.result
    s = opt.final_state
    with torch.no_grad():
        L = float(opt.objective(s["u_re"], s["u_im"], s["z_m"], record=False))
    assert result["loss"] == pytest.approx(L, rel=1e-12)
    assert not {
        "n_factor",
        "n_forward_solves",
        "n_adjoint_solves",
        "n_outer_budget",
    } & set(result)
    assert {"n_outer_iter", "n_closure", "n_evaluations"} <= set(result)
