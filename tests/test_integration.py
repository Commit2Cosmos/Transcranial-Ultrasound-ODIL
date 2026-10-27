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


def _rel_c_error(model, truth, grid) -> float:
    """Relative L2 velocity error on the interior."""
    c = model.c[grid.interior_slice].detach().cpu().numpy().astype(np.float64)
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


def test_run_inverse_tiny_problem_recovers(tiny_experiment):
    """A full tiny run_inverse completes, writes artifacts, and improves the model."""
    cfg = resolve_config(tiny_experiment)

    # Baseline error of the (homogeneous) starting model against the truth.
    problem = build_problem(cfg)
    init_err = _rel_c_error(problem.init_velocity, problem.truth_velocity, problem.grid)

    result = run_inverse(cfg)

    # --- run outcome --------------------------------------------------------- #
    assert result.status == "completed"
    final_err = result.metrics_summary["final_rel_c_error"]
    assert final_err is not None and np.isfinite(final_err)
    # the recovered model is closer to the truth than the starting model.
    assert final_err < init_err

    # --- artifacts ----------------------------------------------------------- #
    run_dir = result.run_dir
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert result.final_c_path.exists()
    c_final = np.load(result.final_c_path)
    assert c_final.shape == tuple(problem.grid.shape)

    # --- loss decreased over the run ---------------------------------------- #
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    losses = [r["loss_total"] for r in records if r.get("loss_total") is not None]
    assert len(losses) >= 2
    assert losses[-1] <= losses[0]


def test_run_inverse_two_bands_chain_models(tiny_experiment, monkeypatch):
    """Each band writes its artifacts and band 2 starts from band 1's model."""
    user_cfg = deep_merge(
        tiny_experiment,
        {
            "run": {"run_id": "two_bands"},
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
    result = run_inverse(cfg)
    assert result.status == "completed"

    # --- one artifact directory per band ------------------------------------ #
    band_dirs = sorted(p for p in (result.run_dir / "bands").iterdir() if p.is_dir())
    assert [p.name[:7] for p in band_dirs] == ["band_00", "band_01"]
    for band_dir, freqs in zip(band_dirs, ([30e3], [40e3])):
        for name in (
            "c_final.npy",
            "c_final_metadata.json",
            "band_config.yaml",
            "band_summary.json",
        ):
            assert (band_dir / name).exists()
        meta = json.loads((band_dir / "c_final_metadata.json").read_text())
        assert meta["frequencies_hz"] == freqs

    # --- band 1 starts from the init model, band 2 from band 1's result ----- #
    assert len(band_inputs) == 2
    assert torch.equal(band_inputs[0].c, init_c)
    assert band_inputs[1] is result.band_results[0].velocity_model
    band0_c = np.load(band_dirs[0] / "c_final.npy")
    np.testing.assert_allclose(
        band_inputs[1].c.detach().cpu().numpy(), band0_c, rtol=1e-6
    )
    # the run's final model is the last band's.
    np.testing.assert_array_equal(
        np.load(result.final_c_path), np.load(band_dirs[1] / "c_final.npy")
    )


def _built_optimiser(experiment, monkeypatch, run_id, loss_cfg):
    """Run a 1-step tiny inversion with ``loss_cfg`` and return the optimiser
    the runner built (captured from ``runner._build_optimiser``)."""
    user_cfg = deep_merge(
        experiment,
        {
            "run": {"run_id": run_id},
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
    assert run_inverse(resolve_config(user_cfg)).status == "completed"
    return built[0]


def _objective_off_truth(opt):
    """Objective at the warm start with the model latent perturbed (so the PDE
    term is non-zero); returns ``(Phi, unweighted pde_term, unweighted data_term)``."""
    u_re, u_im, z = opt.init_params()
    z = z + 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(0))
    L = float(opt.objective(u_re, u_im, z, record=True))
    return L, opt._last["pde_loss"], opt._last["data_loss"]


def test_loss_weights_scale_the_joint_objective(tiny_experiment, monkeypatch):
    """loss.weights.pde / .data set through the run config scale the objective."""
    base = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    L1, p, d = _objective_off_truth(
        _built_optimiser(tiny_experiment, monkeypatch, "w1", base)
    )
    assert p > 0 and d > 0

    pde3 = {"weights": {"pde": 3.0, "data": 1.0, "reg": 0.0}}
    L_pde, *_ = _objective_off_truth(
        _built_optimiser(tiny_experiment, monkeypatch, "w2", pde3)
    )
    np.testing.assert_allclose(L_pde, L1 + 2.0 * p, rtol=1e-10)

    data5 = {"weights": {"pde": 1.0, "data": 5.0, "reg": 0.0}}
    L_dat, *_ = _objective_off_truth(
        _built_optimiser(tiny_experiment, monkeypatch, "w3", data5)
    )
    np.testing.assert_allclose(L_dat, L1 + 4.0 * d, rtol=1e-10)


def test_loss_weights_reg_activates_regulariser(tiny_experiment, monkeypatch):
    """The regulariser contributes only when loss.weights.reg > 0."""

    def loss_cfg(reg):
        return {
            "weights": {"pde": 1.0, "data": 1.0, "reg": reg},
            "regulariser": {"name": "tikhonov"},
        }

    plain = {"weights": {"pde": 1.0, "data": 1.0, "reg": 0.0}}
    L_plain, *_ = _objective_off_truth(
        _built_optimiser(tiny_experiment, monkeypatch, "r0", plain)
    )
    L_off, *_ = _objective_off_truth(
        _built_optimiser(tiny_experiment, monkeypatch, "r1", loss_cfg(0.0))
    )
    L_on, *_ = _objective_off_truth(
        _built_optimiser(tiny_experiment, monkeypatch, "r2", loss_cfg(1e-3))
    )
    assert L_off == L_plain
    assert L_on > L_plain


@pytest.mark.parametrize(
    "weights",
    [{"pde": 1.0, "data": 1.0, "reg": 0.0}, {"pde": 2.0, "data": 10.0, "reg": 0.0}],
)
def test_logged_terms_sum_to_objective(tiny_experiment, weights):
    """Logged pde_term + data_term (+ reg_term = 0) equal the logged objective."""
    user_cfg = deep_merge(
        tiny_experiment,
        {
            "run": {"run_id": "terms"},
            "loss": {"weights": weights},
            "optimiser": {"n_iter": 3},
        },
    )
    result = run_inverse(resolve_config(user_cfg))

    rows = [
        json.loads(line)
        for line in (result.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows
    for r in rows:
        assert r["reg_term"] == 0.0
        np.testing.assert_allclose(
            r["pde_term"] + r["data_term"], r["loss_total"], rtol=1e-10
        )
        np.testing.assert_allclose(
            r["data_term"], weights["data"] * r["data_loss"], rtol=1e-12
        )
