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


def _band_outputs(result, band_index=0):
    """``(band_summary.json, metrics.jsonl rows of that band)`` for one band."""
    band_dir = sorted(p for p in (result.run_dir / "bands").iterdir() if p.is_dir())[
        band_index
    ]
    summary = json.loads((band_dir / "band_summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (result.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    return summary, [r for r in rows if r["band_index"] == band_index]


def test_band_stopped_at_the_start_reports_one_evaluation(tiny_experiment):
    """When the warm start already meets tolerance_grad, L-BFGS evaluates the
    objective once and stops: the band reports exactly that."""
    user_cfg = deep_merge(
        tiny_experiment,
        {"run": {"run_id": "grad_tol"}, "optimiser": {"tolerance_grad": 1e30}},
    )
    result = run_inverse(resolve_config(user_cfg))
    band = result.band_results[0]
    summary, rows = _band_outputs(result)

    assert (band.n_iter_requested, band.n_iter_run) == (10, 1)
    assert band.termination_reason == "tolerance_grad"
    assert summary["n_iter_run"] == 1
    assert summary["termination_reason"] == "tolerance_grad"
    assert summary["optimiser_result"]["n_evaluations"] == 1
    assert summary["optimiser_result"]["n_lbfgs_iter"] == 0
    assert [r["solver_iter"] for r in rows] == [0]
    assert rows[-1]["n_evaluations"] == 1


def test_band_stopped_on_tolerance_change_reports_the_lbfgs_state(tiny_experiment):
    """An L-BFGS step ending on tolerance_change ends the band early; the band
    reports the outer iterations, L-BFGS iterations and evaluations it used."""
    user_cfg = deep_merge(
        tiny_experiment,
        {
            "run": {"run_id": "change_tol"},
            "optimiser": {"inner_max_iter": 5, "tolerance_change": 1e-4},
        },
    )
    result = run_inverse(resolve_config(user_cfg))
    band = result.band_results[0]
    summary, rows = _band_outputs(result)
    opt = summary["optimiser_result"]

    assert (band.n_iter_requested, band.n_iter_run) == (10, 4)
    assert band.termination_reason == "tolerance_change"
    assert summary["termination_reason"] == opt["termination_reason"]
    assert opt["n_outer_iter"] == 4
    # the last step stopped short of its inner_max_iter = 5 iterations.
    assert 3 * 5 < opt["n_lbfgs_iter"] < 4 * 5
    # at least one evaluation per L-BFGS iteration, plus one per step.
    assert opt["n_evaluations"] >= opt["n_lbfgs_iter"] + 4
    assert [r["solver_iter"] for r in rows] == [0, 1, 2, 3]
    evals = [r["n_evaluations"] for r in rows]
    assert evals == sorted(evals) and evals[-1] == opt["n_evaluations"]
    legacy = {"n_outer_budget", "n_closure", "n_factor", "n_forward_solves"}
    assert not legacy & opt.keys()


def test_n_evaluations_accumulate_across_bands(tiny_experiment):
    """The tape's n_evaluations counts the whole run: a band's rows start from
    the earlier bands' totals."""
    user_cfg = deep_merge(
        tiny_experiment,
        {
            "run": {"run_id": "evals"},
            "continuation": {
                "bands": [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]
            },
            "optimiser": {"n_iter": 2},
        },
    )
    result = run_inverse(resolve_config(user_cfg))
    (s0, rows0), (s1, rows1) = _band_outputs(result, 0), _band_outputs(result, 1)
    n0 = s0["optimiser_result"]["n_evaluations"]
    n1 = s1["optimiser_result"]["n_evaluations"]

    assert n0 > 0 and n1 > 0
    assert rows0[-1]["n_evaluations"] == n0
    assert rows1[0]["n_evaluations"] > n0
    assert rows1[-1]["n_evaluations"] == n0 + n1


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
    [
        {"pde": 1.0, "data": 1.0, "reg": 0.0},
        {"pde": 2.0, "data": 10.0, "reg": 0.0},
        {"pde": 1.0, "data": 1.0, "reg": 1e-3},
    ],
)
def test_logged_terms_sum_to_objective(tiny_experiment, weights):
    """Logged pde_term + data_term + reg_term equal the logged objective, with
    reg_term = 0 exactly when the regulariser is off."""
    user_cfg = deep_merge(
        tiny_experiment,
        {
            "run": {"run_id": "terms"},
            "loss": {"weights": weights, "regulariser": {"name": "tikhonov"}},
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
        assert (r["reg_term"] > 0.0) == (weights["reg"] > 0.0)
        assert r["reg_term"] >= 0.0
        np.testing.assert_allclose(
            r["pde_term"] + r["data_term"] + r["reg_term"], r["loss_total"], rtol=1e-10
        )
        np.testing.assert_allclose(
            r["data_term"], weights["data"] * r["data_loss"], rtol=1e-12
        )
