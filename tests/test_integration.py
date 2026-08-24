import json

import numpy as np
import torch

from odil_wave import (
    AcquisitionGeometry,
    FrequencySelection,
    HelmholtzSolver,
    InverseLoss,
    LossConfig,
    SourceSignal,
    VelocityModel,
    WaveEquation,
    Wavefield,
)
from odil_wave.experiment import build_problem, resolve_config, run_inverse

from conftest import make_grid


def _rel_c_error(model, truth, grid) -> float:
    """Relative L2 velocity error on the interior."""
    c = model.c[grid.interior_slice].detach().cpu().numpy().astype(np.float64)
    t = truth.c[grid.interior_slice].detach().cpu().numpy().astype(np.float64)
    return float(np.linalg.norm(c - t) / np.linalg.norm(t))


def test_forward_to_loss_gradient_flows():
    """A Helmholtz forward solve feeds a differentiable inverse loss in ``c``."""
    grid = make_grid(interior_shape=(16, 16))
    fs = FrequencySelection.from_frequencies(grid, [40e3])
    src = SourceSignal(grid, kind="tone_burst", f0=40e3, n_cycles=2.0, offset=5)
    geom = AcquisitionGeometry(
        grid, src, fs, n_receivers=8, n_sources=2, source_spatial="point"
    )

    # Truth data from an overdensity medium.
    truth = VelocityModel(
        grid, profile="overdensity", base=1500.0, contrast=120.0, radius=0.012
    )
    wf_truth = Wavefield(grid, fs, velocity_model=truth)
    sols = HelmholtzSolver(wf_truth, geom, space_order=2).solve(verbose=False)

    # A homogeneous starting model, warm-started at its own Helmholtz field.
    init = VelocityModel(grid, profile="homogeneous", base=1500.0)
    wf = Wavefield(grid, fs, velocity_model=init)
    u0 = torch.stack(
        [
            w.amplitude
            for w in HelmholtzSolver(wf, geom, space_order=2).solve(verbose=False)
        ]
    )

    wave_eq = WaveEquation(wf, space_order=2)
    cfg = LossConfig(
        wave_eq=wave_eq, geometry=geom, weights={"pde": 1.0, "data": 100.0}
    )
    loss = InverseLoss(observed_wavefield=sols, config=cfg)

    c = init.c.clone().requires_grad_(True)
    L = loss.evaluate(u0, c)
    L.backward()
    # The data misfit is non-zero (wrong model) and its gradient reaches c.
    assert float(L.detach()) > 0
    assert c.grad is not None
    assert float(c.grad.abs().max()) > 0


def test_run_inverse_tiny_problem_recovers(tmp_path):
    """A full tiny run_inverse completes, writes artifacts, and improves the model."""
    user_cfg = {
        "run": {"output_root": str(tmp_path), "run_id": "tiny"},
        "runtime": {"dtype": "float64"},
        "grid": {
            "interior_shape": [20, 20],
            "interior_extent": [[0.0, 0.05], [0.0, 0.05]],
            "c_min": 1400.0,
            "c_max": 1750.0,
            "t_max": 6e-5,
            "init_nt": 200,
            "pml_width": 6,
        },
        "source": {"kind": "tone_burst", "f0": 40e3, "n_cycles": 2.0},
        "acquisition": {"n_receivers": 8, "n_sources": 4, "source_spatial": "point"},
        "physics": {"space_order": 2},
        "truth": {
            "profile": "overdensity",
            "base": 1500.0,
            "contrast": 150.0,
            "pml_fill": "edge",
            "extra": {"radius": 0.012},
        },
        "init": {"profile": "homogeneous", "base": 1500.0, "pml_fill": "edge"},
        "observation": {
            "method": "helmholtz",
            "normalize_data": "none",
            "pml_width": None,
        },
        "continuation": {
            "warm_start": "helmholtz",
            "bands": [{"frequencies_hz": [40e3]}],
        },
        "loss": {"weights": {"pde": 1.0, "data": 100.0, "reg": 0.0}},
        "optimiser": {
            "name": "lbfgsb",
            "n_iter": 4,
            "u_steps": 5,
            "c_steps": 1,
            "max_iter": 10,
            "log_every": 1,
            "lbfgsb": {"u_solve": "exact", "c_lr": 1.0, "c_max_iter": 3},
        },
        "metrics": {"ssim": {"mask": "none"}},
    }
    cfg = resolve_config(user_cfg)

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
