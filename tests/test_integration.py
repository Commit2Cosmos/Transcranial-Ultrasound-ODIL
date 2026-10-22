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
from odil_wave.experiment import build_problem, resolve_config, run_inverse, runner

from conftest import make_grid, tiny_run_config


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
    user_cfg = tiny_run_config(tmp_path)
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


def test_run_inverse_two_bands_chain_models(tmp_path, monkeypatch):
    """Each band writes its artifacts and band 2 starts from band 1's model."""
    user_cfg = tiny_run_config(tmp_path, run_id="two_bands")
    user_cfg["continuation"]["bands"] = [
        {"frequencies_hz": [30e3]},
        {"frequencies_hz": [40e3]},
    ]
    user_cfg["optimiser"]["n_iter"] = 3
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
