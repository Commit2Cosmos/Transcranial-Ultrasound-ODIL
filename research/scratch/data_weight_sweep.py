"""Scratch sweep for #97: case 3 of #93 over ``loss.weights.data``.

Case 3: the warm start is replaced by the true wavefield, the initial model is
homogeneous, ``model_preconditioner: none``, regulariser off, inverse crime,
float64. Not package code; run from the repo root, one invocation at a time:

    uv run python research/scratch/data_weight_sweep.py unit 0.1 1 10 100
    uv run python research/scratch/data_weight_sweep.py e2e 0.1 1 10 100

Prints one JSON line per (seam, weight, checkpoint).
"""

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

from conftest import (  # noqa: E402
    F0,
    TINY_RUN_CONFIG,
    make_band,
    make_geometry,
    make_grid,
    make_model,
    make_source,
)

from odil_wave import FrequencySelection, HelmholtzSolver, Wavefield  # noqa: E402
from odil_wave.experiment import deep_merge, resolve_config  # noqa: E402
from odil_wave.experiment.problem import build_problem  # noqa: E402
from odil_wave.experiment.runner import build_optimiser  # noqa: E402
from odil_wave.optimisation import JointFreqODIL  # noqa: E402

UNIT_CHECKPOINTS = (50, 200)
E2E_CHECKPOINTS = (10, 50)


def rel(x, ref):
    return float(torch.linalg.vector_norm(x - ref) / torch.linalg.vector_norm(ref))


def set_true_warm_start(band):
    """Case 3: the wavefield block starts at the true wavefield."""
    u_true = torch.stack([w.amplitude for w in band.true_wavefields]).detach()
    band.u_init = u_true.to(band.u_init.dtype)
    return u_true


def run(opt, band, c_true_int, u_true, checkpoints, seam, weight):
    sl = band.wavefield.grid.interior_slice
    u_true_int = u_true[(...,) + sl]
    rows = []

    def on_log(it):
        if it.iter != 0 and it.iter not in checkpoints:
            return
        rows.append(
            dict(
                seam=seam,
                data_weight=weight,
                iter=it.iter,
                n_eval=it.n_evaluations,
                c_err=rel(it.c_full[sl], c_true_int),
                u_err=rel(it.u[(...,) + sl], u_true_int),
                loss=it.loss,
                pde_term=it.pde_term,
                data_term=it.data_term,
                rel_data_residual=it.rel_data_residual,
                wall_s=round(time.perf_counter() - t0, 2),
            )
        )

    t0 = time.perf_counter()
    opt.minimise(on_log=on_log)
    for r in rows:
        print(json.dumps(r), flush=True)


def unit(weight):
    """``truth_problem`` of tests/test_joint_odil.py, from a homogeneous model."""
    g = make_grid()
    fs = FrequencySelection.from_frequencies(g, [F0])
    geom = make_geometry(g, make_source(g), fs)
    truth = make_model(g, "overdensity", contrast=120.0, radius=0.012)
    observed = HelmholtzSolver(
        Wavefield(g, fs, velocity_model=truth), geom, space_order=2, damping_scale=1.0
    ).solve(verbose=False)
    band = make_band(geom, make_model(g), observed)
    u_true = set_true_warm_start(band)
    opt = JointFreqODIL(
        band,
        n_iter=max(UNIT_CHECKPOINTS),
        inner_max_iter=20,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        lbfgs_lr=1.0,
        pde_weight=1.0,
        data_weight=weight,
        reg_weight=0.0,
        log_every=1,
        verbose=False,
        model_preconditioner="none",
        wavefield_preconditioner="none",
        regulariser=None,
    )
    c_true = truth.c[g.interior_slice].detach()
    run(opt, band, c_true, u_true, UNIT_CHECKPOINTS, "unit", weight)


def e2e_config(weight):
    return resolve_config(
        deep_merge(
            TINY_RUN_CONFIG,
            {
                "loss": {"weights": {"pde": 1.0, "data": weight, "reg": 0.0}},
                "optimiser": {
                    "n_iter": max(E2E_CHECKPOINTS),
                    "log_every": 1,
                    "model_preconditioner": "none",
                },
            },
        )
    )


def e2e(weight):
    """configs/checks/tiny.yaml through the run's own builders
    (build_problem, Problem.make_band, build_optimiser), minus artifact I/O."""
    cfg = e2e_config(weight)
    problem = build_problem(cfg)
    freqs = list(cfg.continuation.bands[0].frequencies_hz)
    band = problem.make_band(freqs, problem.init_velocity)
    u_true = set_true_warm_start(band)
    opt = build_optimiser(problem, band, max(E2E_CHECKPOINTS))
    c_true = problem.truth_velocity.c[problem.grid.interior_slice].detach()
    run(opt, band, c_true, u_true, E2E_CHECKPOINTS, "e2e", weight)


if __name__ == "__main__":
    seam, *weights = sys.argv[1:]
    for w in weights:
        {"unit": unit, "e2e": e2e}[seam](float(w))
