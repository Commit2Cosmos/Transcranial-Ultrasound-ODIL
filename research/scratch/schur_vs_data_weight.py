"""Scratch for #97: the model block's Schur complement against the data weight.

Gauss-Newton blocks from dense per-shot Jacobians of the production residuals
(``JointFreqODIL._residuals_of_u``, scaled as in ``_terms_of_u``):
``S = sum_s (P_s J_z)^T (P_s J_z)``, ``P_s`` the projector off range(J_u,s).
Reports median diag(S)/diag(H_zz) and the spectrum ``mu`` of
``H_zz^-1/2 S H_zz^-1/2``, at the truth and at case 3's start (true wavefield,
homogeneous model). Run from the repo root:

    uv run python research/scratch/schur_vs_data_weight.py unit
    uv run python research/scratch/schur_vs_data_weight.py e2e
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_weight_sweep as dws  # noqa: E402

from odil_wave.experiment.problem import build_problem  # noqa: E402
from odil_wave.experiment.runner import build_optimiser  # noqa: E402

WEIGHTS = (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)


def build(seam):
    """(opt with w_d = 1, u_true, truth c interior)."""
    if seam == "e2e":
        cfg = dws.e2e_config(1.0)
        problem = build_problem(cfg)
        freqs = list(cfg.continuation.bands[0].frequencies_hz)
        band = problem.make_band(freqs, problem.init_velocity)
        u_true = dws.set_true_warm_start(band)
        opt = build_optimiser(problem, band, 1)
        c_true = problem.truth_velocity.c[problem.grid.interior_slice].detach()
        return opt, u_true, c_true
    g = dws.make_grid()
    fs = dws.FrequencySelection.from_frequencies(g, [dws.F0])
    geom = dws.make_geometry(g, dws.make_source(g), fs)
    truth = dws.make_model(g, "overdensity", contrast=120.0, radius=0.012)
    observed = dws.HelmholtzSolver(
        dws.Wavefield(g, fs, velocity_model=truth),
        geom,
        space_order=2,
        damping_scale=1.0,
    ).solve(verbose=False)
    band = dws.make_band(geom, dws.make_model(g), observed)
    u_true = dws.set_true_warm_start(band)
    opt = dws.JointFreqODIL(
        band,
        n_iter=1,
        pde_weight=1.0,
        data_weight=1.0,
        reg_weight=0.0,
        regulariser=None,
    )
    return opt, u_true, truth.c[g.interior_slice].detach()


def jacobians(opt, u, z, s):
    """(J_u, J_z_pde, J_z_data, data row mask) of shot ``s``'s scaled residual
    vector R with sum(R^2) = that shot's part of Phi at w_p = w_d = 1."""
    r_pde, r_data = opt._residuals_of_u(u, z)  # noqa: SLF001 (scratch: the production residuals in physical u)
    a_p = opt.pde_weight * opt.pde_node_factor / (2 * r_pde.numel())
    a_d = 1.0 / (2 * r_data.numel())

    def R(ur, ui, zz):
        uu = u.clone()
        uu[s] = torch.complex(ur, ui)
        rp, rd = opt._residuals_of_u(uu, zz)  # noqa: SLF001 (scratch: the production residuals in physical u)
        rp = (rp / opt.pde_scale)[s] * a_p**0.5
        rd = (rd / opt.data_scale)[s] * a_d**0.5
        return torch.cat(
            [rp.real.flatten(), rp.imag.flatten(), rd.real.flatten(), rd.imag.flatten()]
        )

    n_pde = 2 * r_pde[s].numel()
    Jr, Ji, Jz = torch.autograd.functional.jacobian(
        R, (u[s].real.clone(), u[s].imag.clone(), z), vectorize=True
    )
    M = Jr.shape[0]
    Ju = torch.cat([Jr.reshape(M, -1), Ji.reshape(M, -1)], dim=1)
    return Ju, Jz.reshape(M, -1), n_pde


def schur(opt, u, z):
    """{w_d: stats} at (u, z)."""
    parts = [jacobians(opt, u, z, s) for s in range(opt.n_shots)]
    out = {}
    for w in WEIGHTS:
        S = 0.0
        H = 0.0
        for Ju, Jz, n_pde in parts:
            scale = torch.ones(Ju.shape[0], dtype=Ju.dtype)
            scale[n_pde:] = w**0.5
            Ju_w, Jz_w = Ju * scale[:, None], Jz * scale[:, None]
            Q, _ = torch.linalg.qr(Ju_w)
            Pz = Jz_w - Q @ (Q.T @ Jz_w)
            S = S + Pz.T @ Pz
            H = H + Jz_w.T @ Jz_w
        d = H.diagonal().clamp_min(1e-300)
        Dm = d.rsqrt()
        mu = torch.linalg.eigvalsh(Dm[:, None] * S * Dm[None, :]).clamp_min(0)
        out[w] = dict(
            diagS_over_Hzz_median=float((S.diagonal() / d).median()),
            mu_max=float(mu.max()),
            mu_median=float(mu.median()),
            n_mu_gt_1e_2=int((mu > 1e-2).sum()),
            n_cells=int(mu.numel()),
        )
    return out


if __name__ == "__main__":
    seam = sys.argv[1]
    opt, u_true, c_true = build(seam)
    sl = opt.grid.interior_slice
    z_true = opt.latent.z_of_c(c_true.to(opt.dtype))
    z_init = opt.init_params()[2]
    for label, z in (("truth", z_true), ("case3_start", z_init)):
        for w, st in schur(opt, u_true.to(opt.cdtype), z).items():
            print(
                json.dumps(dict(seam=seam, point=label, data_weight=w, **st)),
                flush=True,
            )
