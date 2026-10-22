"""Tests for the joint full-space ODIL optimiser (``JointFreqODIL``)."""

from types import SimpleNamespace

import pytest
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
from odil_wave.loss import Regulariser
from odil_wave.optimisation import JointFreqODIL, SlownessLatent

from conftest import F0, N_CYCLES, OFFSET, make_grid

C_MIN, C_MAX = 1400.0, 1700.0


@pytest.fixture(scope="module")
def truth_problem() -> SimpleNamespace:
    """Tiny inverse-crime problem: observations are a Helmholtz solve on the truth."""
    g = make_grid()
    fs = FrequencySelection.from_frequencies(g, [F0])
    src = SourceSignal(g, kind="tone_burst", f0=F0, n_cycles=N_CYCLES, offset=OFFSET)
    geom = AcquisitionGeometry(
        g, src, fs, n_receivers=8, n_sources=2, source_spatial="point"
    )
    truth = VelocityModel(
        g, profile="overdensity", base=1500.0, contrast=120.0, radius=0.012
    )
    wf = Wavefield(g, fs, velocity_model=truth)
    observed = HelmholtzSolver(wf, geom, space_order=2).solve(verbose=False)
    return SimpleNamespace(wavefield=wf, geometry=geom, observed=observed)


def _make_optimiser(problem, regulariser=None, reg_weight=0.0) -> JointFreqODIL:
    """Joint optimiser on the truth model, warm-started at the observed fields."""
    wave_eq = WaveEquation(problem.wavefield, space_order=2)
    loss = InverseLoss(
        observed_wavefield=problem.observed,
        config=LossConfig(
            wave_eq=wave_eq,
            geometry=problem.geometry,
            weights={"pde": 1.0, "data": 1.0},
            regulariser=regulariser,
        ),
    )
    return JointFreqODIL(
        problem.wavefield,
        loss,
        u_init=problem.observed,
        c_min=C_MIN,
        c_max=C_MAX,
        reg_weight=reg_weight,
    )


# --------------------------------------------------------------------------- #
# SlownessLatent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("z_scale", [1.0, 10.0])
def test_latent_round_trip(z_scale):
    """z_of_c inverts c_of_z away from the logit clip."""
    latent = SlownessLatent(m_min=1 / C_MAX**2, m_max=1 / C_MIN**2, z_scale=z_scale)
    z = torch.linspace(-10.0, 10.0, 101, dtype=torch.float64) / z_scale
    assert torch.allclose(latent.z_of_c(latent.c_of_z(z)), z, rtol=0, atol=1e-10)


def test_latent_keeps_c_within_bounds():
    """c_of_z stays within [c_min, c_max] even for saturated latents."""
    latent = SlownessLatent(m_min=1 / C_MAX**2, m_max=1 / C_MIN**2, z_scale=10.0)
    z = torch.tensor([-50.0, -1.0, 0.0, 1.0, 50.0], dtype=torch.float64)
    c = latent.c_of_z(z)
    assert torch.all(c >= C_MIN * (1 - 1e-12))
    assert torch.all(c <= C_MAX * (1 + 1e-12))


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def test_objective_vanishes_at_truth(truth_problem):
    """At (observed u, truth c) both the PDE and the data residual vanish."""
    opt = _make_optimiser(truth_problem)
    L = opt.objective(*opt.init_params(), record=True)
    assert float(L) < 1e-20
    assert opt._last["pde_term"] < 1e-20
    assert opt._last["data_term"] < 1e-20


def _directional_check(opt, block):
    """Relative error between autograd and central-difference directional derivatives."""
    u_re, u_im, z = opt.init_params()
    gen = torch.Generator().manual_seed(0)

    def randn_like(t):
        return torch.randn(t.shape, generator=gen, dtype=t.dtype)

    # Move z off the truth so the gradient is non-trivial.
    z = z + 0.1 * randn_like(z)
    d = [randn_like(u_re), randn_like(u_im), randn_like(z)]
    if block == "u":
        d[2] = torch.zeros_like(z)
    elif block == "z":
        d[0], d[1] = torch.zeros_like(u_re), torch.zeros_like(u_im)

    x = [t.clone().requires_grad_(True) for t in (u_re, u_im, z)]
    grads = torch.autograd.grad(opt.objective(*x, record=False), x)
    analytic = sum(float((g * di).sum()) for g, di in zip(grads, d))

    h = 1e-6

    def f(s):
        return float(
            opt.objective(
                *(t + s * di for t, di in zip((u_re, u_im, z), d)), record=False
            )
        )

    fd = (f(h) - f(-h)) / (2 * h)
    return abs(fd - analytic) / abs(analytic)


@pytest.mark.parametrize("block", ["u", "z", "joint"])
def test_objective_gradient_matches_finite_difference(truth_problem, block):
    """Autograd gradient of the objective agrees with central differences."""
    opt = _make_optimiser(truth_problem)
    assert _directional_check(opt, block) < 1e-6


def test_objective_gradient_with_regulariser(truth_problem):
    """The regulariser path is differentiated correctly as well."""
    opt = _make_optimiser(
        truth_problem, regulariser=Regulariser("tikhonov"), reg_weight=1e-6
    )
    # the regulariser term actually contributes at the perturbed point.
    plain = _make_optimiser(truth_problem)
    u_re, u_im, z = opt.init_params()
    z = z + 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(1))
    assert float(opt.objective(u_re, u_im, z, record=False)) != pytest.approx(
        float(plain.objective(u_re, u_im, z, record=False)), rel=1e-3
    )
    assert _directional_check(opt, "joint") < 1e-6
