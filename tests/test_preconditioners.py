"""Tests for the block preconditioners, built directly (no ``JointFreqODIL``)."""

from types import SimpleNamespace

import pytest
import torch

from odil_wave import FrequencySelection, HelmholtzSolver, WaveEquation, Wavefield
from odil_wave.optimisation.preconditioners import (
    FactorSolve,
    InitialModelWavefieldPreconditioner,
    NoModelPreconditioner,
    NoWavefieldPreconditioner,
    PdeDiagonalModelPreconditioner,
    preconditioner_scale,
)

from conftest import F0, make_geometry, make_grid, make_model, make_source

PDE_WEIGHT = 2.5


def _band(space_order) -> SimpleNamespace:
    """A 10x10-interior, 3-shot, two-frequency band on a homogeneous model,
    with its Helmholtz solve and factors, ``u_ref`` (interior RMS per shot and
    frequency) and the PDE scaling the solver derives from them."""
    g = make_grid(interior_shape=(10, 10))
    fs = FrequencySelection.from_frequencies(g, [30e3, F0])
    geom = make_geometry(g, make_source(g), fs, n_transducers=3)
    wf = Wavefield(g, fs, velocity_model=make_model(g))
    operator = dict(space_order=space_order, damping_scale=1.0)
    solved, factors = HelmholtzSolver(wf, geom, **operator).solve(
        verbose=False, return_factors=True
    )
    u0 = torch.stack([w.amplitude for w in solved])
    sl = (slice(None), slice(None)) + g.interior_slice
    u_ref = u0[sl].abs().square().mean(dim=(2, 3)).sqrt()[..., None, None]
    lam_tt = fs.lambda_tt.abs().to(g.dtype).view(1, -1, 1, 1)
    n_terms = g.interior_nx * g.interior_ny * geom.n_shots * fs.n_frequencies
    sources = torch.stack(
        [geom.source_field(i).to(wf.cdtype) for i in range(geom.n_shots)]
    )
    return SimpleNamespace(
        grid=g,
        wave_eq=WaveEquation(wf, **operator),
        c=wf.velocity_model.c.detach(),
        factors=factors,
        u0=u0,
        u_ref=u_ref,
        pde_scale_per_u=lam_tt.expand_as(u_ref).contiguous(),
        n_terms=n_terms,
        term_weight=PDE_WEIGHT / n_terms,
        sources=sources * g.t0**2,
    )


def _wavefield(band, mode):
    if mode == "none":
        return NoWavefieldPreconditioner(
            band.wave_eq, band.c, 1.0 / band.pde_scale_per_u, PDE_WEIGHT, band.n_terms
        )
    return InitialModelWavefieldPreconditioner(
        band.wave_eq, band.c, band.factors, band.pde_scale_per_u, band.term_weight
    )


def _weighted_pde_term(band, u_tilde):
    """Weighted PDE term of the solver's objective as a function of ``u_tilde``
    (source dropped: the term is affine in ``u``, so its Hessian ignores it)."""
    r = band.wave_eq.residual(u_tilde, band.c, torch.zeros_like(u_tilde))
    r = r / band.pde_scale_per_u
    return 0.5 * band.term_weight * (r.real.square() + r.imag.square()).sum()


# --------------------------------------------------------------------------- #
# Wavefield block
# --------------------------------------------------------------------------- #
def test_factor_solve_backward_is_the_adjoint_of_its_forward():
    """u = A(c0)^-1 v from the warm start's splu factors passes gradcheck: its
    backward (lu.solve with trans="H") is the adjoint of its forward."""
    band = _band(2)
    assert len(band.factors) == 2
    solve = FactorSolve(band.factors)
    gen = torch.Generator().manual_seed(3)
    g = band.grid
    v = torch.randn((1, 2, g.nx, g.ny), generator=gen, dtype=torch.complex128)
    assert torch.autograd.gradcheck(solve, (v.requires_grad_(True),))
    assert solve.seconds > 0.0


@pytest.mark.parametrize("mode", ["none", "initial_model"])
def test_v_and_u_tilde_round_trip(mode):
    """v_of_u_tilde inverts u_tilde_of_v for any wavefield, not only the warm
    start; without the preconditioner both are the identity."""
    band = _band(2)
    pre = _wavefield(band, mode)
    gen = torch.Generator().manual_seed(4)
    u_tilde = torch.randn(band.u0.shape, generator=gen, dtype=band.u0.dtype)
    v_re, v_im = pre.v_of_u_tilde(u_tilde)
    assert v_re.dtype == v_im.dtype == band.grid.dtype
    torch.testing.assert_close(
        pre.u_tilde_of_v(v_re, v_im), u_tilde, rtol=0, atol=1e-10
    )
    if mode == "none":
        assert torch.equal(torch.complex(v_re, v_im), u_tilde)
        assert pre.solve_seconds == 0.0
    else:
        assert pre.solve_seconds > 0.0


def test_warm_start_maps_to_the_scaled_source_and_back():
    """For a Helmholtz warm start, v0 = A(c0) u_tilde0 / |lambda_tt| is
    f / pde_scale, and u_tilde_of_v maps v0 back to the warm start."""
    band = _band(2)
    pre = _wavefield(band, "initial_model")
    u_tilde0 = band.u0 / band.u_ref
    v0 = torch.complex(*pre.v_of_u_tilde(u_tilde0))
    expected = band.sources / (band.pde_scale_per_u * band.u_ref)
    atol = 1e-10 * float(expected.abs().max())
    torch.testing.assert_close(v0, expected, rtol=0, atol=atol)
    back = pre.u_tilde_of_v(v0.real, v0.imag)
    rel = torch.linalg.vector_norm(back - u_tilde0) / torch.linalg.vector_norm(u_tilde0)
    assert float(rel) < 1e-12


@pytest.mark.parametrize("space_order", [2, 4])
@pytest.mark.parametrize("part", ["re", "im"])
def test_identity_curvature_matches_the_autograd_hessian_diagonal(space_order, part):
    """Without a preconditioner, pde_curvature is the Hessian diagonal of the
    weighted PDE term w.r.t. v_re (and v_im), at every node of both
    frequencies, PML and the reflect boundary included."""
    band = _band(space_order)
    H_vv = _wavefield(band, "none").pde_curvature()
    g = band.grid
    assert H_vv.shape == (3, 2, g.nx, g.ny)
    u_tilde = band.u0 / band.u_ref
    re, im = u_tilde.real, u_tilde.imag

    def phi(x):
        x = x.view_as(re)
        z = torch.complex(x, im) if part == "re" else torch.complex(re, x)
        return _weighted_pde_term(band, z)

    x0 = (re if part == "re" else im).reshape(-1)
    H = torch.autograd.functional.hessian(phi, x0, vectorize=True)
    torch.testing.assert_close(H_vv.reshape(-1), H.diagonal(), rtol=1e-10, atol=0.0)


def test_initial_model_curvature_is_constant():
    """With the initial model's factors the weighted PDE term at c0 is
    term_weight / 2 * |v - f / pde_scale|^2, so its curvature in v is
    term_weight at every node, whatever the frequency. (Scaled by
    1 / lambda_tt^2 instead, float32 can't resolve the first L-BFGS step on
    the default problem.) The term is quadratic in v, so a gradient difference
    along a unit vector gives the Hessian's row sums."""
    band = _band(2)
    pre = _wavefield(band, "initial_model")
    v_re, v_im = pre.v_of_u_tilde(band.u0 / band.u_ref)

    def grad_v_re(v):
        v = v.clone().requires_grad_(True)
        phi = _weighted_pde_term(band, pre.u_tilde_of_v(v, v_im))
        return torch.autograd.grad(phi, v)[0]

    H_rows = grad_v_re(v_re + 1.0) - grad_v_re(v_re)
    torch.testing.assert_close(
        H_rows, torch.full_like(H_rows, band.term_weight), rtol=1e-6, atol=0.0
    )
    torch.testing.assert_close(
        pre.pde_curvature(), torch.full_like(H_rows, band.term_weight)
    )


@pytest.mark.parametrize("mode", ["none", "initial_model"])
def test_wavefield_info_reports_the_mode(mode):
    assert _wavefield(_band(2), mode).info() == {"wavefield_preconditioner": mode}


# --------------------------------------------------------------------------- #
# Model block
# --------------------------------------------------------------------------- #
def test_preconditioner_scale_equalises_curvature_and_floors_vanishing_cells():
    """Unfloored cells get s^2 H_zz = T; cells with zero curvature (lap u = 0)
    or a saturated sigmoid (H_zz ~ 1e-300) get a finite s, at most 10x the
    median (eps = 1e-2)."""
    H_zz = torch.tensor([0.0, 1e-300, 1e-3, 0.5, 1.0, 2.0, 3.0, 4.0, 40.0])
    H_zz = H_zz.to(torch.float64)
    s, floored = preconditioner_scale(H_zz, target=7.0, eps=1e-2)
    assert floored.tolist() == [True, True, True] + [False] * 6
    torch.testing.assert_close(
        s[~floored].square() * H_zz[~floored], torch.full((6,), 7.0).double()
    )
    assert torch.isfinite(s).all()
    # median(H_zz) = 1, so the floored cells sit at the floor 1e-2.
    torch.testing.assert_close(
        s[floored], torch.full((3,), (7.0 / 1e-2) ** 0.5).double()
    )
    assert float(s.max() / s.median()) <= 10.0 * (1 + 1e-12)


def test_preconditioner_scale_rejects_zero_curvature():
    with pytest.raises(ValueError, match="curvature"):
        preconditioner_scale(torch.zeros(4, dtype=torch.float64), 1.0, 1e-2)


def test_pde_diagonal_maps_z_through_its_scale_and_reports_it():
    H_zz = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    pre = PdeDiagonalModelPreconditioner(H_zz, target=3.0, eps=1e-2)
    s, floored = preconditioner_scale(H_zz, 3.0, 1e-2)
    assert torch.equal(pre.scale, s)
    z = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    assert torch.equal(pre.z_of_y(z), s * z)
    torch.testing.assert_close(pre.z_of_y(pre.y_of_z(z)), z, rtol=1e-15, atol=0.0)
    info = pre.info()
    assert info["model_preconditioner"] == "pde_diagonal"
    assert info["precond_eps"] == 1e-2
    assert info["precond_target"] == 3.0
    assert info["precond_floored_frac"] == pytest.approx(0.2)
    assert info["model_scale_median"] == pytest.approx(float(s.median()))
    assert info["h_zz_max"] == 4.0


def test_no_model_preconditioner_is_the_identity():
    pre = NoModelPreconditioner()
    z = torch.randn(4, dtype=torch.float64)
    assert pre.z_of_y(z) is z and pre.y_of_z(z) is z
    assert pre.info() == {"model_preconditioner": "none"}
