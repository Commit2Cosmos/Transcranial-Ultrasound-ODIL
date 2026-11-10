"""Tests for the joint full-space ODIL optimiser (``JointFreqODIL``)."""

from types import SimpleNamespace

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
from odil_wave.loss import Regulariser
from odil_wave.optimisation import JointFreqODIL, SlownessLatent
from odil_wave.optimisation.preconditioners import NoModelPreconditioner

from conftest import (
    C_MAX,
    C_MIN,
    F0,
    make_geometry,
    make_grid,
    make_model,
    make_source,
    rel_interior_error,
)


@pytest.fixture(scope="module")
def truth_problem() -> SimpleNamespace:
    """Tiny inverse-crime problem: observations are a Helmholtz solve on the truth."""
    g = make_grid()
    fs = FrequencySelection.from_frequencies(g, [F0])
    src = make_source(g)
    geom = make_geometry(g, src, fs)
    truth = make_model(g, "overdensity", contrast=120.0, radius=0.012)
    wf = Wavefield(g, fs, velocity_model=truth)
    observed, factors = HelmholtzSolver(
        wf, geom, space_order=2, damping_scale=1.0
    ).solve(verbose=False, return_factors=True)
    return SimpleNamespace(
        wavefield=wf, geometry=geom, observed=observed, factors=factors
    )


def _make_optimiser(
    problem,
    regulariser=None,
    reg_weight=0.0,
    model_preconditioner="none",
    wavefield_preconditioner="none",
) -> JointFreqODIL:
    """Joint optimiser on the truth model, warm-started at the observed fields.

    The latent velocity bounds are the grid's (C_MIN, C_MAX). ``problem.u_init``
    (if any) replaces the warm start and ``problem.factors`` (if any) are passed
    as the initial model's factors."""
    wave_eq = WaveEquation(problem.wavefield, space_order=2, damping_scale=1.0)
    loss = InverseLoss(
        observed_wavefield=problem.observed,
        config=LossConfig(
            wave_eq=wave_eq,
            geometry=problem.geometry,
            regulariser=regulariser,
        ),
    )
    return JointFreqODIL(
        problem.wavefield,
        loss,
        u_init=getattr(problem, "u_init", problem.observed),
        n_iter=50,
        inner_max_iter=20,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        lbfgs_lr=1.0,
        pde_weight=1.0,
        data_weight=1.0,
        reg_weight=reg_weight,
        log_every=1,
        verbose=False,
        model_preconditioner=model_preconditioner,
        wavefield_preconditioner=wavefield_preconditioner,
        initial_model_factors=getattr(problem, "factors", None),
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


def test_minimise_writes_residual_diagnostics_to_the_loss(truth_problem):
    """Each logged iteration leaves the scalar diagnostics on
    ``loss.last_residuals`` and appends them to the loss's callback tape."""
    opt = _make_optimiser(truth_problem)
    opt.n_iter = 2
    _, tape = opt.minimise()
    assert set(opt.loss.last_residuals) == {
        "pde_rms",
        "data_rms",
        "pde_loss",
        "data_loss",
        "pde_term",
        "data_term",
        "reg_term",
        "pde_rel_residual",
        "n_evaluations",
    }
    assert tape is opt.loss.callback
    assert len(tape.history["pde_rms"]) == len(tape.history["loss"]) >= 2


def test_pde_scale_is_the_wave_term(truth_problem):
    """pde_scale is |lambda_tt| * u_ref per (shot, frequency), u_ref the interior
    RMS of the warm start (ADR-0002), independent of the source."""
    opt = _make_optimiser(truth_problem)
    sl = (slice(None), slice(None)) + opt.grid.interior_slice
    u_ref = opt.u_seed[sl].abs().square().mean(dim=(2, 3)).sqrt()
    lam_tt = truth_problem.wavefield.frequency_selection.lambda_tt.abs()
    expected = lam_tt.view(1, -1) * u_ref
    assert opt.pde_scale.shape == (opt.n_shots, opt.nf, 1, 1)
    torch.testing.assert_close(opt.pde_scale[..., 0, 0], expected.to(opt.dtype))


def test_pde_term_sums_interior_and_pml_over_the_interior_count(truth_problem):
    """The PDE term sums |r / pde_scale|^2 over interior + PML nodes and divides
    by the interior node count (and 2 per complex entry), so residual in the PML
    counts and a thicker PML does not dilute the term."""
    opt = _make_optimiser(truth_problem)
    u_re, u_im, z = opt.init_params()
    z = z + 0.1  # off the truth, so the residual is non-zero
    opt.objective(u_re, u_im, z, record=True)
    r_pde, _ = opt.residuals(u_re, u_im, z)
    n_int = opt.grid.interior_nx * opt.grid.interior_ny
    expected = (r_pde / opt.pde_scale).abs().square().sum() / (
        2 * n_int * opt.n_shots * opt.nf
    )
    assert opt.last_record["pde_term"] == pytest.approx(float(expected), rel=1e-12)
    assert opt.last_record["pde_rel_residual"] == pytest.approx(
        float((2 * expected).sqrt()), rel=1e-12
    )


def test_data_term_ignores_the_transmitting_transducer(truth_problem):
    """Perturbing u only at each shot's own transducer node leaves the data
    term, and its per-(shot, frequency) scale, unchanged."""
    opt = _make_optimiser(truth_problem)
    n = opt.geometry.n_transducers
    assert opt.data_scale.shape == (n, opt.nf, 1)
    u_re, u_im, z = opt.init_params()
    bumped = u_re.clone()
    for s in range(n):
        i, j = opt.geometry.transducer_ij[s]
        bumped[s, :, i, j] += 10.0
    _, r_data = opt.residuals(u_re, u_im, z)
    _, r_bumped = opt.residuals(bumped, u_im, z)
    assert r_data.shape == (n, opt.nf, n - 1)
    assert torch.equal(r_bumped, r_data)


def _directional_check(opt, block):
    """Relative error of autograd vs central-difference directional derivatives."""
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
        truth_problem,
        regulariser=Regulariser(
            kind="tikhonov", eps=1e-4, grid=truth_problem.wavefield.grid
        ),
        reg_weight=1e-2,
    )
    # the regulariser term actually contributes at the perturbed point.
    plain = _make_optimiser(truth_problem)
    u_re, u_im, z = opt.init_params()
    z = z + 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(1))
    assert float(opt.objective(u_re, u_im, z, record=False)) != pytest.approx(
        float(plain.objective(u_re, u_im, z, record=False)), rel=1e-3
    )
    assert _directional_check(opt, "joint") < 1e-6


# --------------------------------------------------------------------------- #
# Model-block preconditioner (``model_preconditioner = "pde_diagonal"``)
# --------------------------------------------------------------------------- #
def test_change_of_variables_keeps_the_objective_and_scales_the_gradient(
    truth_problem,
):
    """L-BFGS optimises y with z_m = s * y: the objective at the same physical
    point is unchanged, and grad_y Phi = s * grad_{z_m} Phi."""
    plain = _make_optimiser(truth_problem)
    pre = _make_optimiser(truth_problem, model_preconditioner="pde_diagonal")
    s = pre.model_precond.scale
    assert isinstance(plain.model_precond, NoModelPreconditioner)
    assert s.shape == pre.init_params()[2].shape
    assert not torch.allclose(s, torch.ones_like(s))

    u_re, u_im, z = plain.init_params()
    z = z + 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(2))
    y = pre.model_precond.y_of_z(z).clone().requires_grad_(True)
    z_leaf = z.clone().requires_grad_(True)
    L_y = pre.objective(u_re, u_im, pre.model_precond.z_of_y(y), record=False)
    L_z = plain.objective(u_re, u_im, z_leaf, record=False)
    assert float(L_y.detach()) == pytest.approx(float(L_z.detach()), rel=1e-12)
    (g_y,) = torch.autograd.grad(L_y, y)
    (g_z,) = torch.autograd.grad(L_z, z_leaf)
    torch.testing.assert_close(g_y, s * g_z, rtol=1e-10, atol=0.0)


def _small_two_frequency_optimiser(
    space_order,
    pde_weight=2.5,
    data_weight=1.0,
    model_preconditioner="none",
    wavefield_preconditioner="none",
) -> JointFreqODIL:
    """A 10x10-interior, 3-shot, two-frequency band, warm-started by a
    Helmholtz solve on a homogeneous start model (observations from a
    heterogeneous truth), small enough for brute-force autograd Hessians. The
    warm start's factors are passed to the optimiser."""
    g = make_grid(interior_shape=(10, 10))
    fs = FrequencySelection.from_frequencies(g, [30e3, F0])
    geom = make_geometry(g, make_source(g), fs, n_transducers=3)
    operator = dict(space_order=space_order, damping_scale=1.0)
    truth = make_model(g, "overdensity", contrast=120.0, radius=0.012)
    observed = HelmholtzSolver(
        Wavefield(g, fs, velocity_model=truth), geom, **operator
    ).solve(verbose=False)
    wf = Wavefield(g, fs, velocity_model=make_model(g))
    u_init, factors = HelmholtzSolver(wf, geom, **operator).solve(
        verbose=False, return_factors=True
    )
    loss = InverseLoss(
        observed_wavefield=observed,
        config=LossConfig(
            wave_eq=WaveEquation(wf, **operator),
            geometry=geom,
            regulariser=None,
        ),
    )
    return JointFreqODIL(
        wf,
        loss,
        u_init=u_init,
        n_iter=3,
        inner_max_iter=5,
        history_size=5,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        lbfgs_lr=1.0,
        pde_weight=pde_weight,
        data_weight=data_weight,
        reg_weight=0.0,
        log_every=1,
        verbose=False,
        model_preconditioner=model_preconditioner,
        wavefield_preconditioner=wavefield_preconditioner,
        initial_model_factors=factors,
    )


@pytest.mark.parametrize("space_order", [2, 4])
def test_model_curvature_in_k_matches_the_autograd_hessian(space_order):
    """H_kk is the full Hessian of the objective w.r.t. k = (c/c0)^2 on the
    interior (the data term does not depend on k): diagonal, including
    pde_scale and pde_weight."""
    opt = _small_two_frequency_optimiser(space_order)
    u_re, u_im, z0 = opt.init_params()
    H_kk, _ = opt.model_curvature(z0)
    k0 = (opt.latent.c_of_z(z0) / opt.c0).square()

    def phi(k):
        return opt.objective(
            u_re, u_im, opt.latent.z_of_c(opt.c0 * k.sqrt()), record=False
        )

    n = k0.numel()
    H = torch.autograd.functional.hessian(
        lambda k: phi(k.view_as(k0)), k0.reshape(-1), vectorize=True
    ).view(n, n)
    torch.testing.assert_close(H_kk.reshape(-1), H.diagonal(), rtol=1e-8, atol=0.0)
    off = H - torch.diag(H.diagonal())
    assert float(off.abs().max()) <= 1e-8 * float(H.diagonal().max())


@pytest.mark.parametrize("space_order", [2, 4])
def test_model_curvature_in_z_matches_autograd_at_the_warm_start(space_order):
    """At the Helmholtz warm start (r = 0) H_zz is the diagonal of the
    objective's Hessian w.r.t. the physical latent z_m."""
    opt = _small_two_frequency_optimiser(space_order)
    u_re, u_im, z0 = opt.init_params()
    _, H_zz = opt.model_curvature(z0)
    H = torch.autograd.functional.hessian(
        lambda z: opt.objective(u_re, u_im, z.view_as(z0), record=False),
        z0.reshape(-1),
        vectorize=True,
    )
    torch.testing.assert_close(H_zz.reshape(-1), H.diagonal(), rtol=1e-6, atol=0.0)


def test_pde_diagonal_scale_matches_the_wavefield_block_and_is_reported():
    """The built model scale puts every unfloored cell at the wavefield block's
    median curvature T and is summarised in scale_info."""
    opt = _small_two_frequency_optimiser(4, model_preconditioner="pde_diagonal")
    z0 = opt.init_params()[2]
    _, H_zz = opt.model_curvature(z0)
    # The curvature without a wavefield preconditioner, at the initial latent's
    # model (tests/test_preconditioners.py checks it against autograd).
    T = float(opt.wavefield_precond.pde_curvature().median())
    info = opt.scale_info
    assert info["model_preconditioner"] == "pde_diagonal"
    assert info["precond_eps"] == 1e-2
    assert info["precond_target"] == pytest.approx(T, rel=1e-12)
    s = opt.model_precond.scale
    unfloored = H_zz >= 1e-2 * H_zz.median()
    torch.testing.assert_close(
        s[unfloored].square() * H_zz[unfloored],
        torch.full_like(H_zz[unfloored], T),
    )
    assert float(s.max() / s.median()) <= 10.0 * (1 + 1e-12)
    assert info["precond_floored_frac"] == pytest.approx(
        float((~unfloored).double().mean())
    )
    for key in ("h_zz", "model_scale"):
        lo, mid, hi = (info[f"{key}_{q}"] for q in ("min", "median", "max"))
        assert 0.0 < lo <= mid <= hi
    assert info["model_scale_median"] == pytest.approx(float(s.median()))


SCALE_INFO_KEYS = [
    "c_min",
    "c_max",
    "m_min",
    "m_max",
    "c_ref",
    "m_ref",
    "z_scale",
    "pde_weight",
    "data_weight",
    "reg_weight",
    "u_ref_min",
    "u_ref_max",
    "u_ref_mean",
    "pde_scale_min",
    "pde_scale_max",
    "pde_scale_mean",
    "pde_scale_rule",
    "pde_node_factor",
    "data_scale_min",
    "data_scale_max",
    "data_scale_mean",
    "data_scale_rule",
    "wavefield_preconditioner",
    "model_preconditioner",
]
MODEL_SCALE_INFO_KEYS = [
    "precond_eps",
    "precond_target",
    "h_zz_min",
    "h_zz_median",
    "h_zz_max",
    "model_scale_min",
    "model_scale_median",
    "model_scale_max",
    "precond_floored_frac",
]


@pytest.mark.parametrize("wavefield_mode", ["none", "initial_model"])
@pytest.mark.parametrize("mode", ["none", "pde_diagonal"])
def test_scale_info_keys(mode, wavefield_mode):
    """scale_info has the same keys, in the same order, under every pair of
    preconditioners: the run summary reports it as is."""
    opt = _small_two_frequency_optimiser(
        2, model_preconditioner=mode, wavefield_preconditioner=wavefield_mode
    )
    extra = MODEL_SCALE_INFO_KEYS if mode == "pde_diagonal" else []
    assert list(opt.scale_info) == SCALE_INFO_KEYS + extra
    assert opt.scale_info["wavefield_preconditioner"] == wavefield_mode
    assert opt.scale_info["model_preconditioner"] == mode


def test_no_preconditioner_reports_only_the_mode(truth_problem):
    info = _make_optimiser(truth_problem).scale_info
    assert info["model_preconditioner"] == "none"
    assert not any(k.startswith(("precond_", "h_zz", "model_scale")) for k in info)


def test_unknown_model_preconditioner_is_rejected(truth_problem):
    with pytest.raises(ValueError, match="model_preconditioner"):
        _make_optimiser(truth_problem, model_preconditioner="diagonal")


@pytest.mark.parametrize("mode", ["none", "pde_diagonal"])
def test_reported_model_gradient_is_with_respect_to_z_m(mode):
    """grad_z_norm on the final iterate is |grad_{z_m} Phi| in both modes, not
    the gradient w.r.t. the optimised variable y."""
    opt = _small_two_frequency_optimiser(4, model_preconditioner=mode)
    logged = []
    opt.minimise(on_log=logged.append)
    state = opt.final_state
    z = state["z_m"].clone().requires_grad_(True)
    (g_z,) = torch.autograd.grad(
        opt.objective(state["u_re"], state["u_im"], z, record=False), z
    )
    assert logged[-1].grad_z_norm == pytest.approx(float(g_z.norm()), rel=1e-10)
    if mode == "pde_diagonal":
        g_y_norm = float((opt.model_precond.scale * g_z).norm())
        assert logged[-1].grad_z_norm != pytest.approx(g_y_norm, rel=1e-3)


# --------------------------------------------------------------------------- #
# Sanity checks at the truth (inverse crime, no regulariser, float64)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("wavefield_mode", ["none", "initial_model"])
@pytest.mark.parametrize("mode", ["none", "pde_diagonal"])
def test_truth_is_a_fixed_point_of_the_joint_solve(truth_problem, mode, wavefield_mode):
    """Started at (true model, true wavefield), the full outer-iteration budget
    leaves both unchanged. Measured: objective 2.6e-29 (threshold 1e-20); c and
    u change by exactly 0 (threshold 1e-8)."""
    opt = _make_optimiser(
        truth_problem,
        model_preconditioner=mode,
        wavefield_preconditioner=wavefield_mode,
    )
    logged = []
    opt.minimise(on_log=logged.append)
    start, final = logged[0], logged[-1]
    assert final.iter == opt.n_iter
    assert final.loss < 1e-20
    grid = opt.grid
    assert rel_interior_error(final.c_full, start.c_full, grid) < 1e-8
    assert rel_interior_error(final.u, start.u, grid) < 1e-8


def test_model_is_recovered_with_the_wavefield_frozen_at_the_truth(truth_problem):
    """L-BFGS over the model latent alone, with the optimiser's settings and the
    wavefield held at the true wavefield, recovers the true model from a
    homogeneous initial model. Measured: c error 3.25e-2 -> 1.3e-6 (threshold
    1e-4)."""
    p = truth_problem
    wf = p.wavefield
    homogeneous = Wavefield(
        wf.grid, wf.frequency_selection, velocity_model=make_model(wf.grid)
    )
    opt = _make_optimiser(
        SimpleNamespace(wavefield=homogeneous, geometry=p.geometry, observed=p.observed)
    )
    u_re, u_im, z0 = (t.detach() for t in opt.init_params())
    u_re0, u_im0 = u_re.clone(), u_im.clone()
    z = torch.nn.Parameter(z0.clone())
    lbfgs = torch.optim.LBFGS(
        [z],
        lr=opt.lbfgs_lr,
        max_iter=opt.inner_max_iter,
        history_size=opt.history_size,
        line_search_fn=opt.line_search_fn,
        tolerance_grad=opt.tolerance_grad,
        tolerance_change=opt.tolerance_change,
    )

    def closure():
        lbfgs.zero_grad()
        L = opt.objective(u_re, u_im, z, record=False)
        L.backward()
        return L

    for _ in range(opt.n_iter):
        lbfgs.step(closure)

    c_full = opt.c_full_of_z(z.detach())
    assert rel_interior_error(c_full, wf.velocity_model.c, wf.grid) < 1e-4
    assert torch.equal(u_re, u_re0) and torch.equal(u_im, u_im0)


# --------------------------------------------------------------------------- #
# Wavefield preconditioner (``wavefield_preconditioner = "initial_model"``)
# --------------------------------------------------------------------------- #
def test_initial_model_preconditioner_needs_the_factors(truth_problem):
    p = truth_problem
    no_factors = SimpleNamespace(
        wavefield=p.wavefield, geometry=p.geometry, observed=p.observed
    )
    with pytest.raises(ValueError, match="initial_model_factors"):
        _make_optimiser(no_factors, wavefield_preconditioner="initial_model")


def test_unknown_wavefield_preconditioner_is_rejected(truth_problem):
    with pytest.raises(ValueError, match="wavefield_preconditioner"):
        _make_optimiser(truth_problem, wavefield_preconditioner="jacobi")


def test_model_scale_target_is_the_curvature_in_v_with_both_preconditioners():
    """With both preconditioners, T is the median curvature of the weighted PDE
    term in v: at c = c0 the scaled residual is v - f / pde_scale, so the
    curvature is pde_weight / (N_int n_s n_f) at every node, whatever the
    frequency. (Scaled by 1 / lambda_tt^2 instead, float32 can't resolve the
    first L-BFGS step on the default problem.) The objective is quadratic in v,
    so a gradient difference along a unit vector gives the Hessian's row sums;
    data_weight = 0 leaves the PDE term."""
    opt = _small_two_frequency_optimiser(
        2,
        data_weight=0.0,
        model_preconditioner="pde_diagonal",
        wavefield_preconditioner="initial_model",
    )
    v_re, v_im, z0 = opt.init_params()

    def grad_v_re(v):
        v = v.clone().requires_grad_(True)
        return torch.autograd.grad(opt.objective(v, v_im, z0, record=False), v)[0]

    H_rows = grad_v_re(v_re + 1.0) - grad_v_re(v_re)
    n_int = opt.grid.interior_nx * opt.grid.interior_ny
    expected = 2.5 / (n_int * opt.n_shots * opt.nf)
    torch.testing.assert_close(
        H_rows, torch.full_like(H_rows, expected), rtol=1e-6, atol=0.0
    )
    info = opt.scale_info
    assert info["wavefield_preconditioner"] == "initial_model"
    assert info["precond_target"] == pytest.approx(float(H_rows.median()), rel=1e-6)


def _frozen_model_iterations_to(target, interior, pml_width):
    """L-BFGS iterations over the wavefield block alone, with the optimiser's
    settings, the model frozen at the true model and the wavefield
    preconditioned by the homogeneous initial model's factors, until the
    interior wavefield error (all shots) is below ``target``; read after each
    outer step, so exact to within inner_max_iter. ``None`` if never reached."""
    g = make_grid(interior_shape=(interior, interior), pml_width=pml_width)
    fs = FrequencySelection.from_frequencies(g, [F0])
    geom = make_geometry(g, make_source(g), fs)
    operator = dict(space_order=2, damping_scale=1.0)
    truth = make_model(g, "overdensity", contrast=120.0, radius=0.012)
    observed = HelmholtzSolver(
        Wavefield(g, fs, velocity_model=truth), geom, **operator
    ).solve(verbose=False)
    wf = Wavefield(g, fs, velocity_model=make_model(g))
    u_init, factors = HelmholtzSolver(wf, geom, **operator).solve(
        verbose=False, return_factors=True
    )
    opt = _make_optimiser(
        SimpleNamespace(
            wavefield=wf,
            geometry=geom,
            observed=observed,
            u_init=u_init,
            factors=factors,
        ),
        wavefield_preconditioner="initial_model",
    )
    v_re0, v_im0, _ = opt.init_params()
    z_true = opt.latent.z_of_c(truth.c[g.interior_slice])
    u_true = torch.stack([w.amplitude for w in observed])
    v_re, v_im = torch.nn.Parameter(v_re0), torch.nn.Parameter(v_im0)
    lbfgs = torch.optim.LBFGS(
        [v_re, v_im],
        lr=opt.lbfgs_lr,
        max_iter=opt.inner_max_iter,
        history_size=opt.history_size,
        line_search_fn=opt.line_search_fn,
        tolerance_grad=opt.tolerance_grad,
        tolerance_change=opt.tolerance_change,
    )

    def closure():
        lbfgs.zero_grad()
        L = opt.objective(v_re, v_im, z_true, record=False)
        L.backward()
        return L

    for _ in range(opt.n_iter):
        lbfgs.step(closure)
        with torch.no_grad():
            u = opt.physical_u(v_re, v_im)
        if rel_interior_error(u, u_true, g) < target:
            return lbfgs.state[v_re]["n_iter"]
    return None


@pytest.mark.parametrize("interior, pml_width", [(20, 4), (39, 8)])
def test_preconditioned_wavefield_converges_with_the_model_frozen_at_the_truth(
    interior, pml_width
):
    """The same physical problem (0.05 m, 10.53 mm PML, 40 kHz, 8 transducers)
    at dx = 2.63 and 1.32 mm reaches a 1e-3 wavefield error within 60 L-BFGS
    iterations. Measured: 37 and 19 (#96: 38 and 19; unpreconditioned 3639 and
    3660)."""
    n_iter = _frozen_model_iterations_to(1e-3, interior, pml_width)
    assert n_iter is not None and n_iter <= 60
