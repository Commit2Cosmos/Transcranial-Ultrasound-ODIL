"""Tests for the ``odil_wave.optimisation`` package.

Covers ``base.py``, ``frequency_continuation.py``, ``helmholtz_utransform.py``,
``joint_odil.py`` and ``u_block_hessian.py``. Pure-math / validation logic is
tested with plain tensors and stub objects; the handful of tests that exercise
real Helmholtz solves (``HelmholtzUTransform``, ``UBlockHessian``, and one
smoke test per optimiser) use a single tiny (12x12) shared grid so they stay
fast (well under a second of actual compute).

This environment has multiple independent OpenMP runtime copies installed
(bundled separately by torch and scikit-learn, plus the conda base env's own
copy, likely from an MKL-linked NumPy/SciPy build); loading two conflicting
copies in one process aborts with "OMP: Error #15" (SIGABRT) -- unrelated to
any odil_wave code. Importing ``odil_wave`` (the top-level package) first
happens to load NumPy/SciPy/matplotlib's OpenMP registration before torch's,
which avoids the conflicting pair on this machine today; it is not a
guaranteed fix (e.g. importing scikit-image before torch still crashes) and
could stop working after a dependency update. Every test module in this
package does the same as a low-cost precaution, not as a principled fix.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import odil_wave  # noqa: F401  (import first: see module docstring)
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from odil_wave.geometry import AcquisitionGeometry
from odil_wave.grid import FrequencySelection, Grid
from odil_wave.loss import InverseLoss, LossConfig, LossTape
from odil_wave.models import VelocityModel
from odil_wave.operator import HelmholtzSolver, WaveEquation
from odil_wave.source import SourceSignal
from odil_wave.wavefield import Wavefield

from odil_wave.optimisation.base import (
    LBFGSB,
    Optimiser,
    StepScheduler,
    _armijo_gd_step_z,
    _c_hat_from_c_param,
    _c_param_bounds,
    _gaussian_smooth_2d,
    _init_c_param,
)
from odil_wave.optimisation.frequency_continuation import (
    BandTimingStats,
    FrequencyBand,
    _as_band,
    _fft_obs_traces,
    run_frequency_continuation,
)
from odil_wave.optimisation.helmholtz_utransform import HelmholtzUTransform
from odil_wave.optimisation.joint_odil import (
    JointFreqODIL,
    SlownessLatent,
    mean_real_sq,
    real_scalar_count,
    rms_abs,
)
from odil_wave.optimisation.u_block_hessian import UBlockHessian


# --------------------------------------------------------------------------- #
# Shared tiny real fixture (12x12 total grid, 1 frequency, 2 shots).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_setup():
    """A minimal real forward/inverse setup, built once and reused read-only.

    Grid/velocity/source/geometry/wavefield/solver/solved-truth are not
    mutated by anything under test (optimisers build their own Parameters and
    return new Wavefield objects), so a single module-scoped instance is safe
    to share across tests.
    """
    grid = Grid(
        interior_shape=(8, 8),
        interior_extent=((-0.02, 0.02), (-0.02, 0.02)),
        c_min=1000.0,
        c_max=2000.0,
        pml_width=2,
        t_max=2e-5,
        init_nt=64,
    )
    vm = VelocityModel(grid, profile="homogeneous", base=1500.0)
    source = SourceSignal(grid, kind="ricker", f0=8e4)
    freq = FrequencySelection.from_frequencies(grid, [8e4])
    geom = AcquisitionGeometry(grid, source, freq, n_receivers=4, n_sources=2)
    wf = Wavefield(grid=grid, frequency_selection=freq, velocity_model=vm)
    solver = HelmholtzSolver(wf, geom, space_order=2, pml_weight=1.0)
    solved = solver.solve(verbose=False)
    obs_traces = torch.stack(
        [geom.extract_observations(w.amplitude) for w in solved], dim=0
    )
    u0 = torch.stack([w.amplitude for w in solved], dim=0)
    return SimpleNamespace(
        grid=grid,
        vm=vm,
        source=source,
        freq=freq,
        geom=geom,
        wf=wf,
        solver=solver,
        solved=solved,
        obs_traces=obs_traces,
        u0=u0,
    )


def _build_loss(tiny_setup, *, weights=None, normalize_data=None, name="loss"):
    """Fresh InverseLoss + LossTape for one test (loss.config.weights and the
    callback are mutated by optimisers, so each test gets its own)."""
    w = weights or {"pde": 1.0, "data": 100.0, "reg": 0.0}
    tape = LossTape(name=name, log_every=1)
    config = LossConfig(
        WaveEquation(tiny_setup.wf, space_order=2, pml_weight=1.0),
        tiny_setup.geom,
        weights=w,
    )
    return InverseLoss(
        config=config,
        observed_traces=tiny_setup.obs_traces,
        callback=tape,
        normalize_data=normalize_data,
    )


# --------------------------------------------------------------------------- #
# base.py: StepScheduler
# --------------------------------------------------------------------------- #
def test_step_scheduler_inactive_from_none_spec():
    sched = StepScheduler.from_spec(5.0, None)
    assert sched.active is False
    assert sched.step(1) == 5.0
    assert sched.step(100) == 5.0


def test_step_scheduler_zero_factor_or_every_n_is_inactive():
    assert StepScheduler(base=3.0, factor=0.0, every_n=1, increase=True).active is False
    assert StepScheduler(base=3.0, factor=2.0, every_n=0, increase=True).active is False


def test_step_scheduler_increase_rescales_every_n_outers():
    sched = StepScheduler(base=1.0, factor=2.0, every_n=2, increase=True)
    assert sched.active is True
    assert sched.step(0) == 1.0  # outer 0 never triggers (requires > 0)
    assert sched.step(1) == 1.0
    assert sched.step(2) == 2.0
    assert sched.step(3) == 2.0
    assert sched.step(4) == 4.0


def test_step_scheduler_decrease_from_spec():
    sched = StepScheduler.from_spec(8.0, (2.0, 1, "decrease"))
    assert sched.step(1) == 4.0
    assert sched.step(2) == 2.0


# --------------------------------------------------------------------------- #
# base.py: _gaussian_smooth_2d
# --------------------------------------------------------------------------- #
def test_gaussian_smooth_2d_sigma_zero_is_identity_noop():
    g = torch.randn(6, 6)
    assert _gaussian_smooth_2d(g, 0.0) is g


def test_gaussian_smooth_2d_preserves_shape_and_constant_field():
    g = torch.full((8, 8), 3.0)
    out = _gaussian_smooth_2d(g, 1.5)
    assert out.shape == g.shape
    assert torch.allclose(out, g, atol=1e-5)


def test_gaussian_smooth_2d_spreads_an_impulse_but_keeps_it_peaked():
    g = torch.zeros(9, 9)
    g[4, 4] = 1.0
    out = _gaussian_smooth_2d(g, 1.0)
    assert 0.0 < out[4, 4].item() < 1.0
    assert out[4, 4] == out.max()


# --------------------------------------------------------------------------- #
# base.py: c-parameterisation helpers
# --------------------------------------------------------------------------- #
def test_c_hat_from_c_param_identity_for_velocity():
    raw = torch.tensor([0.8, 1.2])
    assert torch.equal(_c_hat_from_c_param(raw, "velocity"), raw)


def test_c_hat_from_c_param_squared_slowness_matches_formula():
    raw = torch.tensor([0.25])  # m_hat = 1/c_hat^2
    out = _c_hat_from_c_param(raw, "squared_slowness")
    assert out.item() == pytest.approx(1.0 / math.sqrt(0.25))


def test_c_hat_from_c_param_clamps_extremes_to_finite():
    huge = torch.tensor([1e10])
    tiny = torch.tensor([1e-10])
    assert torch.isfinite(_c_hat_from_c_param(huge, "squared_slowness")).all()
    assert torch.isfinite(_c_hat_from_c_param(tiny, "squared_slowness")).all()


def test_init_c_param_velocity_is_a_clone():
    c_hat0 = torch.tensor([0.9, 1.1])
    out = _init_c_param(c_hat0, "velocity")
    assert torch.equal(out, c_hat0)
    assert out is not c_hat0


def test_init_c_param_squared_slowness_round_trips_through_c_hat_from_c_param():
    c_hat0 = torch.tensor([1.2])
    raw = _init_c_param(c_hat0, "squared_slowness")
    back = _c_hat_from_c_param(raw, "squared_slowness")
    assert back.item() == pytest.approx(c_hat0.item(), rel=1e-5)


def test_c_param_bounds_identity_for_velocity():
    assert _c_param_bounds(0.5, 2.0, "velocity") == (0.5, 2.0)


def test_c_param_bounds_swaps_and_inverts_for_squared_slowness():
    lo, hi = _c_param_bounds(0.5, 2.0, "squared_slowness")
    assert lo == pytest.approx(1.0 / 2.0**2)
    assert hi == pytest.approx(1.0 / 0.5**2)


def test_c_param_bounds_none_passthrough():
    assert _c_param_bounds(None, None, "squared_slowness") == (None, None)


# --------------------------------------------------------------------------- #
# base.py: _armijo_gd_step_z
# --------------------------------------------------------------------------- #
def test_armijo_gd_step_z_requires_populated_gradients():
    z_real = torch.nn.Parameter(torch.tensor([1.0]))
    z_imag = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(RuntimeError):
        _armijo_gd_step_z(
            z_real=z_real,
            z_imag=z_imag,
            pack_z=lambda: torch.complex(z_real, z_imag),
            apply_u=lambda z: z,
            eval_loss=lambda z, u: (u.real**2 + u.imag**2).sum(),
            loss0=torch.tensor(2.0),
            lr=0.1,
        )


def test_armijo_gd_step_z_decreases_a_simple_quadratic():
    z_real = torch.nn.Parameter(torch.tensor([2.0]))
    z_imag = torch.nn.Parameter(torch.tensor([1.0]))

    def pack_z():
        return torch.complex(z_real, z_imag)

    def apply_u(z):
        return z  # identity "solve"

    def eval_loss(z, u):
        return (u.real**2 + u.imag**2).sum()

    z0 = pack_z()
    loss0 = eval_loss(z0, apply_u(z0))
    loss0.backward()
    loss0_value = float(loss0.detach())

    loss1, n_trial = _armijo_gd_step_z(
        z_real=z_real,
        z_imag=z_imag,
        pack_z=pack_z,
        apply_u=apply_u,
        eval_loss=eval_loss,
        loss0=loss0,
        lr=0.1,
    )
    assert n_trial >= 1
    assert float(loss1.detach()) < loss0_value


def test_armijo_gd_step_z_reverts_when_no_step_is_accepted():
    # A single backtrack budget with a pathologically large lr should fail to
    # satisfy Armijo and restore the starting point exactly.
    z_real = torch.nn.Parameter(torch.tensor([2.0]))
    z_imag = torch.nn.Parameter(torch.tensor([1.0]))
    z_r0, z_i0 = z_real.item(), z_imag.item()

    def pack_z():
        return torch.complex(z_real, z_imag)

    def apply_u(z):
        return z

    def eval_loss(z, u):
        return (u.real**2 + u.imag**2).sum()

    loss0 = eval_loss(pack_z(), apply_u(pack_z()))
    loss0.backward()
    loss0_value = float(loss0.detach())

    loss1, n_trial = _armijo_gd_step_z(
        z_real=z_real,
        z_imag=z_imag,
        pack_z=pack_z,
        apply_u=apply_u,
        eval_loss=eval_loss,
        loss0=loss0,
        lr=1e30,
        max_backtracks=1,
    )
    assert n_trial == 1
    assert float(loss1.detach()) == pytest.approx(loss0_value)
    assert z_real.item() == pytest.approx(z_r0)
    assert z_imag.item() == pytest.approx(z_i0)


# --------------------------------------------------------------------------- #
# base.py: Optimiser (abstract base) and LBFGSB option validation
# --------------------------------------------------------------------------- #
def test_optimiser_is_abstract():
    with pytest.raises(TypeError):
        Optimiser(wavefield=None, loss=None)


class _StubGrid:
    c_min = 1000.0
    c_max = 2000.0


class _StubWavefield:
    def __init__(self):
        self.grid = _StubGrid()


def _make_lbfgsb(**opts):
    return LBFGSB(_StubWavefield(), SimpleNamespace(), **opts)


def test_lbfgsb_split_opts_defaults():
    n_iter = _make_lbfgsb()._split_opts()[0]
    assert n_iter == 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"c_param": "bogus"},
        {"u_solve": "bogus"},
        {"u_precond": "bogus"},
        {"z_optim": "bogus"},
        {"z_lr": 0.0},
        {"c_line_search_fn": "bogus"},
        {"u_precond": "z", "z_steps": 0},
        {"u_precond": "z", "c_param": "squared_slowness"},
        {"u_solve": "exact", "u_precond": "z"},
    ],
)
def test_lbfgsb_split_opts_rejects_invalid_combinations(overrides):
    with pytest.raises(ValueError):
        _make_lbfgsb(**overrides)._split_opts()


def test_lbfgsb_seed_complex_rejects_wrong_shot_count(tiny_setup):
    loss = _build_loss(tiny_setup)
    wrong_u_init = [tiny_setup.u0[0]]  # 1 shot, but n_shots below is 2
    opt = LBFGSB(tiny_setup.wf, loss, u_init=wrong_u_init)
    with pytest.raises(ValueError):
        opt._seed_complex(
            n_shots=2, cdtype=tiny_setup.wf.cdtype, device=tiny_setup.grid.device
        )


# --------------------------------------------------------------------------- #
# base.py: LBFGSB integration smoke tests (tiny real grid, 1 outer iteration)
# --------------------------------------------------------------------------- #
def test_lbfgsb_minimise_direct_path_smoke(tiny_setup):
    loss = _build_loss(tiny_setup, normalize_data=None)
    opt = LBFGSB(
        tiny_setup.wf,
        loss,
        clamp=True,
        u_init=list(tiny_setup.u0),
        n_iter=1,
        u_steps=1,
        c_steps=1,
    )
    outputs, tape = opt.minimise()
    assert len(outputs) == tiny_setup.geom.n_sources
    assert tape.result["n_outer_iter"] == 1
    assert math.isfinite(tape.result["loss"])
    # Warm-started exactly at the truth model -> loss should already be ~0.
    assert tape.result["loss"] < 1e-6


def test_lbfgsb_minimise_z_precond_path_smoke(tiny_setup):
    loss = _build_loss(tiny_setup, normalize_data=None)
    opt = LBFGSB(
        tiny_setup.wf,
        loss,
        clamp=True,
        u_init=list(tiny_setup.u0),
        n_iter=1,
        u_precond="z",
        z_steps=1,
        c_steps=1,
    )
    outputs, tape = opt.minimise()
    assert len(outputs) == tiny_setup.geom.n_sources
    assert tape.result["u_precond"] == "z"
    assert math.isfinite(tape.result["loss"])
    assert tape.result["loss"] < 1e-6


def test_lbfgsb_minimise_z_precond_requires_inverse_loss(tiny_setup):
    from odil_wave.loss import ForwardLoss

    tape = LossTape(name="forward", log_every=1)
    config = LossConfig(
        WaveEquation(tiny_setup.wf, space_order=2, pml_weight=1.0),
        tiny_setup.geom,
        weights={"pde": 1.0, "data": 1.0, "reg": 0.0},
    )
    forward_loss = ForwardLoss(config=config, callback=tape)
    opt = LBFGSB(
        tiny_setup.wf, forward_loss, u_precond="z", n_iter=1, z_steps=1, c_steps=1
    )
    with pytest.raises(TypeError):
        opt.minimise()


# --------------------------------------------------------------------------- #
# frequency_continuation.py: FrequencyBand / _as_band / BandTimingStats
# --------------------------------------------------------------------------- #
def test_frequency_band_from_sequence():
    b = FrequencyBand([20e3, 30e3], n_iter=15)
    assert b.frequencies_hz == [20e3, 30e3]
    assert b.n_iter == 15


def test_frequency_band_from_positional_values():
    b = FrequencyBand(20e3, 30e3, n_iter=5)
    assert b.frequencies_hz == [20e3, 30e3]


def test_frequency_band_single_value_no_n_iter():
    b = FrequencyBand(20e3)
    assert b.frequencies_hz == [20e3]
    assert b.n_iter is None


def test_frequency_band_frequencies_hz_kwarg_takes_precedence():
    b = FrequencyBand(999e3, frequencies_hz=[1e3, 2e3])
    assert b.frequencies_hz == [1e3, 2e3]


def test_frequency_band_empty_raises():
    with pytest.raises(ValueError):
        FrequencyBand(frequencies_hz=[])


def test_as_band_passes_through_existing_band():
    b = FrequencyBand([10e3])
    assert _as_band(b) is b


def test_as_band_coerces_raw_sequence():
    coerced = _as_band([10e3, 20e3])
    assert isinstance(coerced, FrequencyBand)
    assert coerced.frequencies_hz == [10e3, 20e3]


def test_band_timing_stats_summary_line_with_loss():
    stats = BandTimingStats(
        band_index=0,
        frequencies_hz=[60e3, 90e3],
        n_iter_requested=10,
        n_iter_run=10,
        helmholtz_s=0.1,
        optimise_s=0.5,
        wall_s=0.6,
        n_u_closure=4,
        n_c_closure=2,
        n_closure=6,
        final_loss=1.23e-4,
    )
    line = stats.summary_line()
    assert "band1" in line
    assert "60, 90" in line
    assert "final_loss=1.230000e-04" in line


def test_band_timing_stats_summary_line_without_loss():
    stats = BandTimingStats(
        band_index=1,
        frequencies_hz=[40e3],
        n_iter_requested=5,
        n_iter_run=3,
        helmholtz_s=0.0,
        optimise_s=0.0,
        wall_s=0.0,
        n_u_closure=0,
        n_c_closure=0,
        n_closure=0,
        final_loss=None,
    )
    assert "final_loss=n/a" in stats.summary_line()


# --------------------------------------------------------------------------- #
# frequency_continuation.py: _fft_obs_traces
# --------------------------------------------------------------------------- #
def test_fft_obs_traces_wrong_ndim_raises():
    freq_sel = SimpleNamespace(n_time=10)
    with pytest.raises(ValueError):
        _fft_obs_traces(freq_sel, torch.zeros(4, 10))


def test_fft_obs_traces_nt_mismatch_raises():
    freq_sel = SimpleNamespace(n_time=10)
    with pytest.raises(ValueError):
        _fft_obs_traces(freq_sel, torch.zeros(2, 8, 4))


def test_fft_obs_traces_real_shapes(tiny_setup):
    freq_sel = tiny_setup.freq
    traces = torch.zeros(2, freq_sel.n_time, 4)
    out = _fft_obs_traces(freq_sel, traces)
    assert out.shape == (2, freq_sel.n_frequencies, 4)
    assert torch.is_complex(out)


# --------------------------------------------------------------------------- #
# frequency_continuation.py: run_frequency_continuation early validation
# --------------------------------------------------------------------------- #
def test_run_frequency_continuation_empty_bands_raises():
    with pytest.raises(ValueError):
        run_frequency_continuation(
            grid=None,
            source=None,
            bands=[],
            observed_time_traces=None,
            velocity_model=None,
        )


def test_run_frequency_continuation_rejects_frequency_selection_kwarg():
    with pytest.raises(ValueError):
        run_frequency_continuation(
            grid=None,
            source=None,
            bands=[FrequencyBand(10e3)],
            observed_time_traces=None,
            velocity_model=None,
            geometry_kwargs={"frequency_selection": object()},
        )


def test_run_frequency_continuation_bad_traces_ndim_raises():
    with pytest.raises(ValueError):
        run_frequency_continuation(
            grid=None,
            source=None,
            bands=[FrequencyBand(10e3)],
            observed_time_traces=torch.zeros(4, 4),
            velocity_model=None,
        )


def test_run_frequency_continuation_shot_count_mismatch_raises():
    with pytest.raises(ValueError):
        run_frequency_continuation(
            grid=None,
            source=None,
            bands=[FrequencyBand(10e3)],
            observed_time_traces=torch.zeros(0, 4, 16),
            velocity_model=None,
        )


# --------------------------------------------------------------------------- #
# helmholtz_utransform.py
# --------------------------------------------------------------------------- #
def test_helmholtz_utransform_inverse_then_apply_round_trips(tiny_setup):
    tf = HelmholtzUTransform(tiny_setup.solver, tiny_setup.vm.c.detach())
    z0 = tf.inverse(tiny_setup.u0)
    u_rt = tf.apply(z0)
    assert torch.allclose(u_rt, tiny_setup.u0, atol=1e-4, rtol=1e-4)


def test_helmholtz_utransform_diagnostics_counts_one_factor(tiny_setup):
    tf = HelmholtzUTransform(tiny_setup.solver, tiny_setup.vm.c.detach())
    d = tf.diagnostics()
    assert d["n_factor"] == 1
    assert d["nf"] == tiny_setup.wf.n_frequencies
    assert d["n_dofs"] == tiny_setup.grid.nx * tiny_setup.grid.ny


def test_helmholtz_utransform_rebuild_increments_factor_count(tiny_setup):
    tf = HelmholtzUTransform(tiny_setup.solver, tiny_setup.vm.c.detach())
    tf.rebuild(tiny_setup.vm.c.detach())
    assert tf.diagnostics()["n_factor"] == 2


def test_helmholtz_utransform_apply_backward_matches_hermitian_solve(tiny_setup):
    tf = HelmholtzUTransform(tiny_setup.solver, tiny_setup.vm.c.detach())
    z0 = tf.inverse(tiny_setup.u0)
    z = z0.detach().clone().requires_grad_(True)
    u = tf.apply(z)
    loss = (u.real.square() + u.imag.square()).sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad.real).all()
    assert torch.isfinite(z.grad.imag).all()


# --------------------------------------------------------------------------- #
# u_block_hessian.py
# --------------------------------------------------------------------------- #
def test_u_block_hessian_matches_autograd_hessian_vector_product(tiny_setup):
    # normalize_data must be None: UBlockHessian's H assumes the *unnormalised*
    # quadratic data term mean|Pu - d|^2 (see the module docstring); the
    # per-receiver-normalised loss is not exactly quadratic in u.
    loss = _build_loss(tiny_setup, normalize_data=None)
    ubh = UBlockHessian(
        tiny_setup.wf, loss, tiny_setup.vm.c.detach(), weights=dict(loss.config.weights)
    )
    result = ubh.verify(tiny_setup.u0.real, tiny_setup.u0.imag, tiny_setup.vm.c.detach())
    assert result["rel_err"] < 1e-4


def test_u_block_hessian_solve_and_apply_shapes(tiny_setup):
    loss = _build_loss(tiny_setup, normalize_data=None)
    ubh = UBlockHessian(
        tiny_setup.wf, loss, tiny_setup.vm.c.detach(), weights=dict(loss.config.weights)
    )
    b = tiny_setup.u0.clone()
    x = ubh.solve(b)
    assert x.shape == b.shape
    assert torch.is_complex(x)
    du = ubh.du_from_grad(b)
    assert torch.allclose(du, -x)


# --------------------------------------------------------------------------- #
# joint_odil.py: pure-math helpers
# --------------------------------------------------------------------------- #
def test_real_scalar_count_real_vs_complex():
    assert real_scalar_count(torch.zeros(3, 4)) == 12
    assert real_scalar_count(torch.zeros(3, 4, dtype=torch.complex64)) == 24


def test_mean_real_sq_matches_manual_computation_real():
    x = torch.tensor([1.0, -2.0, 3.0])
    assert mean_real_sq(x).item() == pytest.approx((1 + 4 + 9) / 3)


def test_mean_real_sq_matches_manual_computation_complex():
    x = torch.tensor([1 + 1j, 2 + 0j])
    expected = (1**2 + 1**2 + 2**2 + 0**2) / 4  # divided by real DOF count = 2*numel
    assert mean_real_sq(x).item() == pytest.approx(expected)


def test_rms_abs_real_no_dim():
    x = torch.tensor([3.0, 4.0])
    assert rms_abs(x).item() == pytest.approx(math.sqrt((9 + 16) / 2))


def test_rms_abs_with_dim():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    out = rms_abs(x, dim=1)
    assert out.shape == (2,)
    assert out[0].item() == pytest.approx(math.sqrt((9 + 16) / 2))
    assert out[1].item() == pytest.approx(0.0)


def test_slowness_latent_m_of_z_stays_within_bounds():
    lat = SlownessLatent(m_min=0.1, m_max=1.0)
    z = torch.linspace(-10.0, 10.0, 21)
    m = lat.m_of_z(z)
    assert torch.all(m >= lat.m_min)
    assert torch.all(m <= lat.m_max)
    assert m.min().item() == pytest.approx(lat.m_min, abs=1e-3)
    assert m.max().item() == pytest.approx(lat.m_max, abs=1e-3)


def test_slowness_latent_z_of_m_inverts_m_of_z():
    lat = SlownessLatent(m_min=0.1, m_max=1.0)
    z = torch.tensor([-2.0, 0.0, 3.0])
    m = lat.m_of_z(z)
    assert torch.allclose(lat.z_of_m(m), z, atol=1e-4)


def test_slowness_latent_c_of_z_is_inverse_sqrt_of_m():
    lat = SlownessLatent(m_min=0.25, m_max=4.0)
    z = torch.tensor([0.5])
    m = lat.m_of_z(z)
    c = lat.c_of_z(z)
    assert c.item() == pytest.approx(1.0 / math.sqrt(m.item()), rel=1e-5)


def test_slowness_latent_dm_dz_matches_autograd():
    lat = SlownessLatent(m_min=0.2, m_max=2.0, z_scale=1.3)
    z = torch.tensor([0.7], requires_grad=True)
    lat.m_of_z(z).backward()
    assert z.grad.item() == pytest.approx(lat.dm_dz(z.detach()).item(), rel=1e-4)


# --------------------------------------------------------------------------- #
# joint_odil.py: JointFreqODIL
# --------------------------------------------------------------------------- #
def test_joint_freq_odil_requires_inverse_loss():
    with pytest.raises(TypeError):
        JointFreqODIL(wavefield=None, loss=object())


def test_joint_freq_odil_rejects_inverted_c_bounds(tiny_setup):
    loss = _build_loss(tiny_setup, normalize_data=None)
    with pytest.raises(ValueError):
        JointFreqODIL(tiny_setup.wf, loss, c_min=2000.0, c_max=1000.0)


def test_joint_freq_odil_minimise_smoke(tiny_setup):
    loss = _build_loss(tiny_setup, normalize_data=None)
    opt = JointFreqODIL(
        tiny_setup.wf,
        loss,
        u_init=list(tiny_setup.u0),
        n_iter=1,
        inner_max_iter=2,
        c_min=1000.0,
        c_max=2000.0,
    )
    outputs, tape = opt.minimise()
    assert len(outputs) == tiny_setup.geom.n_sources
    assert math.isfinite(tape.result["loss"])
    assert tape.result["method"] == "joint"
    # Warm-started exactly at the truth model -> loss should already be ~0.
    assert tape.result["loss"] < 1e-6
