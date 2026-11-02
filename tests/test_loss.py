"""Tests for :mod:`odil_wave.loss` — LossConfig, LossTape, InverseLoss and
the Regulariser.

These affirm:
* ``Regulariser`` computes Tikhonov / anisotropic-TV / isotropic-TV penalties
  (zero on a constant field) and rejects unknown kinds.
* ``LossConfig`` derives device and dtype from the wave equation's grid.
* ``LossTape`` logs scalar diagnostics from dict/tuple residuals, honours the
  c-history cadence, and rebuilds from saved records.
* ``InverseLoss`` sources make the PDE residual vanish at the true wavefield
  and model, its observations (wavefield or trace mode) match the true
  wavefield at the receivers, and it validates its inputs.
"""

from typing import get_args

import numpy as np
import pytest
import torch

from odil_wave import InverseLoss, LossConfig, LossTape, Regulariser
from odil_wave.loss.inverse import mean_abs_sq
from odil_wave.loss.regulariser import RegKind

from conftest import make_grid


# --------------------------------------------------------------------------- #
# mean_abs_sq
# --------------------------------------------------------------------------- #
def test_mean_abs_sq_real_and_complex():
    """mean_abs_sq averages |x|^2 for both real and complex tensors."""
    x = torch.tensor([3.0, 4.0])
    assert float(mean_abs_sq(x)) == pytest.approx((9 + 16) / 2)
    z = torch.tensor([3 + 4j], dtype=torch.complex128)
    assert float(mean_abs_sq(z)) == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# Regulariser
# --------------------------------------------------------------------------- #
# A 10 cm square interior at 1500 m/s background; c0 = c_max = 1600 m/s.
REG_EXTENT = ((0.0, 0.1), (0.0, 0.1))
REG_C0 = 1600.0


def _reg_grid(shape=(32, 32), extent=REG_EXTENT):
    return make_grid(
        interior_shape=shape, interior_extent=extent, c_min=1400.0, c_max=REG_C0
    )


def _interior_xy(grid):
    sl = grid.interior_slice
    return grid.X[sl], grid.Y[sl]


def _ramp(grid, a=300.0, b=-400.0):
    """``c = 1500 + a x + b y`` (m/s, x and y in m) on the interior."""
    x, y = _interior_xy(grid)
    return 1500.0 + a * x + b * y


# Dimensionless gradient of the default ramp: (a, b) * L0 / c0 with L0 = 0.1 m.
RAMP_GX, RAMP_GY = 300.0 * 0.1 / REG_C0, -400.0 * 0.1 / REG_C0
REG_EPS = 1e-4  # the config default


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("tikhonov", RAMP_GX**2 + RAMP_GY**2),
        ("tv_aniso", abs(RAMP_GX) + abs(RAMP_GY)),
        ("tv_iso", np.sqrt(RAMP_GX**2 + RAMP_GY**2 + REG_EPS) - np.sqrt(REG_EPS)),
    ],
)
def test_regulariser_ramp_matches_closed_form(kind, expected):
    """On a linear ramp each kind is the mean of |grad' c'|^p over the interior."""
    grid = _reg_grid()
    reg = Regulariser(kind=kind, eps=REG_EPS, grid=grid)
    assert float(reg(_ramp(grid))) == pytest.approx(expected, rel=1e-10)


KINDS = get_args(RegKind)


@pytest.mark.parametrize("kind", KINDS)
def test_regulariser_zero_on_constant(kind):
    """Every kind vanishes on a constant field at the default eps."""
    grid = _reg_grid()
    reg = Regulariser(kind=kind, eps=REG_EPS, grid=grid)
    c = torch.full(grid.interior_shape, 1500.0, dtype=torch.float64)
    assert float(reg(c)) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("kind", KINDS)
def test_regulariser_non_square_spacing(kind):
    """The same physical ramp gives the same value with dx != dy."""
    square = _reg_grid(shape=(32, 32))
    oblong = _reg_grid(shape=(41, 21))
    assert oblong.dx != oblong.dy
    r_square = Regulariser(kind=kind, eps=REG_EPS, grid=square)(_ramp(square))
    r_oblong = Regulariser(kind=kind, eps=REG_EPS, grid=oblong)(_ramp(oblong))
    assert float(r_oblong) == pytest.approx(float(r_square), rel=1e-10)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("edge", ["last_row", "last_column"])
def test_regulariser_counts_every_interior_edge(kind, edge):
    """A field that is constant except in its last row/column is penalised."""
    grid = _reg_grid()
    c = torch.full(grid.interior_shape, 1500.0, dtype=torch.float64)
    if edge == "last_row":
        c[-1, :] += 50.0
    else:
        c[:, -1] += 50.0
    reg = Regulariser(kind=kind, eps=REG_EPS, grid=grid)
    assert float(reg(c)) > 1e-6


def _gaussian(grid):
    x, y = _interior_xy(grid)
    return 1500.0 + 100.0 * torch.exp(-((x - 0.05) ** 2 + (y - 0.05) ** 2) / 0.015**2)


def _disc(grid):
    x, y = _interior_xy(grid)
    inside = (x - 0.05) ** 2 + (y - 0.05) ** 2 < 0.025**2
    return 1500.0 + 100.0 * inside.double()


@pytest.mark.parametrize(
    "model, kind, rel",
    [
        (_gaussian, "tikhonov", 0.02),
        (_gaussian, "tv_aniso", 0.02),
        (_gaussian, "tv_iso", 0.02),
        # Tikhonov's continuum value is infinite across a jump.
        (_disc, "tv_aniso", 0.05),
        (_disc, "tv_iso", 0.05),
    ],
)
def test_regulariser_is_resolution_independent(model, kind, rel):
    """One physical model gives the same value at n = 64, 128 and 256."""
    values = []
    for n in (64, 128, 256):
        grid = _reg_grid(shape=(n, n))
        values.append(
            float(Regulariser(kind=kind, eps=REG_EPS, grid=grid)(model(grid)))
        )
    assert values[0] > 0
    for v in values[1:]:
        assert v == pytest.approx(values[0], rel=rel)


def test_regulariser_unknown_kind_raises():
    """An unknown regulariser kind is rejected."""
    with pytest.raises(ValueError):
        Regulariser(kind="l1", eps=REG_EPS, grid=_reg_grid())


def test_regulariser_rejects_non_positive_eps():
    """eps must be > 0: tv_iso's square root has no gradient at 0."""
    with pytest.raises(ValueError, match="eps"):
        Regulariser(kind="tv_iso", eps=0.0, grid=_reg_grid())


# --------------------------------------------------------------------------- #
# LossConfig
# --------------------------------------------------------------------------- #
def test_loss_config_device_and_dtype(wave_stack):
    """LossConfig takes device and dtype from the wave equation's grid."""
    cfg = LossConfig(
        wave_eq=wave_stack.wave_eq, geometry=wave_stack.geometry, regulariser=None
    )
    g = wave_stack.grid
    assert cfg.dtype == g.dtype
    assert cfg.device == g.device


# --------------------------------------------------------------------------- #
# LossTape
# --------------------------------------------------------------------------- #
def test_loss_tape_logs_dict_and_tuple():
    """LossTape stores scalar loss/residual series from dict or tensor residuals."""
    tape = LossTape()
    tape.log(1.0, {"pde_rms": 0.5, "data_rms": 0.2}, pde_rel_residual=0.1)
    r_pde = torch.tensor([3.0, 4.0])
    r_data = torch.tensor([1.0, 1.0])
    tape.log(0.5, (r_pde, r_data))
    assert tape.history["loss"] == [1.0, 0.5]
    assert tape.history["pde_rms"][0] == pytest.approx(0.5)
    # second entry derived from the tuple: rms of [3,4] = sqrt(12.5).
    assert tape.history["pde_rms"][1] == pytest.approx(np.sqrt(12.5))


def test_loss_tape_c_history_cadence():
    """log_c honours store_c_history and the c_history_every cadence."""
    tape = LossTape(store_c_history=True, c_history_every=2)
    for i in range(4):
        tape.log(float(i), {"pde_rms": 0.0})
        tape.log_c(np.full((3, 3), float(i)))
    # snapshots kept only when the logged-count is a multiple of 2 -> 2 frames.
    assert len(tape.history["c_history"]) == 2

    off = LossTape(store_c_history=False)
    off.log(0.0, {"pde_rms": 0.0})
    off.log_c(np.zeros((3, 3)))
    assert off.history["c_history"] == []


def test_loss_tape_from_records():
    """from_records rebuilds the scalar series present in tape rows."""
    records = [
        {"loss_total": 2.0, "rmse_brain": 40.0, "ssim_brain": None},
        {"loss_total": 1.0, "rmse_brain": 30.0, "ssim_brain": None},
    ]
    tape = LossTape.from_records(records, name="loaded")
    assert tape.name == "loaded"
    assert tape.history["loss"] == [2.0, 1.0]
    assert tape.history["rmse_brain"] == [40.0, 30.0]
    assert "ssim_brain" not in tape.history


def test_loss_tape_result_property():
    """The optional result object round-trips through the property."""
    tape = LossTape()
    assert tape.result is None
    tape.result = {"loss": 0.1}
    assert tape.result == {"loss": 0.1}


# --------------------------------------------------------------------------- #
# InverseLoss
# --------------------------------------------------------------------------- #
def _inverse_config(wave_stack, regulariser=None):
    return LossConfig(
        wave_eq=wave_stack.wave_eq,
        geometry=wave_stack.geometry,
        regulariser=regulariser,
    )


def test_inverse_loss_sources_satisfy_pde_at_truth(wave_stack):
    """The loss's t0**2-scaled sources make the PDE residual ~0 at the
    Helmholtz solution of the true model."""
    loss = InverseLoss(
        observed_wavefield=wave_stack.sols, config=_inverse_config(wave_stack)
    )
    r = wave_stack.wave_eq.residual(wave_stack.amp, wave_stack.model.c, loss.sources)
    assert float(mean_abs_sq(r).sqrt()) < 1e-8 * loss.src_rms


@pytest.mark.parametrize("mode", ["wavefield", "traces"])
def test_inverse_loss_observations_match_truth_at_receivers(wave_stack, mode):
    """Both observation modes give the true wavefield's receiver traces."""
    obs = (
        {"observed_wavefield": wave_stack.sols}
        if mode == "wavefield"
        else {"observed_traces": wave_stack.obs_traces}
    )
    loss = InverseLoss(**obs, config=_inverse_config(wave_stack))
    syn = wave_stack.geometry.extract_observations(wave_stack.amp)
    assert loss.obs_traces().shape == syn.shape
    assert torch.equal(syn - loss.obs_traces(), torch.zeros_like(syn))


def test_inverse_loss_requires_exactly_one_observation(wave_stack):
    """Exactly one of observed_wavefield / observed_traces must be supplied."""
    cfg = _inverse_config(wave_stack)
    with pytest.raises(ValueError):
        InverseLoss(config=cfg)  # neither
    with pytest.raises(ValueError):
        InverseLoss(
            observed_wavefield=wave_stack.sols,
            observed_traces=wave_stack.obs_traces,
            config=cfg,
        )  # both


def test_inverse_loss_rejects_non_3d_traces(wave_stack):
    """Trace observations must be (n_shots, nf, n_transducers - 1)."""
    cfg = _inverse_config(wave_stack)
    with pytest.raises(ValueError):
        InverseLoss(observed_traces=wave_stack.obs_traces[0], config=cfg)


def test_inverse_loss_data_residual_ignores_the_transmitting_transducer(wave_stack):
    """Perturbing u only at each shot's own transducer node leaves the data
    residual against the observations at zero."""
    loss = InverseLoss(
        observed_wavefield=wave_stack.sols, config=_inverse_config(wave_stack)
    )
    geom = wave_stack.geometry
    amp = wave_stack.amp.clone()
    for s in range(geom.n_shots):
        i, j = geom.transducer_ij[s]
        amp[s, :, i, j] += 10.0
    r_data = geom.extract_observations(amp) - loss.obs_traces()
    assert float(mean_abs_sq(r_data)) < 1e-30


def test_inverse_loss_default_callback_is_a_fresh_tape(wave_stack):
    """Without a callback, each loss gets its own empty LossTape."""
    a = InverseLoss(
        observed_wavefield=wave_stack.sols, config=_inverse_config(wave_stack)
    )
    b = InverseLoss(
        observed_wavefield=wave_stack.sols, config=_inverse_config(wave_stack)
    )
    assert isinstance(a.callback, LossTape)
    assert a.callback is not b.callback
    assert a.callback.history["loss"] == []
