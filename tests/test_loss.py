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

import numpy as np
import pytest
import torch

from odil_wave import InverseLoss, LossConfig, LossTape, Regulariser
from odil_wave.loss.inverse import mean_abs_sq


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
def test_regulariser_zero_on_constant():
    """Every regulariser kind vanishes on a constant field."""
    c = torch.full((6, 6), 1500.0)
    for kind in ("tikhonov", "tv_aniso", "tv_iso"):
        # eps=0 so the isotropic-TV smoothing floor doesn't add a constant.
        reg = Regulariser(kind=kind, eps=0.0)
        assert float(reg(c)) == pytest.approx(0.0, abs=1e-6)


def test_regulariser_tikhonov_ramp():
    """Tikhonov penalises squared first differences of a linear ramp."""
    # c increasing by 1 per row: dx_c = 1 everywhere in the (n-1)x(n-1) stencil.
    n = 4
    c = torch.arange(n, dtype=torch.float64).view(n, 1).expand(n, n).contiguous()
    reg = Regulariser(kind="tikhonov", eps=1e-8)
    # dx_c == 1 over (n-1)*(n-1) cells, dy_c == 0 -> penalty == (n-1)^2.
    assert float(reg(c)) == pytest.approx((n - 1) ** 2)


def test_regulariser_unknown_kind_raises():
    """An unknown regulariser kind is rejected."""
    with pytest.raises(ValueError):
        Regulariser(kind="l1", eps=1e-8)


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
