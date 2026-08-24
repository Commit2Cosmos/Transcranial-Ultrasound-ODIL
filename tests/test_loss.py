"""Tests for :mod:`odil_wave.loss` — LossConfig, LossTape, ForwardLoss,
InverseLoss and the Regulariser.

These affirm:
* ``Regulariser`` computes Tikhonov / anisotropic-TV / isotropic-TV penalties
  (zero on a constant field) and rejects unknown kinds.
* ``LossConfig`` merges default weights and derives the packed wavefield DOF
  offset.
* ``LossTape`` logs scalar diagnostics from dict/tuple residuals, honours the
  c-history cadence, and rebuilds from saved records.
* ``ForwardLoss`` / ``InverseLoss`` are (near) zero at the true wavefield+model,
  scale with the PDE/data weights, support trace observations, per-receiver
  normalisation, ``evaluate_z`` and weight overrides, and validate their inputs.
"""

import numpy as np
import pytest
import torch

from odil_wave import ForwardLoss, InverseLoss, LossConfig, LossTape, Regulariser
from odil_wave.loss.base import mean_abs_sq


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
    reg = Regulariser(kind="tikhonov")
    # dx_c == 1 over (n-1)*(n-1) cells, dy_c == 0 -> penalty == (n-1)^2.
    assert float(reg(c)) == pytest.approx((n - 1) ** 2)


def test_regulariser_unknown_kind_raises():
    """An unknown regulariser kind is rejected."""
    with pytest.raises(ValueError):
        Regulariser(kind="l1")


# --------------------------------------------------------------------------- #
# LossConfig
# --------------------------------------------------------------------------- #
def test_loss_config_merges_weights_and_offset(wave_stack):
    """Missing weight keys default to 1.0 and speed_offset packs the u DOFs."""
    cfg = LossConfig(
        wave_eq=wave_stack.wave_eq,
        geometry=wave_stack.geometry,
        weights={"pde": 2.0},
    )
    assert cfg.weights == {"pde": 2.0, "data": 1.0, "reg": 1.0}
    g = wave_stack.grid
    nf = wave_stack.freq.n_frequencies
    expected = wave_stack.geometry.n_sources * nf * g.nx * g.ny * 2
    assert cfg.speed_offset == expected
    assert cfg.dtype == g.dtype


# --------------------------------------------------------------------------- #
# LossTape
# --------------------------------------------------------------------------- #
def test_loss_tape_logs_dict_and_tuple():
    """LossTape stores scalar loss/residual series from dict or tensor residuals."""
    tape = LossTape()
    tape.log(1.0, {"pde_rms": 0.5, "data_rms": 0.2}, pde_src_ratio=0.1)
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
    """from_records rebuilds the scalar series present in metric rows."""
    records = [
        {"loss_total": 2.0, "pde_rms": 0.4, "data_rms": 0.2},
        {"loss_total": 1.0, "pde_rms": 0.3, "data_rms": 0.1},
    ]
    tape = LossTape.from_records(records, name="loaded")
    assert tape.name == "loaded"
    assert tape.history["loss"] == [2.0, 1.0]
    assert tape.history["pde_rms"] == [0.4, 0.3]


def test_loss_tape_result_property():
    """The optional result object round-trips through the property."""
    tape = LossTape()
    assert tape.result is None
    tape.result = {"loss": 0.1}
    assert tape.result == {"loss": 0.1}


# --------------------------------------------------------------------------- #
# ForwardLoss
# --------------------------------------------------------------------------- #
def test_forward_loss_zero_at_solution(wave_stack):
    """The PDE-residual loss is ~0 at the Helmholtz solution of the truth model."""
    cfg = LossConfig(
        wave_eq=wave_stack.wave_eq,
        geometry=wave_stack.geometry,
        weights={"pde": 1.0, "data": 1.0},
    )
    loss = ForwardLoss(cfg)
    L = loss.evaluate(wave_stack.amp, wave_stack.model.c)
    assert float(L) < 1e-18
    assert not torch.is_complex(L)
    assert loss.evaluations == 1


def test_forward_loss_scales_with_pde_weight(wave_stack):
    """Doubling the PDE weight doubles the forward loss."""
    base = LossConfig(
        wave_eq=wave_stack.wave_eq,
        geometry=wave_stack.geometry,
        weights={"pde": 1.0},
    )
    dbl = LossConfig(
        wave_eq=wave_stack.wave_eq,
        geometry=wave_stack.geometry,
        weights={"pde": 2.0},
    )
    # perturb the field so the residual is non-zero.
    amp = wave_stack.amp * 1.1
    L1 = ForwardLoss(base).evaluate(amp, wave_stack.model.c)
    L2 = ForwardLoss(dbl).evaluate(amp, wave_stack.model.c)
    assert float(L2) == pytest.approx(2.0 * float(L1), rel=1e-6)


# --------------------------------------------------------------------------- #
# InverseLoss
# --------------------------------------------------------------------------- #
def _inverse_config(wave_stack, weights=None):
    return LossConfig(
        wave_eq=wave_stack.wave_eq,
        geometry=wave_stack.geometry,
        weights=weights or {"pde": 1.0, "data": 1.0},
    )


def test_inverse_loss_zero_at_truth(wave_stack):
    """PDE + data misfit is ~0 at the true wavefield and model."""
    loss = InverseLoss(
        observed_wavefield=wave_stack.sols, config=_inverse_config(wave_stack)
    )
    L = loss.evaluate(wave_stack.amp, wave_stack.model.c)
    assert float(L) < 1e-18
    # diagnostics are recorded on _last_residuals.
    assert set(loss._last_residuals) >= {"pde_rms", "data_rms", "pde_src_ratio"}


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
    """Trace observations must be (n_shots, nf, n_receivers)."""
    cfg = _inverse_config(wave_stack)
    with pytest.raises(ValueError):
        InverseLoss(observed_traces=wave_stack.obs_traces[0], config=cfg)


def test_inverse_loss_trace_mode_zero_at_truth(wave_stack):
    """Trace-observation mode is also ~0 at the true wavefield/model."""
    loss = InverseLoss(
        observed_traces=wave_stack.obs_traces, config=_inverse_config(wave_stack)
    )
    L = loss.evaluate(wave_stack.amp, wave_stack.model.c)
    assert float(L) < 1e-18


def test_inverse_loss_data_misfit_positive_when_wrong(wave_stack):
    """A wrong wavefield produces a positive data misfit."""
    cfg = _inverse_config(wave_stack, weights={"pde": 0.0, "data": 1.0})
    loss = InverseLoss(observed_wavefield=wave_stack.sols, config=cfg)
    L = loss.evaluate(wave_stack.amp * 0.5, wave_stack.model.c)
    assert float(L) > 0


def test_inverse_loss_weights_override(wave_stack):
    """weights_override replaces config weights for a single evaluation."""
    loss = InverseLoss(
        observed_wavefield=wave_stack.sols,
        config=_inverse_config(wave_stack, {"pde": 1.0, "data": 1.0}),
    )
    amp = wave_stack.amp * 0.5
    full = float(loss.evaluate(amp, wave_stack.model.c))
    data_only = float(
        loss.evaluate(amp, wave_stack.model.c, weights_override={"pde": 0.0})
    )
    assert data_only < full  # dropping the PDE term lowers the loss


def test_inverse_loss_evaluate_z(wave_stack):
    """evaluate_z routes PDE through z-f and data through u; ~0 at (src, truth)."""
    loss = InverseLoss(
        observed_wavefield=wave_stack.sols, config=_inverse_config(wave_stack)
    )
    Lz = loss.evaluate_z(loss.sources, wave_stack.amp, wave_stack.model.c)
    assert float(Lz) < 1e-18


def test_inverse_loss_per_receiver_normalisation(wave_stack):
    """Per-receiver normalisation rescales the data residual (toggleable)."""
    loss = InverseLoss(
        observed_wavefield=wave_stack.sols,
        config=_inverse_config(wave_stack, {"pde": 0.0, "data": 1.0}),
        normalize_data="per_receiver",
    )
    assert loss._trace_scale is not None
    L_norm = float(loss.evaluate(wave_stack.amp * 0.5, wave_stack.model.c))
    loss.set_normalization(False)
    assert loss._trace_scale is None
    L_raw = float(loss.evaluate(wave_stack.amp * 0.5, wave_stack.model.c))
    # both positive, and normalisation changes the misfit scale.
    assert L_norm > 0 and L_raw > 0 and L_norm != pytest.approx(L_raw)


def test_inverse_loss_regulariser_term(wave_stack):
    """A configured regulariser adds a positive penalty for a varying c_interior."""
    cfg = LossConfig(
        wave_eq=wave_stack.wave_eq,
        geometry=wave_stack.geometry,
        weights={"pde": 1.0, "data": 1.0, "reg": 1.0},
        regulariser=Regulariser("tikhonov"),
    )
    loss = InverseLoss(observed_wavefield=wave_stack.sols, config=cfg)
    g = wave_stack.grid
    c_int = wave_stack.model.c[g.interior_slice].clone()
    c_int[0, 0] += 100.0  # introduce a gradient
    L_with = float(loss.evaluate(wave_stack.amp, wave_stack.model.c, c_interior=c_int))
    L_without = float(loss.evaluate(wave_stack.amp, wave_stack.model.c))
    assert L_with > L_without
