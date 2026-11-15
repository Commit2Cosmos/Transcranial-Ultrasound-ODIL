"""Tests for :mod:`odil_wave.loss` — LossTape and the Regulariser.

These affirm:
* ``Regulariser`` computes Tikhonov / anisotropic-TV / isotropic-TV penalties
  (zero on a constant field) and rejects unknown kinds.
* ``LossTape`` logs an ``Iterate``'s scalars, honours the c-history cadence,
  and rebuilds from saved records into the same plots.
"""

from types import SimpleNamespace
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm

from odil_wave import LossTape, Regulariser
from odil_wave.loss.regulariser import RegKind

from conftest import make_grid, make_model


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
# LossTape
# --------------------------------------------------------------------------- #
def _iterate(loss, c=0.0):
    """The ``Iterate`` fields ``LossTape.log`` reads."""
    return SimpleNamespace(
        loss=loss,
        pde_rel_residual=loss / 10,
        rel_data_residual=loss / 100,
        c_full=torch.full((3, 3), c),
    )


def test_loss_tape_logs_an_iterate():
    tape = LossTape(store_c_history=False)
    tape.log(_iterate(1.0))
    tape.log(_iterate(0.5))
    assert tape.history["loss"] == [1.0, 0.5]
    assert tape.history["pde_rel_residual"] == [0.1, 0.05]
    assert tape.history["rel_data_residual"] == [0.01, 0.005]


def recovery_velocity_norms(truth, recovered, geometry) -> list:
    """The norms of the truth and recovered panels of the recovery figure."""
    LossTape().show_velocity_recovery(truth, recovered, geometry)
    fig = plt.gcf()
    norms = [ax.get_images()[0].norm for ax in fig.axes[:2]]
    plt.close(fig)
    return norms


def test_recovery_of_a_shepp_logan_truth_is_on_the_two_slope_scale(
    grid, geometry, homogeneous_model
):
    truth = make_model(grid, "shepp_logan", scale=0.9)
    for norm in recovery_velocity_norms(truth, homogeneous_model, geometry):
        assert isinstance(norm, TwoSlopeNorm)
        assert (norm.vmin, norm.vcenter, norm.vmax) == (1400.0, 1600.0, 3000.0)


def test_recovery_of_another_truth_shares_a_linear_scale(
    grid, geometry, homogeneous_model
):
    """Truth and recovered share the linear range of both fields."""
    recovered = make_model(grid, "overdensity", contrast=200.0, radius=0.012)
    for norm in recovery_velocity_norms(homogeneous_model, recovered, geometry):
        assert type(norm) is Normalize
        assert (norm.vmin, norm.vmax) == pytest.approx((1500.0, 1700.0))


def test_loss_tape_c_history_cadence():
    """c snapshots honour store_c_history and the c_history_every cadence."""
    tape = LossTape(store_c_history=True, c_history_every=2)
    for i in range(4):
        tape.log(_iterate(float(i), c=float(i)))
    # snapshots kept only when the logged-count is a multiple of 2 -> 2 frames.
    assert [float(c[0, 0]) for c in tape.history["c_history"]] == [1.0, 3.0]

    off = LossTape(store_c_history=False)
    off.log(_iterate(0.0))
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


def test_a_tape_rebuilt_from_records_plots_every_panel(monkeypatch):
    """``show`` on a tape rebuilt from saved rows draws each panel's series,
    as for a live run."""
    monkeypatch.setattr(plt, "show", lambda: None)
    records = [
        {"loss_total": 2.0, "pde_rel_residual": 0.2, "rel_data_residual": 0.5},
        {"loss_total": 1.0, "pde_rel_residual": 0.1, "rel_data_residual": 0.4},
    ]
    LossTape.from_records(records).show()
    fig = plt.gcf()
    assert [len(ax.lines[0].get_ydata()) for ax in fig.axes] == [2, 2, 2]
    plt.close(fig)
