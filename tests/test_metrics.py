"""Tests for the ``odil_wave.metrics`` package.

Covers ``mse``, ``mae``, ``ssim``, ``ssim_map`` and their shared helpers.
Tests use small synthetic grids to keep execution fast.

``matplotlib`` uses the non-interactive ``Agg`` backend so ``ssim_map`` tests
do not block on ``plt.show()``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import odil_wave  # noqa: F401,E402  (import before torch: see test_optimisation.py)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from odil_wave.grid import Grid  # noqa: E402
from odil_wave.metrics.metrics import (  # noqa: E402
    _interior,
    _interior_mask,
    _to_numpy,
)
from odil_wave.metrics import mae, mse, ssim, ssim_map  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures ``ssim_map`` opens so they don't pile up across tests."""
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Shared grids. mse/mae only need slicing, so a small grid is enough; ssim
# needs an interior >= skimage's default 7x7 SSIM window.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_grid():
    """6x6 interior, pml_width=2 -> (10, 10) full grid."""
    return Grid(
        interior_shape=(6, 6),
        c_min=1000.0,
        c_max=2000.0,
        pml_width=2,
        t_max=1e-5,
        init_nt=32,
    )


@pytest.fixture(scope="module")
def ssim_grid():
    """9x9 interior, pml_width=2 -> (13, 13) full grid (>= default win_size=7)."""
    return Grid(
        interior_shape=(9, 9),
        c_min=1000.0,
        c_max=2000.0,
        pml_width=2,
        t_max=1e-5,
        init_nt=32,
    )


def _full_field(grid, interior_values, pml_value=0.0):
    """A full-grid array with ``interior_values`` inside and ``pml_value`` outside."""
    full = np.full(grid.shape, pml_value, dtype=np.float64)
    full[grid.interior_slice] = interior_values
    return full


# --------------------------------------------------------------------------- #
# _to_numpy
# --------------------------------------------------------------------------- #
def test_to_numpy_converts_torch_tensor_to_float64():
    t = torch.tensor([1.0, 2.5], dtype=torch.float32)
    out = _to_numpy(t)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert np.allclose(out, [1.0, 2.5])


def test_to_numpy_detaches_a_tensor_requiring_grad():
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    out = _to_numpy(t)  # must not raise (requires .detach() before .numpy())
    assert np.allclose(out, [1.0, 2.0])


def test_to_numpy_casts_numpy_array_dtype():
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = _to_numpy(arr)
    assert out.dtype == np.float64
    assert np.array_equal(out, [1.0, 2.0, 3.0])


def test_to_numpy_accepts_plain_list():
    out = _to_numpy([1.0, 2.0, 3.0])
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# _interior
# --------------------------------------------------------------------------- #
def test_interior_slices_out_the_pml_region(small_grid):
    interior_vals = np.arange(36, dtype=np.float64).reshape(6, 6)
    full = _full_field(small_grid, interior_vals, pml_value=-999.0)
    out = _interior(full, small_grid)
    assert out.shape == (6, 6)
    assert np.array_equal(out, interior_vals)
    assert -999.0 not in out


def test_interior_leaves_an_already_interior_shaped_array_unchanged(small_grid):
    interior_vals = np.arange(36, dtype=np.float64).reshape(6, 6)
    out = _interior(interior_vals, small_grid)
    assert out.shape == (6, 6)
    assert np.array_equal(out, interior_vals)


def test_interior_is_identity_when_pml_width_is_zero():
    grid = Grid(
        interior_shape=(5, 5),
        c_min=1000.0,
        c_max=2000.0,
        pml_width=0,
        t_max=1e-5,
        init_nt=32,
    )
    vals = np.arange(25, dtype=np.float64).reshape(5, 5)
    assert np.array_equal(_interior(vals, grid), vals)


def test_interior_raises_on_shape_matching_neither_full_nor_interior(small_grid):
    # small_grid: full shape (10, 10), interior shape (6, 6); (7, 5) is neither.
    with pytest.raises(ValueError):
        _interior(np.zeros((7, 5)), small_grid)


# --------------------------------------------------------------------------- #
# _interior_mask
# --------------------------------------------------------------------------- #
def test_interior_mask_slices_a_full_grid_mask(small_grid):
    full_mask = np.zeros(small_grid.shape, dtype=bool)
    full_mask[small_grid.interior_slice] = True
    out = _interior_mask(full_mask, small_grid)
    assert out.shape == (6, 6)
    assert out.all()


def test_interior_mask_leaves_an_already_interior_shaped_mask_unchanged(small_grid):
    interior_mask = np.array([[True, False] * 3] * 6)
    out = _interior_mask(interior_mask, small_grid)
    assert out.shape == (6, 6)
    assert np.array_equal(out, interior_mask)


def test_interior_mask_accepts_torch_bool_tensor(small_grid):
    full_mask = torch.zeros(small_grid.shape, dtype=torch.bool)
    full_mask[small_grid.interior_slice] = True
    out = _interior_mask(full_mask, small_grid)
    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.shape == (6, 6)


def test_interior_mask_coerces_int_mask_to_bool(small_grid):
    interior_mask_int = np.array([[1, 0, 1, 0, 1, 0]] * 6)
    out = _interior_mask(interior_mask_int, small_grid)
    assert out.dtype == bool
    assert np.array_equal(out, interior_mask_int.astype(bool))


def test_interior_mask_raises_on_shape_matching_neither_full_nor_interior(small_grid):
    with pytest.raises(ValueError):
        _interior_mask(np.ones((3, 3), dtype=bool), small_grid)


# --------------------------------------------------------------------------- #
# mse / mae
# --------------------------------------------------------------------------- #
def test_mse_zero_for_identical_fields(small_grid):
    interior_vals = np.random.RandomState(0).rand(6, 6)
    full = _full_field(small_grid, interior_vals)
    assert mse(full, full, small_grid) == 0.0


def test_mse_matches_manual_value_and_ignores_pml(small_grid):
    interior_vals = np.random.RandomState(1).rand(6, 6)
    true_full = _full_field(small_grid, interior_vals, pml_value=0.0)
    pred_full = _full_field(small_grid, interior_vals + 3.0, pml_value=1e6)
    assert mse(pred_full, true_full, small_grid) == pytest.approx(9.0)  # 3.0**2


def test_mae_matches_manual_value_and_ignores_pml(small_grid):
    interior_vals = np.random.RandomState(2).rand(6, 6)
    true_full = _full_field(small_grid, interior_vals, pml_value=0.0)
    pred_full = _full_field(small_grid, interior_vals - 3.0, pml_value=-1e6)
    assert mae(pred_full, true_full, small_grid) == pytest.approx(3.0)


def test_mse_torch_and_numpy_inputs_agree(small_grid):
    interior_vals = np.random.RandomState(3).rand(6, 6)
    true_full = _full_field(small_grid, interior_vals)
    pred_full = _full_field(small_grid, interior_vals + 1.0)
    val_numpy = mse(pred_full, true_full, small_grid)
    val_torch = mse(torch.tensor(pred_full), torch.tensor(true_full), small_grid)
    assert val_torch == pytest.approx(val_numpy)


def test_mse_returns_plain_float(small_grid):
    full = _full_field(small_grid, np.zeros((6, 6)))
    out = mse(full, full, small_grid)
    assert isinstance(out, float)


def test_mse_raises_on_shape_matching_neither_full_nor_interior_grid_shape(small_grid):
    true_full = _full_field(small_grid, np.full((6, 6), 5.0))
    # small_grid: full shape (10, 10), interior shape (6, 6); (3, 3) is neither.
    with pytest.raises(ValueError):
        mse(np.zeros((3, 3)), true_full, small_grid)


def test_mse_accepts_already_interior_shaped_inputs(small_grid):
    interior_vals = np.random.RandomState(17).rand(6, 6)
    true_full = _full_field(small_grid, interior_vals)
    pred_full = _full_field(small_grid, interior_vals + 2.0)
    full_grid_result = mse(pred_full, true_full, small_grid)

    interior_result = mse(interior_vals + 2.0, interior_vals, small_grid)
    assert interior_result == pytest.approx(full_grid_result)
    assert interior_result == pytest.approx(4.0)  # 2.0**2


# --------------------------------------------------------------------------- #
# ssim
# --------------------------------------------------------------------------- #
def test_ssim_is_one_for_identical_fields(ssim_grid):
    interior_vals = np.random.RandomState(4).rand(9, 9)
    full = _full_field(ssim_grid, interior_vals)
    assert ssim(full, full, ssim_grid) == pytest.approx(1.0)


def test_ssim_is_less_than_one_for_different_fields(ssim_grid):
    rng = np.random.RandomState(5)
    true_full = _full_field(ssim_grid, rng.rand(9, 9))
    pred_full = _full_field(ssim_grid, rng.rand(9, 9))
    assert ssim(pred_full, true_full, ssim_grid) < 1.0


def test_ssim_ignores_pml_region(ssim_grid):
    interior_vals = np.random.RandomState(6).rand(9, 9)
    true_full = _full_field(ssim_grid, interior_vals, pml_value=0.0)
    pred_same_interior_garbage_pml = _full_field(
        ssim_grid, interior_vals, pml_value=42.0
    )
    assert ssim(pred_same_interior_garbage_pml, true_full, ssim_grid) == pytest.approx(
        1.0
    )


def test_ssim_default_data_range_matches_true_field_range(ssim_grid):
    rng = np.random.RandomState(7)
    true_interior = rng.rand(9, 9)
    pred_interior = rng.rand(9, 9)
    true_full = _full_field(ssim_grid, true_interior)
    pred_full = _full_field(ssim_grid, pred_interior)
    inferred = ssim(pred_full, true_full, ssim_grid)
    explicit = ssim(
        pred_full,
        true_full,
        ssim_grid,
        data_range=float(true_interior.max() - true_interior.min()),
    )
    assert inferred == pytest.approx(explicit)


def test_ssim_mask_restricts_scoring_to_the_masked_region(ssim_grid):
    rng = np.random.RandomState(8)
    true_full = _full_field(ssim_grid, rng.rand(9, 9))
    pred_full = _full_field(ssim_grid, rng.rand(9, 9))

    mask = np.zeros(ssim_grid.shape, dtype=bool)
    ix0, iy0 = ssim_grid.interior_slice[0].start, ssim_grid.interior_slice[1].start
    mask[ix0 : ix0 + 4, iy0 : iy0 + 9] = True  # top half of the interior

    masked = ssim(pred_full, true_full, ssim_grid, mask=mask)
    unmasked = ssim(pred_full, true_full, ssim_grid)
    assert masked != pytest.approx(unmasked)


def test_ssim_full_grid_mask_and_interior_shaped_mask_agree(ssim_grid):
    rng = np.random.RandomState(9)
    true_full = _full_field(ssim_grid, rng.rand(9, 9))
    pred_full = _full_field(ssim_grid, rng.rand(9, 9))

    full_mask = np.zeros(ssim_grid.shape, dtype=bool)
    full_mask[ssim_grid.interior_slice][:4, :] = True
    interior_mask = np.zeros((9, 9), dtype=bool)
    interior_mask[:4, :] = True

    val_full = ssim(pred_full, true_full, ssim_grid, mask=full_mask)
    val_interior = ssim(pred_full, true_full, ssim_grid, mask=interior_mask)
    assert val_full == pytest.approx(val_interior)


def test_ssim_forwards_kwargs_to_skimage(small_grid):
    # 6x6 interior is smaller than skimage's default win_size=7, so this only
    # succeeds if win_size is actually forwarded through **kwargs.
    interior_vals = np.random.RandomState(10).rand(6, 6)
    full = _full_field(small_grid, interior_vals)
    assert ssim(full, full, small_grid, win_size=5) == pytest.approx(1.0)


def test_ssim_raises_without_win_size_on_too_small_a_grid(small_grid):
    interior_vals = np.random.RandomState(11).rand(6, 6)
    full = _full_field(small_grid, interior_vals)
    with pytest.raises(ValueError):
        ssim(full, full, small_grid)


def test_ssim_mask_with_incompatible_shape_raises(ssim_grid):
    interior_vals = np.random.RandomState(12).rand(9, 9)
    full = _full_field(ssim_grid, interior_vals)
    bad_mask = np.ones((3, 3), dtype=bool)  # neither full-grid nor interior-shaped
    with pytest.raises(ValueError):
        ssim(full, full, ssim_grid, mask=bad_mask)


def test_ssim_accepts_already_interior_shaped_inputs(ssim_grid):
    rng = np.random.RandomState(18)
    true_interior = rng.rand(9, 9)
    pred_interior = rng.rand(9, 9)
    true_full = _full_field(ssim_grid, true_interior)
    pred_full = _full_field(ssim_grid, pred_interior)

    full_grid_result = ssim(pred_full, true_full, ssim_grid)
    interior_result = ssim(pred_interior, true_interior, ssim_grid)
    assert interior_result == pytest.approx(full_grid_result)


def test_ssim_raises_on_pred_shape_matching_neither_full_nor_interior(ssim_grid):
    true_full = _full_field(ssim_grid, np.random.RandomState(19).rand(9, 9))
    with pytest.raises(ValueError):
        ssim(np.zeros((3, 3)), true_full, ssim_grid)


# --------------------------------------------------------------------------- #
# ssim_map
# --------------------------------------------------------------------------- #
def test_ssim_map_shape_and_perfect_recovery(ssim_grid):
    interior_vals = np.random.RandomState(13).rand(9, 9)
    full = _full_field(ssim_grid, interior_vals)
    S = ssim_map(full, full, ssim_grid)
    assert S.shape == (9, 9)
    assert np.allclose(S, 1.0, atol=1e-8)


def test_ssim_map_blanks_outside_mask_with_nan(ssim_grid):
    rng = np.random.RandomState(14)
    true_full = _full_field(ssim_grid, rng.rand(9, 9))
    pred_full = _full_field(ssim_grid, rng.rand(9, 9))

    mask = np.zeros((9, 9), dtype=bool)
    mask[:4, :] = True

    S = ssim_map(pred_full, true_full, ssim_grid, mask=mask)
    assert np.all(np.isnan(S[~mask]))
    assert np.all(~np.isnan(S[mask]))


def test_ssim_map_matches_ssim_over_the_same_mask(ssim_grid):
    rng = np.random.RandomState(15)
    true_full = _full_field(ssim_grid, rng.rand(9, 9))
    pred_full = _full_field(ssim_grid, rng.rand(9, 9))

    mask = np.zeros((9, 9), dtype=bool)
    mask[:5, :6] = True

    S = ssim_map(pred_full, true_full, ssim_grid, mask=mask)
    masked_mean = ssim(pred_full, true_full, ssim_grid, mask=mask)
    assert float(np.nanmean(S)) == pytest.approx(masked_mean)


def test_ssim_map_no_mask_has_no_nans(ssim_grid):
    interior_vals = np.random.RandomState(16).rand(9, 9)
    full = _full_field(ssim_grid, interior_vals)
    S = ssim_map(full, full, ssim_grid)
    assert not np.any(np.isnan(S))
