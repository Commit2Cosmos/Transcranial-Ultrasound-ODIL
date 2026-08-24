"""Tests for the ``odil_wave.wavefield`` package (``Wavefield`` and its helpers).

Covers construction/validation, shapes/dtypes, the ``amplitude`` property
(getter/setter), ``init_ut``/``init_ut_nd``, the private ``_normalize_amplitude``
helper, and the ``show``/``animate`` plotting methods' validation and (lightweight,
Agg-backend) success paths.

Uses only small synthetic grids (no velocity solve) so every test is fast.
``matplotlib`` uses the non-interactive ``Agg`` backend so ``show``/``animate``
don't block or pop up a window; see test_metrics.py's docstring for why.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import odil_wave  # noqa: F401,E402  (import before torch: see test_optimisation.py)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from odil_wave.grid import FrequencySelection, Grid  # noqa: E402
from odil_wave.models import VelocityModel  # noqa: E402
from odil_wave.wavefield import Wavefield  # noqa: E402
from odil_wave.wavefield.base import _normalize_amplitude  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Shared small grid/frequency-selection helpers
# --------------------------------------------------------------------------- #
def _make_grid(**overrides):
    kwargs = dict(
        interior_shape=(6, 6),
        c_min=1000.0,
        c_max=2000.0,
        pml_width=2,
        t_max=1e-5,
        init_nt=32,
    )
    kwargs.update(overrides)
    return Grid(**kwargs)


def _make_freq(grid, freqs=(8e4,)):
    return FrequencySelection.from_frequencies(grid, list(freqs))


# --------------------------------------------------------------------------- #
# _normalize_amplitude
# --------------------------------------------------------------------------- #
def test_normalize_amplitude_none_and_literal_none_string_are_passthrough():
    amp = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(_normalize_amplitude(amp, None), amp)
    assert np.array_equal(_normalize_amplitude(amp, "none"), amp)


def test_normalize_amplitude_global_scales_by_overall_peak():
    amp = np.array([[1.0, -2.0], [3.0, -4.0]])
    out = _normalize_amplitude(amp, "global")
    assert np.abs(out).max() == pytest.approx(1.0)
    assert np.allclose(out, amp / 4.0)


def test_normalize_amplitude_global_zero_array_is_unchanged():
    amp = np.zeros((2, 2))
    out = _normalize_amplitude(amp, "global")
    assert np.array_equal(out, amp)


def test_normalize_amplitude_per_frame_scales_each_frame_independently():
    amp = np.stack([np.array([[1.0, 2.0]]), np.array([[10.0, -20.0]])])  # (2, 1, 2)
    out = _normalize_amplitude(amp, "per_frame")
    assert np.abs(out[0]).max() == pytest.approx(1.0)
    assert np.abs(out[1]).max() == pytest.approx(1.0)
    assert np.allclose(out[0], amp[0] / 2.0)
    assert np.allclose(out[1], amp[1] / 20.0)


def test_normalize_amplitude_per_frame_zero_frame_is_unchanged():
    amp = np.stack([np.zeros((1, 2)), np.array([[4.0, 0.0]])])
    out = _normalize_amplitude(amp, "per_frame")
    assert np.array_equal(out[0], amp[0])  # scale=0 -> divisor forced to 1.0
    assert np.allclose(out[1], amp[1] / 4.0)


def test_normalize_amplitude_invalid_mode_raises():
    with pytest.raises(ValueError):
        _normalize_amplitude(np.zeros((2, 2)), "bogus")


# --------------------------------------------------------------------------- #
# Wavefield construction / validation
# --------------------------------------------------------------------------- #
def test_wavefield_default_amplitude_and_init_ut_are_zero_with_expected_shapes():
    grid = _make_grid()
    freq = _make_freq(grid, [8e4, 9e4])
    wf = Wavefield(grid=grid, frequency_selection=freq)

    assert wf.n_frequencies == 2
    assert wf.amplitude.shape == (2, grid.nx, grid.ny)
    assert torch.is_complex(wf.amplitude)
    assert torch.count_nonzero(wf.amplitude) == 0

    assert wf.init_ut.shape == (grid.nx, grid.ny)
    assert not torch.is_complex(wf.init_ut)
    assert torch.count_nonzero(wf.init_ut) == 0


def test_wavefield_dtype_device_and_cdtype_match_grid():
    grid32 = _make_grid(dtype=torch.float32)
    wf32 = Wavefield(grid=grid32, frequency_selection=_make_freq(grid32))
    assert wf32.dtype == torch.float32
    assert wf32.cdtype == torch.complex64
    assert wf32.device == grid32.device
    assert wf32.amplitude.dtype == torch.complex64

    grid64 = _make_grid(dtype=torch.float64)
    wf64 = Wavefield(grid=grid64, frequency_selection=_make_freq(grid64))
    assert wf64.dtype == torch.float64
    assert wf64.cdtype == torch.complex128
    assert wf64.amplitude.dtype == torch.complex128


def test_wavefield_default_velocity_model_is_homogeneous():
    grid = _make_grid()
    wf = Wavefield(grid=grid, frequency_selection=_make_freq(grid))
    assert isinstance(wf.velocity_model, VelocityModel)
    assert wf.velocity_model.profile == "homogeneous"


def test_wavefield_explicit_velocity_model_is_stored_by_identity():
    grid = _make_grid()
    vm = VelocityModel(grid, profile="homogeneous", base=1234.0)
    wf = Wavefield(grid=grid, frequency_selection=_make_freq(grid), velocity_model=vm)
    assert wf.velocity_model is vm


def test_wavefield_init_amplitude_is_cast_and_stored():
    grid = _make_grid()
    freq = _make_freq(grid, [8e4])
    raw = np.arange(grid.nx * grid.ny, dtype=np.float64).reshape(1, grid.nx, grid.ny)
    wf = Wavefield(grid=grid, frequency_selection=freq, init_amplitude=raw)
    assert wf.amplitude.dtype == wf.cdtype
    assert torch.allclose(wf.amplitude.real, torch.as_tensor(raw, dtype=wf.dtype))
    assert torch.count_nonzero(wf.amplitude.imag) == 0


def test_wavefield_init_velocity_and_init_ut_nd_scaling():
    grid = _make_grid()
    freq = _make_freq(grid, [8e4])
    init_velocity = np.full((grid.nx, grid.ny), 3.0)
    wf = Wavefield(grid=grid, frequency_selection=freq, init_velocity=init_velocity)

    assert torch.allclose(wf.init_ut, torch.full((grid.nx, grid.ny), 3.0, dtype=wf.dtype))
    expected_nd = 3.0 * grid.t0
    assert torch.allclose(
        wf.init_ut_nd, torch.full((grid.nx, grid.ny), expected_nd, dtype=wf.dtype)
    )


def test_wavefield_accepts_a_second_grid_with_matching_metadata():
    grid_a = _make_grid()
    grid_b = _make_grid()  # separate object, identical construction params
    assert grid_a is not grid_b
    freq_from_b = _make_freq(grid_b, [8e4])
    # Must not raise: nt/dt match even though frequency_selection.grid is not
    # this Wavefield's grid object.
    wf = Wavefield(grid=grid_a, frequency_selection=freq_from_b)
    assert wf.amplitude.shape == (1, grid_a.nx, grid_a.ny)


def test_wavefield_rejects_incompatible_frequency_selection():
    grid_a = _make_grid(init_nt=32)
    grid_b = _make_grid(init_nt=48)  # different nt -> incompatible dt/n_time
    freq_from_b = _make_freq(grid_b, [8e4])
    with pytest.raises(ValueError):
        Wavefield(grid=grid_a, frequency_selection=freq_from_b)


# --------------------------------------------------------------------------- #
# amplitude property (getter/setter)
# --------------------------------------------------------------------------- #
def test_amplitude_setter_casts_and_reshapes_numpy_array():
    grid = _make_grid()
    freq = _make_freq(grid, [8e4, 9e4])
    wf = Wavefield(grid=grid, frequency_selection=freq)

    flat = np.arange(2 * grid.nx * grid.ny, dtype=np.complex128)
    wf.amplitude = flat
    assert wf.amplitude.shape == (2, grid.nx, grid.ny)
    assert wf.amplitude.dtype == wf.cdtype
    assert torch.allclose(
        wf.amplitude.reshape(-1), torch.as_tensor(flat, dtype=wf.cdtype)
    )


def test_amplitude_setter_accepts_torch_tensor_and_casts_dtype():
    grid = _make_grid()
    freq = _make_freq(grid, [8e4])
    wf = Wavefield(grid=grid, frequency_selection=freq)

    value = torch.ones(1, grid.nx, grid.ny, dtype=torch.complex64) * (2.0 + 1.0j)
    wf.amplitude = value
    assert wf.amplitude.dtype == wf.cdtype
    assert torch.allclose(wf.amplitude, value.to(wf.cdtype))


def test_amplitude_setter_rejects_wrong_element_count():
    grid = _make_grid()
    freq = _make_freq(grid, [8e4])
    wf = Wavefield(grid=grid, frequency_selection=freq)
    with pytest.raises(RuntimeError):
        wf.amplitude = torch.zeros(3)  # nf*nx*ny != 3 for this grid


# --------------------------------------------------------------------------- #
# show()
# --------------------------------------------------------------------------- #
def test_show_raises_for_out_of_range_idx():
    grid = _make_grid()
    wf = Wavefield(grid=grid, frequency_selection=_make_freq(grid, [8e4]))
    with pytest.raises(ValueError):
        wf.show(idx=5)
    with pytest.raises(ValueError):
        wf.show(idx=-1)


def test_show_raises_for_invalid_normalize_mode():
    grid = _make_grid()
    wf = Wavefield(grid=grid, frequency_selection=_make_freq(grid, [8e4]))
    with pytest.raises(ValueError):
        wf.show(idx=0, normalize="bogus")


def test_show_runs_for_valid_idx_and_normalize_modes():
    grid = _make_grid()
    freq = _make_freq(grid, [8e4, 9e4])
    wf = Wavefield(grid=grid, frequency_selection=freq)
    wf.amplitude = torch.randn(2, grid.nx, grid.ny, dtype=wf.cdtype)
    for normalize in (None, "none", "global", "per_frame"):
        wf.show(idx=1, normalize=normalize)  # must not raise / hang


# --------------------------------------------------------------------------- #
# animate()
# --------------------------------------------------------------------------- #
def test_animate_writes_a_nonempty_gif(tmp_path):
    grid = _make_grid()
    freq = _make_freq(grid, [8e4, 9e4])
    wf = Wavefield(grid=grid, frequency_selection=freq)
    wf.amplitude = torch.randn(2, grid.nx, grid.ny, dtype=wf.cdtype)

    out_path = tmp_path / "wavefield.gif"
    returned = wf.animate(filename=str(out_path), fps=5)

    assert returned == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
