import math

import pytest
import torch

from odil_wave import FrequencySelection, Grid
from odil_wave.grid import complex_dtype, discrete_lambda_t, discrete_lambda_tt

from conftest import make_grid

F0 = 40e3


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #
def test_pml_extends_shape_and_interior_slice(grid):
    """Total shape = interior + 2*pml, and interior_slice recovers the interior."""
    p = grid.pml_width
    assert grid.shape == (grid.interior_nx + 2 * p, grid.interior_ny + 2 * p)
    assert grid.shape == (grid.nx, grid.ny)

    sx, sy = grid.interior_slice
    assert sx == slice(p, p + grid.interior_nx)
    assert sy == slice(p, p + grid.interior_ny)
    # A full-grid tensor sliced by interior_slice has the interior shape.
    full = torch.zeros(grid.shape)
    assert full[grid.interior_slice].shape == grid.interior_shape


def test_spacings_from_extent():
    """dx/dy follow from the *interior* extent and interior node count."""
    g = make_grid(interior_shape=(11, 21), interior_extent=((0.0, 1.0), (0.0, 2.0)))
    assert g.dx == pytest.approx(1.0 / (11 - 1))
    assert g.dy == pytest.approx(2.0 / (21 - 1))
    # coordinate axes span the PML-extended box and have the total length.
    (xmin, xmax), _ = g.extent
    assert g.x[0].item() == pytest.approx(xmin)
    assert g.x[-1].item() == pytest.approx(xmax)
    assert g.x.numel() == g.nx and g.y.numel() == g.ny


def test_cfl_derived_nt_when_init_nt_none():
    """With init_nt unset, nt is chosen so the CFL number stays under the safety."""
    g = make_grid(init_nt=None)
    assert g.nt >= 2
    assert g.dt == pytest.approx(g.t_max / (g.nt - 1))
    # CFL at c_max respects the configured safety factor.
    assert g.cfl(g.c_max) <= g.cfl_safety + 1e-9


def test_nondimensionalisation_defaults():
    """Default L0 = max interior side, c0 = c_max, t0 = L0/c0, spacings scaled."""
    g = make_grid(interior_extent=((0.0, 0.04), (0.0, 0.06)), c_max=1500.0)
    assert g.L0 == pytest.approx(0.06)
    assert g.c0 == pytest.approx(1500.0)
    assert g.t0 == pytest.approx(g.L0 / g.c0)
    assert g.dx_nd == pytest.approx(g.dx / g.L0)
    assert g.dt_nd == pytest.approx(g.dt / g.t0)
    # sigma_nd is sigma * t0 (sigma has units 1/s).
    assert torch.allclose(g.sigma_x_nd, g.sigma_x * g.t0)


def test_explicit_L0_c0_override():
    """Explicit positive L0/c0 are honoured; non-positive values are rejected."""
    g = make_grid(L0=2.0, c0=1000.0)
    assert g.L0 == 2.0 and g.c0 == 1000.0 and g.t0 == pytest.approx(2.0e-3)
    with pytest.raises(ValueError):
        make_grid(L0=-1.0)
    with pytest.raises(ValueError):
        make_grid(c0=0.0)


# --------------------------------------------------------------------------- #
# PML profile
# --------------------------------------------------------------------------- #
def test_pml_profile_zero_interior_positive_border(grid):
    """sigma vanishes in the interior and is strictly positive at the outer edge."""
    total = grid.sigma_x + grid.sigma_y
    assert torch.all(total[grid.interior_slice] == 0)
    # corners of the full grid sit deepest in the PML.
    assert total[0, 0].item() > 0
    assert total[-1, -1].item() > 0


def test_pml_width_zero_gives_no_absorption():
    """pml_width=0 collapses the sponge to zeros and the shape to the interior."""
    g = make_grid(pml_width=0)
    assert g.shape == g.interior_shape
    assert float(g.sigma_x.abs().max()) == 0.0
    assert float(g.sigma_y.abs().max()) == 0.0


def test_bad_pml_R0_raises():
    """pml_R0 must lie strictly in (0, 1)."""
    with pytest.raises(ValueError):
        make_grid(pml_R0=2.0)
    with pytest.raises(ValueError):
        make_grid(pml_R0=0.0)


# --------------------------------------------------------------------------- #
# Helpers / constructors
# --------------------------------------------------------------------------- #
def test_natural_source_amplitude(grid):
    """Amplitude is 2 f0^2 and requires a positive frequency."""
    assert grid.natural_source_amplitude(10.0) == pytest.approx(200.0)
    with pytest.raises(ValueError):
        grid.natural_source_amplitude(0.0)


def test_from_frequency_sets_resolution_and_guards_init_nt():
    """from_frequency picks dx by the points-per-wavelength rule; init_nt banned."""
    f_max, c_min, n_ppw = 40e3, 1400.0, 5
    g = Grid.from_frequency(
        f_max=f_max,
        c_min=c_min,
        interior_extent=((0.0, 0.05), (0.0, 0.05)),
        n_ppw=n_ppw,
        c_max=1600.0,
        t_max=6e-5,
        pml_width=4,
    )
    dx_target = c_min / (f_max * n_ppw)
    # The realised dx rounds to an integer node count, so it lands within a step.
    assert g.dx == pytest.approx(dx_target, abs=dx_target)
    assert g.interior_nx == round(0.05 / dx_target) + 1
    assert g.c_min == c_min
    with pytest.raises(ValueError):
        Grid.from_frequency(f_max=f_max, c_min=c_min, init_nt=100)


def test_summary_is_string(grid):
    """The human-readable summary renders without error."""
    assert isinstance(grid.summary, str)
    assert "Grid interior" in grid.summary


# --------------------------------------------------------------------------- #
# FrequencySelection
# --------------------------------------------------------------------------- #
def test_from_frequencies_maps_to_positive_bin(grid):
    """Requested frequencies map to matching bins/omega; near-0 avoids bin 0."""
    fs = FrequencySelection.from_frequencies(grid, [F0])
    assert fs.n_frequencies == 1
    # angular frequency is 2*pi*f, and omega_nd = omega * t0.
    f_sel = float(fs.frequencies[0])
    assert float(fs.omega[0]) == pytest.approx(2 * math.pi * f_sel)
    assert float(fs.omega_nd[0]) == pytest.approx(float(fs.omega[0]) * grid.t0)
    # A tiny positive request must not collapse onto the DC (0 Hz) bin.
    fs0 = FrequencySelection.from_frequencies(grid, [1.0])
    assert int(fs0.fft_bins[0]) != 0


def test_from_bins_validates_range(grid):
    """Bins outside [0, nt) are rejected."""
    fs = FrequencySelection.from_bins(grid, [2, 3])
    assert fs.fft_bins.tolist() == [2, 3]
    with pytest.raises(ValueError):
        FrequencySelection.from_bins(grid, [grid.nt + 1])
    with pytest.raises(ValueError):
        FrequencySelection.from_bins(grid, [-1])


def test_empty_selection_raises(grid):
    """A selection must contain at least one bin."""
    with pytest.raises(ValueError):
        FrequencySelection.from_bins(grid, [])


def test_symbols_and_broadcast(grid):
    """lambda_t / lambda_tt match their closed forms and broadcast to (1,nf,1,1)."""
    fs = FrequencySelection.from_frequencies(grid, [F0])
    assert torch.allclose(fs.lambda_t, discrete_lambda_t(fs.omega_nd, fs.dt_nd))
    assert torch.allclose(fs.lambda_tt, discrete_lambda_tt(fs.omega_nd, fs.dt_nd))
    lt, ltt = fs.symbols_broadcast()
    assert lt.shape == (1, fs.n_frequencies, 1, 1)
    assert ltt.shape == (1, fs.n_frequencies, 1, 1)
    assert lt.is_complex()


def test_cdtype_matches_grid_dtype(grid):
    """The selection's complex dtype tracks the grid's real dtype."""
    fs = FrequencySelection.from_frequencies(grid, [F0])
    assert fs.cdtype == torch.complex128  # grid is float64


def test_fft_time_series_selects_bins(grid):
    """FFT of a time signal returns only the selected bins along the FFT axis."""
    fs = FrequencySelection.from_bins(grid, [2, 5])
    signal = torch.randn(grid.nt, dtype=grid.dtype)
    spec = fs.fft_time_series(signal, dim=0)
    assert spec.shape == (2,)
    full = torch.fft.fft(signal, dim=0)
    assert torch.allclose(spec, full[torch.tensor([2, 5])])


def test_complex_dtype_helper():
    """complex_dtype promotes real dtypes and rejects unsupported ones."""
    assert complex_dtype(torch.float64) == torch.complex128
    assert complex_dtype(torch.float32) == torch.complex64
    with pytest.raises(ValueError):
        complex_dtype(torch.int32)
