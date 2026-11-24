"""Tests for the ``odil_wave.operator`` package."""

from __future__ import annotations

from types import SimpleNamespace

import odil_wave  # noqa: F401  (import before torch avoids an OpenMP abort)
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from odil_wave.grid import FrequencySelection
from odil_wave.wavefield import Wavefield

from odil_wave.operator.conditions import Pml
from odil_wave.operator.spatial import (
    STAGGERED_D1_COEFFS,
    Laplacian,
    _apply_conv,
    second_difference_stencil,
)
from odil_wave.operator.utils import WaveEquation
from odil_wave.operator.helmholtz import (
    HelmholtzSolver,
    _reflect_index,
    np_complex_dtype,
    second_difference_1d,
)

from conftest import make_geometry, make_grid, make_model, make_source


# --------------------------------------------------------------------------- #
# Shared tiny fixtures
# --------------------------------------------------------------------------- #
def _make_grid(**overrides):
    kwargs = dict(
        interior_shape=(8, 8),
        interior_extent=((-0.02, 0.02), (-0.02, 0.02)),
        c_min=1000.0,
        c_max=2000.0,
        pml_width=2,
        t_max=2e-5,
        init_nt=64,
        dtype=torch.float32,
    )
    kwargs.update(overrides)
    return make_grid(**kwargs)


def _make_wavefield(grid, f0=8e4, **wf_overrides):
    vm = make_model(grid)
    freq = FrequencySelection.from_frequencies(grid, [f0])
    kwargs = dict(grid=grid, frequency_selection=freq, velocity_model=vm)
    kwargs.update(wf_overrides)
    return Wavefield(**kwargs)


@pytest.fixture(scope="module")
def tiny_setup():
    """A minimal real forward setup (grid/velocity/source/geometry/wavefield),
    shared read-only across the Helmholtz integration tests."""
    grid = _make_grid()
    vm = make_model(grid)
    source = make_source(grid, kind="ricker", f0=8e4)
    freq = FrequencySelection.from_frequencies(grid, [8e4])
    geom = make_geometry(grid, source, freq, n_transducers=4, source_spatial="gaussian")
    wf = Wavefield(grid=grid, frequency_selection=freq, velocity_model=vm)
    return SimpleNamespace(grid=grid, vm=vm, source=source, freq=freq, geom=geom, wf=wf)


# --------------------------------------------------------------------------- #
# conditions.py
# --------------------------------------------------------------------------- #
def test_pml_zero_weight_leaves_the_operator_unstretched():
    grid = _make_grid(pml_width=2)
    wf = _make_wavefield(grid)
    pml = Pml(wf, weight=0.0, space_order=2)
    u = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.complex64)
    u_xx = torch.randn_like(u)
    u_yy = torch.randn_like(u)
    lambda_tt = torch.tensor(-3.0 + 0j, dtype=torch.complex64).view(1, 1, 1, 1)
    assert torch.equal(pml.stretched_laplacian(u, u_xx, u_yy), u_xx + u_yy)
    assert torch.equal(pml.mass(lambda_tt), lambda_tt.expand(1, 1, grid.nx, grid.ny))


def test_pml_is_inactive_in_the_interior():
    """``s = 1`` at interior nodes and ``1/s - 1 = 0`` on the faces around
    them, so interior rows see the unstretched operator."""
    grid = _make_grid(pml_width=3)
    pml = Pml(_make_wavefield(grid), weight=1.0, space_order=2)
    sx, sy = grid.interior_slice
    assert torch.equal(pml.s_x[:, sx], torch.ones_like(pml.s_x[:, sx]))
    assert torch.equal(pml.s_y[:, :, sy], torch.ones_like(pml.s_y[:, :, sy]))
    # Faces (ghosts included) k - 1/2 for k = p .. p + n_int: around the
    # interior nodes.
    p = grid.pml_width
    assert (pml.a_x[:, p : p + grid.interior_nx + 1] == 0).all()
    assert (pml.a_x[:, :p] != 0).all() and (pml.a_x[:, -p:] != 0).all()


def test_pml_unsupported_order_raises():
    with pytest.raises(ValueError, match="space order"):
        Pml(_make_wavefield(_make_grid()), weight=1.0, space_order=3)


def test_pml_at_zero_hertz_raises():
    grid = _make_grid(pml_width=2)
    freq = FrequencySelection.from_bins(grid, [0, 3])
    wf = Wavefield(grid=grid, frequency_selection=freq, velocity_model=make_model(grid))
    with pytest.raises(ValueError, match="0 Hz"):
        Pml(wf, weight=1.0, space_order=2)
    Pml(wf, weight=0.0, space_order=2)


# --------------------------------------------------------------------------- #
# spatial.py
# --------------------------------------------------------------------------- #
def test_apply_conv_complex_matches_real_imag_split():
    kernel = torch.randn(1, 1, 3, 1, dtype=torch.float64)
    a = torch.randn(5, 5, dtype=torch.float64)
    b = torch.randn(5, 5, dtype=torch.float64)
    out = _apply_conv(torch.complex(a, b), kernel, (0, 0, 1, 1))
    expected = torch.complex(
        _apply_conv(a, kernel, (0, 0, 1, 1)), _apply_conv(b, kernel, (0, 0, 1, 1))
    )
    assert torch.allclose(out, expected)


def test_second_difference_stencil_6th_order_matches_known_coefficients():
    """The 6th-order stencil carries the textbook coefficients over h**2."""
    k = second_difference_stencil(6, h_nd=0.5)
    assert k.shape == (7,)
    # Fornberg: offsets 1..3, then the centre.
    for off, ck in enumerate([3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0], start=1):
        assert k[3 - off] == pytest.approx(4.0 * ck)
        assert k[3 + off] == pytest.approx(4.0 * ck)
    assert k[3] == pytest.approx(-49.0 / 18.0 * 4.0)


@pytest.mark.parametrize("space_order", [3, 12])
def test_second_difference_stencil_unsupported_order_raises(space_order):
    """Orders without a coefficient table are rejected."""
    with pytest.raises(ValueError):
        second_difference_stencil(space_order, 1.0)


@pytest.mark.parametrize("space_order", [2, 4, 6, 8, 10])
def test_staggered_first_differences_are_exact_to_their_order(space_order):
    """On ``x**k`` about a face, ``k <= P``, the staggered difference gives the
    exact derivative there: 1 for ``k = 1``, else 0."""
    offsets = np.arange(1, space_order // 2 + 1) - 0.5
    coeffs = np.array(STAGGERED_D1_COEFFS[space_order])
    for k in range(space_order + 1):
        d = coeffs @ (offsets**k - (-offsets) ** k)
        assert d == pytest.approx(float(k == 1), abs=1e-12)


@pytest.mark.parametrize("space_order,pad", [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5)])
def test_laplacian_orders_are_exact_on_a_quadratic_interior(space_order, pad):
    n = 2 * pad + 9
    grid = _make_grid(
        interior_shape=(n, n),
        interior_extent=((-1.0, 1.0), (-1.0, 1.0)),
        pml_width=0,
        t_max=1e-5,
        init_nt=16,
        dtype=torch.float64,
    )
    wf = _make_wavefield(grid)
    idx = torch.arange(n, dtype=torch.float64) * grid.dx_nd
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    u = X**2 + Y**2  # analytic Laplacian in non-dimensional coords = 4
    u_xx, u_yy = Laplacian(wf, space_order=space_order).axes(u)
    center = (u_xx + u_yy)[pad : n - pad, pad : n - pad]
    assert torch.allclose(center, torch.full_like(center, 4.0), atol=1e-6)


def test_laplacians_on_different_grids_are_not_equal():
    """The removed per-order ``@dataclass`` classes had no fields, so any two
    instances compared equal."""
    coarse = _make_wavefield(_make_grid(interior_shape=(5, 5)))
    fine = _make_wavefield(_make_grid(interior_shape=(9, 9)))
    assert Laplacian(coarse, space_order=2) != Laplacian(fine, space_order=2)


# --------------------------------------------------------------------------- #
# utils.py: WaveEquation
# --------------------------------------------------------------------------- #
def test_wave_equation_invalid_space_order_raises():
    grid = _make_grid()
    wf = _make_wavefield(grid)
    with pytest.raises(ValueError):
        WaveEquation(wf, space_order=3, damping_scale=1.0)


def test_wave_equation_residual_is_zero_for_zero_amplitude_and_source():
    grid = _make_grid()
    wf = _make_wavefield(grid)
    we = WaveEquation(wf, space_order=2, damping_scale=1.0)
    amp = torch.zeros(1, 1, grid.nx, grid.ny, dtype=wf.cdtype)
    source = torch.zeros_like(amp)
    c = wf.velocity_model.c
    r = we.residual(amp, c, source)
    assert torch.allclose(r, torch.zeros_like(r))


def _conservative_pml_residual(grid, wf, amp, c, source, weight):
    """Order-2 ``λ_tt s_x s_y u - c'^2 [s_y D⁻(D⁺u / s_x) + s_x D⁻(D⁺u / s_y)] - f``
    by explicit loops, ``1/s`` on the faces, mirrored at the outer wall."""
    freq = wf.frequency_selection
    iw = 1j * float(freq.omega_nd[0])
    ltt = float(freq.lambda_tt[0])
    sx = 1 + weight * grid.sigma_x_nd[:, 0].numpy() / iw
    sy = 1 + weight * grid.sigma_y_nd[0, :].numpy() / iw
    # 1/s on faces -1/2 .. n - 1/2, the ghosts mirroring their neighbours.
    fx = 1 / (1 + weight * grid.sigma_x_faces_nd.numpy() / iw)
    fy = 1 / (1 + weight * grid.sigma_y_faces_nd.numpy() / iw)
    fx = np.concatenate([fx[:1], fx, fx[-1:]])
    fy = np.concatenate([fy[:1], fy, fy[-1:]])
    u = np.pad(amp[0, 0].numpy(), 1, mode="reflect")  # u[i + 1, j + 1] is node (i, j)
    nx, ny = grid.shape
    r = np.zeros((nx, ny), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            I, J = i + 1, j + 1  # noqa: E741
            dxx = fx[i + 1] * (u[I + 1, J] - u[I, J]) - fx[i] * (u[I, J] - u[I - 1, J])
            dyy = fy[j + 1] * (u[I, J + 1] - u[I, J]) - fy[j] * (u[I, J] - u[I, J - 1])
            k2 = (float(c[i, j]) / grid.c0) ** 2
            r[i, j] = (
                ltt * sx[i] * sy[j] * u[I, J]
                - k2 * (sy[j] * dxx / grid.dx_nd**2 + sx[i] * dyy / grid.dy_nd**2)
                - source[0, 0, i, j].item()
            )
    return r


def test_wave_equation_residual_is_the_conservative_stretched_form():
    grid = _make_grid(pml_width=3, dtype=torch.float64)
    wf = _make_wavefield(grid)
    we = WaveEquation(wf, space_order=2, damping_scale=1.7)
    torch.manual_seed(0)
    amp = torch.randn(1, 1, grid.nx, grid.ny, dtype=wf.cdtype)
    source = torch.randn(1, 1, grid.nx, grid.ny, dtype=wf.cdtype)
    c = wf.velocity_model.c
    r = we.residual(amp, c, source)
    expected = _conservative_pml_residual(grid, wf, amp, c, source, 1.7)
    assert np.allclose(r[0, 0].numpy(), expected, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("space_order", [2, 4, 6, 8])
def test_pml_leaves_interior_rows_unchanged(space_order):
    """Interior rows are ``λ_tt u - c'^2 (u_xx + u_yy) - f`` exactly, PML on,
    beyond the ``P/2 - 1`` cells the staggered correction reaches in."""
    grid = _make_grid(pml_width=4, dtype=torch.float64)
    wf = _make_wavefield(grid)
    torch.manual_seed(1)
    amp = torch.randn(1, 1, grid.nx, grid.ny, dtype=wf.cdtype)
    source = torch.randn_like(amp)
    c = wf.velocity_model.c
    r = WaveEquation(wf, space_order=space_order, damping_scale=1.0).residual(
        amp, c, source
    )
    u_xx, u_yy = Laplacian(wf, space_order=space_order).axes(amp)
    _, lambda_tt = wf.frequency_selection.symbols_broadcast()
    plain = lambda_tt * amp - (c / grid.c0) ** 2 * (u_xx + u_yy) - source
    p, band = grid.pml_width, space_order // 2 - 1
    sx = slice(p + band, grid.nx - p - band)
    sy = slice(p + band, grid.ny - p - band)
    assert torch.equal(r[..., sx, sy], plain[..., sx, sy])
    assert not torch.allclose(r, plain)


# --------------------------------------------------------------------------- #
# helmholtz.py
# --------------------------------------------------------------------------- #
def test_np_complex_dtype_mappings():
    import numpy as np

    assert np_complex_dtype(torch.complex64) is np.complex64
    assert np_complex_dtype(torch.complex128) is np.complex128
    with pytest.raises(ValueError):
        np_complex_dtype(torch.float32)


def test_reflect_index_n_equals_one_returns_zero_for_any_idx():
    # Regression test for a fixed infinite loop: the `2*n - 2 - idx` fold
    # degenerates to idx <-> -idx for n=1. With a single cell, every offset
    # -- however far out of bounds -- refers to that same cell.
    assert (_reflect_index(np.arange(-5, 6), 1) == 0).all()


@pytest.mark.parametrize("n", [2, 4, 7])
def test_reflect_index_matches_torch_pad_reflect_semantics(n):
    # F.pad(..., mode="reflect") only supports 1-D reflect padding on a 2-D
    # or 3-D input's last dim (a 4-D input needs 2-D reflect padding instead).
    x = torch.arange(n, dtype=torch.float64).view(1, 1, n)
    pad = n - 1
    padded = F.pad(x, (pad, pad), mode="reflect").view(-1)
    idx = np.arange(-pad, n + pad)
    assert (_reflect_index(idx, n) == padded.numpy().astype(int)).all()


def test_reflect_index_raises_for_nonpositive_n():
    with pytest.raises(ValueError):
        _reflect_index(np.arange(1), 0)


@pytest.mark.parametrize("space_order", [2, 4, 6, 8, 10])
def test_sparse_second_differences_match_the_conv_ones(space_order):
    """The sparse second differences the Helmholtz assembly uses equal the
    torch conv ones (reflect boundaries) on a random field."""
    grid = _make_grid(pml_width=0, interior_shape=(11, 9), dtype=torch.float64)
    wf = _make_wavefield(grid)
    torch.manual_seed(space_order)
    u = torch.randn(grid.nx, grid.ny, dtype=torch.float64)
    u_xx, u_yy = Laplacian(wf, space_order=space_order).axes(u)
    Dx = second_difference_1d(grid.nx, space_order, float(grid.dx_nd))
    Dy = second_difference_1d(grid.ny, space_order, float(grid.dy_nd))
    assert np.allclose(u_xx.numpy(), Dx @ u.numpy(), atol=1e-8)
    assert np.allclose(u_yy.numpy(), (Dy @ u.numpy().T).T, atol=1e-8)


def test_second_difference_1d_is_cached():
    D = second_difference_1d(7, 2, 0.1)
    assert D.shape == (7, 7)
    assert second_difference_1d(7, 2, 0.1) is D


def test_second_difference_1d_unsupported_order_raises():
    with pytest.raises(ValueError):
        second_difference_1d(5, 3, 1.0)


@pytest.mark.parametrize("space_order", [2, 4, 6, 8])
def test_assemble_h_sparse_matches_wave_equation_residual(space_order):
    """Sparse and matrix-free operators agree, PML included."""
    grid = _make_grid(pml_width=4, dtype=torch.float64)
    freq = FrequencySelection.from_frequencies(grid, [4e4, 8e4])
    vm = make_model(grid, profile="shepp_logan")
    wf = Wavefield(grid=grid, frequency_selection=freq, velocity_model=vm)
    solver = HelmholtzSolver(wf, None, space_order=space_order, damping_scale=1.3)
    torch.manual_seed(2)
    amp = torch.randn(1, 2, grid.nx, grid.ny, dtype=wf.cdtype)
    r = solver.wave_eq.residual(amp, vm.c, torch.zeros_like(amp))
    for k in range(2):
        H = solver.assemble_H_sparse(vm.c, freq_idx=k)
        Hu = (H @ amp[0, k].reshape(-1).numpy()).reshape(grid.shape)
        assert np.allclose(r[0, k].numpy(), Hu, rtol=1e-10, atol=1e-8)


def test_helmholtz_solver_solve_smoke(tiny_setup):
    solver = HelmholtzSolver(
        tiny_setup.wf, tiny_setup.geom, space_order=2, damping_scale=1.0
    )
    outputs = solver.solve(verbose=False)
    assert len(outputs) == tiny_setup.geom.n_shots
    for wf_out in outputs:
        assert wf_out.amplitude.shape == (1, tiny_setup.grid.nx, tiny_setup.grid.ny)
        assert torch.isfinite(wf_out.amplitude.real).all()
        assert torch.isfinite(wf_out.amplitude.imag).all()

    amp = torch.stack([o.amplitude for o in outputs], dim=0)
    sources = solver.wave_eq.sources(tiny_setup.geom)
    r = solver.wave_eq.residual(amp, tiny_setup.vm.c, sources)
    r_rms = float(torch.mean(torch.abs(r) ** 2).sqrt())
    src_rms = float(torch.mean(torch.abs(sources) ** 2).sqrt())
    assert r_rms / max(src_rms, 1e-30) < 1e-3  # well-converged direct solve
