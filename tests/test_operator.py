"""Tests for the ``odil_wave.operator`` package.

Covers ``base.py``, ``conditions.py``, ``spatial.py``, ``utils.py``
(``WaveEquation``) and ``helmholtz.py``.

Most tests use hand-built tensors or a tiny synthetic :class:`Grid`/
:class:`Wavefield` (no solver) so they run in well under a second; the
Helmholtz integration tests use one small shared fixture (12x12 total grid)
so the handful of real linear solves stay fast too.
"""

from __future__ import annotations

from types import SimpleNamespace

import odil_wave  # noqa: F401  (import before torch avoids an OpenMP abort; see test_optimisation.py)
import pytest
import torch
import torch.nn.functional as F

from odil_wave.grid import FrequencySelection
from odil_wave.wavefield import Wavefield

from odil_wave.operator.base import DenseOperator
from odil_wave.operator.conditions import Conditions, Sponge
from odil_wave.operator.spatial import (
    Laplacian2ndOrder,
    Laplacian4thOrder,
    Laplacian6thOrder,
    Laplacian8thOrder,
    Laplacian10thOrder,
    _apply_conv_laplacian,
    laplacian_stencil,
)
from odil_wave.operator.utils import WaveEquation
from odil_wave.operator.helmholtz import (
    HelmholtzSolver,
    _reflect_index,
    assemble_laplacian_csr,
    np_complex_dtype,
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
    source = make_source(grid, kind="ricker", f0=8e4, offset=0)
    freq = FrequencySelection.from_frequencies(grid, [8e4])
    geom = make_geometry(grid, source, freq, n_transducers=4, source_spatial="gaussian")
    wf = Wavefield(grid=grid, frequency_selection=freq, velocity_model=vm)
    return SimpleNamespace(grid=grid, vm=vm, source=source, freq=freq, geom=geom, wf=wf)


# --------------------------------------------------------------------------- #
# base.py
# --------------------------------------------------------------------------- #
def test_dense_operator_is_abstract():
    with pytest.raises(TypeError):
        DenseOperator(wavefield=None)


# --------------------------------------------------------------------------- #
# conditions.py
# --------------------------------------------------------------------------- #
def test_conditions_is_abstract():
    with pytest.raises(TypeError):
        Conditions()


def test_sponge_apply_residual_matches_manual_formula():
    grid = _make_grid(pml_width=2)
    wf = _make_wavefield(grid)
    sponge = Sponge(wf, weight=2.0)

    fu = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.complex64)
    u = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.complex64)
    lambda_t = torch.tensor(0.5 + 0.25j, dtype=torch.complex64).view(1, 1, 1, 1)

    out = sponge.apply_residual(fu, u, lambda_t)

    sigma_sum = grid.sigma_x_nd + grid.sigma_y_nd
    sigma_prod = grid.sigma_x_nd * grid.sigma_y_nd
    expected = fu + 2.0 * (sigma_sum * lambda_t * u + sigma_prod * u)
    assert torch.allclose(out, expected)


def test_sponge_apply_is_alias_for_apply_residual():
    grid = _make_grid()
    wf = _make_wavefield(grid)
    sponge = Sponge(wf, weight=1.0)
    fu = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.complex64)
    u = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.complex64)
    lambda_t = torch.tensor(0.3j, dtype=torch.complex64).view(1, 1, 1, 1)
    assert torch.equal(
        sponge.apply(fu, u, lambda_t), sponge.apply_residual(fu, u, lambda_t)
    )


def test_sponge_zero_weight_is_a_no_op():
    grid = _make_grid()
    wf = _make_wavefield(grid)
    sponge = Sponge(wf, weight=0.0)
    fu = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.complex64)
    u = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.complex64)
    lambda_t = torch.tensor(0.1j, dtype=torch.complex64).view(1, 1, 1, 1)
    assert torch.equal(sponge.apply_residual(fu, u, lambda_t), fu)


# --------------------------------------------------------------------------- #
# spatial.py
# --------------------------------------------------------------------------- #
def test_apply_conv_laplacian_zero_on_constant_field():
    kernel = torch.zeros(1, 1, 3, 3)
    kernel[0, 0, 0, 1] = 1.0
    kernel[0, 0, 2, 1] = 1.0
    kernel[0, 0, 1, 0] = 1.0
    kernel[0, 0, 1, 2] = 1.0
    kernel[0, 0, 1, 1] = -4.0
    u = torch.full((6, 6), 3.0)
    out = _apply_conv_laplacian(u, kernel, pad=1)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_apply_conv_laplacian_exact_on_quadratic_interior():
    # standard 2nd-order 5-point kernel with unit spacing (cx=cy=1)
    kernel = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    kernel[0, 0, 0, 1] = 1.0
    kernel[0, 0, 2, 1] = 1.0
    kernel[0, 0, 1, 0] = 1.0
    kernel[0, 0, 1, 2] = 1.0
    kernel[0, 0, 1, 1] = -4.0
    n = 9
    idx = torch.arange(n, dtype=torch.float64)
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    u = X**2 + Y**2  # analytic Laplacian = 2 + 2 = 4 everywhere
    out = _apply_conv_laplacian(u, kernel, pad=1)
    center = out[3:6, 3:6]  # away from the reflect-padded edges
    assert torch.allclose(center, torch.full_like(center, 4.0), atol=1e-8)


def test_apply_conv_laplacian_complex_matches_real_imag_split():
    kernel = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    a = torch.randn(5, 5, dtype=torch.float64)
    b = torch.randn(5, 5, dtype=torch.float64)
    u = torch.complex(a, b)
    out = _apply_conv_laplacian(u, kernel, pad=1)
    expected = torch.complex(
        _apply_conv_laplacian(a, kernel, pad=1), _apply_conv_laplacian(b, kernel, pad=1)
    )
    assert torch.allclose(out, expected)


def test_laplacian_stencil_6th_order_matches_known_coefficients():
    """The 6th-order cross stencil carries the textbook 1D coefficients,
    scaled by 1/dx**2 on the x arm and 1/dy**2 on the y arm."""
    K, pad = laplacian_stencil(6, dx_nd=1.0, dy_nd=0.5)
    assert pad == 3 and K.shape == (7, 7)
    # 1D coefficients (Fornberg): offsets 1..3, then the centre; cy = 1/0.5**2.
    for k, ck in enumerate([3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0], start=1):
        assert K[pad - k, pad] == pytest.approx(ck)
        assert K[pad + k, pad] == pytest.approx(ck)
        assert K[pad, pad - k] == pytest.approx(4.0 * ck)
        assert K[pad, pad + k] == pytest.approx(4.0 * ck)
    assert K[pad, pad] == pytest.approx(-49.0 / 18.0 * 5.0)
    # everything off the cross is zero
    off = K.copy()
    off[pad, :] = 0.0
    off[:, pad] = 0.0
    assert (off == 0.0).all()


@pytest.mark.parametrize("space_order", [3, 12])
def test_laplacian_stencil_unsupported_order_raises(space_order):
    """Orders without a coefficient table are rejected."""
    with pytest.raises(ValueError):
        laplacian_stencil(space_order, 1.0, 1.0)


@pytest.mark.parametrize(
    "cls,pad",
    [
        (Laplacian2ndOrder, 1),
        (Laplacian4thOrder, 2),
        (Laplacian6thOrder, 3),
        (Laplacian8thOrder, 4),
        (Laplacian10thOrder, 5),
    ],
)
def test_laplacian_orders_are_exact_on_a_quadratic_interior(cls, pad):
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
    out = cls(wf).apply(u)
    center = out[pad : n - pad, pad : n - pad]
    assert torch.allclose(center, torch.full_like(center, 4.0), atol=1e-6)


# --------------------------------------------------------------------------- #
# utils.py: WaveEquation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("space_order", [2, 4, 6, 8, 10])
def test_wave_equation_selects_matching_laplacian(space_order):
    grid = _make_grid()
    wf = _make_wavefield(grid)
    we = WaveEquation(wf, space_order=space_order, damping_scale=1.0)
    expected_lap = {
        2: Laplacian2ndOrder,
        4: Laplacian4thOrder,
        6: Laplacian6thOrder,
        8: Laplacian8thOrder,
        10: Laplacian10thOrder,
    }[space_order]
    assert isinstance(we._lap, expected_lap)


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


def test_wave_equation_residual_matches_manual_assembly():
    grid = _make_grid(pml_width=2)
    wf = _make_wavefield(grid)
    we = WaveEquation(wf, space_order=2, damping_scale=1.7)

    torch.manual_seed(0)
    amp = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.float64).to(wf.cdtype)
    source = torch.randn(1, 1, grid.nx, grid.ny, dtype=torch.float64).to(wf.cdtype)
    c = wf.velocity_model.c

    r = we.residual(amp, c, source)

    lambda_t, lambda_tt = wf.frequency_selection.symbols_broadcast()
    lambda_t = lambda_t.to(dtype=amp.dtype)
    lambda_tt = lambda_tt.to(dtype=amp.dtype)
    wsp_nd = (c / grid.c0).to(amp.dtype)
    lap = Laplacian2ndOrder(wf).apply(amp)
    core = lambda_tt * amp - wsp_nd**2 * lap - source
    sigma_sum = grid.sigma_x_nd + grid.sigma_y_nd
    sigma_prod = grid.sigma_x_nd * grid.sigma_y_nd
    expected = core + 1.7 * (sigma_sum * lambda_t * amp + sigma_prod * amp)

    assert torch.allclose(r, expected, atol=1e-5)


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
    # Regression test for a fixed infinite loop: before the fix, the
    # `2*n - 2 - idx` fold degenerates to idx <-> -idx for n=1 and never
    # terminates for idx != 0. With a single cell, every offset -- however
    # far out of bounds -- refers to that same cell.
    for idx in range(-5, 6):
        assert _reflect_index(idx, 1) == 0


@pytest.mark.parametrize("n", [4, 7])
def test_reflect_index_matches_torch_pad_reflect_semantics(n):
    # F.pad(..., mode="reflect") only supports 1-D reflect padding on a 2-D
    # or 3-D input's last dim (a 4-D input needs 2-D reflect padding instead).
    x = torch.arange(n, dtype=torch.float64).view(1, 1, n)
    pad = n - 1
    padded = F.pad(x, (pad, pad), mode="reflect").view(-1)
    for idx in range(-pad, n + pad):
        expected = padded[idx + pad].item()
        actual = x.view(-1)[_reflect_index(idx, n)].item()
        assert actual == expected, f"n={n}, idx={idx}"


def test_reflect_index_raises_for_nonpositive_n():
    with pytest.raises(ValueError):
        _reflect_index(0, 0)


@pytest.mark.parametrize("space_order", [2, 4, 6, 8, 10])
def test_sparse_laplacian_matches_dense_conv_laplacian(space_order):
    """The sparse Laplacian the Helmholtz assembly uses equals the torch conv
    Laplacian (reflect boundaries) on a random field -- the key correctness
    invariant the sparse Helmholtz assembly depends on."""
    grid = _make_grid(pml_width=0, interior_shape=(9, 9), dtype=torch.float64)
    wf = _make_wavefield(grid)
    dense_cls = {
        2: Laplacian2ndOrder,
        4: Laplacian4thOrder,
        6: Laplacian6thOrder,
        8: Laplacian8thOrder,
        10: Laplacian10thOrder,
    }[space_order]

    torch.manual_seed(space_order)
    u = torch.randn(grid.nx, grid.ny, dtype=torch.float64)
    dense_out = dense_cls(wf).apply(u)

    L = HelmholtzSolver(
        wf, None, space_order=space_order, damping_scale=1.0
    )._laplacian_csr()
    sparse_out = (L @ u.numpy().reshape(-1)).reshape(grid.nx, grid.ny)

    assert torch.allclose(dense_out, torch.as_tensor(sparse_out), atol=1e-8)


def test_assemble_laplacian_csr_shape():
    grid = _make_grid(pml_width=0, interior_shape=(5, 6))
    kernel, pad = laplacian_stencil(2, grid.dx_nd, grid.dy_nd)
    L = assemble_laplacian_csr(grid.nx, grid.ny, kernel, pad)
    n = grid.nx * grid.ny
    assert L.shape == (n, n)
    assert L.nnz > 0


def test_assemble_h_sparse_matches_wave_equation_residual_at_zero_source(tiny_setup):
    solver = HelmholtzSolver(
        tiny_setup.wf, tiny_setup.geom, space_order=2, damping_scale=1.0
    )
    c = tiny_setup.vm.c.detach()
    H = solver.assemble_H_sparse(c, freq_idx=0)

    torch.manual_seed(2)
    amp = torch.randn(
        1, 1, tiny_setup.grid.nx, tiny_setup.grid.ny, dtype=torch.float64
    ).to(tiny_setup.wf.cdtype)
    source = torch.zeros_like(amp)
    r = solver.wave_eq.residual(amp, c, source)

    flat = amp.reshape(-1).cpu().numpy().astype(H.dtype)
    Hu = (H @ flat).reshape(tiny_setup.grid.nx, tiny_setup.grid.ny)
    assert torch.allclose(
        r[0, 0], torch.as_tensor(Hu, dtype=r.dtype), atol=1e-4, rtol=1e-4
    )


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
    sources = solver._sources()
    r = solver.wave_eq.residual(amp, tiny_setup.vm.c, sources)
    r_rms = float(torch.mean(torch.abs(r) ** 2).sqrt())
    src_rms = float(torch.mean(torch.abs(sources) ** 2).sqrt())
    assert r_rms / max(src_rms, 1e-30) < 1e-3  # well-converged direct solve
