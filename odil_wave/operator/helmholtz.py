from __future__ import annotations

import functools
import time
from typing import List, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import SuperLU, splu

from odil_wave.defaults import library_default
from odil_wave.wavefield import Wavefield
from .spatial import STAGGERED_D1_COEFFS, second_difference_stencil
from .utils import WaveEquation


def np_complex_dtype(torch_cdtype: torch.dtype):
    """Numpy complex dtype for the sparse Helmholtz assembly / SuperLU factor+solve."""
    if torch_cdtype == torch.complex64:
        return np.complex64
    if torch_cdtype == torch.complex128:
        return np.complex128
    raise ValueError(f"Unsupported complex dtype for Helmholtz LU: {torch_cdtype}")


def _reflect_index(idx: np.ndarray, n: int) -> np.ndarray:
    """Map indices to ``[0, n)`` like ``F.pad(..., mode='reflect')``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        # The fold below has period 2*n - 2 == 0; with a single cell, every
        # offset refers to that same cell.
        return np.zeros_like(idx)
    period = 2 * n - 2
    m = np.mod(idx, period)
    return np.where(m >= n, period - m, m)


@functools.lru_cache(maxsize=16)
def second_difference_1d(n: int, space_order: int, h_nd: float) -> sp.csr_matrix:
    """Sparse ``(n, n)`` order-P ``d²/dx²`` with Neumann-mirror (reflect)
    boundaries, cached. Shared between callers: don't modify it in place."""
    kernel = second_difference_stencil(space_order, h_nd)
    pad = len(kernel) // 2
    i = np.arange(n)
    rows, cols, data = [], [], []
    # Within a row, COO -> CSR sums entries that reflect onto the same column
    # in the order given, so the stencil order fixes the rounding.
    for d in range(-pad, pad + 1):
        rows.append(i)
        cols.append(_reflect_index(i + d, n))
        data.append(np.full(n, kernel[d + pad]))
    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def _staggered_difference_matrix(
    rows: int, cols: int, coeffs: Tuple[float, ...]
) -> sp.csr_matrix:
    """The ``(rows, cols)`` staggered differences, row ``k`` centred between
    columns ``k + P/2 - 1`` and ``k + P/2``, unit spacing."""
    half = len(coeffs)
    return sp.diags(
        [-c for c in reversed(coeffs)] + list(coeffs),
        list(range(2 * half)),
        shape=(rows, cols),
        format="csr",
    )


@functools.lru_cache(maxsize=16)
def _staggered_differences_1d(
    n: int, space_order: int
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """``(D⁺, D⁻)``: order-P staggered, nodes to the ``n + P - 1`` faces (ghosts
    included, reflect) and back, unit spacing."""
    coeffs = STAGGERED_D1_COEFFS[space_order]
    p = space_order
    nodes = np.arange(1 - p, n - 1 + p)
    reflect = sp.csr_matrix(
        (np.ones(len(nodes)), (np.arange(len(nodes)), _reflect_index(nodes, n))),
        shape=(len(nodes), n),
    )
    d_plus = _staggered_difference_matrix(n + p - 1, len(nodes), coeffs) @ reflect
    d_minus = _staggered_difference_matrix(n, n + p - 1, coeffs)
    return d_plus.tocsr(), d_minus


def stretched_second_difference_1d(
    n: int, space_order: int, h_nd: float, a_faces: np.ndarray
) -> sp.csr_matrix:
    """Sparse ``Λ̃ = Λ + D⁻ diag(a) D⁺ / h²`` along one axis, ``a_faces`` the
    PML's ``1/s - 1`` on the ``n + P - 1`` faces (see
    :class:`~odil_wave.operator.conditions.Pml`)."""
    d_plus, d_minus = _staggered_differences_1d(n, space_order)
    corr = d_minus @ sp.diags(a_faces) @ d_plus / h_nd**2
    return (second_difference_1d(n, space_order, h_nd) + corr).tocsr()


class HelmholtzSolver:
    """Solve H(c, omega) u = f_hat for complex frequency-domain wavefields."""

    def __init__(
        self,
        wavefield: Wavefield,
        geometry,
        *,
        space_order: int,
        damping_scale: float = library_default(WaveEquation, "damping_scale"),
    ) -> None:
        """Bind to a wavefield/geometry and build the matching WaveEquation.

        ``space_order`` and ``damping_scale`` are as in :class:`WaveEquation`.
        """
        self.wavefield = wavefield
        self.geometry = geometry
        self.space_order = space_order
        self.damping_scale = float(damping_scale)
        self.wave_eq = WaveEquation(
            wavefield, space_order=space_order, damping_scale=damping_scale
        )

    def assemble_H_sparse(self, c: torch.Tensor, freq_idx: int) -> sp.csr_matrix:
        """Sparse H at full-grid ``c`` and frequency index ``freq_idx``: the
        matrix of the matrix-free residual."""
        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        nx, ny = grid.nx, grid.ny
        n = nx * ny
        c0 = float(grid.c0)
        c_nd2 = (c.detach().cpu().numpy().astype(np.float64) / c0) ** 2
        c_nd2 = c_nd2.reshape(-1)
        lam_tt = float(freq.lambda_tt[freq_idx].detach().cpu().numpy())

        pml = self.wave_eq.pml
        s_x = pml.s_x[freq_idx, :, 0].detach().cpu().numpy().astype(np.complex128)
        s_y = pml.s_y[freq_idx, 0, :].detach().cpu().numpy().astype(np.complex128)
        a_x = pml.a_x[freq_idx, :, 0].detach().cpu().numpy().astype(np.complex128)
        a_y = pml.a_y[freq_idx, 0, :].detach().cpu().numpy().astype(np.complex128)
        Lx = sp.kron(
            stretched_second_difference_1d(
                nx, self.space_order, float(grid.dx_nd), a_x
            ),
            sp.identity(ny),
        )
        Ly = sp.kron(
            sp.identity(nx),
            stretched_second_difference_1d(
                ny, self.space_order, float(grid.dy_nd), a_y
            ),
        )
        # Row-major (nx, ny) flattening: x is the slow index.
        sx_nodes = np.repeat(s_x, ny)
        sy_nodes = np.tile(s_y, nx)
        lap = sp.diags(sy_nodes) @ Lx + sp.diags(sx_nodes) @ Ly

        cd = np_complex_dtype(self.wavefield.cdtype)
        mass = lam_tt * sx_nodes * sy_nodes
        H = (sp.diags(mass) - sp.diags(c_nd2) @ lap).astype(cd)
        assert H.shape == (n, n)
        return H.tocsr()

    def solve(
        self, verbose: bool = False, return_factors: bool = False
    ) -> Union[List[Wavefield], Tuple[List[Wavefield], List[SuperLU]]]:
        """Solve H(c) u = f for every shot and frequency; one Wavefield per shot,
        and with ``return_factors`` also H's ``splu`` factor per frequency.
        ``verbose`` prints timings and the residual (an extra evaluation)."""
        wf = self.wavefield
        grid = wf.grid
        freq = wf.frequency_selection
        nf = freq.n_frequencies
        Nx, Ny = grid.shape
        n = Nx * Ny
        n_shots = self.geometry.n_shots
        cdtype = wf.cdtype
        device = grid.device
        c = wf.velocity_model.c.to(device=device)

        sources = self.wave_eq.sources(self.geometry)
        amp = torch.zeros(n_shots, nf, Nx, Ny, dtype=cdtype, device=device)

        assemble_times: list[float] = []
        factor_times: list[float] = []
        solve_times: list[float] = []
        factors: List[SuperLU] = []
        t_all0 = time.perf_counter()

        for k in range(nf):
            t0 = time.perf_counter()
            H = self.assemble_H_sparse(c, k)
            t_asm = time.perf_counter() - t0
            assemble_times.append(t_asm)

            t0 = time.perf_counter()
            lu = splu(H.tocsc())
            t_fac = time.perf_counter() - t0
            factor_times.append(t_fac)
            if return_factors:
                factors.append(lu)

            # Batched RHS: columns are shots  (n, n_shots)
            F = (
                sources[:, k]
                .detach()
                .cpu()
                .numpy()
                .reshape(n_shots, n)
                .T.astype(np_complex_dtype(cdtype), copy=False)
            )
            t0 = time.perf_counter()
            U = lu.solve(F)
            t_sol = time.perf_counter() - t0
            solve_times.append(t_sol)

            U_t = torch.as_tensor(
                U.T.reshape(n_shots, Nx, Ny), dtype=cdtype, device=device
            )
            amp[:, k] = U_t

            if verbose:
                f_hz = float(freq.frequencies[k])
                print(
                    f"  freq[{k}] f={f_hz:.1f} Hz | "
                    f"assemble={t_asm:.3f}s  factor={t_fac:.3f}s  "
                    f"solve({n_shots} RHS)={t_sol:.3f}s  nnz={H.nnz}"
                )

        t_total = time.perf_counter() - t_all0

        if verbose:
            with torch.no_grad():
                r = self.wave_eq.residual(amp, c, sources)
                r_rms = float(torch.mean(torch.abs(r) ** 2).sqrt())
                src_rms = float(torch.mean(torch.abs(sources) ** 2).sqrt())
            d = {
                "ratio": r_rms / max(src_rms, 1e-30),
                "u_absmax": float(amp.abs().max()),
                "assemble_total_s": float(sum(assemble_times)),
                "factor_total_s": float(sum(factor_times)),
                "solve_total_s": float(sum(solve_times)),
                "total_s": float(t_total),
            }
            print(
                f"helmholtz ({n_shots} shots, {nf} freqs, "
                f"space_order={self.space_order}, n={n}): "
                f"|r|/|src| = {d['ratio']:.3e}, |u|_max = {d['u_absmax']:.3e}"
            )
            print(
                "  timings: "
                f"assemble={d['assemble_total_s']:.3f}s  "
                f"factor={d['factor_total_s']:.3f}s  "
                f"solve={d['solve_total_s']:.3f}s  "
                f"total={d['total_s']:.3f}s"
            )

        outputs: List[Wavefield] = []
        for s in range(n_shots):
            out = Wavefield(
                grid=grid,
                frequency_selection=freq,
                velocity_model=wf.velocity_model,
            )
            out.amplitude = amp[s]
            outputs.append(out)
        if return_factors:
            return outputs, factors
        return outputs
