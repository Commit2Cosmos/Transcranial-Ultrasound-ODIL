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
from .spatial import laplacian_stencil
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


@functools.lru_cache(maxsize=8)
def laplacian_csr(
    nx: int, ny: int, space_order: int, dx_nd: float, dy_nd: float
) -> sp.csr_matrix:
    """Sparse Laplacian with Neumann-mirror (reflect) boundaries, cached per grid.

    Row-major ``(nx, ny)`` flattening, as ``tensor.reshape(-1)``. The matrix is
    shared between callers: don't modify it in place.
    """
    kernel, pad = laplacian_stencil(space_order, dx_nd, dy_nd)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i, j = i.reshape(-1), j.reshape(-1)
    rows, cols, data = [], [], []
    # Within a row, COO -> CSR sums entries that reflect onto the same column
    # in the order given, so the stencil order fixes the rounding.
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            coeff = kernel[di + pad, dj + pad]
            if coeff == 0.0:
                continue
            rows.append(i * ny + j)
            cols.append(_reflect_index(i + di, nx) * ny + _reflect_index(j + dj, ny))
            data.append(np.full(nx * ny, coeff))
    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nx * ny, nx * ny),
    )


class HelmholtzSolver:
    """Solve H(c, omega) u = f_hat for complex frequency-domain wavefields.

    For fixed ``c``, assembles a sparse Helmholtz matrix once per frequency,
    factorises with ``splu``, then solves all shot RHS in one batched call.
    """

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

    def _laplacian_csr(self) -> sp.csr_matrix:
        grid = self.wavefield.grid
        return laplacian_csr(
            grid.nx, grid.ny, self.space_order, float(grid.dx_nd), float(grid.dy_nd)
        )

    def assemble_H_sparse(self, c: torch.Tensor, freq_idx: int) -> sp.csr_matrix:
        """Assemble the sparse Helmholtz matrix H for one frequency.

        Parameters
        ----------
        c :
            Full-grid velocity field to assemble H at.
        freq_idx :
            Index into the wavefield's frequency selection.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse Helmholtz matrix H at that frequency (matches the
            matrix-free residual).
        """
        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        nx, ny = grid.nx, grid.ny
        n = nx * ny
        c0 = float(grid.c0)
        c_nd2 = (c.detach().cpu().numpy().astype(np.float64) / c0) ** 2
        c_nd2 = c_nd2.reshape(-1)

        lam_t = complex(freq.lambda_t[freq_idx].detach().cpu().numpy())
        lam_tt = float(freq.lambda_tt[freq_idx].detach().cpu().numpy())

        sig_sum = (
            (grid.sigma_x_nd + grid.sigma_y_nd)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ).reshape(-1)
        sig_prod = (
            (grid.sigma_x_nd * grid.sigma_y_nd)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ).reshape(-1)

        cd = np_complex_dtype(self.wavefield.cdtype)
        # Diagonal: lambda_tt + w * (sigma_sum * lambda_t + sigma_prod)
        diag = (lam_tt + self.damping_scale * (sig_sum * lam_t + sig_prod)).astype(cd)

        L = self._laplacian_csr()
        # H = diag(a) - diag(c_nd^2) @ L
        H = sp.diags(diag, format="csr", dtype=cd) - sp.diags(c_nd2, format="csr").dot(
            L
        ).astype(cd)
        assert H.shape == (n, n)
        return H.tocsr()

    def solve(
        self, verbose: bool = False, return_factors: bool = False
    ) -> Union[List[Wavefield], Tuple[List[Wavefield], List[SuperLU]]]:
        """Solve H(c) u = f for every shot and frequency; one Wavefield per shot.

        Assembles and factorises H once per frequency, then solves all shots'
        sources in one batched call at that frequency.

        Parameters
        ----------
        verbose :
            When True, print per-frequency assemble/factor/solve timings plus
            a final residual/timing summary; also computes that summary
            (an extra residual evaluation), which is skipped when False.
        return_factors :
            When True, also return the per-frequency ``splu`` factors of H
            (the wavefield preconditioner reuses them for the band).

        Returns
        -------
        List[Wavefield]
            One solved Wavefield per shot, each holding the amplitude at
            every selected frequency; with ``return_factors``, the tuple
            ``(wavefields, factors)``, ``factors[k]`` the factor at frequency
            ``k``.
        """
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

        t_lap0 = time.perf_counter()
        _ = self._laplacian_csr()
        t_lap = time.perf_counter() - t_lap0

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
                f"  timings: lap_csr={t_lap:.3f}s  "
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
