from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import SuperLU, splu

from odil_wave.wavefield import Wavefield
from .spatial import _C6, _C6_CENTER, _C8, _C8_CENTER, _C10, _C10_CENTER
from .utils import WaveEquation


def np_complex_dtype(torch_cdtype: torch.dtype):
    """Numpy complex dtype for the sparse Helmholtz assembly / SuperLU factor+solve."""
    if torch_cdtype == torch.complex64:
        return np.complex64
    if torch_cdtype == torch.complex128:
        return np.complex128
    raise ValueError(f"Unsupported complex dtype for Helmholtz LU: {torch_cdtype}")


class HelmholtzFactorCache:
    """Cached SuperLU factors of ``H(c, omega_k)`` for a fixed medium ``c``."""

    def __init__(self, solver: "HelmholtzSolver", c: torch.Tensor) -> None:
        """Factorise H at c for every frequency bin, one SuperLU LU per bin."""
        self.solver = solver
        wf = solver.wavefield
        grid = wf.grid
        self.nx, self.ny = grid.nx, grid.ny
        self.n = self.nx * self.ny
        self.nf = wf.frequency_selection.n_frequencies
        self.cdtype = wf.cdtype
        self.np_cdtype = np_complex_dtype(self.cdtype)
        self.device = grid.device
        self._H: list[sp.csr_matrix] = []
        self._lu: list[SuperLU] = []
        self.n_factor = 0
        self.n_forward_solves = 0
        self.n_adjoint_solves = 0

        c_det = c.detach()
        for k in range(self.nf):
            H = solver.assemble_H_sparse(c_det, k)
            lu = splu(H.tocsc())
            self.n_factor += 1
            self._H.append(H.tocsr())
            self._lu.append(lu)

    def reset_counters(self) -> None:
        """Zero the forward/adjoint solve counters."""
        self.n_forward_solves = 0
        self.n_adjoint_solves = 0

    def matvec(self, u: torch.Tensor) -> torch.Tensor:
        """Apply sparse ``H`` (no solve): ``z = H u``.

        ``u`` shape ``(n_shots, nf, nx, ny)`` complex.
        """
        if u.ndim != 4 or u.shape[1] != self.nf:
            raise ValueError(
                f"expected u shape (n_shots, {self.nf}, {self.nx}, {self.ny}), "
                f"got {tuple(u.shape)}"
            )
        n_shots = u.shape[0]
        out = torch.empty_like(u)
        for k in range(self.nf):
            U = (
                u[:, k]
                .detach()
                .cpu()
                .numpy()
                .reshape(n_shots, self.n)
                .T.astype(self.np_cdtype, copy=False)
            )
            Z = self._H[k] @ U
            out[:, k] = torch.as_tensor(
                Z.T.reshape(n_shots, self.nx, self.ny),
                dtype=self.cdtype,
                device=self.device,
            )
        return out

    def solve(
        self,
        rhs: torch.Tensor,
        *,
        trans: str = "N",
    ) -> torch.Tensor:
        """Solve ``H X = RHS`` (``trans='N'``) or ``H^H X = RHS`` (``trans='H'``).

        ``rhs`` / return shape ``(n_shots, nf, nx, ny)`` complex. Each frequency
        uses one batched multi-RHS SuperLU call (counts as one linear solve per
        frequency toward ``n_forward_solves`` / ``n_adjoint_solves``).
        """
        if trans not in ("N", "H", "T"):
            raise ValueError(f"trans must be 'N', 'H', or 'T'; got {trans!r}")
        if rhs.ndim != 4 or rhs.shape[1] != self.nf:
            raise ValueError(
                f"expected rhs shape (n_shots, {self.nf}, {self.nx}, {self.ny}), "
                f"got {tuple(rhs.shape)}"
            )
        n_shots = rhs.shape[0]
        out = torch.empty_like(rhs)
        is_adj = trans in ("H", "T")
        for k in range(self.nf):
            F = (
                rhs[:, k]
                .detach()
                .cpu()
                .numpy()
                .reshape(n_shots, self.n)
                .T.astype(self.np_cdtype, copy=False)
            )
            X = self._lu[k].solve(F, trans=trans)
            out[:, k] = torch.as_tensor(
                X.T.reshape(n_shots, self.nx, self.ny),
                dtype=self.cdtype,
                device=self.device,
            )
        if is_adj:
            self.n_adjoint_solves += self.nf
        else:
            self.n_forward_solves += self.nf
        return out


def _reflect_index(idx: int, n: int) -> int:
    """Map an index to ``[0, n)`` with the same rule as ``F.pad(..., mode='reflect')``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        # 2*n-2 == 0 degenerates the fold below into idx <-> -idx, which
        # never terminates for idx != 0. With a single cell, every offset
        # refers to that same cell.
        return 0
    while idx < 0 or idx >= n:
        if idx < 0:
            idx = -idx
        else:
            idx = 2 * n - 2 - idx
    return idx


def _laplacian_kernel(
    space_order: int, dx_nd: float, dy_nd: float
) -> Tuple[np.ndarray, int]:
    """Return ``(kernel[hx, hy], pad)`` matching :mod:`odil_wave.operator.spatial`."""
    cx = 1.0 / dx_nd**2
    cy = 1.0 / dy_nd**2
    if space_order == 2:
        pad = 1
        K = np.zeros((3, 3), dtype=np.float64)
        K[0, 1] = cx
        K[2, 1] = cx
        K[1, 0] = cy
        K[1, 2] = cy
        K[1, 1] = -2.0 * cx - 2.0 * cy
        return K, pad
    if space_order == 4:
        pad = 2
        K = np.zeros((5, 5), dtype=np.float64)
        K[0, 2] = -1.0 / 12.0 * cx
        K[1, 2] = 16.0 / 12.0 * cx
        K[3, 2] = 16.0 / 12.0 * cx
        K[4, 2] = -1.0 / 12.0 * cx
        K[2, 0] = -1.0 / 12.0 * cy
        K[2, 1] = 16.0 / 12.0 * cy
        K[2, 3] = 16.0 / 12.0 * cy
        K[2, 4] = -1.0 / 12.0 * cy
        K[2, 2] = -30.0 / 12.0 * (cx + cy)
        return K, pad
    wide = {6: (_C6, _C6_CENTER), 8: (_C8, _C8_CENTER), 10: (_C10, _C10_CENTER)}
    if space_order in wide:
        coeffs, center = wide[space_order]
        pad = len(coeffs)
        n = 2 * pad + 1
        K = np.zeros((n, n), dtype=np.float64)
        for k, ck in enumerate(coeffs):
            offset = k + 1
            K[pad - offset, pad] = ck * cx
            K[pad + offset, pad] = ck * cx
            K[pad, pad - offset] = ck * cy
            K[pad, pad + offset] = ck * cy
        K[pad, pad] = center * (cx + cy)
        return K, pad
    raise ValueError(f"Unsupported space_order for sparse Helmholtz: {space_order}")


def assemble_laplacian_csr(
    nx: int, ny: int, kernel: np.ndarray, pad: int
) -> sp.csr_matrix:
    """Sparse Laplacian with Neumann-mirror (reflect) boundaries."""
    kh, kw = kernel.shape
    assert kh == 2 * pad + 1 and kw == 2 * pad + 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    coeff = float(kernel[di + pad, dj + pad])
                    if coeff == 0.0:
                        continue
                    ii = _reflect_index(i + di, nx)
                    jj = _reflect_index(j + dj, ny)
                    rows.append(row)
                    cols.append(ii * ny + jj)
                    data.append(coeff)
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
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
        space_order: int = 2,
        pml_weight: float = 1.0,
    ) -> None:
        """Bind to a wavefield/geometry and build the matching WaveEquation."""
        self.wavefield = wavefield
        self.geometry = geometry
        self.space_order = space_order
        self.pml_weight = float(pml_weight)
        self.wave_eq = WaveEquation(
            wavefield, space_order=space_order, pml_weight=pml_weight
        )
        self._L_csr: Optional[sp.csr_matrix] = None

    def factorize(self, c: torch.Tensor) -> HelmholtzFactorCache:
        """Build a :class:`HelmholtzFactorCache` for fixed ``c`` (all frequencies)."""
        return HelmholtzFactorCache(self, c)

    def _sources(self) -> torch.Tensor:
        """Stack per-shot injection sources, scaled by t0^2 for non-dimensional time."""
        grid = self.wavefield.grid
        cdtype = self.wavefield.cdtype
        t0 = grid.t0
        return (
            torch.stack(
                [
                    self.geometry.source_field(i).to(dtype=cdtype, device=grid.device)
                    for i in range(self.geometry.n_sources)
                ]
            )
            * t0**2
        )

    def _laplacian_csr(self) -> sp.csr_matrix:
        """Return the cached sparse Laplacian CSR, building it on first call."""
        if self._L_csr is None:
            grid = self.wavefield.grid
            K, pad = _laplacian_kernel(
                self.space_order, float(grid.dx_nd), float(grid.dy_nd)
            )
            self._L_csr = assemble_laplacian_csr(grid.nx, grid.ny, K, pad)
        return self._L_csr

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
        diag = (lam_tt + self.pml_weight * (sig_sum * lam_t + sig_prod)).astype(cd)

        L = self._laplacian_csr()
        # H = diag(a) - diag(c_nd^2) @ L
        H = sp.diags(diag, format="csr", dtype=cd) - sp.diags(c_nd2, format="csr").dot(
            L
        ).astype(cd)
        assert H.shape == (n, n)
        return H.tocsr()

    def solve(self, verbose: bool = True) -> List[Wavefield]:
        """Solve H(c) u = f for every shot and frequency, returning one Wavefield per shot.

        Assembles and factorises H once per frequency, then solves all shots'
        sources in one batched call at that frequency.

        Parameters
        ----------
        verbose :
            When True, print per-frequency assemble/factor/solve timings plus
            a final residual/timing summary; also computes that summary
            (an extra residual evaluation), which is skipped when False.

        Returns
        -------
        List[Wavefield]
            One solved Wavefield per shot, each holding the amplitude at
            every selected frequency.
        """
        wf = self.wavefield
        grid = wf.grid
        freq = wf.frequency_selection
        nf = freq.n_frequencies
        Nx, Ny = grid.shape
        n = Nx * Ny
        n_shots = self.geometry.n_sources
        cdtype = wf.cdtype
        device = grid.device
        c = wf.velocity_model.c.to(device=device)

        sources = self._sources()
        amp = torch.zeros(n_shots, nf, Nx, Ny, dtype=cdtype, device=device)

        assemble_times: list[float] = []
        factor_times: list[float] = []
        solve_times: list[float] = []
        t_all0 = time.perf_counter()

        # Cache Laplacian CSR once (geometry / space_order only).
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
        return outputs
