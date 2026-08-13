"""Exact frozen-c wavefield-subproblem Hessian ``H`` and its inverse action.

The frozen-``c`` wavefield objective (the u-block subproblem of :class:`LBFGSB`)

    J(u) = w_pde · mean|A(c) u − f|²      (mean over shots × freqs × grid)
         + w_data · mean|D (P u − d)|²    (mean over shots × freqs × receivers)

is *exactly quadratic* in ``u``, so its Hessian is the constant sparse operator

    H = α · AᴴA  +  β · Pᴴ|D|²P ,
        α = 2 w_pde  / N_pde ,     N_pde  = n_shots · nf · nx · ny
        β = 2 w_data / N_data ,    N_data = n_shots · nf · n_recv

where ``A`` is the sparse Helmholtz operator (``HelmholtzSolver.assemble_H_sparse``
— the same operator the matrix-free residual uses), ``P`` samples ``u`` at the
receiver cells and ``D`` is the loss's per-element data weighting (frequency
weights ``f_weights`` and the ``normalize_data`` trace scaling, so ``H`` matches
the *actual* loss Hessian, not just the un-normalised sandbox form). ``Pᴴ|D|²P``
is diagonal on the grid, non-zero only at receiver cells, and — through ``D`` —
may differ per shot, so the objective is block-diagonal across shots.

The realised L-BFGS u-block converges to ``u* = argmin J = u0 − H⁻¹ g(u0)``, so
the wavefield update each *term* induces (its gradient deconvolved by the shared
Hessian) is ``du_term = −H⁻¹ g_term`` with ``du_pde + du_data = −H⁻¹ g_total``.
This is exactly why a receiver/annulus-localised gradient produces a delocalised,
physically reasonable update: ``H⁻¹`` (dominated by ``A⁻¹A⁻ᴴ`` off the receivers)
is a global Green's-function operator.

Implementation: assemble ``H`` per frequency as a sparse matrix and factorise it
once with SuperLU — the *exact* ``H⁻¹``. A direct factorisation (rather than a
PDE-only preconditioned CG) is used deliberately: ``w_data`` is averaged over
~``n_recv`` cells while ``w_pde`` is averaged over the whole grid, so the data
block is enormously heavier per DOF and a PDE-only preconditioner would be
hopeless. Factorisations are cached by ``(frequency, data-diagonal)`` so shots
that share a data weighting (e.g. ``normalize_data`` in ``{None, "none"}``)
reuse one factor. :meth:`verify` checks the sparse ``H`` against the autograd
Hessian-vector product (which, ``J`` being quadratic, equals ``H v`` exactly).

Ported from ``sandbox/profiling/optim_block_diag/hess_split.py``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu

from odil_wave.operator.helmholtz import HelmholtzSolver, np_complex_dtype


def _global_norm(x: torch.Tensor) -> float:
    v = x.real.square() + x.imag.square() if x.is_complex() else x.square()
    return float(v.sum().sqrt().cpu())


class UBlockHessian:
    """Exact frozen-c u-subproblem Hessian ``H`` and its inverse action ``−H⁻¹g``.

    Built once per outer (``c`` frozen); ``H`` is constant in ``u`` (quadratic
    objective), so the same factorisation serves every u-solve depth.

    Parameters
    ----------
    wavefield:
        The inverse :class:`~odil_wave.wavefield.Wavefield` (grid, frequency
        selection, complex dtype). Its velocity model is irrelevant — ``c_full``
        is passed explicitly to ``assemble_H_sparse``.
    loss:
        The :class:`~odil_wave.loss.InverseLoss`; supplies the geometry (receiver
        cells), the source count, the data normalisation (``_trace_scale`` /
        ``_f_weights``) and the default weights.
    c_full:
        Physical full-grid velocity ``(nx, ny)`` at which to freeze ``H``.
    weights:
        Optional ``{"pde": …, "data": …}`` override (defaults to
        ``loss.config.weights``) so a scheduled ``pde`` weight is honoured.
    """

    def __init__(self, wavefield, loss, c_full: torch.Tensor, weights=None):
        self.wavefield = wavefield
        self.loss = loss
        grid = wavefield.grid
        wave_eq = loss.config.wave_eq

        self.nx, self.ny = int(grid.nx), int(grid.ny)
        self.n = self.nx * self.ny
        self.nf = int(wavefield.n_frequencies)
        self.cdtype = wavefield.cdtype
        self.np_cdtype = np_complex_dtype(self.cdtype)
        self.device = grid.device

        c_det = c_full.detach()
        solver = HelmholtzSolver(
            wavefield,
            loss.config.geometry,
            space_order=wave_eq.space_order,
            pml_weight=wave_eq.pml_weight,
        )

        # Term weights + mean-normalisation element counts (mean over the full
        # tensor, matching InverseLoss.evaluate's mean_abs_sq).
        w = dict(weights) if weights is not None else dict(loss.config.weights)
        w_pde, w_data = float(w.get("pde", 1.0)), float(w.get("data", 1.0))
        n_shots = int(loss.sources.shape[0])
        recv_ij = loss.config.geometry.recv_ij
        n_recv = int(recv_ij.shape[0])
        n_pde = n_shots * self.nf * self.n
        n_data = n_shots * self.nf * n_recv
        self.alpha = 2.0 * w_pde / max(n_pde, 1)
        self.beta = 2.0 * w_data / max(n_data, 1)
        self.n_shots = n_shots

        # Receiver flat indices (x-major, row = i*ny + j — matching
        # assemble_laplacian_csr / the batched-column convention below).
        ri = recv_ij[:, 0].detach().cpu().numpy().astype(int)
        rj = recv_ij[:, 1].detach().cpu().numpy().astype(int)
        self._recv_flat = ri * self.ny + rj

        # Per-(shot, freq, receiver) squared data weight |D|² = |f_w / scale|²,
        # so β·Pᴴ|D|²P is the *actual* data Hessian (incl. normalisation).
        w2 = np.ones((n_shots, self.nf, n_recv), dtype=np.float64)
        fw = getattr(loss, "_f_weights", None)
        if fw is not None:
            w2 = w2 * (
                fw.detach().cpu().numpy().astype(np.float64).reshape(1, -1, 1) ** 2
            )
        scale = getattr(loss, "_trace_scale", None)
        if scale is not None:
            sc = (
                scale.detach().cpu().numpy().astype(np.float64)
            )  # (n_shots,1,n_recv)|(…,1,1)
            w2 = w2 / (sc**2)
        self._w2 = w2  # (n_shots, nf, n_recv)

        # Per-frequency Helmholtz operator + AᴴA (shared across shots).
        self._A: List[sp.csr_matrix] = []
        self._AhA: List[sp.csr_matrix] = []
        for k in range(self.nf):
            A = solver.assemble_H_sparse(c_det, k).tocsr().astype(self.np_cdtype)
            self._A.append(A)
            self._AhA.append((A.getH() @ A).tocsr())

        # Factorisation cache keyed by (freq, data-diagonal signature).
        self._lu_cache: Dict[tuple, object] = {}

    # ---- data diagonal + factor (cached per (freq, shot-weighting)) -------- #
    def _diag_key(self, k: int, s: int) -> tuple:
        return (k, self._w2[s, k].tobytes())

    def _factor(self, k: int, s: int):
        key = self._diag_key(k, s)
        lu = self._lu_cache.get(key)
        if lu is None:
            d = np.zeros(self.n, dtype=self.np_cdtype)
            d[self._recv_flat] = self.beta * self._w2[s, k]
            H = (self.alpha * self._AhA[k] + sp.diags(d, format="csr")).tocsc()
            lu = splu(H)
            self._lu_cache[key] = lu
        return lu

    # ---- reshape helpers (batched columns, x-major) ------------------------ #
    def _cols(self, field_k: torch.Tensor) -> np.ndarray:
        """``(n_shots, nx, ny)`` complex -> ``(n, n_shots)`` numpy columns."""
        return (
            field_k.detach()
            .cpu()
            .numpy()
            .reshape(self.n_shots, self.n)
            .T.astype(self.np_cdtype, copy=False)
        )

    def _from_cols(self, cols: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            cols.T.reshape(self.n_shots, self.nx, self.ny),
            dtype=self.cdtype,
            device=self.device,
        )

    # ---- exact inverse action --------------------------------------------- #
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """``H⁻¹ b`` for ``b`` of shape ``(n_shots, nf, nx, ny)`` (complex)."""
        out = torch.empty_like(b)
        for k in range(self.nf):
            B = self._cols(b[:, k])  # (n, n_shots)
            X = np.empty_like(B)
            # Group shots that share a factor (identical data weighting) so a
            # single solve handles them together (one factor when un-normalised).
            groups: Dict[tuple, List[int]] = {}
            for s in range(self.n_shots):
                groups.setdefault(self._diag_key(k, s), []).append(s)
            for key, shots in groups.items():
                lu = self._factor(k, shots[0])
                X[:, shots] = lu.solve(B[:, shots])
            out[:, k] = self._from_cols(X)
        return out

    def du_from_grad(self, g: torch.Tensor) -> torch.Tensor:
        """Wavefield update ``−H⁻¹ g`` induced by the (weighted) gradient ``g``."""
        return -self.solve(g)

    # ---- exact sparse H apply (verification only) -------------------------- #
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(v)
        for k in range(self.nf):
            V = self._cols(v[:, k])  # (n, n_shots)
            AhA_V = self._AhA[k] @ V
            Z = self.alpha * AhA_V
            for s in range(self.n_shots):
                d = np.zeros(self.n, dtype=self.np_cdtype)
                d[self._recv_flat] = self.beta * self._w2[s, k]
                Z[:, s] = Z[:, s] + d * V[:, s]
            out[:, k] = self._from_cols(Z)
        return out

    # ---- correctness gate: sparse H vs autograd Hessian-vector product ----- #
    def _u_grad(
        self, u_re: torch.Tensor, u_im: torch.Tensor, c_full: torch.Tensor, weights
    ) -> torch.Tensor:
        a = u_re.detach().clone().requires_grad_(True)
        b = u_im.detach().clone().requires_grad_(True)
        L = self.loss.evaluate(
            torch.complex(a, b), c_full, None, weights_override=weights
        )
        ga, gb = torch.autograd.grad(L, (a, b))
        return torch.complex(ga.detach(), gb.detach())

    def verify(
        self,
        u_re: torch.Tensor,
        u_im: torch.Tensor,
        c_full: torch.Tensor,
        weights=None,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Assert the sparse ``H`` equals the autograd Hvp (exact for a quadratic).

        ``H v = grad_u J(u0 + v) − grad_u J(u0)`` holds exactly because ``J`` is
        quadratic in ``u``; compare against the sparse operator on a random ``v``.
        """
        w = dict(weights) if weights is not None else dict(self.loss.config.weights)
        torch.manual_seed(seed)
        vr = torch.randn(u_re.shape, dtype=u_re.dtype, device=u_re.device)
        vi = torch.randn(u_im.shape, dtype=u_im.dtype, device=u_im.device)
        v = torch.complex(vr, vi).to(self.cdtype)
        g0 = self._u_grad(u_re, u_im, c_full, w)
        g1 = self._u_grad(u_re + vr, u_im + vi, c_full, w)
        Hv_auto = g1 - g0
        Hv_sparse = self.apply(v)
        rel = _global_norm(Hv_sparse - Hv_auto) / max(_global_norm(Hv_auto), 1e-300)
        return {"rel_err": rel, "alpha": self.alpha, "beta": self.beta}
