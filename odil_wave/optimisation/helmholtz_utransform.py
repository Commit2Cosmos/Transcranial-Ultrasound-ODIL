"""Frequency-domain wavefield reparameterisation ``u = H(c, ω)^{-1} z``.

Used by :class:`~odil_wave.optimisation.base.LBFGSB` when ``u_precond="z"``.

While ``c`` is fixed, one SuperLU factorisation per frequency is reused for
forward ``H^{-1}`` and Hermitian-adjoint ``H^{-H}`` solves. For the c-stage,
:meth:`HelmholtzUTransform.apply_diff_c` re-factorises at the live ``c`` and
back-propagates data gradients into ``c`` via implicit differentiation.
"""

from __future__ import annotations

import numpy as np
import torch

from odil_wave.operator.helmholtz import HelmholtzFactorCache, HelmholtzSolver


def _helmholtz_solve(
    cache: HelmholtzFactorCache, rhs: torch.Tensor, *, trans: str = "N"
) -> torch.Tensor:
    """Shared SuperLU triangular solve (``trans='N'`` or ``'H'``)."""
    return cache.solve(rhs, trans=trans)


def _apply_laplacian(cache: HelmholtzFactorCache, u: torch.Tensor) -> torch.Tensor:
    """Apply the cached sparse Laplacian ``L`` (geometry only) to ``u``."""
    L = cache.solver._laplacian_csr()
    n_shots = u.shape[0]
    out = torch.empty_like(u)
    for k in range(cache.nf):
        U = (
            u[:, k]
            .detach()
            .cpu()
            .numpy()
            .reshape(n_shots, cache.n)
            .T
            .astype(np.complex128, copy=False)
        )
        LU = L @ U
        out[:, k] = torch.as_tensor(
            LU.T.reshape(n_shots, cache.nx, cache.ny),
            dtype=cache.cdtype,
            device=cache.device,
        )
    return out


class _HelmholtzInvFn(torch.autograd.Function):
    """``u = H^{-1} z`` with backward ``g_z = H^{-H} g_u`` (``c`` frozen)."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, cache: HelmholtzFactorCache) -> torch.Tensor:
        ctx.cache = cache
        with torch.no_grad():
            return _helmholtz_solve(cache, z, trans="N")

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        cache: HelmholtzFactorCache = ctx.cache
        with torch.no_grad():
            g_z = _helmholtz_solve(cache, grad_out.contiguous(), trans="H")
        return g_z, None


class _HelmholtzInvDiffCFn(torch.autograd.Function):
    """``u = H(c)^{-1} z`` with gradients w.r.t. real ``c`` only (``z`` frozen).

    Forward factorises at the trial ``c``. Backward uses
    ``λ = H^{-H} g_u`` and ``dH[δc] u = -(2 c_nd δc_nd) ⊙ (L u)``.
    """

    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        c_full: torch.Tensor,
        transform: "HelmholtzUTransform",
    ) -> torch.Tensor:
        cache = transform.solver.factorize(c_full.detach())
        transform._n_factor_total += cache.n_factor
        with torch.no_grad():
            u = _helmholtz_solve(cache, z.detach(), trans="N")
        transform._n_forward_total += cache.n_forward_solves
        cache.reset_counters()
        ctx.transform = transform
        ctx.cache = cache
        ctx.save_for_backward(u, c_full.detach())
        return u

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        transform: HelmholtzUTransform = ctx.transform
        cache: HelmholtzFactorCache = ctx.cache
        u, c_full = ctx.saved_tensors
        with torch.no_grad():
            lam = _helmholtz_solve(cache, grad_out.contiguous(), trans="H")
            transform._n_adjoint_total += cache.n_adjoint_solves
            cache.reset_counters()
            Lu = _apply_laplacian(cache, u)
            c0 = float(transform.solver.wavefield.grid.c0)
            c_nd = c_full / c0
            # dL = Re <λ, (2 c_nd / c0) ⊙ Lu ⊙ δc>  (PyTorch complex convention)
            coeff = (2.0 * c_nd / c0) * Lu
            g_c = (lam.conj() * coeff).real.sum(dim=(0, 1))
        return None, g_c, None


class HelmholtzUTransform:
    """Coordinate change ``u = H(c)^{-1} z`` using cached sparse Helmholtz LU.

    * :meth:`apply` — z-stage (``c`` frozen): autograd through ``H^{-H}``.
    * :meth:`apply_diff_c` — c-stage (``z`` frozen): autograd into ``c``.
    * :meth:`inverse` — seed ``z0 = H u0`` (sparse matvec).

    Call :meth:`rebuild` after accepted ``c`` updates so the next z-stage
    reuses a factorisation of the new medium.
    """

    def __init__(self, solver: HelmholtzSolver, c_full: torch.Tensor) -> None:
        self.solver = solver
        self._n_factor_total = 0
        self._n_forward_total = 0
        self._n_adjoint_total = 0
        self.cache = solver.factorize(c_full.detach())
        self._n_factor_total += self.cache.n_factor

    def _flush_solve_counts(self) -> None:
        """Fold the live cache's solve counters into the cumulative totals."""
        self._n_forward_total += self.cache.n_forward_solves
        self._n_adjoint_total += self.cache.n_adjoint_solves
        self.cache.reset_counters()

    def rebuild(self, c_full: torch.Tensor) -> None:
        """Refactorise after a medium update (invalidates previous LU)."""
        self._flush_solve_counts()
        self.cache = self.solver.factorize(c_full.detach())
        self._n_factor_total += self.cache.n_factor

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable ``u = H^{-1} z`` w.r.t. ``z`` only (``c`` frozen)."""
        return _HelmholtzInvFn.apply(z, self.cache)

    def apply_diff_c(self, z: torch.Tensor, c_full: torch.Tensor) -> torch.Tensor:
        """``u = H(c)^{-1} z`` with gradients w.r.t. ``c_full`` (``z`` frozen).

        Re-factorises at the current ``c``. Intended for the c-stage; do not
        detach ``u`` before the data loss.
        """
        return _HelmholtzInvDiffCFn.apply(z, c_full, self)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """``z = H u`` (exact sparse matvec, no grad) — seed ``z`` from ``u``."""
        with torch.no_grad():
            return self.cache.matvec(u)

    def diagnostics(self) -> dict:
        return {
            "n_factor": self._n_factor_total,
            "n_forward_solves": self._n_forward_total + self.cache.n_forward_solves,
            "n_adjoint_solves": self._n_adjoint_total + self.cache.n_adjoint_solves,
            "n_linear_solves": (
                self._n_forward_total
                + self._n_adjoint_total
                + self.cache.n_forward_solves
                + self.cache.n_adjoint_solves
            ),
            "nf": self.cache.nf,
            "n_dofs": self.cache.n,
        }

