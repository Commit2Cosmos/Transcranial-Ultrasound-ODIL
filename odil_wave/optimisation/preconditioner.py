"""Exact leapfrog preconditioner for the wavefield block.

``TimeStepPreconditioner`` implements the exact inverse of the reduced PDE
block ``A`` by one sequential forward-substitution (leapfrog) sweep.
``TimeStepUTransform`` wraps it as a differentiable coordinate change
``u = A^{-1} z`` so that L-BFGS on ``z`` sees a well-conditioned system.

Public API
----------
TimeStepPreconditioner   -- exact A^{-1} and A^{-T} by leapfrog sweep
TimeStepUTransform       -- differentiable u = A^{-1} z coordinate change
u_block_ops              -- returns {"matvec": callable} for A u (seeding)
"""

from __future__ import annotations
from typing import Callable, Dict

import torch


# Helper: differentiable linear map via custom autograd Function

class _LinearMapFn(torch.autograd.Function):
    """Wrap a linear map ``M`` so that autograd sees ``forward = M`` and
    ``backward = M^T``.  Avoids storing the full computation graph of the
    sequential sweep, keeping memory O(NT * NX * NY) instead of O(NT^2 …).
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, M: "TimeStepPreconditioner") -> torch.Tensor:
        ctx.M = M
        with torch.no_grad():
            return M.apply(z)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # dL/dz = (du/dz)^T dL/du = (A^{-1})^T dL/du = A^{-T} dL/du
        return ctx.M.apply_T(grad_out), None


# Vectorised matvec for the reduced PDE block A

def u_block_ops(loss, c_full: torch.Tensor) -> Dict[str, Callable]:
    """Return matrix-free operators for the reduced PDE block ``A``.

    ``A`` is the block lower-triangular leapfrog operator on the inner time
    rows ``(S, NT-2, Nx, Ny)``.  From the forward-substitution recurrence
    ``denom * x_{j+1} = v_j + B0*x_j + k*L(x_j) [+ ot4] - Cc*x_{j-1}``
    the row-``j`` action of ``A`` on a vector ``x`` is::

        (A x)_j = denom*x_j - B0*x_{j-1} - k*L(x_{j-1}) [- ot4] + Cc*x_{j-2}

    with ``x_{-1} = x_{-2} = 0`` (zero ICs of the reduced block).  This is
    a banded-in-time operation that can be evaluated in a single pass with
    no sequential dependency.

    Returns
    -------
    dict with key ``"matvec"``: a callable ``(S, n, Nx, Ny) -> (S, n, Nx, Ny)``.
    """
    wave_eq = loss.config.wave_eq
    g = wave_eq.wavefield.grid
    dt = g.dt_nd
    c0 = g.c0
    lap = wave_eq._lap
    ot4 = wave_eq.ot4
    w = wave_eq.pml_weight

    sig_s = w * (g.sigma_x_nd + g.sigma_y_nd)
    sig_p = w * (g.sigma_x_nd * g.sigma_y_nd)
    denom = 1.0 / dt**2 + sig_s / (2.0 * dt)   # (Nx, Ny)
    Cc    = 1.0 / dt**2 - sig_s / (2.0 * dt)
    B0    = 2.0 / dt**2 - sig_p
    k = (c_full.detach().to(dtype=g.dtype, device=g.device) / c0) ** 2
    gam = dt**2 / 12.0

    def matvec(u: torch.Tensor) -> torch.Tensor:
        """Vectorised ``A u``: ``(S, n, Nx, Ny) -> (S, n, Nx, Ny)``."""
        z0 = torch.zeros_like(u[:, :1])
        u_m1 = torch.cat([z0,       u[:, :-1]],  dim=1)  # u_{j-1}
        u_m2 = torch.cat([z0, z0,   u[:, :-2]],  dim=1)  # u_{j-2}
        lap_m1 = lap.apply(u_m1)
        Au = denom * u - B0 * u_m1 - k * lap_m1 + Cc * u_m2
        if ot4:
            Au = Au - gam * k * lap.apply(k * lap_m1)
        return Au

    return {"matvec": matvec}


# Exact preconditioner: A^{-1} by forward substitution

class TimeStepPreconditioner:
    """Exact ``M = A^{-1}`` by forward substitution (a leapfrog sweep).

    The reduced PDE block ``A`` (equation rows ``1..nt-2``, unknown rows
    ``2..nt-1``, zero ICs) is block *lower-triangular* in time: row ``t``
    couples ``x_{t+1}`` through the pointwise-invertible coefficient
    ``1/dt'^2 + sigma_s/(2 dt')`` and ``x_t, x_{t-1}`` through the leapfrog
    stencil. One forward-substitution sweep -- the existing leapfrog kernel
    driven by the residual as a source -- therefore applies ``A^{-1}``
    *exactly*: arbitrary velocity contrast, true reflect boundaries, PML and
    OT4 rows included. ``A^{-T}`` is the reverse-time sweep, obtained here by
    autograd through the forward loop (exact transpose of every patch).

    As a split preconditioner ``P = M M^T = (A^T A)^{-1}`` the PDE block of
    the normal equations becomes the identity: ``M^T H M = a I +
    B (S A^{-1})^T (S A^{-1})``, i.e. identity plus a positive semi-definite
    data term -- clustered spectrum, contrast-independent. The price is ``nt``
    *sequential* steps per application (no parallel-in-time), so cost scales
    linearly with ``nt`` where the ParaDiag FFT round is log-parallel.
    """

    def __init__(self, wave_eq, c_full: torch.Tensor) -> None:
        self.wave_eq = wave_eq
        g = wave_eq.wavefield.grid
        self._dt = g.dt_nd
        self._c0 = g.c0
        self._ot4 = wave_eq.ot4
        self._lap = wave_eq._lap
        w = wave_eq.pml_weight
        self._sig_s = w * (g.sigma_x_nd + g.sigma_y_nd)
        self._sig_p = w * (g.sigma_x_nd * g.sigma_y_nd)
        self._denom = 1.0 / self._dt**2 + self._sig_s / (2.0 * self._dt)
        self._Cc = 1.0 / self._dt**2 - self._sig_s / (2.0 * self._dt)
        self._B0 = 2.0 / self._dt**2 - self._sig_p
        self.rebuild(c_full)

    def rebuild(self, c_full: torch.Tensor) -> None:
        """Refresh the medium (cheap, elementwise). Call per outer c-update."""
        g = self.wave_eq.wavefield.grid
        self._k = (c_full.detach().to(dtype=g.dtype, device=g.device)
                   / self._c0) ** 2

    def _sweep_with_k(self, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Forward substitution ``A(k) x = v`` for ``v`` of shape ``(S, n, Nx, Ny)``.

        ``k`` may require grad (used by :meth:`apply_diff_c`). Row ``t = j+1``::

            denom x_{t+1} = v_j + (2/dt'^2 - sig_p) x_t + k L x_t
                          + (dt^2/12) k L (k L x_t) - (1/dt'^2 - sig_s/2dt') x_{t-1}
        """
        n = v.shape[1]
        gam = self._dt**2 / 12.0
        x_t = torch.zeros_like(v[:, 0])
        x_tm = torch.zeros_like(v[:, 0])
        xs = []
        for j in range(n):
            lap = self._lap.apply(x_t)
            rhs = v[:, j] + self._B0 * x_t + k * lap - self._Cc * x_tm
            if self._ot4:
                rhs = rhs + gam * k * self._lap.apply(k * lap)
            x_new = rhs / self._denom
            xs.append(x_new)
            x_tm, x_t = x_t, x_new
        return torch.stack(xs, dim=1)

    def _sweep(self, v: torch.Tensor) -> torch.Tensor:
        """Forward substitution with the cached (detached) medium ``self._k``."""
        return self._sweep_with_k(v, self._k)

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """Exact ``A^{-1} v`` (one sequential leapfrog sweep, no grad)."""
        with torch.no_grad():
            return self._sweep(v)

    def apply_diff_c(self, z: torch.Tensor, c_full: torch.Tensor) -> torch.Tensor:
        """``u = A(c)^{-1} z`` with gradients flowing into ``c_full``.

        Used for the ``frozen_z`` c-step where ``z`` is held fixed (typically
        detached) and L-BFGS updates ``c``.  Does **not** use
        :class:`_LinearMapFn` — that map only backprops through ``z``.
        """
        g = self.wave_eq.wavefield.grid
        k = (c_full.to(dtype=g.dtype, device=g.device) / self._c0) ** 2
        return self._sweep_with_k(z, k)

    def apply_T(self, v: torch.Tensor) -> torch.Tensor:
        """Exact ``A^{-T} v``: autograd through the (linear) forward sweep."""
        x = torch.zeros_like(v)
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self._sweep(x)
            (g,) = torch.autograd.grad((y * v).sum(), x)
        return g

    def apply_T_then_apply(self, v: torch.Tensor) -> torch.Tensor:
        """SPD action ``P v = A^{-1} A^{-T} v = (A^T A)^{-1} v`` (exact)."""
        return self.apply(self.apply_T(v))

# Coordinate-change transform: u = A^{-1} z

class TimeStepUTransform:
    """Exact inverse-operator metric for the wavefield block: ``u = A^{-1} z``.

    ``M`` is :class:`TimeStepPreconditioner` -- the exact inverse of the
    reduced PDE block by sequential forward substitution (one leapfrog
    sweep; arbitrary velocity contrast, PML and OT4 rows included). L-BFGS
    on ``z`` therefore sees::

        M^T H_u M = a I + b (S A^{-1})^T (S A^{-1})

    *exactly*: the conditioning floor for any ``u = M z`` metric built from
    an approximation of ``A^{-1}`` (its smallest eigenvalue is exactly
    ``a``, the PDE normaliser, since the data term is rank-deficient). The
    price is ``nt`` *sequential* time steps per closure forward plus a
    reverse-mode sweep per backward -- no parallel-in-time, against
    ParaDiag's log-parallel FFT rounds.

    Seeding is exact and cheap: ``z0 = M^{-1} u0 = A u0`` is one residual
    evaluation. Like the ParaDiag transform this is a fixed coordinate
    change, valid across closed-form c-updates, but built for one medium --
    rebuild after a large c change.
    """

    def __init__(self, loss, c_full: torch.Tensor) -> None:
        c_full = c_full.detach()
        self.M = TimeStepPreconditioner(loss.config.wave_eq, c_full)
        self._A = u_block_ops(loss, c_full)["matvec"]

    def rebuild(self, loss, c_full: torch.Tensor) -> None:
        """Rebuild for a new velocity after a c-update."""
        c_full = c_full.detach()
        self.M.rebuild(c_full)
        self._A = u_block_ops(loss, c_full)["matvec"]

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable ``u = A^{-1} z`` w.r.t. ``z`` only (``c`` frozen in ``M``).

        Uses :class:`_LinearMapFn` so the sequential sweep is not stored in the
        autograd graph — correct and cheap for the z-step.
        """
        return _LinearMapFn.apply(z, self.M)

    def apply_diff_c(self, z: torch.Tensor, c_full: torch.Tensor) -> torch.Tensor:
        """``u = A(c)^{-1} z`` with gradients w.r.t. ``c_full`` (for the c-step).

        ``z`` should be detached.  Recomputes the leapfrog sweep with a live
        ``k = (c/c0)^2`` so L-BFGS trial ``c`` values see a valid closure.
        """
        return self.M.apply_diff_c(z, c_full)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """``z = A u`` (exact, no grad) -- used to seed ``z`` from a ``u`` guess."""
        with torch.no_grad():
            return self._A(u)