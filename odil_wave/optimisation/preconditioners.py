"""Block preconditioners of the joint solve: one object per option of each block.

Each object owns its block's change of variables, i.e. the map from the
optimised variable L-BFGS updates to the block's own unknown and back, plus
whatever that map needs (curvature, diagnostics). "No preconditioner" is an
object too, so :class:`~odil_wave.optimisation.JointFreqODIL` never branches on
which one is on.

Wavefield block (``WAVEFIELD_PRECONDITIONERS``), mapping ``v <-> u_tilde``
(``u = u_ref * u_tilde`` stays in the solver):

* :class:`NoWavefieldPreconditioner` (``"none"``): ``u_tilde = v``.
* :class:`InitialModelWavefieldPreconditioner` (``"initial_model"``):

      u_tilde[s, w] = A(c_init, w)^-1 (|lambda_tt| v)[s, w],
      v0 = A(c_init) u_tilde0 / |lambda_tt|,

  ``c_init`` the band's initial model and ``A`` the operator of
  ``WaveEquation.residual`` (``HelmholtzSolver.assemble_H_sparse``). The
  ``|lambda_tt|`` (``pde_scale / u_ref``) puts ``v`` in the PDE term's own
  scale (ADR-0002); without it the curvature in ``v`` is ``1 / lambda_tt^2``
  smaller and float32 can't resolve L-BFGS's first step on the default
  problem. For a Helmholtz warm start ``v0 = f / pde_scale``. The solve reuses
  the warm start's ``splu`` factors (nothing is factorised here) with all
  shots batched per frequency, on the CPU, and its backward is the adjoint
  solve. At ``c = c_init`` the wavefield block's PDE term is then the identity
  up to one scalar; with the model frozen at the true model its iteration
  count stayed flat across resolutions (#96).

Both give ``pde_curvature()``, the diagonal of the weighted PDE term in their
optimised variable: the column norms of ``A`` at the initial model (through
the latent) without a preconditioner, the uniform ``pde_weight / (N_int n_s
n_f)`` with it (the scaled residual at ``c_init`` is ``v - f / pde_scale``).

Model block (``MODEL_PRECONDITIONERS``), mapping ``y <-> z_m``:

* :class:`NoModelPreconditioner` (``"none"``): ``z_m = y``.
* :class:`PdeDiagonalModelPreconditioner` (``"pde_diagonal"``):
  ``z_m = s * y``, ``s`` from :func:`preconditioner_scale` with the target the
  median of the wavefield preconditioner's ``pde_curvature()`` (see the
  solver's module docstring for ``H_zz``).
"""

from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from odil_wave.operator import WaveEquation

WAVEFIELD_PRECONDITIONERS = ("none", "initial_model")
MODEL_PRECONDITIONERS = ("none", "pde_diagonal")


def abs_sq(x: torch.Tensor) -> torch.Tensor:
    """Elementwise ``|x|^2`` (complex-aware, no sqrt)."""
    if x.is_complex():
        return x.real.square() + x.imag.square()
    return x.square()


def _real_parts(x: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    return x.real.contiguous().to(dtype=dtype), x.imag.contiguous().to(dtype=dtype)


# --------------------------------------------------------------------------- #
# Factor solve
# --------------------------------------------------------------------------- #
class _FactorSolveFn(torch.autograd.Function):
    """``A^-1 v`` forward, ``A^-H g`` backward (the adjoint of a linear map)."""

    @staticmethod
    def forward(ctx, v, solve):
        ctx.solve = solve
        return solve.apply_factors(v, "N")

    @staticmethod
    def backward(ctx, grad):
        return ctx.solve.apply_factors(grad, "H"), None


class FactorSolve:
    """Differentiable ``u = A^-1 v`` from fixed per-frequency ``splu`` factors.

    ``v`` is ``(n_shots, nf, Nx, Ny)`` complex; frequency ``k`` uses
    ``factors[k]`` with all shots batched as right-hand sides. The solves run
    on the CPU in SciPy (copied from ``v``'s device and back), so every device
    works. ``seconds`` accumulates the wall time of all forward and adjoint
    triangular solves.
    """

    def __init__(self, factors: Sequence) -> None:
        self.factors = list(factors)
        self.seconds = 0.0

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return _FactorSolveFn.apply(v, self)

    def apply_factors(self, v: torch.Tensor, trans: str) -> torch.Tensor:
        """``A^-1 v`` (``trans="N"``) or ``A^-H v`` (``trans="H"``), no autograd."""
        n_shots, nf, nx, ny = v.shape
        if nf != len(self.factors):
            raise ValueError(f"{len(self.factors)} factors for {nf} frequencies")
        t0 = time.perf_counter()
        x = v.detach().cpu().numpy()
        out = np.empty_like(x)
        for k, lu in enumerate(self.factors):
            rhs = np.ascontiguousarray(x[:, k].reshape(n_shots, nx * ny).T)
            out[:, k] = lu.solve(rhs, trans=trans).T.reshape(n_shots, nx, ny)
        u = torch.from_numpy(out).to(device=v.device)
        self.seconds += time.perf_counter() - t0
        return u


# --------------------------------------------------------------------------- #
# Wavefield block
# --------------------------------------------------------------------------- #
class NoWavefieldPreconditioner:
    """``u_tilde = v``: L-BFGS optimises the scaled wavefield itself.

    ``c_init`` is the full-grid initial model through the band's initial
    latent (the solver's ``c_full_of_z(z0)``, the model L-BFGS starts from),
    ``pde_scale_per_u`` the ``(n_shots, nf, 1, 1)`` tensor ``pde_scale /
    u_ref`` and ``term_weight`` the PDE term's weight per scaled residual
    entry, ``pde_weight / (N_int n_s n_f)``; they set :meth:`pde_curvature`.
    """

    name = "none"

    def __init__(
        self,
        wave_eq: WaveEquation,
        c_init: torch.Tensor,
        pde_scale_per_u: torch.Tensor,
        term_weight: float,
    ) -> None:
        self.wave_eq = wave_eq
        self.c_init = c_init
        self.pde_scale_per_u = pde_scale_per_u
        self.term_weight = float(term_weight)
        self.solve_seconds = 0.0

    def u_tilde_of_v(self, v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
        return torch.complex(v_re, v_im)

    def v_of_u_tilde(self, u_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _real_parts(u_tilde, self.pde_scale_per_u.dtype)

    def pde_curvature(self) -> torch.Tensor:
        """Diagonal of the weighted PDE term w.r.t. ``v_re`` (equal for ``v_im``).

        Shape ``(n_shots, nf, Nx, Ny)``, PML included, at ``c_init``. ``r`` is
        affine in ``u``, so this is ``term_weight * sum_i |A_ij|^2 /
        pde_scale_per_u^2`` with the column norms of ``A`` taken exactly from
        the solver's own operator: combs of unit impulses spaced past the
        stencil's footprint (half-width ``space_order // 2`` per axis, reflect
        boundaries included), each summed over its own box.
        """
        nf, nx, ny = self.pde_scale_per_u.shape[1], *self.c_init.shape
        dtype, device = self.pde_scale_per_u.dtype, self.pde_scale_per_u.device
        cdtype = self.wave_eq.wavefield.cdtype
        half = self.wave_eq.space_order // 2
        stride = 2 * half + 1
        shape = (1, nf, nx, ny)
        no_src = torch.zeros(shape, dtype=cdtype, device=device)
        box = torch.ones(1, 1, stride, stride, dtype=dtype, device=device)
        col_sq = torch.zeros(shape[1:], dtype=dtype, device=device)
        with torch.no_grad():
            for a in range(stride):
                for b in range(stride):
                    comb = torch.zeros(shape, dtype=cdtype, device=device)
                    comb[..., a::stride, b::stride] = 1.0
                    Ae_sq = abs_sq(self.wave_eq.residual(comb, self.c_init, no_src))
                    boxed = torch.nn.functional.conv2d(
                        Ae_sq.view(nf, 1, nx, ny), box, padding=half
                    ).view(shape[1:])
                    col_sq[:, a::stride, b::stride] = boxed[:, a::stride, b::stride]
        return self.term_weight * col_sq / self.pde_scale_per_u.square()

    def info(self) -> Dict[str, object]:
        return {"wavefield_preconditioner": self.name}


class InitialModelWavefieldPreconditioner:
    """``u_tilde = A(c_init)^-1 (pde_scale_per_u * v)`` (see the module doc).

    ``c_init`` is the full-grid initial model as given (the solver's
    ``vm_in.c``, not through the latent: the model ``factors`` were built at),
    which the operator ``wave_eq`` is applied at for the inverse map,
    ``pde_scale_per_u`` the ``(n_shots, nf, 1, 1)`` tensor ``pde_scale /
    u_ref``, ``term_weight`` the PDE term's weight per scaled residual entry,
    ``pde_weight / (N_int n_s n_f)``, and ``factors`` the per-frequency
    ``splu`` factors of ``A(c_init)`` (the warm start's).
    """

    name = "initial_model"

    def __init__(
        self,
        wave_eq: WaveEquation,
        c_init: torch.Tensor,
        pde_scale_per_u: torch.Tensor,
        term_weight: float,
        factors: Sequence,
    ) -> None:
        self.wave_eq = wave_eq
        self.c_init = c_init
        self.pde_scale_per_u = pde_scale_per_u
        self.term_weight = float(term_weight)
        self.solve = FactorSolve(factors)

    @property
    def solve_seconds(self) -> float:
        """Wall time of the triangular solves so far (forward and adjoint)."""
        return self.solve.seconds

    def u_tilde_of_v(self, v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
        return self.solve(torch.complex(v_re, v_im) * self.pde_scale_per_u)

    def v_of_u_tilde(self, u_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            v = (
                self.wave_eq.residual(u_tilde, self.c_init, torch.zeros_like(u_tilde))
                / self.pde_scale_per_u
            )
        return _real_parts(v, self.pde_scale_per_u.dtype)

    def pde_curvature(self) -> torch.Tensor:
        """Diagonal of the weighted PDE term w.r.t. ``v_re`` at ``c = c_init``:
        uniform ``term_weight``, shape ``(n_shots, nf, Nx, Ny)``."""
        n_shots, nf = self.pde_scale_per_u.shape[:2]
        return torch.full(
            (n_shots, nf, *self.c_init.shape),
            self.term_weight,
            dtype=self.pde_scale_per_u.dtype,
            device=self.pde_scale_per_u.device,
        )

    def info(self) -> Dict[str, object]:
        return {"wavefield_preconditioner": self.name}


# --------------------------------------------------------------------------- #
# Model block
# --------------------------------------------------------------------------- #
def preconditioner_scale(
    H_zz: torch.Tensor, target: float, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cell scale ``s = sqrt(target / max(H_zz, eps * median(H_zz)))``.

    Every unfloored cell gets preconditioned curvature ``s**2 * H_zz == target``;
    the floor keeps ``s`` finite where the curvature vanishes and bounds
    ``max(s) / median(s)`` by ``1 / sqrt(eps)``. Returns ``(s, floored)``, the
    second a boolean mask of the floored cells.
    """
    floor = eps * H_zz.median()
    if not float(floor) > 0.0:
        raise ValueError(
            "model preconditioner needs a positive median model-block curvature "
            "(is the PDE weight zero?)"
        )
    floored = H_zz < floor
    return (target / H_zz.clamp_min(floor)).sqrt(), floored


class NoModelPreconditioner:
    """``z_m = y``: L-BFGS optimises the model latent itself."""

    name = "none"

    def z_of_y(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def y_of_z(self, z_m: torch.Tensor) -> torch.Tensor:
        return z_m

    def info(self) -> Dict[str, object]:
        return {"model_preconditioner": self.name}


class PdeDiagonalModelPreconditioner:
    """``z_m = scale * y``, ``scale`` putting the model block's PDE curvature
    ``H_zz`` at ``target`` (:func:`preconditioner_scale`, floor ``eps``)."""

    name = "pde_diagonal"

    def __init__(self, H_zz: torch.Tensor, target: float, eps: float) -> None:
        self.scale, floored = preconditioner_scale(H_zz, target, eps)
        s = self.scale
        self._info = {
            "model_preconditioner": self.name,
            "precond_eps": eps,
            "precond_target": target,
            "h_zz_min": float(H_zz.min()),
            "h_zz_median": float(H_zz.median()),
            "h_zz_max": float(H_zz.max()),
            "model_scale_min": float(s.min()),
            "model_scale_median": float(s.median()),
            "model_scale_max": float(s.max()),
            "precond_floored_frac": float(floored.double().mean()),
        }

    def z_of_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.scale * y

    def y_of_z(self, z_m: torch.Tensor) -> torch.Tensor:
        return z_m / self.scale

    def info(self) -> Dict[str, object]:
        return dict(self._info)
