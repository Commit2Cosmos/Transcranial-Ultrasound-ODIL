"""Pure joint full-space frequency-domain ODIL.

A single joint optimiser over **both** the complex wavefield block ``u`` (all
sources and frequencies) and a squared-slowness model latent ``z_m`` — no
alternating updates, no closed-form / reduced-FWI / WRI / augmented-Lagrangian
c-block, no multigrid, no adaptive loss balancing. It exists to test whether a
properly scaled, verified full-space joint formulation recovers ``m``/``c`` from
receiver-trace data, and to diagnose precisely why it does or does not.

Formulation
-----------
* Model latent ``z_m`` on interior cells, bounded squared slowness

      m(z_m) = m_min + (m_max - m_min) * sigmoid(z_scale * z_m),
      m_min = 1 / c_max**2 ,   m_max = 1 / c_min**2 ,   c = 1 / sqrt(m),

  with ``c_min`` / ``c_max`` the grid's velocity bounds.

  ``c_full = velocity_model.build_full_c(c)`` freezes PML cells at ``pml_c`` so
  only interior cells are optimised. Bounds are enforced *structurally* by the
  sigmoid — every L-BFGS line-search trial stays in-bounds (no post-step
  projection).

* Scaled wavefield ``u = u_ref * u_tilde`` with a fixed per-(shot, frequency)
  scalar ``u_ref[s, w] = max(RMS_active|u_init[s, w]|, eps_u)``. ``u_tilde`` is
  two real tensors (Re, Im).

* Residuals use the repository's exact operator (inverse-crime consistent):

      r_pde[s, w]  = WaveEquation.residual(u, c_full, q)[s, w]     (full grid)
      r_data[s, w] = P_s u[s, w] - d_obs[s, w]                     (receivers)

  where ``P_s`` samples the ``n_transducers - 1`` transducers recording shot
  ``s`` (never its own), so ``data_scale`` below excludes the self-trace too.

  fixed scales ``pde_scale[s, w] = max(|lambda_tt[w]| * u_ref[s, w], eps_p)``
  (the wave term's magnitude, ADR-0002: independent of the source
  normalisation, grid spacing and PML width; NOT the near-zero warm-start PDE
  residual) and ``data_scale[s, w] = max(RMS|d_obs[s, w]|, eps_d)``.

* Objective:

      Phi = pde_weight  * sum|r_pde / pde_scale|^2 / (2 N_int n_s n_f)
          + data_weight * mean|r_data / data_scale|^2
          + w_reg  * R(c_interior)

  The PDE sum runs over interior **and** PML nodes but is normalised by the
  interior node count ``N_int``, so a thicker PML does not dilute it; the data
  term is a mean over the count of **real** scalar entries (``2x`` complex).

  with ``pde_weight``, ``data_weight`` and ``w_reg = reg_weight`` fixed run-level
  hyperparameters (never adapted). The run layer fills all three from
  ``loss.weights.{pde,data,reg}`` (defaults 1 / 1 / 0); the regulariser is
  applied only when ``reg_weight != 0``. Each term is the per-(shot, freq)
  scaled relative residual above, not the raw ``InverseLoss`` mean.

* One persistent ``torch.optim.LBFGS([v_re, v_im, z_m])`` (strong-Wolfe) — a
  single joint optimiser over the full space, no alternation. It is advanced by
  ``n_iter`` outer ``.step`` calls, each running up to ``inner_max_iter`` L-BFGS
  iterations on the shared history (total ~= ``n_iter * inner_max_iter``), so the
  per-step trajectory is logged while the solve stays continuous. (A single
  iteration per step, ``inner_max_iter=1``, stalls the strong-Wolfe line search
  and is *not* representative of L-BFGS on this problem.) The objective is
  matrix-free (one conv2d Laplacian per evaluation); ``A(c)`` is never factorised
  inside optimisation.

* Block preconditioners (:mod:`odil_wave.optimisation.preconditioners`). Each
  block's optimised variable is mapped to its own unknown by one object, held
  as ``wavefield_precond`` (``v <-> u_tilde``; ``u = u_ref * u_tilde`` stays
  here) and ``model_precond`` (``y <-> z_m``); without a preconditioner the
  object is the identity map. ``wavefield_preconditioner="initial_model"``
  (default ``"none"``) optimises ``v`` with ``u_tilde = A(c_init)^-1
  (|lambda_tt| v)`` from the warm start's ``splu`` factors
  (``initial_model_factors``); see that module for the map and its curvature.

* Optional model-block preconditioner (``model_preconditioner="pde_diagonal"``;
  default ``"none"``). L-BFGS models the Hessian outside its history as one
  ``gamma * I`` over both blocks, so the model block is rescaled per cell to the
  wavefield block's curvature: L-BFGS optimises ``y`` with ``z_m = s * y``,

      H_kk[i] = pde_weight * sum_{s,f} |lap(u)[s,f,i] / pde_scale[s,f]|^2
                / (N_int n_s n_f)                         (r is linear in k)
      H_zz[i] = (dk/dz_m[i])^2 * H_kk[i]                  (exact where r = 0)
      T       = median of the weighted PDE term's diagonal over v_re (all
                shots, frequencies and nodes, PML included)
      s[i]    = sqrt(T / max(H_zz[i], eps_precond * median(H_zz)))

  with ``k = (c / c0)^2``, evaluated once per band at the warm start (never
  refreshed, which would invalidate the L-BFGS history). The data term and the
  regulariser are left out of both curvatures. ``z_scale`` cancels from
  ``z_scale * s``. ``H_zz`` comes from :meth:`JointFreqODIL.model_curvature`;
  ``T`` is the median of ``wavefield_precond.pde_curvature()``, the curvature
  in the variable L-BFGS optimises: the column norms of ``A`` over
  ``|lambda_tt|`` without the wavefield preconditioner, the uniform
  ``pde_weight / (N_int n_s n_f)`` with it. ``T``, and so ``s``, then
  differs by about the size of ``A``'s column norms over ``|lambda_tt|``.

The class is pure library code (no file IO). Per-iteration diagnostics are
exposed on :attr:`diag_rows` and mirrored into the loss callback history; the
``sandbox/joint_odil`` drivers persist the full §10 artifact set.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from odil_wave.loss import InverseLoss
from odil_wave.loss.utils import LossTape
from odil_wave.models import VelocityModel
from odil_wave.wavefield import Wavefield

from .base import Optimiser
from .preconditioners import (
    InitialModelWavefieldPreconditioner,
    MODEL_PRECONDITIONERS,
    NoModelPreconditioner,
    NoWavefieldPreconditioner,
    PdeDiagonalModelPreconditioner,
    WAVEFIELD_PRECONDITIONERS,
    abs_sq,
)


# --------------------------------------------------------------------------- #
# Real-scalar reductions and complex real inner product
# --------------------------------------------------------------------------- #
def real_scalar_count(x: torch.Tensor) -> int:
    """Number of real scalar entries in ``x`` (2x for complex)."""
    return x.numel() * (2 if x.is_complex() else 1)


def mean_real_sq(x: torch.Tensor) -> torch.Tensor:
    """Mean of ``|x|^2`` over the count of **real** scalar entries.

    For complex ``x`` this divides ``sum(re^2 + im^2)`` by ``2 * numel`` — the
    real degrees of freedom (real + imaginary), as required by the task's
    objective averaging. Equals ``0.5 * mean(|x|^2)`` for complex ``x``.
    """
    if x.is_complex():
        ssq = x.real.square().sum() + x.imag.square().sum()
    else:
        ssq = x.square().sum()
    return ssq / real_scalar_count(x)


def rms_abs(x: torch.Tensor, dim=None) -> torch.Tensor:
    """RMS of ``|x|`` = sqrt(mean(|x|^2)) over ``dim`` (complex-aware)."""
    v = abs_sq(x)
    return v.mean(dim=dim).sqrt() if dim is not None else v.mean().sqrt()


# --------------------------------------------------------------------------- #
# Latent squared-slowness map
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SlownessLatent:
    """Bounded squared-slowness reparameterisation ``m = m(z_m)``.

    ``m(z) = m_min + (m_max - m_min) * sigmoid(z_scale * z)`` with velocity
    bounds converted as ``m_min = 1/c_max**2``, ``m_max = 1/c_min**2``.
    ``z_scale`` is a fixed model-coordinate scale (default 1). The inverse
    (logit) is used once for initialisation, clipped away from 0/1.
    """

    m_min: float
    m_max: float
    z_scale: float = 1.0
    logit_clip: float = 1e-6

    def m_of_z(self, z: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.z_scale * z)
        return self.m_min + (self.m_max - self.m_min) * s

    def c_of_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.m_of_z(z).clamp_min(1e-30).rsqrt()

    def sigmoid_deriv(self, z: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.z_scale * z)
        return s * (1.0 - s)

    def dm_dz(self, z: torch.Tensor) -> torch.Tensor:
        """Elementwise derivative of ``m_of_z``."""
        return (self.m_max - self.m_min) * self.z_scale * self.sigmoid_deriv(z)

    def z_of_m(self, m: torch.Tensor) -> torch.Tensor:
        """Stable inverse (logit) so that ``m(z_of_m(m)) == m`` (clipped)."""
        frac = (m - self.m_min) / (self.m_max - self.m_min)
        frac = frac.clamp(self.logit_clip, 1.0 - self.logit_clip)
        return torch.log(frac / (1.0 - frac)) / self.z_scale

    def z_of_c(self, c: torch.Tensor) -> torch.Tensor:
        return self.z_of_m(c.clamp_min(1e-30) ** -2)


@dataclass(frozen=True)
class Iterate:
    """The joint solve at one logged point: what ``minimise(on_log=...)`` gets.

    Logged points are the start (``iter == 0``, the warm start before any
    update), every ``log_every``-th outer iteration and the final iterate.
    The objective terms come from one evaluation at this point, so
    ``pde_term + data_term + reg_term == loss``; ``reg_term`` is
    ``w_reg * R(c_interior)``, ``0.0`` without a regulariser.
    """

    iter: int  # outer iterations done in this solve
    n_evaluations: int  # L-BFGS closures so far in this solve
    loss: float
    pde_term: float
    data_term: float
    reg_term: float
    pde_rel_residual: float  # RMS(r_pde / pde_scale), interior + PML over N_int
    rel_data_residual: float  # ||d_syn - d_obs|| / ||d_obs||
    grad_u_norm: float
    grad_z_norm: float
    sat_frac: float  # cells whose sigmoid is within 1e-3 of 0 or 1
    c_full: torch.Tensor  # full-grid velocity, detached
    u: torch.Tensor  # (n_shots, nf, Nx, Ny) complex wavefield, detached


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class JointFreqODIL(Optimiser):
    """Pure joint full-space frequency-domain ODIL solver.

    Every parameter the run config sets (the ``optimiser`` block, the
    ``loss.weights`` and ``runtime.verbose``) is a required keyword (ADR 0001);
    the numerical-method details (``eps_*``, ``z_scale``, ``logit_clip``) keep
    library defaults. See the module docstring for the formulation. ``loss``
    must be an :class:`InverseLoss` (the operator, geometry, sources and
    observations are read from it, so the PDE / receiver operators are
    identical to truth-data generation). ``u_init`` (per-shot warm-start
    wavefields, normally a Helmholtz solve on the current model) is required:
    it seeds ``u`` and sets ``u_ref``. ``initial_model_factors`` (the
    per-frequency ``splu`` factors of ``A`` on the wavefield's velocity model,
    from ``HelmholtzSolver.solve(return_factors=True)``) is required by
    ``wavefield_preconditioner="initial_model"``.

    ``init_params``, ``objective``, ``residuals`` and ``physical_u`` take the
    optimised wavefield variable ``(v_re, v_im)``, as do the ``u_re`` /
    ``u_im`` entries of ``final_state``; the logged ``Iterate.u`` and the
    returned wavefields are the physical ``u``.
    """

    def __init__(
        self,
        wavefield: Wavefield,
        loss: InverseLoss,
        *,
        u_init: Sequence[Wavefield] | torch.Tensor,
        n_iter: int,
        inner_max_iter: int,
        history_size: int,
        line_search_fn: Optional[str],
        tolerance_grad: float,
        tolerance_change: float,
        lbfgs_lr: float,
        pde_weight: float,
        data_weight: float,
        reg_weight: float,
        log_every: int,
        verbose: bool,
        model_preconditioner: str,
        wavefield_preconditioner: str,
        initial_model_factors: Optional[Sequence] = None,
        # Fixed-scale controls.
        eps_u: float = 1e-30,
        eps_data: float = 1e-30,
        eps_pde: float = 1e-30,
        z_scale: float = 1.0,
        logit_clip: float = 1e-6,
        eps_grad: float = 1e-30,
        eps_precond: float = 1e-2,
    ) -> None:
        super().__init__(wavefield, loss)
        if not isinstance(loss, InverseLoss):
            raise TypeError("JointFreqODIL requires an InverseLoss")
        if u_init is None:
            # A u = 0 seed collapses u_ref to eps_u and leaves an exactly zero
            # model gradient (the PDE residual is -q, independent of c).
            raise ValueError(
                "JointFreqODIL requires a warm-start wavefield u_init "
                "(e.g. a Helmholtz solve on the current model)"
            )
        if model_preconditioner not in MODEL_PRECONDITIONERS:
            raise ValueError(
                f"model_preconditioner must be one of {list(MODEL_PRECONDITIONERS)}, "
                f"got {model_preconditioner!r}"
            )
        if wavefield_preconditioner not in WAVEFIELD_PRECONDITIONERS:
            raise ValueError(
                "wavefield_preconditioner must be one of "
                f"{list(WAVEFIELD_PRECONDITIONERS)}, got {wavefield_preconditioner!r}"
            )
        if (
            wavefield_preconditioner == "initial_model"
            and initial_model_factors is None
        ):
            raise ValueError(
                "wavefield_preconditioner 'initial_model' needs "
                "initial_model_factors (the warm-start HelmholtzSolver's splu "
                "factors, solve(return_factors=True))"
            )

        grid = wavefield.grid
        self.grid = grid
        self.dtype = grid.dtype
        self.cdtype = wavefield.cdtype
        self.device = grid.device
        self.wave_eq = loss.config.wave_eq
        self.geometry = loss.config.geometry
        self.n_shots = self.geometry.n_shots
        self.nf = wavefield.n_frequencies
        self.Nx, self.Ny = grid.shape
        self.c0 = float(grid.c0)

        self.u_init = u_init
        self.n_iter = int(n_iter)
        self.inner_max_iter = int(inner_max_iter)
        self.history_size = int(history_size)
        self.line_search_fn = line_search_fn
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        self.lbfgs_lr = float(lbfgs_lr)
        self.data_weight = float(data_weight)
        # Run-level PDE-penalty weight; never adapted from the instantaneous
        # losses. lambda_data (== data_weight) stays fixed.
        self.pde_weight = float(pde_weight)
        self.reg_weight = float(reg_weight)
        self.eps_grad = float(eps_grad)
        self.log_every = max(1, int(log_every))
        self.verbose = bool(verbose)

        # Grid velocity bounds -> squared-slowness bounds (reversed ordering).
        if grid.c_min is None:
            raise ValueError("JointFreqODIL requires a grid with c_min set")
        cmin = float(grid.c_min)
        cmax = float(grid.c_max)
        if not (0.0 < cmin < cmax):
            raise ValueError(f"require 0 < c_min < c_max; got {cmin}, {cmax}")
        self.c_min, self.c_max = cmin, cmax
        self.m_min = 1.0 / cmax**2
        self.m_max = 1.0 / cmin**2
        self.latent = SlownessLatent(
            m_min=self.m_min,
            m_max=self.m_max,
            z_scale=float(z_scale),
            logit_clip=float(logit_clip),
        )

        self.vm_in = wavefield.velocity_model
        self.pml_c = float(self.vm_in.pml_c)

        # ---- fixed reference model / wavefield / residual scales ---------- #
        self.sources = loss.sources  # (n_shots, nf, Nx, Ny), t0**2-scaled, no m-dep
        # (n_shots, nf, n_transducers - 1): no shot is recorded at its source.
        self._obs_traces = loss.obs_traces().detach()

        # Active (non-PML interior) mask over the full grid, for RMS_active.
        self._interior_slice = grid.interior_slice

        # u_init warm-start wavefields -> u_ref.
        u_seed = self._stack_u_init()  # (n_shots, nf, Nx, Ny) complex
        self.u_seed = u_seed
        u_int = u_seed[:, :, self._interior_slice[0], self._interior_slice[1]]
        u_ref = rms_abs(u_int, dim=(2, 3)).clamp_min(eps_u)  # (n_shots, nf)
        self.u_ref = u_ref.view(self.n_shots, self.nf, 1, 1)

        # |lambda_tt| * u_ref: the symbol the residual uses (ADR-0002).
        _, lambda_tt = wavefield.frequency_selection.symbols_broadcast()
        lam_tt = lambda_tt.abs().to(
            dtype=self.dtype, device=self.device
        )  # (1, nf, 1, 1)
        pde_scale = (lam_tt.view(1, self.nf) * u_ref).clamp_min(
            eps_pde
        )  # (n_shots, nf)
        self.pde_scale = pde_scale.view(self.n_shots, self.nf, 1, 1)
        # Full-grid mean -> sum over interior + PML / N_int.
        self.n_int = u_int.shape[-2] * u_int.shape[-1]
        self.pde_node_factor = (self.Nx * self.Ny) / self.n_int
        data_scale = rms_abs(self._obs_traces, dim=(2,)).clamp_min(eps_data)
        self.data_scale = data_scale.view(self.n_shots, self.nf, 1)
        # ||d_obs||_2 over shots x frequencies x receivers, for rel_data_residual.
        self.obs_norm = float(torch.linalg.vector_norm(self._obs_traces))
        # Fixed model reference for reporting: c_ref = homogeneous background.
        c0_int = self.vm_in.c[self._interior_slice].detach()
        self.c_ref = float(c0_int.mean().item())
        self.m_ref = 1.0 / self.c_ref**2

        # Diagnostics buffers.
        self.diag_rows: List[Dict[str, float]] = []
        self.scale_info: Dict[str, object] = {}
        self._build_diag_scale_info(u_ref, pde_scale, data_scale)

        # Wavefield preconditioner (v <-> u_tilde), then the model one (y <-> z_m),
        # whose target is the wavefield block's curvature in v.
        pde_scale_per_u = self.pde_scale / self.u_ref
        term_weight = self.pde_weight / (self.n_int * self.n_shots * self.nf)
        if wavefield_preconditioner == "initial_model":
            c_init = self.vm_in.c.detach().to(dtype=self.dtype, device=self.device)
            self.wavefield_precond = InitialModelWavefieldPreconditioner(
                self.wave_eq,
                c_init,
                pde_scale_per_u,
                term_weight,
                initial_model_factors,
            )
        else:
            c_init = self.c_full_of_z(self._initial_latent())
            self.wavefield_precond = NoWavefieldPreconditioner(
                self.wave_eq, c_init, pde_scale_per_u, term_weight
            )
        self.scale_info.update(self.wavefield_precond.info())

        self.eps_precond = float(eps_precond)
        if model_preconditioner == "pde_diagonal":
            _, H_zz = self.model_curvature(self._initial_latent())
            target = float(self.wavefield_precond.pde_curvature().median())
            self.model_precond = PdeDiagonalModelPreconditioner(
                H_zz, target, self.eps_precond
            )
        else:
            self.model_precond = NoModelPreconditioner()
        self.scale_info.update(self.model_precond.info())

    # ------------------------------------------------------------------ #
    def _stack_u_init(self) -> torch.Tensor:
        """Warm-start wavefields as a detached ``(n_shots, nf, Nx, Ny)`` tensor."""
        if isinstance(self.u_init, (list, tuple)):
            stack = torch.stack(
                [
                    w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                    for w in self.u_init
                ]
            )
        else:
            stack = torch.as_tensor(self.u_init)
        stack = stack.detach().to(dtype=self.cdtype, device=self.device)
        if stack.ndim == 3:
            stack = stack.unsqueeze(0).expand(self.n_shots, -1, -1, -1)
        if tuple(stack.shape) != (self.n_shots, self.nf, self.Nx, self.Ny):
            raise ValueError(
                f"u_init shape {tuple(stack.shape)} != "
                f"({self.n_shots}, {self.nf}, {self.Nx}, {self.Ny})"
            )
        return stack.contiguous()

    def _build_diag_scale_info(self, u_ref, pde_scale, data_scale) -> None:
        self.scale_info = {
            "c_min": self.c_min,
            "c_max": self.c_max,
            "m_min": self.m_min,
            "m_max": self.m_max,
            "c_ref": self.c_ref,
            "m_ref": self.m_ref,
            "z_scale": self.latent.z_scale,
            "pde_weight": self.pde_weight,
            "data_weight": self.data_weight,
            "reg_weight": self.reg_weight,
            "u_ref_min": float(u_ref.min()),
            "u_ref_max": float(u_ref.max()),
            "u_ref_mean": float(u_ref.mean()),
            "pde_scale_min": float(pde_scale.min()),
            "pde_scale_max": float(pde_scale.max()),
            "pde_scale_mean": float(pde_scale.mean()),
            "pde_scale_rule": "|lambda_tt| * u_ref per (shot, freq), fixed",
            "pde_node_factor": self.pde_node_factor,
            "data_scale_min": float(data_scale.min()),
            "data_scale_max": float(data_scale.max()),
            "data_scale_mean": float(data_scale.mean()),
            "data_scale_rule": "RMS(|d_obs|) per (shot, freq), fixed",
        }

    # ------------------------------------------------------------------ #
    # Block curvatures of the weighted PDE term at the warm start
    # ------------------------------------------------------------------ #
    def model_curvature(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diagonal ``(H_kk, H_zz)`` of the weighted PDE term on interior cells.

        At ``(u_seed, z0)``, ``z0`` the band's initial latent (``init_params``):
        ``H_kk`` is w.r.t. ``k = (c/c0)^2`` (exact, as ``r`` is linear in ``k``),
        ``H_zz`` w.r.t. ``z_m`` (exact where the PDE residual vanishes, i.e. for
        a Helmholtz warm start).
        """
        c_full = self.c_full_of_z(z0)
        no_src = torch.zeros_like(self.u_seed)
        with torch.no_grad():
            # dr/dk = -lap(u): the residual at k = 0 minus the residual at k = 1.
            lap_u = self.wave_eq.residual(
                self.u_seed, torch.zeros_like(c_full), no_src
            ) - self.wave_eq.residual(
                self.u_seed, torch.full_like(c_full, self.c0), no_src
            )
            lap_u = lap_u[:, :, self._interior_slice[0], self._interior_slice[1]]
            H_kk = (abs_sq(lap_u) / self.pde_scale.square()).sum(dim=(0, 1))
            H_kk = self.pde_weight * H_kk / (self.n_int * self.n_shots * self.nf)
            # k = 1 / (c0^2 m)
            m = self.latent.m_of_z(z0)
            dk_dz = -self.latent.dm_dz(z0) / (self.c0**2 * m.square())
        return H_kk, dk_dz.square() * H_kk

    # ------------------------------------------------------------------ #
    # Parameter <-> physical helpers
    # ------------------------------------------------------------------ #
    def init_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial ``(v_re, v_im, z_m)`` tensors (not yet Parameters).

        ``v`` is the optimised wavefield variable of the scaled warm start
        (``wavefield_precond.v_of_u_tilde``).
        """
        v_re, v_im = self.wavefield_precond.v_of_u_tilde(self.u_seed / self.u_ref)
        return v_re, v_im, self._initial_latent()

    def _initial_latent(self) -> torch.Tensor:
        """The band's initial latent ``z_m``, from the wavefield's velocity model."""
        c0_int = (
            self.vm_in.c[self._interior_slice]
            .detach()
            .to(dtype=self.dtype, device=self.device)
        )
        return self.latent.z_of_c(c0_int).contiguous().to(dtype=self.dtype)

    @property
    def wavefield_solve_s(self) -> float:
        """Wall time of the wavefield preconditioner's triangular solves so far
        (forward and adjoint); ``0.0`` without it."""
        return self.wavefield_precond.solve_seconds

    def physical_u(self, v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
        """Physical wavefield ``u`` of the optimised wavefield variable."""
        return self.u_ref * self.wavefield_precond.u_tilde_of_v(v_re, v_im)

    def c_full_of_z(self, z_m: torch.Tensor) -> torch.Tensor:
        return self.vm_in.build_full_c(self.latent.c_of_z(z_m))

    @property
    def last_record(self) -> Dict[str, float]:
        """Diagnostics of the latest ``objective(..., record=True)`` call."""
        return dict(self._last)

    # ------------------------------------------------------------------ #
    # Residuals and objective (the production code path).
    # ------------------------------------------------------------------ #
    def residuals(
        self, v_re: torch.Tensor, v_im: torch.Tensor, z_m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(r_pde, r_data)`` on physical variables (unscaled)."""
        return self._residuals_of_u(self.physical_u(v_re, v_im), z_m)

    def _residuals_of_u(
        self, u: torch.Tensor, z_m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c_full = self.c_full_of_z(z_m)
        r_pde = self.wave_eq.residual(u, c_full, self.sources)
        u_recv = self.geometry.extract_observations(u)
        r_data = u_recv - self._obs_traces
        return r_pde, r_data

    def objective(
        self,
        v_re: torch.Tensor,
        v_im: torch.Tensor,
        z_m: torch.Tensor,
        *,
        record: bool = True,
    ) -> torch.Tensor:
        """Phi at the optimised wavefield variable and the physical latent."""
        return self._objective_of_u(self.physical_u(v_re, v_im), z_m, record=record)

    def _objective_of_u(
        self, u: torch.Tensor, z_m: torch.Tensor, *, record: bool
    ) -> torch.Tensor:
        r_pde, r_data = self._residuals_of_u(u, z_m)
        pde_term = self.pde_node_factor * mean_real_sq(r_pde / self.pde_scale)
        data_term = mean_real_sq(r_data / self.data_scale)
        L = self.pde_weight * pde_term + self.data_weight * data_term
        reg_term = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.reg_weight != 0.0 and self.loss.config.regulariser is not None:
            c_int = self.latent.c_of_z(z_m)
            reg_term = self.reg_weight * self.loss.config.regulariser(c_int)
            L = L + reg_term
        if L.is_complex():
            L = L.real
        if record:
            with torch.no_grad():
                self._last = {
                    "pde_weight": self.pde_weight,
                    "pde_term": float(pde_term.detach()),
                    "pde_term_weighted": float((self.pde_weight * pde_term).detach()),
                    "data_term": float((self.data_weight * data_term).detach()),
                    "reg_term": float(reg_term.detach()),
                    "pde_loss": float(pde_term.detach()),
                    "data_loss": float(data_term.detach()),
                    "pde_rms_raw": float(rms_abs(r_pde).detach()),
                    "data_rms_raw": float(rms_abs(r_data).detach()),
                    "pde_rms_scaled": float(rms_abs(r_pde / self.pde_scale).detach()),
                    "data_rms_scaled": float(
                        rms_abs(r_data / self.data_scale).detach()
                    ),
                    "pde_rel_residual": math.sqrt(2.0 * float(pde_term.detach())),
                    "rel_data_residual": float(torch.linalg.vector_norm(r_data))
                    / max(self.obs_norm, 1e-30),
                }
        return L

    # ------------------------------------------------------------------ #
    def _model_stats(self, z_m: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            m = self.latent.m_of_z(z_m)
            c = m.clamp_min(1e-30).rsqrt()
            sderiv = self.latent.sigmoid_deriv(z_m)
            # near-bound fraction: sigmoid within 1e-3 of 0 or 1
            s = torch.sigmoid(self.latent.z_scale * z_m)
            near = ((s < 1e-3) | (s > 1.0 - 1e-3)).float().mean()
            return {
                "m_min": float(m.min()),
                "m_max": float(m.max()),
                "c_min": float(c.min()),
                "c_max": float(c.max()),
                "sigmoid_deriv_mean": float(sderiv.mean()),
                "sigmoid_deriv_min": float(sderiv.min()),
                "sat_frac": float(near),
            }

    def _grad_norms(
        self, v_re: torch.Tensor, v_im: torch.Tensor, z_m: torch.Tensor
    ) -> Dict[str, float]:
        gu = 0.0
        if v_re.grad is not None:
            gu += float(v_re.grad.pow(2).sum())
        if v_im.grad is not None:
            gu += float(v_im.grad.pow(2).sum())
        gu = math.sqrt(gu)
        gz = float(z_m.grad.norm()) if z_m.grad is not None else 0.0
        return {
            "grad_u_norm": gu,
            "grad_z_norm": gz,
            "grad_ratio": gz / max(gu, self.eps_grad),
        }

    def _diag_grads(self, v_re_val, v_im_val, z_m_val):
        """Φ and *physical* block-gradient norms (∇_{u_tilde}, ∇_{z_m}) at a point.

        Uses fresh leaves on ``u_tilde`` and ``z_m`` so the reported gradients are
        always with respect to the scaled wavefield and the physical latent —
        comparable across runs with and without either preconditioner — and
        records ``self._last``.
        """
        with torch.no_grad():
            u_tilde = self.wavefield_precond.u_tilde_of_v(
                v_re_val.detach(), v_im_val.detach()
            )
        a = u_tilde.real.clone().requires_grad_(True)
        b = u_tilde.imag.clone().requires_grad_(True)
        c = z_m_val.detach().clone().requires_grad_(True)
        L = self._objective_of_u(self.u_ref * torch.complex(a, b), c, record=True)
        ga, gb, gc = torch.autograd.grad(L, (a, b, c))
        gu = math.sqrt(float(ga.pow(2).sum() + gb.pow(2).sum()))
        gz = float(gc.norm())
        return float(L.detach()), {
            "grad_u_norm": gu,
            "grad_z_norm": gz,
            "grad_ratio": gz / max(gu, self.eps_grad),
        }

    def _snapshot(self, n_done: int, v_re, v_im, z_m):
        """``(Iterate, grad norms, model stats)`` after ``n_done`` outer
        iterations; the evaluation is not counted as a closure."""
        z_m = z_m.detach()
        loss_scalar, gnorms = self._diag_grads(v_re, v_im, z_m)
        mstats = self._model_stats(z_m)
        with torch.no_grad():
            c_full = self.c_full_of_z(z_m)
            u = self.physical_u(v_re.detach(), v_im.detach())
        last = self._last
        it = Iterate(
            iter=n_done,
            n_evaluations=self._n_eval,
            loss=loss_scalar,
            pde_term=last["pde_term_weighted"],
            data_term=last["data_term"],
            reg_term=last["reg_term"],
            pde_rel_residual=last["pde_rel_residual"],
            rel_data_residual=last["rel_data_residual"],
            grad_u_norm=gnorms["grad_u_norm"],
            grad_z_norm=gnorms["grad_z_norm"],
            sat_frac=mstats["sat_frac"],
            c_full=c_full,
            u=u,
        )
        return it, gnorms, mstats

    def minimise(self, on_log=None, **overrides) -> Tuple[List[Wavefield], LossTape]:
        """Run ``n_iter`` outer L-BFGS steps.

        ``on_log(iterate)`` receives an :class:`Iterate` at the start (before
        any update) and at every logged point: each ``log_every``-th outer
        iteration and always the final iterate.
        """
        for k, v in overrides.items():
            setattr(self, k, v)
        v_re0, v_im0, z0 = self.init_params()
        v_re = torch.nn.Parameter(v_re0)
        v_im = torch.nn.Parameter(v_im0)
        # L-BFGS optimises y; the objective and diagnostics see z_m = z_of_y(y)
        # (model_precond).
        z_param = torch.nn.Parameter(self.model_precond.y_of_z(z0))

        def z_of():
            return self.model_precond.z_of_y(z_param)

        opt = torch.optim.LBFGS(
            [v_re, v_im, z_param],
            lr=self.lbfgs_lr,
            max_iter=self.inner_max_iter,
            history_size=self.history_size,
            line_search_fn=self.line_search_fn,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
        )

        self._n_eval = 0

        def closure():
            self._n_eval += 1
            opt.zero_grad()
            L = self.objective(v_re, v_im, z_of(), record=True)
            L.backward()
            return L

        t0 = time.perf_counter()
        if on_log is not None:
            on_log(self._snapshot(0, v_re, v_im, z_of())[0])
        for i in range(self.n_iter):
            opt.step(closure)

            # Diagnostic evaluation at the current (post-step) point.
            n_done = i + 1
            if n_done % self.log_every != 0 and n_done != self.n_iter:
                continue
            it, gnorms, mstats = self._snapshot(n_done, v_re, v_im, z_of())
            loss_scalar = it.loss
            self.diag_rows.append(
                {
                    "iter": n_done,
                    "wall_s": time.perf_counter() - t0,
                    "loss": loss_scalar,
                    "n_eval": self._n_eval,
                    **self._last,
                    **gnorms,
                    **mstats,
                }
            )
            self.loss.last_residuals = {
                "pde_rms": self._last["pde_rms_raw"],
                "data_rms": self._last["data_rms_raw"],
                "pde_loss": self._last["pde_loss"],
                "data_loss": self._last["data_loss"],
                # Weighted terms as minimised (weights applied here).
                "pde_term": self._last["pde_term_weighted"],
                "data_term": self._last["data_term"],
                "reg_term": self._last["reg_term"],
                "pde_rel_residual": self._last["pde_rel_residual"],
                # Closures so far in this solve (diagnostics excluded).
                "n_evaluations": self._n_eval,
            }
            self.loss.callback.log(
                loss_scalar,
                self.loss.last_residuals,
                pde_rel_residual=self._last["pde_rel_residual"],
            )
            hist = self.loss.callback.history
            for key in ("grad_u_norm", "grad_z_norm", "grad_ratio"):
                hist.setdefault(key, []).append(gnorms[key])
            hist.setdefault("sat_frac", []).append(mstats["sat_frac"])
            hist.setdefault("pde_rms_scaled", []).append(self._last["pde_rms_scaled"])
            hist.setdefault("data_rms_scaled", []).append(self._last["data_rms_scaled"])
            if self.loss.callback.store_c_history:
                self.loss.callback.log_c(it.c_full.cpu().numpy())
            if self.verbose:
                print(
                    f"[joint] iter {n_done:4d} | Phi={loss_scalar:.6e} | "
                    f"pde_rel={self._last['pde_rel_residual']:.3e} | "
                    f"|gz|/|gu|={gnorms['grad_ratio']:.3e} | "
                    f"c=[{mstats['c_min']:.1f},{mstats['c_max']:.1f}]"
                )
            if on_log is not None:
                on_log(it)

        wall_s = time.perf_counter() - t0

        # Final recovered model + wavefield.
        with torch.no_grad():
            z_final = z_of().detach()
            c_full_final = self.c_full_of_z(z_final)
            vm_out = VelocityModel.from_field(
                self.grid,
                c_full_final,
                pml_c=self.vm_in.pml_c,
            )
            u_final = self.physical_u(v_re.detach(), v_im.detach())
            # ``LBFGS.step`` returns the loss at the *start* of the step; report
            # the objective at the final iterate instead (not counted as a closure).
            loss_final = float(
                self.objective(v_re.detach(), v_im.detach(), z_final, record=False)
            )

        outputs: List[Wavefield] = []
        for s in range(self.n_shots):
            wf = Wavefield(
                grid=self.grid,
                frequency_selection=self.wavefield.frequency_selection,
                velocity_model=vm_out,
            )
            wf.amplitude = u_final[s]
            outputs.append(wf)

        self.final_state = {
            "u_re": v_re.detach(),
            "u_im": v_im.detach(),
            "z_m": z_final,
            "u_final": u_final,
            "c_full_final": c_full_final,
        }
        self.loss.callback.result = {
            "loss": loss_final,
            "n_outer_iter": self.n_iter,
            "n_closure": self._n_eval,
            "n_evaluations": self._n_eval,
            "wall_s": wall_s,
            "method": "joint",
            "data_weight": self.data_weight,
            "scale_info": self.scale_info,
        }
        return outputs, self.loss.callback
