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
      m_min = 1 / c_max**2 ,   m_max = 1 / c_min**2 ,   c = 1 / sqrt(m).

  ``c_full = velocity_model.build_full_c(c)`` freezes PML cells at ``pml_c`` so
  only interior cells are optimised. Bounds are enforced *structurally* by the
  sigmoid — every L-BFGS line-search trial stays in-bounds (no post-step
  projection).

* Scaled wavefield ``u = u_ref * u_tilde`` with a fixed per-(shot, frequency)
  scalar ``u_ref[s, w] = max(RMS_active|u_init[s, w]|, eps_u)``. ``u_tilde`` is
  two real tensors (Re, Im).

* Residuals use the repository's exact operator (inverse-crime consistent):

      r_pde[s, w]  = WaveEquation.residual(u, c_full, q)[s, w]     (full grid)
      r_data[s, w] = P u[s, w] - d_obs[s, w]                       (receivers)

  fixed scales ``pde_scale[s, w] = max(RMS_active|q[s, w]|, eps_p)`` (source
  based; NOT the near-zero warm-start PDE residual) and
  ``data_scale[s, w] = max(RMS|d_obs[s, w]|, eps_d)``.

* Objective (means over the count of **real** scalar residual entries):

      Phi = pde_weight  * mean|r_pde  / pde_scale |^2
          + data_weight * mean|r_data / data_scale|^2
          + w_reg  * R(c_interior)

  with ``pde_weight = 1``, ``data_weight = data_weight``, ``w_reg = reg_weight`` (default
  0). ``data_weight`` is a fixed run-level hyperparameter (never adapted).

* One persistent ``torch.optim.LBFGS([u_re, u_im, z_m])`` (strong-Wolfe) — a
  single joint optimiser over the full space, no alternation. It is advanced by
  ``n_iter`` outer ``.step`` calls, each running up to ``inner_max_iter`` L-BFGS
  iterations on the shared history (total ~= ``n_iter * inner_max_iter``), so the
  per-step trajectory is logged while the solve stays continuous. (A single
  iteration per step, ``inner_max_iter=1``, stalls the strong-Wolfe line search
  and is *not* representative of L-BFGS on this problem.) The objective is
  matrix-free (one conv2d Laplacian per evaluation); ``A(c)`` is never factorised
  inside optimisation.

The class is pure library code (no file IO). Per-iteration diagnostics are
exposed on :attr:`diag_rows` and mirrored into the loss callback history; the
``sandbox/joint_odil`` drivers persist the full §10 artifact set.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from odil_wave.loss import InverseLoss
from odil_wave.loss.utils import LossTape
from odil_wave.models import VelocityModel
from odil_wave.wavefield import Wavefield

from .base import Optimiser


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
    if x.is_complex():
        v = x.real.square() + x.imag.square()
    else:
        v = x.square()
    return v.mean(dim=dim).sqrt() if dim is not None else v.mean().sqrt()


# --------------------------------------------------------------------------- #
# Latent squared-slowness map
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SlownessLatent:
    """Bounded squared-slowness reparameterisation ``m = m(z_m)``.

    ``m(z) = m_min + (m_max - m_min) * sigmoid(z_scale * z)`` with velocity
    bounds converted as ``m_min = 1/c_max**2``, ``m_max = 1/c_min**2``.
    ``z_scale`` is a fixed model-coordinate scale (default 1; used by the scaling
    sweep). The inverse (logit) is used once for initialisation, clipped away
    from 0/1.
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

    def dm_dz(self, z: torch.Tensor) -> torch.Tensor:
        """Analytic dm/dz = (m_max - m_min) * z_scale * s * (1 - s)."""
        s = torch.sigmoid(self.z_scale * z)
        return (self.m_max - self.m_min) * self.z_scale * s * (1.0 - s)

    def sigmoid_deriv(self, z: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.z_scale * z)
        return s * (1.0 - s)

    def z_of_m(self, m: torch.Tensor) -> torch.Tensor:
        """Stable inverse (logit) so that ``m(z_of_m(m)) == m`` (clipped)."""
        frac = (m - self.m_min) / (self.m_max - self.m_min)
        frac = frac.clamp(self.logit_clip, 1.0 - self.logit_clip)
        return torch.log(frac / (1.0 - frac)) / self.z_scale

    def z_of_c(self, c: torch.Tensor) -> torch.Tensor:
        return self.z_of_m(c.clamp_min(1e-30) ** -2)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class JointFreqODIL(Optimiser):
    """Pure joint full-space frequency-domain ODIL solver.

    Parameters mirror the ``optimiser.joint`` config block; see the module
    docstring for the formulation. ``loss`` must be an :class:`InverseLoss`
    (the operator, geometry, sources and observations are read from it, so the
    PDE / receiver operators are identical to truth-data generation).
    """

    def __init__(
        self,
        wavefield: Wavefield,
        loss: InverseLoss,
        *,
        clamp: bool = True,  # bounds always via latent map; kept for API parity
        u_init=None,
        n_iter: int = 50,
        inner_max_iter: int = 20,
        history_size: int = 20,
        line_search_fn: Optional[str] = "strong_wolfe",
        tolerance_grad: float = 1e-12,
        tolerance_change: float = 1e-14,
        lbfgs_lr: float = 1.0,
        data_weight: float = 1.0,
        pde_weight: float = 1.0,
        reg_weight: float = 0.0,
        c_min: Optional[float] = None,
        c_max: Optional[float] = None,
        # Fixed-scale controls + scaling-sweep perturbation factors.
        eps_u: float = 1e-30,
        eps_data: float = 1e-30,
        eps_pde: float = 1e-30,
        u_scale_factor: float = 1.0,
        pde_scale_factor: float = 1.0,
        data_scale_factor: float = 1.0,
        z_scale: float = 1.0,
        logit_clip: float = 1e-6,
        eps_grad: float = 1e-30,
        log_every: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__(wavefield, loss)
        if not isinstance(loss, InverseLoss):
            raise TypeError("JointFreqODIL requires an InverseLoss")

        grid = wavefield.grid
        self.grid = grid
        self.dtype = grid.dtype
        self.cdtype = wavefield.cdtype
        self.device = grid.device
        self.wave_eq = loss.config.wave_eq
        self.geometry = loss.config.geometry
        self.n_shots = self.geometry.n_sources
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
        # Run-level PDE-penalty weight pde_weight (default 1.0 -> bit-identical to the
        # established objective). A *predeclared, run-level* schedule may set this
        # per continuation stage (Question A); it is never adapted from the
        # instantaneous losses. lambda_data (== data_weight) stays fixed.
        self.pde_weight = float(pde_weight)
        self.reg_weight = float(reg_weight)
        self.eps_grad = float(eps_grad)
        self.log_every = max(1, int(log_every))
        self.verbose = bool(verbose)

        # Velocity bounds -> squared-slowness bounds (reversed ordering).
        cmin = float(grid.c_min if c_min is None else c_min)
        cmax = float(grid.c_max if c_max is None else c_max)
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
        self._obs_traces = loss._obs_traces().detach()  # (n_shots, nf, n_recv)
        ri = self.geometry.recv_ij[:, 0]
        rj = self.geometry.recv_ij[:, 1]
        self._ri, self._rj = ri, rj

        # Active (non-PML interior) mask over the full grid, for RMS_active.
        self._interior_slice = grid.interior_slice

        # u_init warm-start wavefields -> u_ref.
        u_seed = self._stack_u_init()  # (n_shots, nf, Nx, Ny) complex
        self.u_seed = u_seed
        u_int = u_seed[:, :, self._interior_slice[0], self._interior_slice[1]]
        u_ref = rms_abs(u_int, dim=(2, 3)).clamp_min(eps_u)  # (n_shots, nf)
        self.u_ref = (u_ref * float(u_scale_factor)).view(self.n_shots, self.nf, 1, 1)

        q_int = self.sources[:, :, self._interior_slice[0], self._interior_slice[1]]
        pde_scale = rms_abs(q_int, dim=(2, 3)).clamp_min(eps_pde)  # (n_shots, nf)
        self.pde_scale = (pde_scale * float(pde_scale_factor)).view(
            self.n_shots, self.nf, 1, 1
        )
        data_scale = rms_abs(self._obs_traces, dim=(2,)).clamp_min(eps_data)
        self.data_scale = (data_scale * float(data_scale_factor)).view(
            self.n_shots, self.nf, 1
        )
        # Fixed model reference for reporting: c_ref = homogeneous background.
        c0_int = self.vm_in.c[self._interior_slice].detach()
        self.c_ref = float(c0_int.mean().item())
        self.m_ref = 1.0 / self.c_ref**2

        # Source RMS for pde/src ratio reporting.
        self.src_rms = float(rms_abs(self.sources).item())

        # Diagnostics buffers.
        self.diag_rows: List[Dict[str, float]] = []
        self.scale_info: Dict[str, object] = {}
        self._build_diag_scale_info(u_ref, pde_scale, data_scale)

    # ------------------------------------------------------------------ #
    def _stack_u_init(self) -> torch.Tensor:
        """Warm-start wavefields as a detached ``(n_shots, nf, Nx, Ny)`` tensor."""
        if self.u_init is None:
            seed = self.wavefield.amplitude.detach()
            if seed.ndim == 3:
                seed = seed.unsqueeze(0).expand(self.n_shots, -1, -1, -1)
            return seed.to(dtype=self.cdtype, device=self.device).contiguous()
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
            "data_weight": self.data_weight,
            "reg_weight": self.reg_weight,
            "u_ref_min": float(u_ref.min()),
            "u_ref_max": float(u_ref.max()),
            "u_ref_mean": float(u_ref.mean()),
            "pde_scale_min": float(pde_scale.min()),
            "pde_scale_max": float(pde_scale.max()),
            "pde_scale_mean": float(pde_scale.mean()),
            "pde_scale_rule": "RMS_active(|q|)  (source based, fixed)",
            "data_scale_min": float(data_scale.min()),
            "data_scale_max": float(data_scale.max()),
            "data_scale_mean": float(data_scale.mean()),
            "data_scale_rule": "RMS(|d_obs|) per (shot, freq), fixed",
            "src_rms": self.src_rms,
        }

    # ------------------------------------------------------------------ #
    # Parameter <-> physical helpers
    # ------------------------------------------------------------------ #
    def init_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial ``(u_re, u_im, z_m)`` tensors (not yet Parameters)."""
        u_tilde0 = self.u_seed / self.u_ref
        c0_int = (
            self.vm_in.c[self._interior_slice]
            .detach()
            .to(dtype=self.dtype, device=self.device)
        )
        z0 = self.latent.z_of_c(c0_int)
        return (
            u_tilde0.real.contiguous().to(dtype=self.dtype),
            u_tilde0.imag.contiguous().to(dtype=self.dtype),
            z0.contiguous().to(dtype=self.dtype),
        )

    def physical_u(self, u_re: torch.Tensor, u_im: torch.Tensor) -> torch.Tensor:
        return self.u_ref * torch.complex(u_re, u_im)

    def c_full_of_z(self, z_m: torch.Tensor) -> torch.Tensor:
        return self.vm_in.build_full_c(self.latent.c_of_z(z_m))

    # ------------------------------------------------------------------ #
    # Residuals and objective (the production code path).
    # ------------------------------------------------------------------ #
    def residuals(
        self, u_re: torch.Tensor, u_im: torch.Tensor, z_m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(r_pde, r_data)`` on physical variables (unscaled)."""
        u = self.physical_u(u_re, u_im)
        c_full = self.c_full_of_z(z_m)
        r_pde = self.wave_eq.residual(u, c_full, self.sources)
        u_recv = u[:, :, self._ri, self._rj]
        r_data = u_recv - self._obs_traces
        return r_pde, r_data

    def lap(self, u: torch.Tensor) -> torch.Tensor:
        """Non-dimensional Laplacian used inside the PDE residual (same BC)."""
        return self.wave_eq._lap.apply(u, bc=self.wave_eq._bc)

    def _zero_pad_interior(self, interior: torch.Tensor) -> torch.Tensor:
        """Embed an interior-shaped field in a full grid padded with **zeros**.

        (``build_full_c`` pads with ``pml_c``; a *perturbation* ``dk`` must be
        zero in PML because PML ``c`` is frozen.)
        """
        full = torch.zeros(
            self.Nx, self.Ny, dtype=interior.dtype, device=interior.device
        )
        full[self._interior_slice] = interior
        return full

    def residual_jvp_analytic(
        self,
        u_re: torch.Tensor,
        u_im: torch.Tensor,
        z_m: torch.Tensor,
        du_re: torch.Tensor,
        du_im: torch.Tensor,
        dz_m: torch.Tensor,
    ) -> torch.Tensor:
        """Hand-derived directional derivative ``D r_pde[du, dm]`` (no autograd).

        For the repository operator ``r_pde = H(k)u - q`` with ``k = 1/(c0**2 m)``
        (see the module / audit notes),

            D r_pde[du, dm] = H(k)·du  -  ∇'^2 u · dk ,
            H(k)·du = WaveEquation.residual(du, c, 0),
            dk = (∂k/∂m)·dm,  ∂k/∂m = -1/(c0**2 m^2),
            dm = (dm/dz)·dz_m   (interior only; dk = 0 in PML).

        Directions ``du_*`` / ``dz_m`` are in **scaled** coordinates
        (``du`` physical = ``u_ref·(du_re + i du_im)``).
        """
        u = self.physical_u(u_re, u_im)
        du = self.u_ref * torch.complex(du_re, du_im)
        c_full = self.c_full_of_z(z_m)
        Ju = self.wave_eq.residual(du, c_full, torch.zeros_like(self.sources))
        m = self.latent.m_of_z(z_m)
        dm = self.latent.dm_dz(z_m) * dz_m
        dk_int = -(dm / (m * m)) / (self.c0**2)
        dk_full = self._zero_pad_interior(dk_int)
        Jm = -self.lap(u) * dk_full
        return Ju + Jm

    def objective(
        self,
        u_re: torch.Tensor,
        u_im: torch.Tensor,
        z_m: torch.Tensor,
        *,
        record: bool = True,
    ) -> torch.Tensor:
        r_pde, r_data = self.residuals(u_re, u_im, z_m)
        pde_term = mean_real_sq(r_pde / self.pde_scale)
        data_term = mean_real_sq(r_data / self.data_scale)
        L = self.pde_weight * pde_term + self.data_weight * data_term
        if self.reg_weight != 0.0 and self.loss.config.regulariser is not None:
            c_int = self.latent.c_of_z(z_m)
            L = L + self.reg_weight * self.loss.config.regulariser(c_int)
        if L.is_complex():
            L = L.real
        if record:
            with torch.no_grad():
                self._last = {
                    "pde_weight": self.pde_weight,
                    "pde_term": float(pde_term.detach()),
                    "pde_term_weighted": float((self.pde_weight * pde_term).detach()),
                    "data_term": float((self.data_weight * data_term).detach()),
                    "pde_loss": float(pde_term.detach()),
                    "data_loss": float(data_term.detach()),
                    "pde_rms_raw": float(rms_abs(r_pde).detach()),
                    "data_rms_raw": float(rms_abs(r_data).detach()),
                    "pde_rms_scaled": float(rms_abs(r_pde / self.pde_scale).detach()),
                    "data_rms_scaled": float(
                        rms_abs(r_data / self.data_scale).detach()
                    ),
                    "pde_src_ratio": float(rms_abs(r_pde).detach())
                    / max(self.src_rms, 1e-30),
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
        self, u_re: torch.Tensor, u_im: torch.Tensor, z_m: torch.Tensor
    ) -> Dict[str, float]:
        gu = 0.0
        if u_re.grad is not None:
            gu += float(u_re.grad.pow(2).sum())
        if u_im.grad is not None:
            gu += float(u_im.grad.pow(2).sum())
        gu = math.sqrt(gu)
        gz = float(z_m.grad.norm()) if z_m.grad is not None else 0.0
        return {
            "grad_u_norm": gu,
            "grad_z_norm": gz,
            "grad_ratio": gz / max(gu, self.eps_grad),
        }

    def _diag_grads(self, u_re_val, u_im_val, z_m_val):
        """Φ and *physical* block-gradient norms (∇_u, ∇_{z_m}) at a point.

        Uses fresh leaves on ``z_m`` so the reported model-gradient is always with
        respect to the physical latent — comparable across baseline / preconditioned
        runs — and records ``self._last``.
        """
        a = u_re_val.detach().clone().requires_grad_(True)
        b = u_im_val.detach().clone().requires_grad_(True)
        c = z_m_val.detach().clone().requires_grad_(True)
        L = self.objective(a, b, c, record=True)
        ga, gb, gc = torch.autograd.grad(L, (a, b, c))
        gu = math.sqrt(float(ga.pow(2).sum() + gb.pow(2).sum()))
        gz = float(gc.norm())
        return float(L.detach()), {
            "grad_u_norm": gu,
            "grad_z_norm": gz,
            "grad_ratio": gz / max(gu, self.eps_grad),
        }

    def minimise(
        self, on_iteration=None, **overrides
    ) -> Tuple[List[Wavefield], LossTape]:
        for k, v in overrides.items():
            setattr(self, k, v)
        u_re0, u_im0, z0 = self.init_params()
        u_re = torch.nn.Parameter(u_re0)
        u_im = torch.nn.Parameter(u_im0)
        z_param = torch.nn.Parameter(z0)

        def z_of():
            return z_param

        opt = torch.optim.LBFGS(
            [u_re, u_im, z_param],
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
            L = self.objective(u_re, u_im, z_of(), record=True)
            L.backward()
            return L

        loss_value = None
        t0 = time.perf_counter()
        for i in range(self.n_iter):
            loss_value = opt.step(closure)

            # Diagnostic evaluation at the current (post-step) point.
            should_log = (i % self.log_every == 0) or (i == self.n_iter - 1)
            if should_log or on_iteration is not None:
                z_m = z_of().detach()
                loss_scalar, gnorms = self._diag_grads(u_re, u_im, z_m)
                mstats = self._model_stats(z_m)
                c_full_now = self.c_full_of_z(z_m)

                if on_iteration is not None:
                    on_iteration(i, c_full_now)

                row = {
                    "iter": i,
                    "wall_s": time.perf_counter() - t0,
                    "loss": loss_scalar,
                    "n_eval": self._n_eval,
                    **self._last,
                    **gnorms,
                    **mstats,
                }
                self.diag_rows.append(row)

                if should_log:
                    self.loss._last_residuals = {
                        "pde_rms": self._last["pde_rms_raw"],
                        "data_rms": self._last["data_rms_raw"],
                        "pde_loss": self._last["pde_loss"],
                        "data_loss": self._last["data_loss"],
                        "pde_src_ratio": self._last["pde_src_ratio"],
                    }
                    self.loss.callback.log(
                        loss_scalar,
                        self.loss._last_residuals,
                        pde_src_ratio=self._last["pde_src_ratio"],
                    )
                    hist = self.loss.callback.history
                    for key in (
                        "grad_u_norm",
                        "grad_z_norm",
                        "grad_ratio",
                    ):
                        hist.setdefault(key, []).append(gnorms[key])
                    hist.setdefault("sat_frac", []).append(mstats["sat_frac"])
                    hist.setdefault("pde_rms_scaled", []).append(
                        self._last["pde_rms_scaled"]
                    )
                    hist.setdefault("data_rms_scaled", []).append(
                        self._last["data_rms_scaled"]
                    )
                    if self.loss.callback.store_c_history:
                        self.loss.callback.log_c(c_full_now.detach().cpu().numpy())
                    if self.verbose:
                        print(
                            f"[joint] iter {i:4d} | Phi={loss_scalar:.6e} | "
                            f"|r_pde|/|src|={self._last['pde_src_ratio']:.3e} | "
                            f"|gz|/|gu|={gnorms['grad_ratio']:.3e} | "
                            f"c=[{mstats['c_min']:.1f},{mstats['c_max']:.1f}]"
                        )

        wall_s = time.perf_counter() - t0

        # Final recovered model + wavefield.
        with torch.no_grad():
            z_final = z_of().detach()
            c_full_final = self.c_full_of_z(z_final)
            vm_out = VelocityModel.from_field(
                self.grid,
                c_full_final,
                pml_c=self.vm_in.pml_c,
                pml_fill=self.vm_in.pml_fill,
            )
            u_final = self.physical_u(u_re.detach(), u_im.detach())

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
            "u_re": u_re.detach(),
            "u_im": u_im.detach(),
            "z_m": z_final,
            "u_final": u_final,
            "c_full_final": c_full_final,
        }
        self.loss.callback.result = {
            "loss": (float(loss_value.detach()) if loss_value is not None else None),
            "n_outer_iter": self.n_iter,
            "n_outer_budget": self.n_iter,
            "n_closure": self._n_eval,
            "n_evaluations": self._n_eval,
            "wall_s": wall_s,
            "method": "joint",
            "data_weight": self.data_weight,
            "n_factor": 0,
            "n_forward_solves": 0,
            "n_adjoint_solves": 0,
            "scale_info": self.scale_info,
        }
        return outputs, self.loss.callback
