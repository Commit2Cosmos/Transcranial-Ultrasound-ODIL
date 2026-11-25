"""Pure joint full-space frequency-domain ODIL.

A single joint optimiser over **both** the complex wavefield block ``u`` (all
sources and frequencies) and a squared-slowness model latent ``z_m``: no
alternating updates, no reduced c-block (ADR 0003), no multigrid, no adaptive
loss balancing.

Formulation
-----------
* Model latent ``z_m`` on interior cells, bounded squared slowness

      m(z_m) = m_min + (m_max - m_min) * sigmoid(z_scale * z_m),
      m_min = 1 / c_max**2 ,   m_max = 1 / c_min**2 ,   c = 1 / sqrt(m),

  with ``c_min`` / ``c_max`` the grid's velocity bounds. PML cells replicate
  the interior edge (``VelocityModel.build_full_c``). The sigmoid keeps every
  L-BFGS line-search trial in bounds (no post-step projection).

* Scaled wavefield ``u = u_ref * u_tilde`` with a fixed per-(shot, frequency)
  scalar ``u_ref[s, w] = max(RMS_active|u_init[s, w]|, eps_u)``.

* Residuals use the repository's exact operator (inverse-crime consistent):

      r_pde[s, w]  = WaveEquation.residual(u, c_full, q)[s, w]     (full grid)
      r_data[s, w] = P_s u[s, w] - d_obs[s, w]                     (receivers)

  where ``P_s`` samples the ``n_transducers - 1`` transducers recording shot
  ``s``, with fixed scales ``pde_scale[s, w] = max(|lambda_tt[w]| * u_ref[s,
  w], eps_p)`` (the wave term's magnitude, ADR-0002) and ``data_scale[s, w] =
  max(RMS|d_obs[s, w]|, eps_d)``.

* Objective:

      Phi = pde_weight  * sum|r_pde / pde_scale|^2 / (2 N_int n_s n_f)
          + data_weight * mean|r_data / data_scale|^2
          + w_reg  * R(c_interior)

  The PDE sum runs over interior **and** PML nodes but is normalised by the
  interior node count ``N_int``, so a thicker PML does not dilute it; the data
  term is a mean over the count of **real** scalar entries (``2x`` complex).
  The weights are fixed for the run (never adapted).

* One persistent ``torch.optim.LBFGS`` over the full space, advanced by
  ``n_iter`` outer ``.step`` calls of up to ``inner_max_iter`` iterations each
  on the shared history, so the trajectory is logged while the solve stays
  continuous.

* Block preconditioners (:mod:`odil_wave.optimisation.preconditioners`): each
  block's optimised variable maps to its own unknown through one object,
  ``wavefield_precond`` (``v <-> u_tilde``) and ``model_precond``
  (``y <-> z_m``); without a preconditioner it is the identity.

  ``model_preconditioner="pde_diagonal"``: L-BFGS models the Hessian outside
  its history as one ``gamma * I`` over both blocks, so the model block is
  rescaled per cell to the wavefield block's curvature, ``z_m = s * y``:

      H_kk[i] = pde_weight * sum_{s,f} |lap(u)[s,f,i] / pde_scale[s,f]|^2
                / (N_int n_s n_f)                         (r is linear in k)
      H_zz[i] = (dk/dz_m[i])^2 * H_kk[i]                  (exact where r = 0)
      T       = median of the weighted PDE term's diagonal in v
      s[i]    = sqrt(T / max(H_zz[i], eps_precond * median(H_zz)))

  with ``k = (c / c0)^2``, evaluated once per band at the warm start (never
  refreshed, which would make the L-BFGS history invalid). The data term and the
  regulariser are left out of both curvatures. ``z_scale`` cancels from
  ``z_scale * s``. ``H_zz`` comes from
  :meth:`JointFreqODIL.model_curvature`, ``T`` from
  ``wavefield_precond.pde_curvature()``.

The class is pure library code (no file IO). ``minimise`` reports through
``on_log`` (an :class:`Iterate` per logged point) and returns a
:class:`BandSolve`; the run's tape and artifacts are written in
``odil_wave.experiment``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from odil_wave.band import Band
from odil_wave.loss import Regulariser
from odil_wave.models import VelocityModel

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
    """Mean of ``|x|^2`` over the count of **real** scalar entries (``2 *
    numel`` for complex ``x``)."""
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
    ``z_scale`` is a fixed model-coordinate scale.
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


@dataclass(frozen=True)
class ObjectiveTerms:
    """The objective's terms before weighting, from
    :meth:`JointFreqODIL.objective_terms`:
    ``loss = pde_weight * pde + data_weight * data + reg_weight * reg``."""

    pde: float
    data: float
    reg: float  # R(c_interior); 0.0 without a regulariser


@dataclass(frozen=True)
class BandSolve:
    """What one band's joint solve produced: what ``minimise`` returns."""

    velocity_model: VelocityModel  # the recovered model, full grid
    u: torch.Tensor  # (n_shots, nf, Nx, Ny) complex physical wavefield, detached
    params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # final (v_re, v_im, z_m)
    loss: float  # the objective at the final iterate
    n_evaluations: int  # L-BFGS closures in this solve


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class JointFreqODIL:
    """Pure joint full-space frequency-domain ODIL solver; defaults follow
    ADR 0001. See the module docstring for the formulation. The operator,
    sources, observed data, initial model and warm start all come from
    ``band``; ``wavefield_preconditioner="initial_model"`` needs a band built
    with ``keep_factors``.

    ``init_params``, ``objective``, ``objective_terms``, ``residuals`` and
    ``physical_u`` take the optimised wavefield variable ``(v_re, v_im)``, as
    does ``BandSolve.params``; ``Iterate.u`` and ``BandSolve.u`` are the
    physical ``u``.
    """

    def __init__(
        self,
        band: Band,
        *,
        n_iter: int,
        pde_weight: float,
        data_weight: float,
        reg_weight: float,
        regulariser: Optional[Regulariser],
        inner_max_iter: int = 20,
        history_size: int = 20,
        line_search_fn: Optional[str] = "strong_wolfe",
        tolerance_grad: float = 1e-12,
        tolerance_change: float = 1e-14,
        lbfgs_lr: float = 1.0,
        log_every: int = 5,
        verbose: bool = False,
        model_preconditioner: str = "none",
        wavefield_preconditioner: str = "none",
        eps_u: float = 1e-30,
        eps_data: float = 1e-30,
        eps_pde: float = 1e-30,
        z_scale: float = 1.0,
        logit_clip: float = 1e-6,
        eps_precond: float = 1e-2,
    ) -> None:
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
        if wavefield_preconditioner == "initial_model" and band.factors is None:
            raise ValueError(
                "wavefield_preconditioner 'initial_model' needs a band built "
                "with keep_factors=True"
            )

        self.band = band
        self.regulariser = regulariser
        wavefield = band.wavefield
        grid = wavefield.grid
        self.grid = grid
        self.dtype = grid.dtype
        self.cdtype = wavefield.cdtype
        self.device = grid.device
        self.wave_eq = band.wave_eq
        self.geometry = band.geometry
        self.n_shots = self.geometry.n_shots
        self.nf = wavefield.n_frequencies
        self.Nx, self.Ny = grid.shape
        self.c0 = float(grid.c0)

        self.n_iter = int(n_iter)
        self.inner_max_iter = int(inner_max_iter)
        self.history_size = int(history_size)
        self.line_search_fn = line_search_fn
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        self.lbfgs_lr = float(lbfgs_lr)
        self.data_weight = float(data_weight)
        self.pde_weight = float(pde_weight)
        self.reg_weight = float(reg_weight)
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

        # ---- fixed reference model / wavefield / residual scales ---------- #
        self.sources = band.sources  # (n_shots, nf, Nx, Ny), t0**2-scaled, no m-dep
        # (n_shots, nf, n_transducers - 1): no shot is recorded at its source.
        self._obs_traces = band.observed_traces.detach()

        # Active (non-PML interior) mask over the full grid, for RMS_active.
        self._interior_slice = grid.interior_slice

        u_seed = band.u_init.to(dtype=self.cdtype, device=self.device).contiguous()
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
                band.factors,
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
    def _build_diag_scale_info(self, u_ref, pde_scale, data_scale) -> None:
        self.scale_info = {
            "c_min": self.c_min,
            "c_max": self.c_max,
            "m_min": self.m_min,
            "m_max": self.m_max,
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
        a Helmholtz warm start). An edge cell's curvature counts the PML cells
        replicating it.
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
            H_full = (abs_sq(lap_u) / self.pde_scale.square()).sum(dim=(0, 1))
        # Each PML cell's k is its edge cell's, so the edge cell sums their
        # curvature: the adjoint of the (linear) PML padding.
        k_int = torch.zeros_like(z0, requires_grad=True)
        (H_kk,) = torch.autograd.grad(self.vm_in.build_full_c(k_int), k_int, H_full)
        with torch.no_grad():
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
        (forward and adjoint, device copies included); ``0.0`` without it. It
        counts the logged rows' diagnostic solves too, so it is not the L-BFGS
        evaluations' cost alone."""
        return self.wavefield_precond.solve_seconds

    def physical_u(self, v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
        """Physical wavefield ``u`` of the optimised wavefield variable."""
        return self.u_ref * self.wavefield_precond.u_tilde_of_v(v_re, v_im)

    def c_full_of_z(self, z_m: torch.Tensor) -> torch.Tensor:
        return self.vm_in.build_full_c(self.latent.c_of_z(z_m))

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
        self, v_re: torch.Tensor, v_im: torch.Tensor, z_m: torch.Tensor
    ) -> torch.Tensor:
        """Phi at the optimised wavefield variable and the physical latent."""
        pde, data, reg, _ = self._terms_of_u(self.physical_u(v_re, v_im), z_m)
        return self._loss(pde, data, reg)

    def objective_terms(
        self, v_re: torch.Tensor, v_im: torch.Tensor, z_m: torch.Tensor
    ) -> ObjectiveTerms:
        """The terms of ``objective`` at the same point, unweighted."""
        with torch.no_grad():
            pde, data, reg, _ = self._terms_of_u(self.physical_u(v_re, v_im), z_m)
        return ObjectiveTerms(pde=float(pde), data=float(data), reg=float(reg))

    def _terms_of_u(self, u: torch.Tensor, z_m: torch.Tensor):
        """Unweighted ``(pde, data, reg)`` terms and the data residual."""
        r_pde, r_data = self._residuals_of_u(u, z_m)
        pde = self.pde_node_factor * mean_real_sq(r_pde / self.pde_scale)
        data = mean_real_sq(r_data / self.data_scale)
        reg = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.reg_weight != 0.0 and self.regulariser is not None:
            reg = self.regulariser(self.latent.c_of_z(z_m))
        return pde, data, reg, r_data

    def _loss(self, pde, data, reg) -> torch.Tensor:
        L = self.pde_weight * pde + self.data_weight * data
        if self.reg_weight != 0.0 and self.regulariser is not None:
            L = L + self.reg_weight * reg
        return L

    # ------------------------------------------------------------------ #
    def _snapshot(self, n_done: int, v_re, v_im, z_m) -> Iterate:
        """The :class:`Iterate` after ``n_done`` outer iterations; the
        evaluation is not counted as a closure.

        Gradients are taken on fresh leaves in ``u_tilde`` and ``z_m``, so
        they are comparable across runs with and without either
        preconditioner.
        """
        z_m = z_m.detach()
        with torch.no_grad():
            u_tilde = self.wavefield_precond.u_tilde_of_v(v_re.detach(), v_im.detach())
        a = u_tilde.real.clone().requires_grad_(True)
        b = u_tilde.imag.clone().requires_grad_(True)
        z = z_m.clone().requires_grad_(True)
        pde, data, reg, r_data = self._terms_of_u(self.u_ref * torch.complex(a, b), z)
        L = self._loss(pde, data, reg)
        ga, gb, gz = torch.autograd.grad(L, (a, b, z))
        with torch.no_grad():
            s = torch.sigmoid(self.latent.z_scale * z_m)
            return Iterate(
                iter=n_done,
                n_evaluations=self._n_eval,
                loss=float(L),
                pde_term=float(self.pde_weight * pde),
                data_term=float(self.data_weight * data),
                reg_term=float(self.reg_weight * reg),
                pde_rel_residual=math.sqrt(2.0 * float(pde)),
                rel_data_residual=float(torch.linalg.vector_norm(r_data))
                / max(self.obs_norm, 1e-30),
                grad_u_norm=math.sqrt(float(ga.pow(2).sum() + gb.pow(2).sum())),
                grad_z_norm=float(gz.norm()),
                sat_frac=float(((s < 1e-3) | (s > 1.0 - 1e-3)).float().mean()),
                c_full=self.c_full_of_z(z_m),
                u=self.physical_u(v_re.detach(), v_im.detach()),
            )

    def minimise(self, on_log: Optional[Callable[[Iterate], None]] = None) -> BandSolve:
        """Run ``n_iter`` outer L-BFGS steps.

        ``on_log(iterate)`` receives an :class:`Iterate` at the start (before
        any update) and at every logged point: each ``log_every``-th outer
        iteration and always the final iterate.
        """
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
            L = self.objective(v_re, v_im, z_of())
            L.backward()
            return L

        reports = on_log is not None or self.verbose
        if on_log is not None:
            on_log(self._snapshot(0, v_re, v_im, z_of()))
        for i in range(self.n_iter):
            opt.step(closure)

            n_done = i + 1
            if not reports or (n_done % self.log_every != 0 and n_done != self.n_iter):
                continue
            it = self._snapshot(n_done, v_re, v_im, z_of())
            if self.verbose:
                c = it.c_full[self._interior_slice]
                print(
                    f"[joint] iter {n_done:4d} | Phi={it.loss:.6e} | "
                    f"pde_rel={it.pde_rel_residual:.3e} | "
                    f"|gu|={it.grad_u_norm:.3e} |gz|={it.grad_z_norm:.3e} | "
                    f"c=[{float(c.min()):.1f},{float(c.max()):.1f}]"
                )
            if on_log is not None:
                on_log(it)

        with torch.no_grad():
            v_re_final, v_im_final = v_re.detach(), v_im.detach()
            z_final = z_of().detach()
            # ``LBFGS.step`` returns the loss at the *start* of the step; report
            # the objective at the final iterate instead (not counted as a closure).
            loss_final = float(self.objective(v_re_final, v_im_final, z_final))
            velocity_model = VelocityModel.from_field(
                self.grid, self.c_full_of_z(z_final)
            )
            u_final = self.physical_u(v_re_final, v_im_final)
        return BandSolve(
            velocity_model=velocity_model,
            u=u_final,
            params=(v_re_final, v_im_final, z_final),
            loss=loss_final,
            n_evaluations=self._n_eval,
        )
