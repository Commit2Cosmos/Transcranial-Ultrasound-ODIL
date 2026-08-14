from abc import ABC, abstractmethod
import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as _F

from odil_wave.loss import DiscreteLoss, ForwardLoss, InverseLoss
from odil_wave.loss.utils import LossTape
from odil_wave.models import VelocityModel
from odil_wave.operator import HelmholtzSolver
from odil_wave.optimisation.helmholtz_utransform import HelmholtzUTransform
from odil_wave.wavefield import Wavefield


def _gaussian_smooth_2d(g: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth a 2-D gradient tensor with a Gaussian kernel (reflect padding).

    Parameters
    ----------
    g : Tensor of shape (Nx, Ny)
    sigma : kernel standard deviation in grid cells

    Returns
    -------
    A new tensor of the same shape as *g*, smoothed.
    """
    if sigma <= 0.0:
        return g
    ks = 2 * int(math.ceil(3.0 * sigma)) + 1
    x = torch.arange(ks, dtype=g.dtype, device=g.device) - ks // 2
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, ks, ks)
    pad = ks // 2
    g_in = _F.pad(g.view(1, 1, *g.shape), (pad, pad, pad, pad), mode="reflect")
    return _F.conv2d(g_in, k2d).view(g.shape)


class StepScheduler:
    """Multiplicative outer-loop scheduler for a single scalar knob.

    ``value`` starts at ``base`` and, when *active*, is rescaled by ``factor``
    (``increase``) or ``1 / factor`` (otherwise) at the start of every outer
    whose index is a positive multiple of ``every_n``. Inactive -- a no-op
    leaving ``value`` at its base -- when ``factor`` or ``every_n`` is 0.
    """

    def __init__(self, base: float, factor: float, every_n: int, increase: bool):
        """Build a scheduler starting at ``base`` (inactive if ``factor`` or ``every_n`` is 0)."""
        self.value = float(base)
        self.factor = float(factor)
        self.every_n = int(every_n)
        self.increase = bool(increase)

    @classmethod
    def from_spec(cls, base: float, spec) -> "StepScheduler":
        """Build from a ``(factor, every_n, direction)`` tuple (direction is
        ``"increase"`` / ``"decrease"``). ``None`` -> an inactive scheduler."""
        if spec is None:
            return cls(base, 0.0, 0, True)
        factor, every_n, direction = spec
        return cls(base, factor, every_n, str(direction).lower() != "decrease")

    @property
    def active(self) -> bool:
        """Whether the scheduler rescales ``value`` at all (``False`` = does nothing)."""
        return self.factor != 0 and self.every_n != 0

    def step(self, outer_i: int) -> float:
        """Advance to outer ``outer_i`` and return the value in effect for it."""
        if self.active and outer_i > 0 and outer_i % self.every_n == 0:
            self.value = (
                self.value * self.factor if self.increase else self.value / self.factor
            )
        return self.value


def _armijo_gd_step_z(
    *,
    z_real: torch.nn.Parameter,
    z_imag: torch.nn.Parameter,
    pack_z,
    apply_u,
    eval_loss,
    loss0: torch.Tensor,
    lr: float,
    armijo_c: float = 1e-4,
    max_backtracks: int = 20,
) -> Tuple[torch.Tensor, int]:
    """One steepest-descent step on complex ``z`` with Armijo backtracking.

    Assumes ``z_real.grad`` / ``z_imag.grad`` are already populated at the
    current point and ``loss0`` is that point's loss. Updates parameters
    in place.

    Returns
    -------
    loss :
        Loss at the accepted point (or ``loss0`` if no step accepted).
    n_trial_evals :
        Number of line-search loss evaluations (excluding ``loss0``).
    """
    g_r = z_real.grad
    g_i = z_imag.grad
    if g_r is None or g_i is None:
        raise RuntimeError("z GD step requires gradients on z_real and z_imag")

    with torch.no_grad():
        g_sq = float(g_r.pow(2).sum() + g_i.pow(2).sum())
        L0 = float(loss0.detach())
        z_r0 = z_real.data.clone()
        z_i0 = z_imag.data.clone()
        step = float(lr)
        n_trial = 0
        for _ in range(max_backtracks):
            z_real.data.copy_(z_r0 - step * g_r)
            z_imag.data.copy_(z_i0 - step * g_i)
            z_trial = pack_z()
            u_trial = apply_u(z_trial)
            L_trial = eval_loss(z_trial, u_trial)
            n_trial += 1
            if float(L_trial.detach()) <= L0 - armijo_c * step * g_sq:
                return L_trial, n_trial
            step *= 0.5
        z_real.data.copy_(z_r0)
        z_imag.data.copy_(z_i0)
        return loss0, n_trial


_C_PARAMS = frozenset({"velocity", "squared_slowness"})

# Numerical safety rails for c_param="squared_slowness" (distinct from the
# user-facing c_min/c_max in _c_param_bounds): clamp the raw m_hat = 1/c_hat^2
# to stop float32 overflow/NaN when a line search trials it near zero
# (c_hat -> +inf via rsqrt). 100x headroom never constrains normal
# optimisation but guards every closure, including line-search trials.
_SQ_SLOWNESS_RATIO_MIN = 1e-2
_SQ_SLOWNESS_RATIO_MAX = 1e2
_SQ_SLOWNESS_PARAM_MIN = _SQ_SLOWNESS_RATIO_MAX**-2  # = 1e-4
_SQ_SLOWNESS_PARAM_MAX = _SQ_SLOWNESS_RATIO_MIN**-2  # = 1e4


def _c_hat_from_c_param(raw: torch.Tensor, c_param: str) -> torch.Tensor:
    """Map the c-block's raw optimisation variable to normalised velocity c_hat.

    ``c_param="velocity"`` (default): ``raw`` already *is* c_hat (identity).

    ``c_param="squared_slowness"``: ``raw`` is m_hat = 1/c_hat^2 (squared
    slowness normalised by ``c_ref``, same as c_hat). ``raw`` is clamped to
    ``[_SQ_SLOWNESS_PARAM_MIN, _SQ_SLOWNESS_PARAM_MAX]`` first (see the
    safety-rail note above), bounding c_hat to
    ``[_SQ_SLOWNESS_RATIO_MIN, _SQ_SLOWNESS_RATIO_MAX]``. Returns
    c_hat = m_hat^(-1/2); autograd differentiates through this transform, so
    gradients w.r.t. ``raw`` are correctly ``dL/dm_hat``.
    """
    if c_param == "squared_slowness":
        raw_safe = raw.clamp(min=_SQ_SLOWNESS_PARAM_MIN, max=_SQ_SLOWNESS_PARAM_MAX)
        return torch.rsqrt(raw_safe)
    return raw


def _init_c_param(c_hat0: torch.Tensor, c_param: str) -> torch.Tensor:
    """Inverse of :func:`_c_hat_from_c_param`: raw parameter from initial c_hat0.

    For ``c_param="squared_slowness"``, ``c_hat0`` is clamped to
    ``[_SQ_SLOWNESS_RATIO_MIN, _SQ_SLOWNESS_RATIO_MAX]`` first, as the same
    numerical safety rail (protects against a near-zero initial c_hat0
    blowing up m_hat0).
    """
    if c_param == "squared_slowness":
        c_hat0_safe = c_hat0.clamp(
            min=_SQ_SLOWNESS_RATIO_MIN, max=_SQ_SLOWNESS_RATIO_MAX
        )
        return torch.reciprocal(c_hat0_safe * c_hat0_safe)
    return c_hat0.clone()


def _c_param_bounds(
    c_min: Optional[float], c_max: Optional[float], c_param: str
) -> Tuple[Optional[float], Optional[float]]:
    """Map c_hat bounds to raw-parameter bounds (identity unless squared_slowness).

    ``m = 1/c_hat^2`` is monotonically *decreasing* in c_hat, so the bounds
    swap: ``m_min = 1/c_max^2``, ``m_max = 1/c_min^2``.
    """
    if c_param != "squared_slowness":
        return c_min, c_max
    m_min = None if c_max is None else 1.0 / (c_max * c_max)
    m_max = None if c_min is None else 1.0 / (c_min * c_min)
    return m_min, m_max


class Optimiser(ABC):
    """Base optimiser class."""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        """Store the wavefield to optimise and the loss to minimise it against."""
        self.loss = loss
        self.wavefield = wavefield

    @abstractmethod
    def minimise(self, **kwargs) -> Tuple[List[Wavefield], LossTape]:
        """Run the optimiser and return the solved wavefields and loss tape."""
        raise NotImplementedError


class LBFGSB(Optimiser):
    """Block-coordinate dual optimiser for frequency-domain ODIL.

    Each outer iteration updates the wavefield block then the ``c`` block
    (``u`` or ``z`` held fixed as appropriate). Complex wavefield unknowns are
    stored as two real tensors. ``c`` is always updated with L-BFGS
    (``c_max_iter``, ``c_lr``, ...). Before each c-block L-BFGS step, its
    gradient is optionally Gaussian-smoothed by ``c_grad_smooth_sigma`` (grid
    cells; ``0`` = off) -- the only gradient smoothing applied to c.

    Wavefield block solver (``u_solve``, direct path only):

    * ``"optim"`` (default) -- one fresh L-BFGS u-block step (``u_steps`` x
      ``max_iter``).
    * ``"exact"`` -- replace the u-block by the exact minimiser
      ``u* = u0 - H^{-1} g(u0)`` of the frozen-c quadratic wavefield subproblem,
      via a direct sparse factorisation of the full Hessian
      ``H = alpha * A^H A + beta * P^H P``
      (:class:`~odil_wave.optimisation.u_block_hessian.UBlockHessian`). No L-BFGS
      runs for the u block. Inverse-only.

    Outer-loop schedulers (``pde_weight_schedule`` /
    ``c_grad_smooth_sigma_schedule``): each is a ``(factor, every_n,
    direction)`` spec (or ``None``) that multiplicatively rescales the ``pde``
    loss weight / the c-gradient smoothing sigma every ``every_n`` outers.
    Inactive (no-op) when ``factor`` or ``every_n`` is 0.

    Wavefield mode (``u_precond``):

    * ``None`` / ``False`` (default) -- direct L-BFGS on physical ``u``
      (``u_steps``, ``max_iter``).
    * ``"z"`` -- reparameterisation ``u = A(c)^{-1} z``. The ``z`` block is
      controlled by:

      - ``z_optim``: ``"gd"`` (default) -- one steepest-descent step with
        Armijo backtracking (initial step ``z_lr``); or ``"lbfgs"`` --
        PyTorch L-BFGS using ``max_iter`` / ``history_size``.
      - ``z_steps``: number of z updates per outer iteration (recommended: 1).
      - ``z_lr``: Armijo initial step when ``z_optim="gd"`` (recommended: 1.0).

      PDE loss is ``mean(|z-f|^2)``; data loss is ``mean(|P A(c)^{-1} z - d|^2)``.
      Recommended weights: ``pde_weight=100``, ``data_weight=1``. Prefer
      ``z_optim="gd"``: with the usual per-outer z-history reset,
      ``z_optim="lbfgs"`` and ``max_iter=1`` is nearly equivalent and does not
      improve recovery.
    """

    _DEFAULT_OPTS = {
        "n_iter": 100,
        "u_steps": 1,
        "c_steps": 1,
        "z_steps": 1,
        "z_optim": "gd",
        "z_lr": 1.0,
        "u_precond": None,
        "max_iter": 4,
        "history_size": 10,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
        "c_lr": 1.0,
        "c_max_iter": 4,
        "c_history_size": 10,
        # c-gradient Gaussian smoothing width in grid cells; 0 disables it
        # (the only c-gradient preconditioner).
        "c_grad_smooth_sigma": 0.0,
        "reset_c_history": True,
        # Wavefield block solver: "optim" (L-BFGS) or "exact" (u* via direct H^-1).
        "u_solve": "optim",
        # Outer-loop schedulers: (factor, every_n, direction) tuples, or None
        # for inactive. Only pde_weight and c_grad_smooth_sigma schedulable.
        "pde_weight_schedule": None,
        "c_grad_smooth_sigma_schedule": None,
        # c-block optimisation variable: "velocity" (default, c_hat itself) or
        # "squared_slowness" (optimise m_hat = 1/c_hat^2).
        "c_param": "velocity",
    }
    _LBFGS_KEYS = frozenset(
        {
            "lr",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search_fn",
        }
    )
    _Z_OPTIMS = frozenset({"gd", "lbfgs"})

    def __init__(
        self,
        wavefield: Wavefield,
        loss: DiscreteLoss,
        clamp: bool = False,
        u_init=None,
        **opts,
    ) -> None:
        """Set up the optimiser; ``clamp`` enables ``c`` bounds, ``opts``
        override the defaults in ``_DEFAULT_OPTS``."""
        super().__init__(wavefield, loss)
        grid = wavefield.grid
        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.u_init = u_init
        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self):
        """Validate ``self.opts`` and unpack it into the individual loop
        settings and the separate LBFGS option dicts for the u/c blocks."""
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        u_steps = int(opts.pop("u_steps", 1))
        c_steps = int(opts.pop("c_steps", 1))
        z_steps = int(opts.pop("z_steps", 1))
        z_optim = str(opts.pop("z_optim", "gd")).lower()
        z_lr = float(opts.pop("z_lr", 1.0))
        u_precond = opts.pop("u_precond", None)
        c_lr = float(opts.pop("c_lr", 1.0))
        c_max_iter = int(opts.pop("c_max_iter", opts.get("max_iter", 4)))
        c_history_size = int(opts.pop("c_history_size", opts.get("history_size", 10)))
        c_grad_smooth_sigma = float(opts.pop("c_grad_smooth_sigma", 0.0))
        u_solve = str(opts.pop("u_solve", "optim")).lower()
        pde_weight_schedule = opts.pop("pde_weight_schedule", None)
        c_grad_smooth_sigma_schedule = opts.pop("c_grad_smooth_sigma_schedule", None)
        reset_c_history = bool(opts.pop("reset_c_history", True))
        c_param = str(opts.pop("c_param", "velocity")).lower()
        # Legacy keys (unused; popped so they don't leak into u_torch_opts).
        opts.pop("u_precond_stab", None)
        for _legacy in (
            "c_precond",
            "c_precond_type",
            "c_precond_sigma",
            "c_precond_stab",
        ):
            opts.pop(_legacy, None)

        if c_param not in _C_PARAMS:
            raise ValueError(
                f"c_param must be one of {sorted(_C_PARAMS)}; got {c_param!r}"
            )
        if u_solve not in ("optim", "exact"):
            raise ValueError(f"u_solve must be 'optim' or 'exact'; got {u_solve!r}")

        if u_precond in (None, False):
            u_precond_mode = None
        elif u_precond == "z":
            u_precond_mode = "z"
        else:
            raise ValueError(
                f"u_precond must be None, False, or 'z'; got {u_precond!r}. "
                "Legacy u_precond=True is no longer supported."
            )
        if u_precond_mode == "z" and z_steps < 1:
            raise ValueError(f"z_steps must be >= 1 when u_precond='z'; got {z_steps}")
        if z_optim not in self._Z_OPTIMS:
            raise ValueError(
                f"z_optim must be one of {sorted(self._Z_OPTIMS)}; got {z_optim!r}"
            )
        if z_lr <= 0.0:
            raise ValueError(f"z_lr must be > 0; got {z_lr}")
        if u_precond_mode == "z" and c_param == "squared_slowness":
            raise ValueError(
                "c_param='squared_slowness' is not supported with u_precond='z' "
                "yet; keep u_precond=None (the direct wavefield block)."
            )
        if u_solve == "exact" and u_precond_mode == "z":
            raise ValueError(
                "u_solve='exact' is incompatible with u_precond='z' "
                "(the exact solve replaces the direct wavefield block)."
            )

        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}
        c_torch_opts = dict(u_torch_opts)
        c_torch_opts["lr"] = c_lr
        c_torch_opts["max_iter"] = c_max_iter
        c_torch_opts["history_size"] = c_history_size
        c_torch_opts["line_search_fn"] = None

        return (
            n_iter,
            u_steps,
            c_steps,
            z_steps,
            z_optim,
            z_lr,
            u_precond_mode,
            u_solve,
            u_torch_opts,
            c_torch_opts,
            c_grad_smooth_sigma,
            pde_weight_schedule,
            c_grad_smooth_sigma_schedule,
            reset_c_history,
            c_param,
        )

    def _seed_complex(self, n_shots, cdtype, device) -> torch.Tensor:
        """``(n_shots, nf, nx, ny)`` complex seed."""
        nf = self.wavefield.n_frequencies
        Nx, Ny = self.wavefield.grid.shape
        if self.u_init is None:
            seed = (
                self.wavefield.amplitude.detach()
                .clone()
                .to(dtype=cdtype, device=device)
            )
            if seed.ndim == 3:
                seed = seed.unsqueeze(0).expand(n_shots, -1, -1, -1)
            return seed.contiguous()
        if isinstance(self.u_init, (list, tuple)):
            stack = torch.stack(
                [
                    w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                    for w in self.u_init
                ]
            )
        else:
            stack = torch.as_tensor(self.u_init)
        stack = stack.detach().to(dtype=cdtype, device=device)
        if stack.ndim == 3:
            stack = stack.unsqueeze(0).expand(n_shots, -1, -1, -1)
        if stack.shape[0] != n_shots:
            raise ValueError(
                f"u_init provides {stack.shape[0]} shots, expected {n_shots}."
            )
        if tuple(stack.shape[1:]) != (nf, Nx, Ny):
            raise ValueError(
                f"u_init shape {tuple(stack.shape)} incompatible with "
                f"(n_shots={n_shots}, nf={nf}, nx={Nx}, ny={Ny})"
            )
        return stack.contiguous()

    def minimise(
        self,
        on_iteration=None,
        diagnostics=None,
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Run wavefield L-BFGS then c-LBFGS block-coordinate loop.

        ``diagnostics`` (LBFGSB direct path only) is an optional duck-typed
        collector; when given, per-outer block-diagnostic snapshots are handed to
        it (``bind`` / ``record_outer`` / ``finalise``) with no effect on the
        trajectory. It stays ``None`` in normal runs, keeping this a no-op.
        """
        self.opts.update(overrides)
        (
            n_iter,
            u_steps,
            c_steps,
            z_steps,
            z_optim,
            z_lr,
            u_precond_mode,
            u_solve,
            u_torch_opts,
            c_torch_opts,
            c_grad_smooth_sigma,
            pde_weight_schedule,
            c_grad_smooth_sigma_schedule,
            reset_c_history,
            c_param,
        ) = self._split_opts()

        if u_precond_mode == "z":
            return self._minimise_z(
                n_iter=n_iter,
                z_steps=z_steps,
                c_steps=c_steps,
                z_optim=z_optim,
                z_lr=z_lr,
                u_torch_opts=u_torch_opts,
                c_torch_opts=c_torch_opts,
                c_grad_smooth_sigma=c_grad_smooth_sigma,
                pde_weight_schedule=pde_weight_schedule,
                c_grad_smooth_sigma_schedule=c_grad_smooth_sigma_schedule,
                reset_c_history=reset_c_history,
                on_iteration=on_iteration,
            )

        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        dtype = grid.dtype
        cdtype = self.wavefield.cdtype
        device = grid.device
        n_shots = self.loss.config.geometry.n_sources

        u_seed = self._seed_complex(n_shots, cdtype, device)
        u_real = torch.nn.Parameter(u_seed.real.contiguous().to(dtype=dtype))
        u_imag = torch.nn.Parameter(u_seed.imag.contiguous().to(dtype=dtype))

        def pack_u() -> torch.Tensor:
            return torch.complex(u_real, u_imag)

        def pack_u_detached() -> torch.Tensor:
            return torch.complex(u_real.detach(), u_imag.detach())

        vm_in = self.wavefield.velocity_model
        vm_c_const = vm_in.c.detach().to(dtype=dtype, device=device)

        is_inverse = isinstance(self.loss, InverseLoss)
        if is_inverse:
            c0_int = vm_c_const[grid.interior_slice].detach().clone()
            c_ref = float(c0_int.mean().item())
            c_hat0 = c0_int / c_ref
            # Raw c-block optimisation variable: c_hat itself (c_param="velocity",
            # default) or m_hat = 1/c_hat^2 (c_param="squared_slowness"). See
            # _c_hat_from_c_param for the (autograd-differentiable) map back to c_hat.
            c_interior_param = torch.nn.Parameter(_init_c_param(c_hat0, c_param))
        else:
            c_ref = None
            c_interior_param = None

        def make_u_optimiser():
            return torch.optim.LBFGS([u_real, u_imag], **u_torch_opts)

        def make_c_optimiser():
            return torch.optim.LBFGS([c_interior_param], **c_torch_opts)

        u_solve_exact = is_inverse and u_solve == "exact"
        u_optimiser = None if u_solve_exact else make_u_optimiser()
        c_optimiser = make_c_optimiser() if (is_inverse and c_steps > 0) else None

        c_min = (
            self.c_min / c_ref
            if (self.c_min is not None and c_ref is not None)
            else None
        )
        c_max = (
            self.c_max / c_ref
            if (self.c_max is not None and c_ref is not None)
            else None
        )
        # The lbfgs c-block clamps the raw parameter, so c_min/c_max are mapped
        # into that same space (identity unless squared_slowness).
        c_param_min, c_param_max = _c_param_bounds(c_min, c_max, c_param)
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        n_u_closure = 0
        n_c_closure = 0
        n_outer_done = 0

        # Outer-loop schedulers. Bases are read from the freshly built loss /
        # c-block sigma so an inactive scheduler re-writes the value it read
        # (exact no-op). ``sigma_box`` carries the sigma the c-closure smooths
        # with, updated once per outer.
        base_pde_w = (
            float(self.loss.config.weights.get("pde", 1.0)) if is_inverse else 1.0
        )
        pde_sched = StepScheduler.from_spec(base_pde_w, pde_weight_schedule)
        sigma_sched = StepScheduler.from_spec(
            float(c_grad_smooth_sigma), c_grad_smooth_sigma_schedule
        )
        sigma_box = [float(c_grad_smooth_sigma)]

        # Block diagnostics (LBFGSB direct path). Duck-typed collector; a no-op
        # when None. bind() supplies the run-time optimiser context; record_init
        # stamps the warm-start state as the outer-0 reference.
        diag_on = diagnostics is not None and is_inverse
        if diag_on:
            diagnostics.bind(
                loss=self.loss,
                wavefield=self.wavefield,
                c_ref=c_ref,
                c_param=c_param,
                build_full_c=vm_in.build_full_c,
                u_torch_opts=dict(u_torch_opts),
                u_steps=u_steps,
                c_torch_opts=dict(c_torch_opts),
                c_steps=c_steps,
                c_param_bounds=(c_param_min, c_param_max),
                u_solve=u_solve,
            )
            diagnostics.record_init(
                c_interior_param.detach().clone(),
                u_real.detach().clone(),
                u_imag.detach().clone(),
            )

        for i in range(n_iter):
            n_outer_done = i + 1

            # Outer-loop schedulers, applied at the start of the outer.
            if is_inverse:
                self.loss.config.weights["pde"] = pde_sched.step(i)
            sigma_box[0] = sigma_sched.step(i)

            if diag_on:
                u_before_re = u_real.detach().clone()
                u_before_im = u_imag.detach().clone()
                c_raw_before = c_interior_param.detach().clone()

            if u_solve_exact:
                loss_value = self._exact_u_block(
                    u_real, u_imag, c_interior_param, c_param, c_ref, vm_in
                )
            else:
                for _ in range(u_steps):

                    def u_closure():
                        nonlocal n_u_closure
                        n_u_closure += 1
                        u_optimiser.zero_grad()
                        amps = pack_u()
                        if is_inverse:
                            # PDE uses physical c; regulariser uses normalised c_hat.
                            c_hat_fixed = _c_hat_from_c_param(
                                c_interior_param.detach(), c_param
                            )
                            c_full = vm_in.build_full_c(c_hat_fixed * c_ref)
                            L = self.loss.evaluate(amps, c_full, c_hat_fixed)
                        else:
                            L = self.loss.evaluate(amps, vm_c_const)
                        L.backward()
                        return L

                    loss_value = u_optimiser.step(u_closure)

            if diag_on:
                u_after_re = u_real.detach().clone()
                u_after_im = u_imag.detach().clone()

            if is_inverse and c_steps > 0:
                if reset_c_history:
                    c_optimiser = make_c_optimiser()

                for _ in range(c_steps):

                    def c_closure():
                        nonlocal n_c_closure
                        n_c_closure += 1
                        c_optimiser.zero_grad()
                        # c_hat = c / c_ref (raw param directly, or via
                        # m_hat=1/c_hat^2 for squared_slowness); convert here
                        # so physics/regulariser see velocity, rescaled by
                        # c_ref below for physical c.
                        c_hat = _c_hat_from_c_param(c_interior_param, c_param)
                        c_phys = c_hat * c_ref
                        c_full = vm_in.build_full_c(c_phys)
                        amps_fixed = pack_u_detached()
                        L_c = self.loss.evaluate(amps_fixed, c_full, c_hat)
                        L_c.backward()
                        _sigma = sigma_box[0]
                        if _sigma > 0.0 and c_interior_param.grad is not None:
                            with torch.no_grad():
                                c_interior_param.grad.copy_(
                                    _gaussian_smooth_2d(c_interior_param.grad, _sigma)
                                )
                        return L_c

                    loss_value = c_optimiser.step(c_closure)
                    with torch.no_grad():
                        if c_param_min is not None or c_param_max is not None:
                            c_interior_param.clamp_(min=c_param_min, max=c_param_max)

                if not u_solve_exact:
                    u_optimiser = make_u_optimiser()

            if diag_on:
                diagnostics.record_outer(
                    i,
                    u_before_re,
                    u_before_im,
                    u_after_re,
                    u_after_im,
                    c_raw_before,
                    c_interior_param.detach().clone(),
                    c_lr=float(c_torch_opts["lr"]),
                    c_grad_smooth_sigma=float(sigma_box[0]),
                )

            should_log = (i % log_every == 0) or (i == n_iter - 1)
            c_full_now = None
            if is_inverse and (should_log or on_iteration is not None):
                c_hat_now = _c_hat_from_c_param(c_interior_param.detach(), c_param)
                c_full_now = vm_in.build_full_c(c_hat_now * c_ref)

            if on_iteration is not None:
                on_iteration(i, c_full_now)

            loss_scalar = (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            )

            if should_log and loss_scalar is not None:
                ratio = self.loss.pde_src_ratio()
                self.loss.callback.log(
                    loss_scalar, self.loss._last_residuals, pde_src_ratio=ratio
                )
                if c_full_now is not None:
                    self.loss.callback.log_c(c_full_now.cpu().numpy())
                print(
                    f"Iteration: {i} | loss = {loss_scalar:.6e} | "
                    f"|r_pde|/|src| = {ratio:.3e}"
                )

        if diag_on:
            diagnostics.finalise()

        if isinstance(self.loss, ForwardLoss):
            vm_out = vm_in
        else:
            c_hat_final = _c_hat_from_c_param(c_interior_param.detach(), c_param)
            c_full_final = vm_in.build_full_c(c_hat_final * c_ref)
            vm_out = VelocityModel.from_field(
                grid, c_full_final, pml_c=vm_in.pml_c, pml_fill=vm_in.pml_fill
            )

        u_final = pack_u_detached()
        outputs: List[Wavefield] = []
        for s in range(n_shots):
            wf = Wavefield(
                grid=grid,
                frequency_selection=freq,
                velocity_model=vm_out,
            )
            wf.amplitude = u_final[s]
            outputs.append(wf)

        self.loss.callback.result = {
            "loss": (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            ),
            "n_outer_iter": n_outer_done,
            "n_outer_budget": n_iter,
            "n_u_closure": n_u_closure,
            "n_c_closure": n_c_closure,
            "n_closure": n_u_closure + n_c_closure,
            "u_precond": None,
            "u_solve": u_solve,
            "c_update": "lbfgs",
            "c_param": c_param,
            "n_factor": 0,
            "n_forward_solves": 0,
            "n_adjoint_solves": 0,
        }
        return outputs, self.loss.callback

    def _exact_u_block(
        self, u_real, u_imag, c_interior_param, c_param, c_ref, vm_in
    ) -> torch.Tensor:
        """Exact frozen-c wavefield block: overwrite ``(u_real, u_imag)`` with
        the quadratic minimiser ``u* = u0 - H^{-1} g(u0)``.

        The frozen-c u-subproblem ``J(u) = pde_weight*mean|A(c)u-f|^2 +
        data_weight*mean|Pu-d|^2`` is exactly quadratic in ``u`` with constant
        Hessian ``H = alpha * A^H A + beta * P^H P``, independent of the warm
        start. ``H`` is assembled and factorised by :class:`UBlockHessian` at
        the run's current loss weights (so a scheduled ``pde`` weight is
        honoured). Returns the loss at ``u*``.
        """
        from odil_wave.optimisation.u_block_hessian import UBlockHessian

        c_hat_fixed = _c_hat_from_c_param(c_interior_param.detach(), c_param)
        c_full = vm_in.build_full_c(c_hat_fixed * c_ref)
        a = u_real.detach().clone().requires_grad_(True)
        b = u_imag.detach().clone().requires_grad_(True)
        L = self.loss.evaluate(torch.complex(a, b), c_full, c_hat_fixed)
        ga, gb = torch.autograd.grad(L, (a, b))
        g = torch.complex(ga.detach(), gb.detach())  # total weighted u-gradient at u0
        ubh = UBlockHessian(
            self.wavefield, self.loss, c_full, weights=dict(self.loss.config.weights)
        )
        u0 = torch.complex(a.detach(), b.detach())
        u_star = u0 + ubh.du_from_grad(g)  # u0 - H^{-1} g(u0)
        with torch.no_grad():
            u_real.copy_(u_star.real)
            u_imag.copy_(u_star.imag)
        return self.loss.evaluate(
            torch.complex(u_real.detach(), u_imag.detach()), c_full, c_hat_fixed
        )

    def _minimise_z(
        self,
        *,
        n_iter: int,
        z_steps: int,
        c_steps: int,
        z_optim: str,
        z_lr: float,
        u_torch_opts: dict,
        c_torch_opts: dict,
        c_grad_smooth_sigma: float,
        pde_weight_schedule,
        c_grad_smooth_sigma_schedule,
        reset_c_history: bool,
        on_iteration=None,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Block-coordinate loop with ``u = A(c)^{-1} z`` (``u_precond='z'``).

        The c-block is always L-BFGS here. The outer-loop schedulers (``pde``
        weight and ``c_grad_smooth_sigma``) apply here too; the exact u-solve
        and block diagnostics do not (they are direct-path only).
        """
        if not isinstance(self.loss, InverseLoss):
            raise TypeError("u_precond='z' requires an InverseLoss")

        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        dtype = grid.dtype
        cdtype = self.wavefield.cdtype
        device = grid.device
        n_shots = self.loss.config.geometry.n_sources
        vm_in = self.wavefield.velocity_model
        wave_eq = self.loss.config.wave_eq

        u_seed = self._seed_complex(n_shots, cdtype, device)
        c0_int = vm_in.c[grid.interior_slice].detach().clone().to(device=device)
        c_ref = float(c0_int.mean().item())
        c_interior_param = torch.nn.Parameter(c0_int / c_ref)

        def c_full_from_hat(c_hat: torch.Tensor) -> torch.Tensor:
            return vm_in.build_full_c(c_hat * c_ref)

        c_full0 = c_full_from_hat(c_interior_param.detach())
        helm = HelmholtzSolver(
            self.wavefield,
            self.loss.config.geometry,
            space_order=wave_eq.space_order,
            pml_weight=wave_eq.pml_weight,
        )
        tf = HelmholtzUTransform(helm, c_full0)
        z0 = tf.inverse(u_seed)
        z_real = torch.nn.Parameter(z0.real.contiguous().to(dtype=dtype))
        z_imag = torch.nn.Parameter(z0.imag.contiguous().to(dtype=dtype))

        def pack_z() -> torch.Tensor:
            return torch.complex(z_real, z_imag)

        def pack_z_detached() -> torch.Tensor:
            return torch.complex(z_real.detach(), z_imag.detach())

        def make_z_optimiser():
            return torch.optim.LBFGS([z_real, z_imag], **u_torch_opts)

        def make_c_optimiser():
            return torch.optim.LBFGS([c_interior_param], **c_torch_opts)

        z_optimiser = make_z_optimiser() if z_optim == "lbfgs" else None
        c_optimiser = make_c_optimiser() if c_steps > 0 else None

        c_min = self.c_min / c_ref if self.c_min is not None else None
        c_max = self.c_max / c_ref if self.c_max is not None else None
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        n_u_closure = 0
        n_c_closure = 0
        n_outer_done = 0
        t_wall0 = time.perf_counter()

        # Outer-loop schedulers (bases from the freshly built loss / c-block sigma).
        pde_sched = StepScheduler.from_spec(
            float(self.loss.config.weights.get("pde", 1.0)), pde_weight_schedule
        )
        sigma_sched = StepScheduler.from_spec(
            float(c_grad_smooth_sigma), c_grad_smooth_sigma_schedule
        )
        sigma_box = [float(c_grad_smooth_sigma)]

        for i in range(n_iter):
            n_outer_done = i + 1

            # Outer-loop schedulers, applied at the start of the outer.
            self.loss.config.weights["pde"] = pde_sched.step(i)
            sigma_box[0] = sigma_sched.step(i)

            c_hat_fixed = c_interior_param.detach()
            c_full_fixed = c_full_from_hat(c_hat_fixed)

            for _ in range(z_steps):
                if z_optim == "gd":
                    if z_real.grad is not None:
                        z_real.grad = None
                    if z_imag.grad is not None:
                        z_imag.grad = None
                    z = pack_z()
                    u = tf.apply(z)
                    L = self.loss.evaluate_z(z, u, c_full_fixed, c_hat_fixed)
                    L.backward()
                    n_u_closure += 1

                    def _eval_z(z_t, u_t):
                        return self.loss.evaluate_z(z_t, u_t, c_full_fixed, c_hat_fixed)

                    loss_value, n_trial = _armijo_gd_step_z(
                        z_real=z_real,
                        z_imag=z_imag,
                        pack_z=pack_z,
                        apply_u=tf.apply,
                        eval_loss=_eval_z,
                        loss0=L,
                        lr=z_lr,
                    )
                    n_u_closure += n_trial
                else:

                    def z_closure():
                        nonlocal n_u_closure
                        n_u_closure += 1
                        z_optimiser.zero_grad()
                        z = pack_z()
                        u = tf.apply(z)
                        L = self.loss.evaluate_z(z, u, c_full_fixed, c_hat_fixed)
                        L.backward()
                        return L

                    loss_value = z_optimiser.step(z_closure)

            if c_steps > 0:
                z_fixed = pack_z_detached()
                if reset_c_history:
                    c_optimiser = make_c_optimiser()

                for _ in range(c_steps):

                    def c_closure():
                        nonlocal n_c_closure
                        n_c_closure += 1
                        c_optimiser.zero_grad()
                        c_full = c_full_from_hat(c_interior_param)
                        # Freeze z; recompute u = A(c)^{-1} z (do not detach u).
                        u = tf.apply_diff_c(z_fixed, c_full)
                        L_c = self.loss.evaluate_z(z_fixed, u, c_full, c_interior_param)
                        L_c.backward()
                        _sigma = sigma_box[0]
                        if _sigma > 0.0 and c_interior_param.grad is not None:
                            with torch.no_grad():
                                c_interior_param.grad.copy_(
                                    _gaussian_smooth_2d(c_interior_param.grad, _sigma)
                                )
                        return L_c

                    loss_value = c_optimiser.step(c_closure)
                    with torch.no_grad():
                        if c_min is not None or c_max is not None:
                            c_interior_param.clamp_(min=c_min, max=c_max)

                # c changed -- rebuild factors for next z-stage; reset z history.
                with torch.no_grad():
                    tf.rebuild(c_full_from_hat(c_interior_param.detach()))
                if z_optim == "lbfgs":
                    z_optimiser = make_z_optimiser()

            should_log = (i % log_every == 0) or (i == n_iter - 1)
            c_full_now = None
            if should_log or on_iteration is not None:
                c_full_now = c_full_from_hat(c_interior_param.detach())

            if on_iteration is not None:
                on_iteration(i, c_full_now)

            loss_scalar = (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            )

            if should_log and loss_scalar is not None:
                ratio = self.loss.pde_src_ratio()
                self.loss.callback.log(
                    loss_scalar, self.loss._last_residuals, pde_src_ratio=ratio
                )
                if c_full_now is not None:
                    self.loss.callback.log_c(c_full_now.cpu().numpy())
                print(
                    f"Iteration: {i} | loss = {loss_scalar:.6e} | "
                    f"|r_pde|/|src| = {ratio:.3e}"
                )

        wall_s = time.perf_counter() - t_wall0
        c_full_final = c_full_from_hat(c_interior_param.detach())
        vm_out = VelocityModel.from_field(
            grid, c_full_final, pml_c=vm_in.pml_c, pml_fill=vm_in.pml_fill
        )
        with torch.no_grad():
            u_final = tf.cache.solve(pack_z_detached(), trans="N")
        d = tf.diagnostics()
        last = getattr(self.loss, "_last_residuals", {}) or {}

        outputs: List[Wavefield] = []
        for s in range(n_shots):
            wf = Wavefield(
                grid=grid,
                frequency_selection=freq,
                velocity_model=vm_out,
            )
            wf.amplitude = u_final[s]
            outputs.append(wf)

        self.loss.callback.result = {
            "loss": (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            ),
            "n_outer_iter": n_outer_done,
            "n_outer_budget": n_iter,
            "n_u_closure": n_u_closure,
            "n_c_closure": n_c_closure,
            "n_closure": n_u_closure + n_c_closure,
            "u_precond": "z",
            "c_update": "lbfgs",
            "z_steps": z_steps,
            "z_optim": z_optim,
            "z_lr": z_lr,
            "n_factor": d["n_factor"],
            "n_forward_solves": d["n_forward_solves"],
            "n_adjoint_solves": d["n_adjoint_solves"],
            "wall_s": wall_s,
            "pde_loss": last.get("pde_loss"),
            "data_loss": last.get("data_loss"),
        }
        return outputs, self.loss.callback
