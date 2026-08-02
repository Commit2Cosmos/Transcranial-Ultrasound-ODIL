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
    Tensor of same shape as *g*, smoothed in-place copy.
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


class Optimiser(ABC):
    """Base optimiser class."""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        self.loss = loss
        self.wavefield = wavefield

    @abstractmethod
    def minimise(self, **kwargs) -> Tuple[List[Wavefield], LossTape]:
        raise NotImplementedError


class LBFGSB(Optimiser):
    """Block-coordinate dual optimiser for frequency-domain ODIL.

    Each outer iteration updates the wavefield block then the ``c`` block
    (``u`` or ``z`` held fixed as appropriate). Complex wavefield unknowns are
    stored as two real tensors. ``c`` is always updated with L-BFGS
    (``c_max_iter``, ``c_lr``, …).

    Wavefield mode (``u_precond``):

    * ``None`` / ``False`` (default) — direct L-BFGS on physical ``u``
      (``u_steps``, ``max_iter``).
    * ``"z"`` — reparameterisation ``u = A(c)^{-1} z``. The ``z`` block is
      controlled by:

      - ``z_optim``: ``"gd"`` (default) — one steepest-descent step with
        Armijo backtracking (initial step ``z_lr``); or ``"lbfgs"`` —
        PyTorch L-BFGS using ``max_iter`` / ``history_size``.
      - ``z_steps``: number of z updates per outer iteration (recommended: 1).
      - ``z_lr``: Armijo initial step when ``z_optim="gd"`` (recommended: 1.0).

      PDE loss is ``mean(|z-f|^2)``; data loss is ``mean(|P A(c)^{-1} z - d|^2)``.
      Recommended weights (caller-side): ``w_pde=100``, ``w_data=1``.
      Prefer ``z_optim="gd"``: with the usual per-outer z-history reset,
      ``z_optim="lbfgs"`` and ``max_iter=1`` is nearly the same as one
      steepest-descent + Wolfe step and does not improve recovery.

    The closed-form c-update (``c_update="closed_form"``) requires the direct
    wavefield block and is rejected with ``u_precond="z"``: the reparameterisation
    slaves ``u = A(c)^{-1} z`` so the PDE residual it projects against is ``z - f``,
    which the z-block drives to ~0 (c would never move). Use ``c_update="lbfgs"``
    with ``u_precond="z"``, or the direct block (``u_precond=None``) for closed form.

    Optional early stopping (disabled by default: ``early_stop_rtol=0``):
    after ``early_stop_min_iter`` outer steps, stop if relative loss improvement
    stays below ``early_stop_rtol`` for ``early_stop_patience`` consecutive
    outer iterations. Existing call sites are unchanged unless these kwargs
    are set explicitly.
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
        "c_precond": False,
        "c_precond_type": "energy",
        "c_precond_sigma": 2.0,
        "c_precond_stab": 1e-2,
        "reset_c_history": True,
        # c-block rule: "lbfgs" (default) or "closed_form" variable projection.
        "c_update": "lbfgs",
        "c_update_every": 1,
        "c_relax": 1.0,
        "illum_rel_floor": 1e-6,
        # Optional; early_stop_rtol <= 0 disables (default).
        "early_stop_rtol": 0.0,
        "early_stop_min_iter": 0,
        "early_stop_patience": 3,
        # Print / store per-outer [grad_c] diagnostics (off by default).
        "debug_c": False,
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
        free_mask=None,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        grid = wavefield.grid
        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.u_init = u_init
        self.free_mask = free_mask
        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self):
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
        c_precond = bool(opts.pop("c_precond", False))
        c_precond_type = str(opts.pop("c_precond_type", "energy"))
        c_precond_sigma = float(opts.pop("c_precond_sigma", 2.0))
        c_precond_stab = float(opts.pop("c_precond_stab", 1e-2))
        reset_c_history = bool(opts.pop("reset_c_history", True))
        c_update = str(opts.pop("c_update", "lbfgs")).lower()
        c_update_every = int(opts.pop("c_update_every", 1))
        c_relax = float(opts.pop("c_relax", 1.0))
        illum_rel_floor = float(opts.pop("illum_rel_floor", 1e-6))
        early_stop_rtol = float(opts.pop("early_stop_rtol", 0.0))
        early_stop_min_iter = int(opts.pop("early_stop_min_iter", 0))
        early_stop_patience = int(opts.pop("early_stop_patience", 3))
        debug_c = bool(opts.pop("debug_c", False))
        # Legacy time-domain stab key (unused).
        opts.pop("u_precond_stab", None)

        if c_update not in ("lbfgs", "closed_form"):
            raise ValueError(
                f"c_update must be 'lbfgs' or 'closed_form'; got {c_update!r}"
            )

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
        if u_precond_mode == "z" and c_update == "closed_form":
            raise ValueError(
                "u_precond='z' is incompatible with c_update='closed_form'. "
                "The closed-form (variable-projection) c-update minimises the "
                "PDE residual of a free wavefield, but the z-reparameterisation "
                "slaves u = A(c)^-1 z, so that residual is z - f — which the "
                "z-block drives to ~0, leaving c frozen. Use the direct "
                "wavefield block (u_precond=None, e.g. the 'cf' optimiser) for "
                "closed-form c, or keep u_precond='z' with c_update='lbfgs'."
            )

        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}
        c_torch_opts = dict(u_torch_opts)
        c_torch_opts["lr"] = c_lr
        c_torch_opts["max_iter"] = c_max_iter
        c_torch_opts["history_size"] = c_history_size

        return (
            n_iter,
            u_steps,
            c_steps,
            z_steps,
            z_optim,
            z_lr,
            u_precond_mode,
            u_torch_opts,
            c_torch_opts,
            c_precond,
            c_precond_type,
            c_precond_sigma,
            c_precond_stab,
            reset_c_history,
            c_update,
            c_update_every,
            c_relax,
            illum_rel_floor,
            early_stop_rtol,
            early_stop_min_iter,
            early_stop_patience,
            debug_c,
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

    def _prox_regularise(
        self, c_star: torch.Tensor, illum: torch.Tensor
    ) -> torch.Tensor:
        """Illumination-weighted proximal application of the loss regulariser.

        Solves ``min_c 0.5 Σ w (c - c*)² + λ R(c)`` on the interior map with
        ``w = illum / mean(illum)``. No-op without a configured regulariser or a
        non-positive ``reg`` weight. Used only by the closed-form c-update and
        operating on **physical** ``c`` (the L-BFGS c-block, by contrast,
        regularises the normalised ĉ). :class:`MODILInversion` carries a
        matching proximal step.
        """
        reg = self.loss.config.regulariser
        lam = float(self.loss.config.weights.get("reg", 0.0))
        if reg is None or lam <= 0.0:
            return c_star
        w = illum / illum.mean().clamp(min=1e-30)
        c = c_star.detach().clone().requires_grad_(True)
        prox_opt = torch.optim.LBFGS(
            [c], max_iter=50, history_size=10, line_search_fn="strong_wolfe"
        )

        def prox_closure():
            with torch.enable_grad():
                prox_opt.zero_grad()
                F = 0.5 * (w * (c - c_star) ** 2).sum() + lam * reg(c)
                F.backward()
            return F

        prox_opt.step(prox_closure)
        return c.detach()

    def _closed_form_c_update(
        self,
        *,
        amps_fixed: torch.Tensor,
        c_interior_param: torch.nn.Parameter,
        c_ref: float,
        vm_in,
        grid,
        wave_eq,
        c_min,
        c_max,
        free_mask,
        c_frozen_init,
        c_relax: float,
        illum_rel_floor: float,
    ) -> None:
        """Write one closed-form variable-projection c-update into ``c_interior_param``.

        ``amps_fixed`` is the (detached) wavefield ``u``; the update is the exact
        per-cell argmin of the PDE misfit over ``c`` for that ``u`` (see
        :meth:`odil_wave.operator.utils.WaveEquation.c_closed_form`) — no step
        size, preconditioner or line search to tune. It is optionally relaxed
        (``c ← (1-α) c + α c*``), regularised (a proximal step) and clamped to
        ``[c_min, c_max]``. The result honours this optimiser's ĉ = c/c_ref
        normalisation and any frozen (masked) cells; ``c_min``/``c_max`` are in
        ĉ units. Replaces the L-BFGS c-block when ``c_update='closed_form'``.
        """
        with torch.no_grad():
            c_phys_int = c_interior_param.detach() * c_ref
            c_full_cur = vm_in.build_full_c(c_phys_int)
            c_star_full, illum_full = wave_eq.c_closed_form(
                amps_fixed,
                self.loss.sources,
                c_current=c_full_cur,
                illum_rel_floor=illum_rel_floor,
            )
            c_star = c_star_full[grid.interior_slice]
            c_star = self._prox_regularise(c_star, illum_full[grid.interior_slice])
            if c_relax != 1.0:
                c_star = (1.0 - c_relax) * c_phys_int + c_relax * c_star
            c_hat = c_star / c_ref
            if free_mask is not None:
                c_hat = c_hat.clone()
                c_hat[~free_mask] = c_frozen_init
            if c_min is not None or c_max is not None:
                c_hat = c_hat.clamp(min=c_min, max=c_max)
            c_interior_param.data.copy_(c_hat)

    def minimise(
        self,
        on_iteration=None,
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Run wavefield L-BFGS then c-LBFGS block-coordinate loop."""
        self.opts.update(overrides)
        (
            n_iter,
            u_steps,
            c_steps,
            z_steps,
            z_optim,
            z_lr,
            u_precond_mode,
            u_torch_opts,
            c_torch_opts,
            c_precond,
            c_precond_type,
            c_precond_sigma,
            c_precond_stab,
            reset_c_history,
            c_update,
            c_update_every,
            c_relax,
            illum_rel_floor,
            early_stop_rtol,
            early_stop_min_iter,
            early_stop_patience,
            debug_c,
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
                c_precond=c_precond,
                c_precond_type=c_precond_type,
                c_precond_sigma=c_precond_sigma,
                c_precond_stab=c_precond_stab,
                reset_c_history=reset_c_history,
                c_update=c_update,
                early_stop_rtol=early_stop_rtol,
                early_stop_min_iter=early_stop_min_iter,
                early_stop_patience=early_stop_patience,
                debug_c=debug_c,
                on_iteration=on_iteration,
            )

        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        dtype = grid.dtype
        cdtype = self.wavefield.cdtype
        device = grid.device
        n_shots = self.loss.config.geometry.n_sources
        wave_eq = self.loss.config.wave_eq
        closed_form_c = c_update == "closed_form"

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
            c_interior_param = torch.nn.Parameter(c0_int / c_ref)
            if self.free_mask is not None:
                _free_mask = self.free_mask.to(dtype=torch.bool, device=device)
                _c_frozen_init = (c0_int / c_ref)[~_free_mask].clone().detach()
            else:
                _free_mask = None
                _c_frozen_init = None
        else:
            c_ref = None
            _free_mask = None
            _c_frozen_init = None
            c_interior_param = None

        def make_u_optimiser():
            return torch.optim.LBFGS([u_real, u_imag], **u_torch_opts)

        def make_c_optimiser():
            return torch.optim.LBFGS([c_interior_param], **c_torch_opts)

        u_optimiser = make_u_optimiser()
        c_optimiser = (
            make_c_optimiser()
            if (is_inverse and c_steps > 0 and not closed_form_c)
            else None
        )

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
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        _prec: Optional[torch.Tensor] = None
        _eps_prec: float = 0.0
        n_u_closure = 0
        n_c_closure = 0
        prev_loss: Optional[float] = None
        stall_count = 0
        stopped_early = False
        n_outer_done = 0

        for i in range(n_iter):
            n_outer_done = i + 1
            for _ in range(u_steps):

                def u_closure():
                    nonlocal n_u_closure
                    n_u_closure += 1
                    u_optimiser.zero_grad()
                    amps = pack_u()
                    if is_inverse:
                        # PDE uses physical c; regulariser uses normalised ĉ.
                        c_hat_fixed = c_interior_param.detach()
                        c_full = vm_in.build_full_c(c_hat_fixed * c_ref)
                        L = self.loss.evaluate(amps, c_full, c_hat_fixed)
                    else:
                        L = self.loss.evaluate(amps, vm_c_const)
                    L.backward()
                    return L

                loss_value = u_optimiser.step(u_closure)

            if is_inverse and closed_form_c:
                if c_update_every > 0 and (i + 1) % c_update_every == 0:
                    self._closed_form_c_update(
                        amps_fixed=pack_u_detached(),
                        c_interior_param=c_interior_param,
                        c_ref=c_ref,
                        vm_in=vm_in,
                        grid=grid,
                        wave_eq=wave_eq,
                        c_min=c_min,
                        c_max=c_max,
                        free_mask=_free_mask,
                        c_frozen_init=_c_frozen_init,
                        c_relax=c_relax,
                        illum_rel_floor=illum_rel_floor,
                    )
                    with torch.no_grad():
                        c_full_new = vm_in.build_full_c(
                            c_interior_param.detach() * c_ref
                        )
                        loss_value = self.loss.evaluate(
                            pack_u_detached(), c_full_new, c_interior_param.detach()
                        )
                    # c changed — reset u LBFGS history
                    u_optimiser = make_u_optimiser()

            elif is_inverse and c_steps > 0:
                if c_precond:
                    with torch.no_grad():
                        u_sq = pack_u_detached().abs().square().mean(dim=(0, 1))
                        _prec = u_sq[grid.interior_slice]
                        _eps_prec = c_precond_stab * float(_prec.max().clamp(min=1e-30))

                if reset_c_history:
                    c_optimiser = make_c_optimiser()

                logged_grad_c = False
                for _ in range(c_steps):

                    def c_closure():
                        nonlocal logged_grad_c, n_c_closure
                        n_c_closure += 1
                        c_optimiser.zero_grad()
                        # Optimise ĉ = c / c_ref; PDE sees c = ĉ c_ref.
                        # Pass ĉ into evaluate so Tikhonov is scale-stable.
                        c_phys = c_interior_param * c_ref
                        c_full = vm_in.build_full_c(c_phys)
                        amps_fixed = pack_u_detached()
                        L_c = self.loss.evaluate(amps_fixed, c_full, c_interior_param)
                        L_c.backward()
                        if _free_mask is not None and c_interior_param.grad is not None:
                            c_interior_param.grad[~_free_mask] = 0.0
                        if (
                            debug_c
                            and c_interior_param.grad is not None
                            and not logged_grad_c
                        ):
                            with torch.no_grad():
                                g = c_interior_param.grad
                                n = g.numel()
                                pct_pos = 100.0 * float((g > 0).sum()) / n
                                pct_neg = 100.0 * float((g < 0).sum()) / n
                                g_mean = float(g.mean())
                                g_min = float(g.min())
                                g_max = float(g.max())
                                print(
                                    f"    [grad_c] mean={g_mean:+.3e}  "
                                    f"min={g_min:+.3e}  max={g_max:+.3e}  "
                                    f"pos={pct_pos:.1f}%  neg={pct_neg:.1f}%"
                                )
                                hist = self.loss.callback.history
                                hist.setdefault("grad_c_mean", []).append(g_mean)
                                hist.setdefault("grad_c_min", []).append(g_min)
                                hist.setdefault("grad_c_max", []).append(g_max)
                                hist.setdefault("grad_c_pct_pos", []).append(pct_pos)
                                hist.setdefault("grad_c_pct_neg", []).append(pct_neg)
                                # Interior ĉ-gradient map (physical ∂L/∂c = g / c_ref)
                                hist.setdefault("grad_c_maps", []).append(
                                    g.detach().cpu().clone()
                                )
                                logged_grad_c = True
                        elif not debug_c:
                            logged_grad_c = True
                        if c_precond and c_interior_param.grad is not None:
                            with torch.no_grad():
                                if c_precond_type == "gaussian":
                                    g_smooth = _gaussian_smooth_2d(
                                        c_interior_param.grad, c_precond_sigma
                                    )
                                    c_interior_param.grad.copy_(g_smooth)
                                elif _prec is not None:
                                    c_interior_param.grad.div_(_prec + _eps_prec)
                        return L_c

                    loss_value = c_optimiser.step(c_closure)
                    with torch.no_grad():
                        if _free_mask is not None:
                            c_interior_param.data[~_free_mask] = _c_frozen_init
                        if c_min is not None or c_max is not None:
                            c_interior_param.clamp_(min=c_min, max=c_max)

                # c changed — reset u LBFGS history
                u_optimiser = make_u_optimiser()

            should_log = (i % log_every == 0) or (i == n_iter - 1)
            c_full_now = None
            if is_inverse and (should_log or on_iteration is not None):
                c_full_now = vm_in.build_full_c(c_interior_param.detach() * c_ref)

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

            # Relative loss improvement early stop (disabled when rtol <= 0).
            if (
                early_stop_rtol > 0.0
                and loss_scalar is not None
                and prev_loss is not None
            ):
                denom = max(abs(prev_loss), 1e-30)
                rel_improve = (prev_loss - loss_scalar) / denom
                if rel_improve < early_stop_rtol:
                    stall_count += 1
                else:
                    stall_count = 0
                if (
                    n_outer_done >= early_stop_min_iter
                    and stall_count >= early_stop_patience
                ):
                    stopped_early = True
                    print(
                        f"  early stop at outer iter {i}: "
                        f"rel_improve < {early_stop_rtol:g} for "
                        f"{early_stop_patience} iters "
                        f"(min_iter={early_stop_min_iter})"
                    )
                    break
            if loss_scalar is not None:
                prev_loss = loss_scalar

        if isinstance(self.loss, ForwardLoss):
            vm_out = vm_in
        else:
            c_full_final = vm_in.build_full_c(c_interior_param.detach() * c_ref)
            vm_out = VelocityModel.from_field(grid, c_full_final, pml_c=vm_in.pml_c)

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
            "stopped_early": stopped_early,
            "n_u_closure": n_u_closure,
            "n_c_closure": n_c_closure,
            "n_closure": n_u_closure + n_c_closure,
            "u_precond": None,
            "c_update": c_update,
            "n_factor": 0,
            "n_forward_solves": 0,
            "n_adjoint_solves": 0,
        }
        return outputs, self.loss.callback

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
        c_precond: bool,
        c_precond_type: str,
        c_precond_sigma: float,
        c_precond_stab: float,
        reset_c_history: bool,
        c_update: str,
        early_stop_rtol: float,
        early_stop_min_iter: int,
        early_stop_patience: int,
        debug_c: bool,
        on_iteration=None,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Block-coordinate loop with ``u = A(c)^{-1} z`` (``u_precond='z'``).

        The c-block is always L-BFGS here: ``c_update='closed_form'`` is rejected
        upstream (see :meth:`_split_opts`) because the reparameterisation zeroes
        the PDE residual the closed form projects against. ``c_update`` is kept
        only to record the mode in the result payload.
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
        if self.free_mask is not None:
            _free_mask = self.free_mask.to(dtype=torch.bool, device=device)
            _c_frozen_init = (c0_int / c_ref)[~_free_mask].clone().detach()
        else:
            _free_mask = None
            _c_frozen_init = None

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
        _prec: Optional[torch.Tensor] = None
        _eps_prec: float = 0.0
        n_u_closure = 0
        n_c_closure = 0
        prev_loss: Optional[float] = None
        stall_count = 0
        stopped_early = False
        n_outer_done = 0
        t_wall0 = time.perf_counter()

        for i in range(n_iter):
            n_outer_done = i + 1
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
                if c_precond:
                    with torch.no_grad():
                        u_sq = tf.apply(z_fixed).abs().square().mean(dim=(0, 1))
                        _prec = u_sq[grid.interior_slice]
                        _eps_prec = c_precond_stab * float(_prec.max().clamp(min=1e-30))

                if reset_c_history:
                    c_optimiser = make_c_optimiser()

                logged_grad_c = False
                for _ in range(c_steps):

                    def c_closure():
                        nonlocal logged_grad_c, n_c_closure
                        n_c_closure += 1
                        c_optimiser.zero_grad()
                        c_full = c_full_from_hat(c_interior_param)
                        # Freeze z; recompute u = A(c)^{-1} z (do not detach u).
                        u = tf.apply_diff_c(z_fixed, c_full)
                        L_c = self.loss.evaluate_z(z_fixed, u, c_full, c_interior_param)
                        L_c.backward()
                        if _free_mask is not None and c_interior_param.grad is not None:
                            c_interior_param.grad[~_free_mask] = 0.0
                        if (
                            debug_c
                            and c_interior_param.grad is not None
                            and not logged_grad_c
                        ):
                            with torch.no_grad():
                                g = c_interior_param.grad
                                n = g.numel()
                                pct_pos = 100.0 * float((g > 0).sum()) / n
                                pct_neg = 100.0 * float((g < 0).sum()) / n
                                g_mean = float(g.mean())
                                g_min = float(g.min())
                                g_max = float(g.max())
                                print(
                                    f"    [grad_c] mean={g_mean:+.3e}  "
                                    f"min={g_min:+.3e}  max={g_max:+.3e}  "
                                    f"pos={pct_pos:.1f}%  neg={pct_neg:.1f}%"
                                )
                                hist = self.loss.callback.history
                                hist.setdefault("grad_c_mean", []).append(g_mean)
                                hist.setdefault("grad_c_min", []).append(g_min)
                                hist.setdefault("grad_c_max", []).append(g_max)
                                hist.setdefault("grad_c_pct_pos", []).append(pct_pos)
                                hist.setdefault("grad_c_pct_neg", []).append(pct_neg)
                                hist.setdefault("grad_c_maps", []).append(
                                    g.detach().cpu().clone()
                                )
                                logged_grad_c = True
                        elif not debug_c:
                            logged_grad_c = True
                        if c_precond and c_interior_param.grad is not None:
                            with torch.no_grad():
                                if c_precond_type == "gaussian":
                                    g_smooth = _gaussian_smooth_2d(
                                        c_interior_param.grad, c_precond_sigma
                                    )
                                    c_interior_param.grad.copy_(g_smooth)
                                elif _prec is not None:
                                    c_interior_param.grad.div_(_prec + _eps_prec)
                        return L_c

                    loss_value = c_optimiser.step(c_closure)
                    with torch.no_grad():
                        if _free_mask is not None:
                            c_interior_param.data[~_free_mask] = _c_frozen_init
                        if c_min is not None or c_max is not None:
                            c_interior_param.clamp_(min=c_min, max=c_max)

                # c changed — rebuild factors for next z-stage; reset z history.
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

            if (
                early_stop_rtol > 0.0
                and loss_scalar is not None
                and prev_loss is not None
            ):
                denom = max(abs(prev_loss), 1e-30)
                rel_improve = (prev_loss - loss_scalar) / denom
                if rel_improve < early_stop_rtol:
                    stall_count += 1
                else:
                    stall_count = 0
                if (
                    n_outer_done >= early_stop_min_iter
                    and stall_count >= early_stop_patience
                ):
                    stopped_early = True
                    print(
                        f"  early stop at outer iter {i}: "
                        f"rel_improve < {early_stop_rtol:g} for "
                        f"{early_stop_patience} iters "
                        f"(min_iter={early_stop_min_iter})"
                    )
                    break
            if loss_scalar is not None:
                prev_loss = loss_scalar

        wall_s = time.perf_counter() - t_wall0
        c_full_final = c_full_from_hat(c_interior_param.detach())
        vm_out = VelocityModel.from_field(grid, c_full_final, pml_c=vm_in.pml_c)
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
            "stopped_early": stopped_early,
            "n_u_closure": n_u_closure,
            "n_c_closure": n_c_closure,
            "n_closure": n_u_closure + n_c_closure,
            "u_precond": "z",
            "c_update": c_update,
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
