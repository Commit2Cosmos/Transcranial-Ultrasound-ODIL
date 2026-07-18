from abc import ABC, abstractmethod
import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as _F

from odil_wave.loss import DiscreteLoss, ForwardLoss, InverseLoss
from odil_wave.loss.utils import LossTape
from odil_wave.models import VelocityModel
from odil_wave.wavefield import Wavefield
from .preconditioner import TimeStepUTransform


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


class Optimiser(ABC):
    """Base optimiser class."""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        self.loss = loss
        self.wavefield = wavefield

    @abstractmethod
    def minimise(self, **kwargs) -> Tuple[List[Wavefield], LossTape]:
        raise NotImplementedError


class LBFGSB(Optimiser):
    """`torch.optim.LBFGS`-based optimiser with separate u and c LBFGS phases.

    Each outer iteration runs ``u_steps`` LBFGS steps on the wavefield (c detached)
    followed by ``c_steps`` LBFGS steps on the velocity interior (u detached).
    After every c phase the u curvature history is reset, because the Hessian
    approximation built for the old c is stale for the new one.

    Tip: LBFGS builds its curvature history across consecutive calls.  Setting
    ``u_steps > 1`` (default 1) with ``c_steps=0`` is therefore much more
    efficient for forward problems, while small ``c_steps`` (e.g. 1 every few
    outer iterations via on_iteration) suit inverse problems.
    """

    _DEFAULT_OPTS = {
        "n_iter": 100,
        "u_steps": 1,
        "c_steps": 1,
        # u-LBFGS knobs
        "max_iter": 4,
        "history_size": 10,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
        # c-LBFGS knobs (independent from u-phase)
        "c_lr": 1.0,
        "c_max_iter": 4,
        "c_history_size": 10,
        # c-gradient preconditioning
        # c_precond_type: "energy" (divide by mean u^2) or "gaussian" (smooth gradient)
        "c_precond": False,
        "c_precond_type": "energy",
        "c_precond_sigma": 2.0,   # Gaussian sigma in grid cells (used when type="gaussian")
        "c_precond_stab": 1e-2,
        # u-block preconditioning: reparameterise u = A^{-1} z (leapfrog transform)
        "u_precond": False,
        # c-step mode:
        #   "frozen_u" — classic: u detached, L = PDE + data (grad through A(c)u)
        #   "frozen_z" — z detached, u = A(c)^{-1} z live in c, L = data (+reg)
        #                requires u_precond=True; keep c_precond=False until
        #                the raw direct data gradient is validated
        "c_update": "frozen_u",
        # whether to discard the c LBFGS curvature history after each u-phase
        # (True = safe default; False = carry history across outer iterations)
        "reset_c_history": True,
    }
    _LBFGS_KEYS = frozenset({
        "lr", "max_iter", "max_eval",
        "tolerance_grad", "tolerance_change",
        "history_size", "line_search_fn",
    })
    _SCHEDULE_KEYS = frozenset({
        "u_steps", "c_steps", "c_lr", "c_max_iter", "c_history_size",
        "c_precond", "c_precond_type", "c_precond_sigma", "c_precond_stab",
        "u_precond", "c_update", "reset_c_history",
    })

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
        # Optional boolean mask (shape = interior grid) indicating which c
        # cells are free to be optimised.  Cells where free_mask=False are
        # held fixed at their initial value throughout the run (gradient is
        # zeroed before each LBFGS curvature update; value is hard-reset after
        # each step so the line-search cannot move them either).
        self.free_mask = free_mask

        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self):
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        u_steps = int(opts.pop("u_steps", 1))
        c_steps = int(opts.pop("c_steps", 1))
        c_lr = float(opts.pop("c_lr", 1.0))
        c_max_iter = int(opts.pop("c_max_iter", opts.get("max_iter", 4)))
        c_history_size = int(opts.pop("c_history_size", opts.get("history_size", 10)))
        c_precond = bool(opts.pop("c_precond", False))
        c_precond_type = str(opts.pop("c_precond_type", "energy"))
        c_precond_sigma = float(opts.pop("c_precond_sigma", 2.0))
        c_precond_stab = float(opts.pop("c_precond_stab", 1e-2))
        u_precond = bool(opts.pop("u_precond", False))
        c_update = str(opts.pop("c_update", "frozen_u"))
        reset_c_history = bool(opts.pop("reset_c_history", True))

        if c_update not in ("frozen_u", "frozen_z"):
            raise ValueError(
                f"c_update must be 'frozen_u' or 'frozen_z', got {c_update!r}"
            )
        if c_update == "frozen_z" and not u_precond:
            raise ValueError("c_update='frozen_z' requires u_precond=True")

        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}

        c_torch_opts = dict(u_torch_opts)
        c_torch_opts["lr"] = c_lr
        c_torch_opts["max_iter"] = c_max_iter
        c_torch_opts["history_size"] = c_history_size

        return (n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts,
                c_precond, c_precond_type, c_precond_sigma, c_precond_stab,
                u_precond, c_update, reset_c_history)

    def _seed_amplitudes(self, n_shots, dtype, device) -> torch.Tensor:
        """(n_shots, NT-2, NX, NY) initial inner rows from u_init/wavefield."""
        if self.u_init is None:
            seed = (
                self.wavefield.amplitude[2:]
                .detach()
                .clone()
                .to(dtype=dtype, device=device)
            )
            return seed.unsqueeze(0).expand(n_shots, -1, -1, -1).contiguous()
        if isinstance(self.u_init, (list, tuple)):
            stack = torch.stack(
                [
                    w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                    for w in self.u_init
                ]
            )
        else:
            stack = torch.as_tensor(self.u_init)
        stack = stack.detach().to(dtype=dtype, device=device)
        if stack.shape[0] != n_shots:
            raise ValueError(
                f"u_init provides {stack.shape[0]} shots, expected {n_shots}."
            )
        return stack[:, 2:].contiguous()

    def minimise(
        self,
        on_iteration=None,
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Run the block-coordinate LBFGS optimisation loop.

        When ``u_precond=True`` the wavefield is reparameterised as
        ``u = A^{-1} z`` (one leapfrog sweep per closure call).  L-BFGS
        then optimises ``z``; the transformed Hessian block is
        ``A^{-T} H_u A^{-1} = a I + (data term)``, which is much better
        conditioned than the raw ``H_u``.  Seeding is exact:
        ``z_0 = A u_0``.

        ``c_update``:
          - ``"frozen_u"`` (default): after the z/u phase, freeze ``u`` and
            minimise PDE+data w.r.t. ``c``.  Then rebuild ``A`` and choose
            ``z_new`` so ``A(c_new)^{-1} z_new = u_old`` (``u`` preserved).
          - ``"frozen_z"``: freeze ``z``, set ``u = A(c)^{-1} z`` inside
            every L-BFGS closure (live in ``c``), minimise data (+reg) only.
            Leave ``z`` unchanged; rebuild the transform for the next z-step.
        """
        self.opts.update(overrides)
        (n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts,
         c_precond, c_precond_type, c_precond_sigma, c_precond_stab,
         u_precond, c_update, reset_c_history) = self._split_opts()

        grid = self.wavefield.grid
        dtype = grid.dtype
        device = grid.device
        Nx, Ny = grid.shape
        n_shots = self.loss.config.geometry.n_sources

        zero_row = torch.zeros(1, Nx, Ny, dtype=dtype, device=device)
        init_ut = self.wavefield.init_ut.detach().to(dtype=dtype, device=device)
        ic_row = (grid.dt * init_ut).unsqueeze(0)

        zero_row_S = zero_row.unsqueeze(0).expand(n_shots, -1, -1, -1)
        ic_row_S = ic_row.unsqueeze(0).expand(n_shots, -1, -1, -1)

        u_seed = self._seed_amplitudes(n_shots, dtype, device)

        vm_in = self.wavefield.velocity_model
        vm_c_const = vm_in.c.detach().to(dtype=dtype, device=device)

        is_inverse = isinstance(self.loss, InverseLoss)
        if is_inverse:
            c0_int = vm_c_const[grid.interior_slice].detach().clone()
            # Reparameterise: optimise ĉ = c / c_ref (dimensionless, order ~1).
            # This brings the c gradient into the same order of magnitude as u,
            # which is required for LBFGS/GD line searches to succeed.
            c_ref = float(c0_int.mean().item())
            c_interior_param = torch.nn.Parameter(c0_int / c_ref)

            # free_mask: which interior cells are allowed to move.
            # None means all cells are free (default behaviour).
            if self.free_mask is not None:
                _free_mask = self.free_mask.to(dtype=torch.bool, device=device)
                # Normalised initial values for frozen cells (reset after each step).
                _c_frozen_init = (c0_int / c_ref)[~_free_mask].clone().detach()
            else:
                _free_mask = None
                _c_frozen_init = None
        else:
            c_ref = None
            _free_mask = None
            _c_frozen_init = None
            c_interior_param = None

        # u parameterisation: direct (u_inner_param) or transformed (z_param)
        if u_precond:
            c_full_init = (
                vm_in.build_full_c(c0_int) if is_inverse else vm_c_const
            )
            u_transform = TimeStepUTransform(self.loss, c_full_init)
            z_seed = u_transform.inverse(u_seed)
            z_param = torch.nn.Parameter(z_seed)
            u_inner_param = None  # computed on-the-fly via transform

            def get_u_inner() -> torch.Tensor:
                return u_transform.apply(z_param)

            def get_u_inner_detached() -> torch.Tensor:
                with torch.no_grad():
                    return u_transform.apply(z_param.detach())

            def optim_param():
                return z_param
        else:
            u_transform = None
            u_inner_param = torch.nn.Parameter(u_seed)

            def get_u_inner() -> torch.Tensor:
                return u_inner_param

            def get_u_inner_detached() -> torch.Tensor:
                return u_inner_param.detach()

            def optim_param():
                return u_inner_param

        def make_u_optimiser():
            return torch.optim.LBFGS([optim_param()], **u_torch_opts)

        def make_c_optimiser():
            return torch.optim.LBFGS([c_interior_param], **c_torch_opts)

        u_optimiser = make_u_optimiser()
        c_optimiser = make_c_optimiser() if (is_inverse and c_steps > 0) else None

        c_min = self.c_min / c_ref if (self.c_min is not None and c_ref is not None) else None
        c_max = self.c_max / c_ref if (self.c_max is not None and c_ref is not None) else None
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        _prec: Optional[torch.Tensor] = None
        _eps_prec: float = 0.0

        for i in range(n_iter):

            for _ in range(u_steps):
                def u_closure():
                    u_optimiser.zero_grad()
                    amps = torch.cat([zero_row_S, ic_row_S, get_u_inner()], dim=1)
                    if is_inverse:
                        c_fixed_phys = c_interior_param.detach() * c_ref
                        c_full = vm_in.build_full_c(c_fixed_phys)
                        L = self.loss.evaluate(amps, c_full, c_fixed_phys)
                    else:
                        L = self.loss.evaluate(amps, vm_c_const)
                    L.backward()
                    return L
                loss_value = u_optimiser.step(u_closure)

            if is_inverse and c_steps > 0:
                # Wavefield-energy preconditioner: prec[x,y] = mean over shots
                # and time of u^2.  Dividing grad_c by this dampens updates in
                # well-illuminated regions and amplifies them where energy is low.
                # Skip for frozen_z until the raw gradient is validated.
                if c_precond and c_update == "frozen_u":
                    with torch.no_grad():
                        u_sq = get_u_inner_detached().pow(2).mean(dim=(0, 1))  # (NX, NY)
                        _prec = u_sq[grid.interior_slice]
                        _eps_prec = c_precond_stab * float(
                            _prec.max().clamp(min=1e-30)
                        )

                # u/z just changed — reset c's curvature history unless the caller
                # explicitly asked to carry it across outer iterations.
                if reset_c_history:
                    c_optimiser = make_c_optimiser()

                # Freeze z for the whole c-phase (frozen_z).
                z_fixed = (
                    z_param.detach()
                    if (c_update == "frozen_z" and u_precond)
                    else None
                )

                for _ in range(c_steps):
                    def c_closure():
                        c_optimiser.zero_grad()
                        c_phys = c_interior_param * c_ref
                        c_full = vm_in.build_full_c(c_phys)

                        if c_update == "frozen_z":
                            # Recompute u = A(c)^{-1} z inside every L-BFGS trial
                            # so the graph from c → u is live.
                            u_inner_live = u_transform.apply_diff_c(z_fixed, c_full)
                            amps = torch.cat(
                                [zero_row_S, ic_row_S, u_inner_live], dim=1
                            )
                            # At fixed z, ||z - f|| is independent of c, so drop PDE.
                            L_c = self.loss.evaluate(
                                amps,
                                c_full,
                                c_phys,
                                weights_override={"pde": 0.0},
                            )
                        else:
                            # frozen_u: u detached; data term is constant in c.
                            amps_fixed = torch.cat(
                                [zero_row_S, ic_row_S, get_u_inner_detached()], dim=1
                            )
                            L_c = self.loss.evaluate(
                                amps_fixed, c_full, c_phys
                            )

                        L_c.backward()
                        # Zero out gradients for frozen cells so LBFGS does
                        # not build curvature info for them.
                        if _free_mask is not None and c_interior_param.grad is not None:
                            c_interior_param.grad[~_free_mask] = 0.0
                        if c_interior_param.grad is not None:
                            with torch.no_grad():
                                g_raw = c_interior_param.grad
                                _n = g_raw.numel()
                                _pct_pos = 100.0 * float((g_raw > 0).sum()) / _n
                                _pct_neg = 100.0 * float((g_raw < 0).sum()) / _n
                                tag = "frozen_z" if c_update == "frozen_z" else "frozen_u"
                                print(
                                    f"    [grad_c {tag}]  "
                                    f"mean={g_raw.mean():+.3e}  "
                                    f"min={g_raw.min():+.3e}  "
                                    f"max={g_raw.max():+.3e}  "
                                    f"pos={_pct_pos:.1f}%  neg={_pct_neg:.1f}%"
                                )
                        if (
                            c_precond
                            and c_update == "frozen_u"
                            and c_interior_param.grad is not None
                        ):
                            with torch.no_grad():
                                if c_precond_type == "gaussian":
                                    # Smooth gradient with a Gaussian kernel.
                                    # Does not depend on u — safe when u is wrong.
                                    print(
                                        f"    [prec gaussian] sigma={c_precond_sigma:.1f} cells"
                                    )
                                    g_smooth = _gaussian_smooth_2d(
                                        c_interior_param.grad, c_precond_sigma
                                    )
                                    c_interior_param.grad.copy_(g_smooth)
                                else:
                                    # Default: energy preconditioner (divide by mean u^2).
                                    if _prec is not None:
                                        print(
                                            f"    [prec energy]   "
                                            f"min={_prec.min():.3e}  "
                                            f"mean={_prec.mean():.3e}  "
                                            f"max={_prec.max():.3e}  "
                                            f"eps={_eps_prec:.3e}"
                                        )
                                        c_interior_param.grad.div_(_prec + _eps_prec)
                                g_pre = c_interior_param.grad
                                _np = g_pre.numel()
                                _pct_pos_p = 100.0 * float((g_pre > 0).sum()) / _np
                                _pct_neg_p = 100.0 * float((g_pre < 0).sum()) / _np
                                print(
                                    f"    [grad_c PRECOND] "
                                    f"mean={g_pre.mean():+.3e}  "
                                    f"min={g_pre.min():+.3e}  "
                                    f"max={g_pre.max():+.3e}  "
                                    f"pos={_pct_pos_p:.1f}%  neg={_pct_neg_p:.1f}%"
                                )
                        return L_c
                    loss_value = c_optimiser.step(c_closure)
                    with torch.no_grad():
                        # Hard-reset frozen cells — the line-search may have
                        # moved them slightly despite zero gradient.
                        if _free_mask is not None:
                            c_interior_param.data[~_free_mask] = _c_frozen_init
                        if c_min is not None or c_max is not None:
                            c_at_min = (c_interior_param <= c_min + 1e-8).float().mean().item() if c_min is not None else 0.0
                            c_at_max = (c_interior_param >= c_max - 1e-8).float().mean().item() if c_max is not None else 0.0
                            print(
                                f"    [clamp]  c_min={c_min:.4f}  c_max={c_max:.4f}  "
                                f"frac@min={c_at_min:.3f}  frac@max={c_at_max:.3f}  "
                                f"c_param: mean={c_interior_param.mean():.4f}  "
                                f"min={c_interior_param.min():.4f}  max={c_interior_param.max():.4f}"
                            )
                            c_interior_param.clamp_(min=c_min, max=c_max)

                # c changed — rebuild A for the new medium.
                if u_precond and u_transform is not None:
                    # For frozen_u, save u from the OLD transform before rebuild.
                    # After rebuild, get_u_inner() would use A(c_new)^{-1} z_old
                    # and reseeding would incorrectly leave z unchanged.
                    u_fixed_before_rebuild = None
                    if c_update == "frozen_u":
                        with torch.no_grad():
                            u_fixed_before_rebuild = u_transform.apply(
                                z_param.detach()
                            ).clone()

                    c_full_new = vm_in.build_full_c(
                        c_interior_param.detach() * c_ref
                    )
                    u_transform.rebuild(self.loss, c_full_new)

                    if c_update == "frozen_u":
                        # Choose z_new so A(c_new)^{-1} z_new = u_old.
                        with torch.no_grad():
                            z_param.data.copy_(
                                u_transform.inverse(u_fixed_before_rebuild)
                            )
                    # frozen_z: leave z unchanged; next z-step uses
                    # u = A(c_new)^{-1} z.
                    u_optimiser = make_u_optimiser()

            should_log = (i % log_every == 0) or (i == n_iter - 1)

            c_full_now = None
            if is_inverse and (should_log or on_iteration is not None):
                c_full_now = vm_in.build_full_c(c_interior_param.detach() * c_ref)

            if on_iteration is not None:
                on_iteration(i, c_full_now)

            if should_log:
                loss_scalar = float(loss_value.detach().cpu())
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

        if isinstance(self.loss, ForwardLoss):
            vm_out = vm_in
        else:
            c_full_final = vm_in.build_full_c(c_interior_param.detach() * c_ref)
            vm_out = VelocityModel.from_field(grid, c_full_final, pml_c=vm_in.pml_c)

        u_inner_final = get_u_inner_detached()
        outputs: List[Wavefield] = []
        for s in range(n_shots):
            wf = Wavefield(grid=grid, velocity_model=vm_out)
            amp_full = torch.cat([zero_row, ic_row, u_inner_final[s]], dim=0)
            wf.amplitude = amp_full
            outputs.append(wf)

        self.loss.callback.result = {
            "loss": (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            ),
            "n_outer_iter": n_iter,
        }
        return outputs, self.loss.callback


class JointLBFGS(Optimiser):
    """Joint L-BFGS optimisation of the wavefield ``u`` and velocity ``c``.

    For an inverse problem, a single ``torch.optim.LBFGS`` instance receives
    both parameters::

        [u_inner_param, c_interior_param]

    Therefore every closure evaluation differentiates the same total loss with
    respect to both variables simultaneously. Neither variable is detached.

    For a forward problem, only ``u_inner_param`` is optimised.

    
    * c is represented as c / c_ref so that it is dimensionless and
      closer in numerical scale to u.
    * Projecting ``c`` with clamping can invalidate L-BFGS curvature pairs. If
      projection changes ``c``, the optimiser is rebuilt to discard stale
      history while still retaining a single joint optimiser.
    * c_precond=False should stay False for joint L-BFGS, useful for block L-BFGS instead.
    """

    _DEFAULT_OPTS = {
        "n_iter": 100,
        "lr": 1.0,
        "max_iter": 4,
        "history_size": 10,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
        "c_precond": False,
        "c_precond_stab": 1e-2,
        "print_grad_stats": False,
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

    def __init__(
        self,
        wavefield: Wavefield,
        loss: DiscreteLoss,
        clamp: bool = False,
        u_init=None,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        grid = wavefield.grid

        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.u_init = u_init

        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self) -> Tuple[int, dict, bool, float, bool]:
        opts = dict(self.opts)

        n_iter = int(opts.pop("n_iter"))
        c_precond = bool(opts.pop("c_precond", False))
        c_precond_stab = float(opts.pop("c_precond_stab", 1e-2))
        print_grad_stats = bool(opts.pop("print_grad_stats", False))

        torch_opts = {key: value for key, value in opts.items() if key in self._LBFGS_KEYS}
        return n_iter, torch_opts, c_precond, c_precond_stab, print_grad_stats

    def _seed_amplitudes(self, n_shots, dtype, device) -> torch.Tensor:
        """Return initial unknown wavefield rows with shape ``(S, NT-2, NX, NY)``."""
        if self.u_init is None:
            seed = (
                self.wavefield.amplitude[2:]
                .detach()
                .clone()
                .to(dtype=dtype, device=device)
            )
            return seed.unsqueeze(0).expand(n_shots, -1, -1, -1).contiguous()

        if isinstance(self.u_init, (list, tuple)):
            stack = torch.stack(
                [
                    item.amplitude if isinstance(item, Wavefield) else torch.as_tensor(item)
                    for item in self.u_init
                ]
            )
        else:
            stack = torch.as_tensor(self.u_init)

        stack = stack.detach().to(dtype=dtype, device=device)
        if stack.shape[0] != n_shots:
            raise ValueError(
                f"u_init provides {stack.shape[0]} shots, expected {n_shots}."
            )

        return stack[:, 2:].contiguous()

    @staticmethod
    def _grad_stats(name: str, grad: Optional[torch.Tensor]) -> None:
        if grad is None:
            print(f"{name} grad is None")
            return

        with torch.no_grad():
            print(f"{name} grad norm:", grad.norm().item())
            print(f"{name} grad min/max:", grad.min().item(), grad.max().item())
            print(f"{name} grad RMS:", grad.square().mean().sqrt().item())

    def minimise(
        self,
        on_iteration=None,
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Run joint L-BFGS over ``u`` and, for inverse problems, ``c``."""
        self.opts.update(overrides)
        (
            n_iter,
            torch_opts,
            c_precond,
            c_precond_stab,
            print_grad_stats,
        ) = self._split_opts()

        grid = self.wavefield.grid
        dtype = grid.dtype
        device = grid.device
        nx, ny = grid.shape
        n_shots = self.loss.config.geometry.n_sources

        zero_row = torch.zeros(1, nx, ny, dtype=dtype, device=device)
        init_ut = self.wavefield.init_ut.detach().to(dtype=dtype, device=device)
        ic_row = (grid.dt * init_ut).unsqueeze(0)

        zero_row_s = zero_row.unsqueeze(0).expand(n_shots, -1, -1, -1)
        ic_row_s = ic_row.unsqueeze(0).expand(n_shots, -1, -1, -1)

        u_inner_param = torch.nn.Parameter(
            self._seed_amplitudes(n_shots, dtype, device)
        )

        vm_in = self.wavefield.velocity_model
        vm_c_const = vm_in.c.detach().to(dtype=dtype, device=device)
        is_inverse = isinstance(self.loss, InverseLoss)

        c_ref: Optional[float]
        c_interior_param: Optional[torch.nn.Parameter]

        if is_inverse:
            c0_int = vm_c_const[grid.interior_slice].detach().clone()
            c_ref = float(c0_int.mean().item())
            if c_ref == 0.0:
                raise ValueError("The reference velocity c_ref must be non-zero.")
            c_interior_param = torch.nn.Parameter(c0_int / c_ref)
            joint_params = [u_inner_param, c_interior_param]
        else:
            c_ref = None
            c_interior_param = None
            joint_params = [u_inner_param]

        def make_joint_optimiser() -> torch.optim.LBFGS:
            return torch.optim.LBFGS(joint_params, **torch_opts)

        joint_optimiser = make_joint_optimiser()

        c_min_scaled = (
            self.c_min / c_ref
            if self.c_min is not None and c_ref is not None
            else None
        )
        c_max_scaled = (
            self.c_max / c_ref
            if self.c_max is not None and c_ref is not None
            else None
        )

        log_every = max(1, int(self.loss.callback.log_every))
        loss_value: Optional[torch.Tensor] = None

        def current_state():
            amps = torch.cat([zero_row_s, ic_row_s, u_inner_param], dim=1)

            if is_inverse:
                assert c_interior_param is not None
                assert c_ref is not None
                c_phys = c_interior_param * c_ref
                c_full = vm_in.build_full_c(c_phys)
                return amps, c_full, c_phys

            return amps, vm_c_const, None

        for i in range(n_iter):

            def closure():
                joint_optimiser.zero_grad()
                amps, c_full, c_phys = current_state()

                if is_inverse:
                    loss = self.loss.evaluate(amps, c_full, c_phys)
                else:
                    loss = self.loss.evaluate(amps, c_full)

                loss.backward()

                if is_inverse and c_interior_param is not None:
                    if print_grad_stats:
                        self._grad_stats("u raw", u_inner_param.grad)
                        self._grad_stats("c raw", c_interior_param.grad)

                    if c_precond and c_interior_param.grad is not None:
                        # Recomputed at each closure evaluation because u changes
                        # during the joint L-BFGS line search.
                        with torch.no_grad():
                            illumination = u_inner_param.detach().square().mean(dim=(0, 1))
                            illumination = illumination[grid.interior_slice]
                            eps = c_precond_stab * float(
                                illumination.max().clamp(min=1e-30).item()
                            )
                            c_interior_param.grad.div_(illumination + eps)

                        if print_grad_stats:
                            self._grad_stats("c preconditioned", c_interior_param.grad)

                return loss

            joint_optimiser.step(closure)

            # Project velocity bounds after the complete line-search step.
            # If projection changed c, discard curvature pairs generated for the
            # unprojected path.
            if is_inverse and c_interior_param is not None:
                with torch.no_grad():
                    c_before = c_interior_param.clone()
                    if c_min_scaled is not None or c_max_scaled is not None:
                        c_interior_param.clamp_(
                            min=c_min_scaled,
                            max=c_max_scaled,
                        )
                    projection_changed_c = not torch.equal(
                        c_before, c_interior_param
                    )

                if projection_changed_c:
                    joint_optimiser = make_joint_optimiser()

            should_log = i % log_every == 0 or i == n_iter - 1

            # Evaluate the actual post-step state. optimizer.step(closure) returns
            # the loss from its first closure call, which may be stale.
            if should_log or on_iteration is not None:
                with torch.no_grad():
                    amps_now, c_full_now, c_phys_now = current_state()
                    if is_inverse:
                        loss_value = self.loss.evaluate(
                            amps_now, c_full_now, c_phys_now
                        )
                    else:
                        loss_value = self.loss.evaluate(amps_now, c_full_now)
            else:
                c_full_now = None

            if on_iteration is not None:
                on_iteration(i, c_full_now if is_inverse else None)

            if should_log:
                assert loss_value is not None
                loss_scalar = float(loss_value.detach().cpu())
                ratio = self.loss.pde_src_ratio()

                self.loss.callback.log(
                    loss_scalar,
                    self.loss._last_residuals,
                    pde_src_ratio=ratio,
                )

                if is_inverse and c_full_now is not None:
                    self.loss.callback.log_c(c_full_now.detach().cpu().numpy())

                print(
                    f"Iteration: {i} | loss = {loss_scalar:.6e} | "
                    f"|r_pde|/|src| = {ratio:.3e}"
                )

        if isinstance(self.loss, ForwardLoss):
            vm_out = vm_in
        else:
            assert c_interior_param is not None
            assert c_ref is not None
            c_full_final = vm_in.build_full_c(
                c_interior_param.detach() * c_ref
            )
            vm_out = VelocityModel.from_field(
                grid,
                c_full_final,
                pml_c=vm_in.pml_c,
            )

        u_inner_final = u_inner_param.detach()
        outputs: List[Wavefield] = []

        for shot in range(n_shots):
            wf = Wavefield(grid=grid, velocity_model=vm_out)
            wf.amplitude = torch.cat(
                [zero_row, ic_row, u_inner_final[shot]],
                dim=0,
            )
            outputs.append(wf)

        self.loss.callback.result = {
            "loss": (
                float(loss_value.detach().cpu())
                if loss_value is not None
                else None
            ),
            "n_outer_iter": n_iter,
        }

        return outputs, self.loss.callback


class FirstOrderOptimiser(Optimiser):
    """Shared minimise loop for first-order optimisers (Adam, GD).

    Matches :class:`LBFGSB` setup: batched ``(n_shots, NT-2, NX, NY)`` amplitudes
    and hard zero + velocity IC rows. For inverse problems, ``u`` and ``c`` are
    updated alternately with separate PyTorch optimisers (block coordinate descent).
    """

    _TORCH_OPTIM_CLS: Optional[Type[torch.optim.Optimizer]] = None
    _DEFAULT_OPTS: dict = {}
    _TORCH_OPT_KEYS: frozenset = frozenset()
    _SCHEDULE_KEYS = frozenset({
        "u_steps", "c_steps", "c_lr", "c_max_grad_norm",
        "c_precond", "c_precond_stab",
    })

    def __init__(
        self,
        wavefield: Wavefield,
        loss: DiscreteLoss,
        clamp: bool = False,
        u_init=None,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        grid = wavefield.grid
        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.u_init = u_init

        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self) -> Tuple[int, dict, Optional[float], Optional[float]]:
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        max_grad_norm = opts.pop("max_grad_norm", None)
        if max_grad_norm is not None:
            max_grad_norm = float(max_grad_norm)
        c_max_grad_norm = opts.pop("c_max_grad_norm", None)
        if c_max_grad_norm is not None:
            c_max_grad_norm = float(c_max_grad_norm)
        for key in self._SCHEDULE_KEYS:
            opts.pop(key, None)
        torch_opts = {k: v for k, v in opts.items() if k in self._TORCH_OPT_KEYS}
        return n_iter, torch_opts, max_grad_norm, c_max_grad_norm

    def _seed_amplitudes(self, n_shots, dtype, device) -> torch.Tensor:
        """(n_shots, NT-2, NX, NY) initial inner rows from u_init/wavefield."""
        if self.u_init is None:
            seed = (
                self.wavefield.amplitude[2:]
                .detach()
                .clone()
                .to(dtype=dtype, device=device)
            )
            return seed.unsqueeze(0).expand(n_shots, -1, -1, -1).contiguous()
        if isinstance(self.u_init, (list, tuple)):
            stack = torch.stack(
                [
                    w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                    for w in self.u_init
                ]
            )
        else:
            stack = torch.as_tensor(self.u_init)
        stack = stack.detach().to(dtype=dtype, device=device)
        if stack.shape[0] != n_shots:
            raise ValueError(
                f"u_init provides {stack.shape[0]} shots, expected {n_shots}."
            )
        return stack[:, 2:].contiguous()

    def minimise(
        self,
        on_iteration=None,
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        if self._TORCH_OPTIM_CLS is None:
            raise NotImplementedError(
                "Subclasses must set _TORCH_OPTIM_CLS before calling minimise."
            )

        self.opts.update(overrides)
        n_iter, torch_opts, max_grad_norm, c_max_grad_norm = self._split_opts()

        u_steps = int(self.opts.get("u_steps", 1))
        c_steps = int(self.opts.get("c_steps", 1))
        c_lr = float(self.opts.get("c_lr", torch_opts.get("lr", 1e-4)))
        if c_max_grad_norm is None:
            c_max_grad_norm = max_grad_norm
        c_precond = bool(self.opts.get("c_precond", False))
        c_precond_stab = float(self.opts.get("c_precond_stab", 1e-2))

        grid = self.wavefield.grid
        dtype = grid.dtype
        device = grid.device
        Nx, Ny = grid.shape
        n_shots = self.loss.config.geometry.n_sources

        zero_row = torch.zeros(1, Nx, Ny, dtype=dtype, device=device)
        init_ut = self.wavefield.init_ut.detach().to(dtype=dtype, device=device)
        ic_row = (grid.dt * init_ut).unsqueeze(0)

        zero_row_S = zero_row.unsqueeze(0).expand(n_shots, -1, -1, -1)
        ic_row_S = ic_row.unsqueeze(0).expand(n_shots, -1, -1, -1)

        u_inner_param = torch.nn.Parameter(
            self._seed_amplitudes(n_shots, dtype, device)
        )

        vm_in = self.wavefield.velocity_model
        vm_c_const = vm_in.c.detach().to(dtype=dtype, device=device)

        is_inverse = isinstance(self.loss, InverseLoss)
        if is_inverse:
            c0_int = vm_c_const[grid.interior_slice].detach().clone()
            # Reparameterise: optimise c_new = c / c_ref (dimensionless, order 1).
            c_ref = float(c0_int.mean().item())
            c_interior_param = torch.nn.Parameter(c0_int / c_ref)
        else:
            c_ref = None
            c_interior_param = None

        u_optimiser = self._TORCH_OPTIM_CLS([u_inner_param], **torch_opts)

        if is_inverse:
            c_torch_opts = dict(torch_opts)
            c_torch_opts["lr"] = c_lr
            c_optimiser = self._TORCH_OPTIM_CLS([c_interior_param], **c_torch_opts)
        else:
            c_optimiser = None

        c_min = self.c_min / c_ref if (self.c_min is not None and c_ref is not None) else None
        c_max = self.c_max / c_ref if (self.c_max is not None and c_ref is not None) else None
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        _prec: Optional[torch.Tensor] = None
        _eps_prec: float = 0.0

        for i in range(n_iter):
            for _ in range(u_steps):
                u_optimiser.zero_grad()
                amps = torch.cat([zero_row_S, ic_row_S, u_inner_param], dim=1)
                if is_inverse:
                    c_fixed_phys = c_interior_param.detach() * c_ref
                    c_full = vm_in.build_full_c(c_fixed_phys)
                    L_u = self.loss.evaluate(amps, c_full, c_fixed_phys)
                else:
                    L_u = self.loss.evaluate(amps, vm_c_const)
                L_u.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [u_inner_param],
                        max_norm=max_grad_norm,
                    )
                u_optimiser.step()
                loss_value = L_u

            if is_inverse and c_steps > 0:
                if c_precond:
                    with torch.no_grad():
                        u_sq = u_inner_param.detach().pow(2).mean(dim=(0, 1))  # (NX, NY)
                        _prec = u_sq[grid.interior_slice]
                        _eps_prec = c_precond_stab * float(
                            _prec.max().clamp(min=1e-30)
                        )

                # u just changed — reset c-optimiser so stale momentum
                # from the previous u does not bias the c update.
                c_optimiser = self._TORCH_OPTIM_CLS(
                    [c_interior_param], **{**torch_opts, "lr": c_lr}
                )
                for _ in range(c_steps):
                    c_optimiser.zero_grad()
                    amps_fixed = torch.cat(
                        [zero_row_S, ic_row_S, u_inner_param.detach()],
                        dim=1,
                    )
                    c_phys = c_interior_param * c_ref
                    c_full = vm_in.build_full_c(c_phys)
                    L_c = self.loss.evaluate(
                        amps_fixed,
                        c_full,
                        c_phys,
                    )
                    L_c.backward()
                    if (
                        c_precond
                        and _prec is not None
                        and c_interior_param.grad is not None
                    ):
                        with torch.no_grad():
                            c_interior_param.grad.div_(_prec + _eps_prec)
                    if c_max_grad_norm is not None and c_max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [c_interior_param],
                            max_norm=c_max_grad_norm,
                        )
                    c_optimiser.step()
                    with torch.no_grad():
                        if c_min is not None or c_max is not None:
                            c_interior_param.clamp_(min=c_min, max=c_max)
                    loss_value = L_c

            should_log = (i % log_every == 0) or (i == n_iter - 1)

            c_full_now = None
            if is_inverse and (should_log or on_iteration is not None):
                c_full_now = vm_in.build_full_c(c_interior_param.detach() * c_ref)

            if on_iteration is not None:
                on_iteration(i, c_full_now)

            if should_log:
                loss_scalar = float(loss_value.detach().cpu())
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

        if isinstance(self.loss, ForwardLoss):
            vm_out = vm_in
        else:
            c_full_final = vm_in.build_full_c(c_interior_param.detach() * c_ref)
            vm_out = VelocityModel.from_field(grid, c_full_final, pml_c=vm_in.pml_c)

        u_inner_final = u_inner_param.detach()
        outputs: List[Wavefield] = []
        for s in range(n_shots):
            wf = Wavefield(grid=grid, velocity_model=vm_out)
            amp_full = torch.cat([zero_row, ic_row, u_inner_final[s]], dim=0)
            wf.amplitude = amp_full
            outputs.append(wf)

        self.loss.callback.result = {
            "loss": (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            ),
            "n_outer_iter": n_iter,
        }
        return outputs, self.loss.callback


class AdamOptimiser(FirstOrderOptimiser):
    """Adam optimiser for forward and inverse ODIL problems.

    Inverse problems alternate Adam steps on the wavefield ``u`` and velocity
    ``c`` with separate learning rates. Each outer step is cheaper than
    L-BFGS but typically needs more iterations.
    """

    _TORCH_OPTIM_CLS = torch.optim.Adam
    _TORCH_OPT_KEYS = frozenset({"lr", "betas", "eps", "weight_decay", "amsgrad"})
    _DEFAULT_OPTS = {
        "n_iter": 200,
        "lr": 1e-3,
        "c_lr": 1e-4,
        "u_steps": 1,
        "c_steps": 1,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": None,
        "c_max_grad_norm": None,
        "c_precond": False,
        "c_precond_stab": 1e-2,
    }


class GradientDescent(FirstOrderOptimiser):
    """Plain gradient descent baseline. Convergence is slow and the learning
    rate is very sensitive.
    """

    _TORCH_OPTIM_CLS = torch.optim.SGD
    _TORCH_OPT_KEYS = frozenset(
        {"lr", "momentum", "dampening", "weight_decay", "nesterov"}
    )
    _DEFAULT_OPTS = {
        "n_iter": 100,
        "lr": 1e-4,
        "c_lr": 1e-5,
        "u_steps": 1,
        "c_steps": 1,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": None,
        "c_max_grad_norm": None,
        "c_precond": False,
        "c_precond_stab": 1e-2,
    }
