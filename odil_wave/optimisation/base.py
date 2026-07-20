from abc import ABC, abstractmethod
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as _F

from odil_wave.loss import DiscreteLoss, ForwardLoss, InverseLoss
from odil_wave.loss.utils import LossTape
from odil_wave.models import VelocityModel
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


class Optimiser(ABC):
    """Base optimiser class."""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        self.loss = loss
        self.wavefield = wavefield

    @abstractmethod
    def minimise(self, **kwargs) -> Tuple[List[Wavefield], LossTape]:
        raise NotImplementedError


class LBFGSB(Optimiser):
    """Block-coordinate dual L-BFGS for frequency-domain ODIL.

    Each outer iteration: ``u_steps`` of u-LBFGS (``c`` fixed) then
    ``c_steps`` of c-LBFGS (``u`` fixed). Complex ``u`` is stored as two real
    tensors ``u_real`` / ``u_imag`` owned by the same u-optimiser. ``c`` is real.

    No Adam, joint optimiser, closed-form ``c``, or Helmholtz ``H^{-1}`` reparam.
    """

    _DEFAULT_OPTS = {
        "n_iter": 100,
        "u_steps": 1,
        "c_steps": 1,
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
    }
    _LBFGS_KEYS = frozenset({
        "lr", "max_iter", "max_eval",
        "tolerance_grad", "tolerance_change",
        "history_size", "line_search_fn",
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
        reset_c_history = bool(opts.pop("reset_c_history", True))
        # Silently drop legacy keys from older call sites
        opts.pop("u_precond", None)
        opts.pop("c_update", None)

        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}
        c_torch_opts = dict(u_torch_opts)
        c_torch_opts["lr"] = c_lr
        c_torch_opts["max_iter"] = c_max_iter
        c_torch_opts["history_size"] = c_history_size

        return (
            n_iter,
            u_steps,
            c_steps,
            u_torch_opts,
            c_torch_opts,
            c_precond,
            c_precond_type,
            c_precond_sigma,
            c_precond_stab,
            reset_c_history,
        )

    def _seed_complex(self, n_shots, cdtype, device) -> torch.Tensor:
        """``(n_shots, nf, nx, ny)`` complex seed."""
        nf = self.wavefield.n_frequencies
        Nx, Ny = self.wavefield.grid.shape
        if self.u_init is None:
            seed = self.wavefield.amplitude.detach().clone().to(
                dtype=cdtype, device=device
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
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Run u-LBFGS then c-LBFGS block-coordinate loop."""
        self.opts.update(overrides)
        (
            n_iter,
            u_steps,
            c_steps,
            u_torch_opts,
            c_torch_opts,
            c_precond,
            c_precond_type,
            c_precond_sigma,
            c_precond_stab,
            reset_c_history,
        ) = self._split_opts()

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
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        _prec: Optional[torch.Tensor] = None
        _eps_prec: float = 0.0

        for i in range(n_iter):
            for _ in range(u_steps):
                def u_closure():
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

            if is_inverse and c_steps > 0:
                if c_precond:
                    with torch.no_grad():
                        u_sq = pack_u_detached().abs().square().mean(dim=(0, 1))
                        _prec = u_sq[grid.interior_slice]
                        _eps_prec = c_precond_stab * float(
                            _prec.max().clamp(min=1e-30)
                        )

                if reset_c_history:
                    c_optimiser = make_c_optimiser()

                logged_grad_c = False
                for _ in range(c_steps):
                    def c_closure():
                        nonlocal logged_grad_c
                        c_optimiser.zero_grad()
                        # Optimise ĉ = c / c_ref; PDE sees c = ĉ c_ref.
                        # Pass ĉ into evaluate so Tikhonov is scale-stable.
                        c_phys = c_interior_param * c_ref
                        c_full = vm_in.build_full_c(c_phys)
                        amps_fixed = pack_u_detached()
                        L_c = self.loss.evaluate(
                            amps_fixed, c_full, c_interior_param
                        )
                        L_c.backward()
                        if _free_mask is not None and c_interior_param.grad is not None:
                            c_interior_param.grad[~_free_mask] = 0.0
                        if c_interior_param.grad is not None and not logged_grad_c:
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
                        if (
                            c_precond
                            and c_interior_param.grad is not None
                        ):
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
            "n_outer_iter": n_iter,
        }
        return outputs, self.loss.callback


