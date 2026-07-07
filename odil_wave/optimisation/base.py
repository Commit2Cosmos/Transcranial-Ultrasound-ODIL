from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

import torch

from odil_wave.loss import DiscreteLoss, ForwardLoss, InverseLoss
from odil_wave.loss.utils import LossTape
from odil_wave.models import VelocityModel
from odil_wave.wavefield import Wavefield


class Optimiser(ABC):
    """Base optimiser class."""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        self.loss = loss
        self.wavefield = wavefield

    @abstractmethod
    def minimise(self, **kwargs) -> Tuple[List[Wavefield], LossTape]:
        raise NotImplementedError


class LBFGSB(Optimiser):
    """`torch.optim.LBFGS`-based optimiser with a nested c-update scheme."""

    _DEFAULT_OPTS = {
        "n_iter": 100,
        "max_iter": 4,
        "history_size": 10,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
    }

    def __init__(
        self,
        wavefield: Wavefield,
        loss: DiscreteLoss,
        clamp: bool = False,
        c_update_every: int = 300,
        c_update_alpha: float = 1.0,
        u_init=None,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        grid = wavefield.grid
        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.c_update_every = int(c_update_every)
        self.c_update_alpha = float(c_update_alpha)
        self.u_init = u_init

        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self) -> Tuple[int, dict]:
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        # Only forward keys that torch.optim.LBFGS accepts.
        allowed = {
            "lr",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search_fn",
        }
        torch_opts = {k: v for k, v in opts.items() if k in allowed}
        return n_iter, torch_opts

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

    def _prox_regularise(
        self, c_star: torch.Tensor, illum: torch.Tensor
    ) -> torch.Tensor:
        """Solves ``min_c 0.5 * sum(w * (c - c*)**2) + lam * R(c)`` on the
        interior map with ``w = illum / mean(illum)``
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
            # minimise() calls this under no_grad; the prox solve needs
            # its own (tiny) graph on the 2D map.
            with torch.enable_grad():
                prox_opt.zero_grad()
                F = 0.5 * (w * (c - c_star) ** 2).sum() + lam * reg(c)
                F.backward()
            return F

        prox_opt.step(prox_closure)
        return c.detach()

    def minimise(
        self,
        on_iteration=None,
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Run the optimisation loop."""
        self.opts.update(overrides)
        n_iter, torch_opts = self._split_opts()

        grid = self.wavefield.grid
        dtype = grid.dtype
        device = grid.device
        Nx, Ny = grid.shape
        n_shots = self.loss.config.geometry.n_sources

        zero_row = torch.zeros(1, Nx, Ny, dtype=dtype, device=device)
        init_ut = self.wavefield.init_ut.detach().to(dtype=dtype, device=device)
        ic_row = (grid.dt * init_ut).unsqueeze(0)  # (1, Nx, Ny)

        zero_row_S = zero_row.unsqueeze(0).expand(n_shots, -1, -1, -1)
        ic_row_S = ic_row.unsqueeze(0).expand(n_shots, -1, -1, -1)

        # Single (n_shots, NT-2, NX, NY) Parameter — the only L-BFGS variable.
        u_inner_param = torch.nn.Parameter(
            self._seed_amplitudes(n_shots, dtype, device)
        )

        vm_in = self.wavefield.velocity_model
        # Detach c once for the forward path so the closure can't accidentally
        # carry a stale subgraph through vm_in.c if it were ever requires_grad.
        vm_c_const = vm_in.c.detach().to(dtype=dtype, device=device)

        is_inverse = isinstance(self.loss, InverseLoss)
        if is_inverse:
            # Plain tensor: c is updated in closed form, never by L-BFGS.
            c_interior = vm_c_const[grid.interior_slice].clone()
        else:
            c_interior = None

        def make_optimiser():
            return torch.optim.LBFGS([u_inner_param], **torch_opts)

        optimiser = make_optimiser()
        c_min, c_max = self.c_min, self.c_max
        alpha = self.c_update_alpha

        def closure():
            optimiser.zero_grad()
            amps = torch.cat([zero_row_S, ic_row_S, u_inner_param], dim=1)
            if is_inverse:
                c_full = vm_in.build_full_c(c_interior)
                L = self.loss.evaluate(amps, c_full, c_interior)
            else:
                L = self.loss.evaluate(amps, vm_c_const)
            L.backward()
            return L

        log_every = max(1, int(self.loss.callback.log_every))

        for i in range(n_iter):

            loss_value = optimiser.step(closure)

            if (
                is_inverse
                and self.c_update_every > 0
                and (i + 1) % self.c_update_every == 0
            ):
                with torch.no_grad():
                    amps = torch.cat(
                        [zero_row_S, ic_row_S, u_inner_param.detach()], dim=1
                    )
                    c_star_full, illum_full = self.loss.config.wave_eq.c_closed_form(
                        amps,
                        self.loss.sources,
                        c_current=vm_in.build_full_c(c_interior),
                    )
                    c_star = c_star_full[grid.interior_slice]
                    c_star = self._prox_regularise(
                        c_star, illum_full[grid.interior_slice]
                    )
                    dc_rms = float((c_star - c_interior).pow(2).mean().sqrt())
                    ratio_pre = self.loss.pde_src_ratio()
                    c_interior.mul_(1.0 - alpha).add_(alpha * c_star)
                    if c_min is not None or c_max is not None:
                        c_interior.clamp_(min=c_min, max=c_max)
                    # Same u, new c, before the optimiser reacts. With
                    # alpha=1 and no clamp/prox this can only drop (c* is
                    # the argmin given u); the size of the drop measures
                    # how much c-signal the u-phase accumulated.
                    r_post = self.loss.config.wave_eq.residual(
                        amps, vm_in.build_full_c(c_interior), self.loss.sources
                    )
                    ratio_post = float(r_post.pow(2).mean().sqrt()) / max(
                        self.loss.src_rms, 1e-30
                    )
                # The L-BFGS curvature history refers to the old c: reset it.
                optimiser = make_optimiser()
                print(
                    f"Iteration: {i} | closed-form c-update "
                    f"(alpha={alpha:g}, |c* - c|_rms = {dc_rms:.3e}, "
                    f"|r_pde|/|src| {ratio_pre:.3e} -> {ratio_post:.3e})"
                )

            should_log = (i % log_every == 0) or (i == n_iter - 1)

            # Build the on-device c_full only when something will read it.
            # When neither log_c nor on_iteration needs it, skip the work.
            c_full_now = None
            if is_inverse and (should_log or on_iteration is not None):
                c_full_now = vm_in.build_full_c(c_interior).detach()

            if on_iteration is not None:
                on_iteration(i, c_full_now)

            if should_log:
                # Single device->host sync per logging step.
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

        # Build returned wavefields with hard zero IC row, all sharing one
        # VelocityModel reference so we don't carry n_shots copies of c.
        if isinstance(self.loss, ForwardLoss):
            vm_out = vm_in  # medium was fixed; reuse the input model
        else:
            c_full_final = vm_in.build_full_c(c_interior).detach()
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


class FirstOrderOptimiser(Optimiser):
    """Shared minimise loop for first-order optimisers (Adam, GD).

    Matches :class:`LBFGSB` setup: batched ``(n_shots, NT-2, NX, NY)`` amplitudes
    and hard zero + velocity IC rows. For inverse problems, ``u`` and ``c`` are
    updated alternately with separate PyTorch optimisers (block coordinate descent).
    """

    _TORCH_OPTIM_CLS: Optional[Type[torch.optim.Optimizer]] = None
    _DEFAULT_OPTS: dict = {}
    _TORCH_OPT_KEYS: frozenset = frozenset()
    _SCHEDULE_KEYS = frozenset({"u_steps", "c_steps", "c_lr", "c_max_grad_norm"})

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
            c_interior_param = torch.nn.Parameter(
                vm_c_const[grid.interior_slice].detach().clone()
            )
        else:
            c_interior_param = None

        u_optimiser = self._TORCH_OPTIM_CLS([u_inner_param], **torch_opts)

        if is_inverse:
            c_torch_opts = dict(torch_opts)
            c_torch_opts["lr"] = c_lr
            c_optimiser = self._TORCH_OPTIM_CLS([c_interior_param], **c_torch_opts)
        else:
            c_optimiser = None

        c_min, c_max = self.c_min, self.c_max
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None

        for i in range(n_iter):
            for _ in range(u_steps):
                u_optimiser.zero_grad()
                amps = torch.cat([zero_row_S, ic_row_S, u_inner_param], dim=1)
                if is_inverse:
                    c_fixed = c_interior_param.detach()
                    c_full = vm_in.build_full_c(c_fixed)
                    L_u = self.loss.evaluate(amps, c_full, c_fixed)
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
                for _ in range(c_steps):
                    c_optimiser.zero_grad()
                    amps_fixed = torch.cat(
                        [zero_row_S, ic_row_S, u_inner_param.detach()],
                        dim=1,
                    )
                    c_full = vm_in.build_full_c(c_interior_param)
                    L_c = self.loss.evaluate(
                        amps_fixed,
                        c_full,
                        c_interior_param,
                    )
                    L_c.backward()
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
                c_full_now = vm_in.build_full_c(c_interior_param.detach())

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
            c_full_final = vm_in.build_full_c(c_interior_param.detach())
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
    }