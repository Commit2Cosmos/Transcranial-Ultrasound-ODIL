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
    """`torch.optim.LBFGS`-based optimiser with separate u and c LBFGS phases.

    Each
    outer iteration runs ``u_steps`` LBFGS steps on the wavefield (c detached)
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
    }
    _LBFGS_KEYS = frozenset({
        "lr", "max_iter", "max_eval",
        "tolerance_grad", "tolerance_change",
        "history_size", "line_search_fn",
    })
    _SCHEDULE_KEYS = frozenset({"u_steps", "c_steps", "c_lr", "c_max_iter", "c_history_size"})

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

    def _split_opts(self) -> Tuple[int, int, int, dict, dict]:
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        u_steps = int(opts.pop("u_steps", 1))
        c_steps = int(opts.pop("c_steps", 1))
        c_lr = float(opts.pop("c_lr", 1.0))
        c_max_iter = int(opts.pop("c_max_iter", opts.get("max_iter", 4)))
        c_history_size = int(opts.pop("c_history_size", opts.get("history_size", 10)))

        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}

        c_torch_opts = dict(u_torch_opts)
        c_torch_opts["lr"] = c_lr
        c_torch_opts["max_iter"] = c_max_iter
        c_torch_opts["history_size"] = c_history_size

        return n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts

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
        """Run the block-coordinate LBFGS optimisation loop."""
        self.opts.update(overrides)
        n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts = self._split_opts()

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
            # Reparameterise: optimise ĉ = c / c_ref (dimensionless, order ~1).
            # This brings the c gradient into the same order of magnitude as u,
            # which is required for LBFGS/GD line searches to succeed.
            c_ref = float(c0_int.mean().item())
            c_interior_param = torch.nn.Parameter(c0_int / c_ref)
        else:
            c_ref = None
            c_interior_param = None

        def make_u_optimiser():
            return torch.optim.LBFGS([u_inner_param], **u_torch_opts)

        def make_c_optimiser():
            return torch.optim.LBFGS([c_interior_param], **c_torch_opts)

        u_optimiser = make_u_optimiser()
        c_optimiser = make_c_optimiser() if (is_inverse and c_steps > 0) else None

        c_min = self.c_min / c_ref if (self.c_min is not None and c_ref is not None) else None
        c_max = self.c_max / c_ref if (self.c_max is not None and c_ref is not None) else None
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None

        for i in range(n_iter):

            for _ in range(u_steps):
                def u_closure():
                    u_optimiser.zero_grad()
                    amps = torch.cat([zero_row_S, ic_row_S, u_inner_param], dim=1)
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
                for _ in range(c_steps):
                    def c_closure():
                        c_optimiser.zero_grad()
                        amps_fixed = torch.cat(
                            [zero_row_S, ic_row_S, u_inner_param.detach()], dim=1
                        )
                        c_phys = c_interior_param * c_ref
                        c_full = vm_in.build_full_c(c_phys)
                        L_c = self.loss.evaluate(
                            amps_fixed, c_full, c_phys
                        )
                        L_c.backward()
                        with torch.no_grad():
                            print(
                                "c loss", float(L_c.detach().cpu()),
                                "c grad is None?", c_interior_param.grad is None,
                                "c grad norm", None if c_interior_param.grad is None else float(c_interior_param.grad.norm().detach().cpu()),
                                "c min/max", float(c_phys.min().detach().cpu()), float(c_phys.max().detach().cpu()),
                            )
                        return L_c
                    loss_value = c_optimiser.step(c_closure)
                    with torch.no_grad():
                        if c_min is not None or c_max is not None:
                            c_interior_param.clamp_(min=c_min, max=c_max)
                # u curvature history is stale after c changes — reset.
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