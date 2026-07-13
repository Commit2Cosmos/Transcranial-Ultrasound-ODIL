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
        # c-gradient preconditioning: divide grad_c by (mean u^2 energy + stab * max)
        "c_precond": False,
        "c_precond_stab": 1e-2,
    }
    _LBFGS_KEYS = frozenset({
        "lr", "max_iter", "max_eval",
        "tolerance_grad", "tolerance_change",
        "history_size", "line_search_fn",
    })
    _SCHEDULE_KEYS = frozenset({
        "u_steps", "c_steps", "c_lr", "c_max_iter", "c_history_size",
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

    def _split_opts(self) -> Tuple[int, int, int, dict, dict, bool, float]:
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        u_steps = int(opts.pop("u_steps", 1))
        c_steps = int(opts.pop("c_steps", 1))
        c_lr = float(opts.pop("c_lr", 1.0))
        c_max_iter = int(opts.pop("c_max_iter", opts.get("max_iter", 4)))
        c_history_size = int(opts.pop("c_history_size", opts.get("history_size", 10)))
        c_precond = bool(opts.pop("c_precond", False))
        c_precond_stab = float(opts.pop("c_precond_stab", 1e-2))

        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}

        c_torch_opts = dict(u_torch_opts)
        c_torch_opts["lr"] = c_lr
        c_torch_opts["max_iter"] = c_max_iter
        c_torch_opts["history_size"] = c_history_size

        return n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts, c_precond, c_precond_stab

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
        n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts, c_precond, c_precond_stab = self._split_opts()

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
            # Reparameterise: optimise c new = c / c_ref (dimensionless, order 1).
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
        _prec: Optional[torch.Tensor] = None
        _eps_prec: float = 0.0

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
                # Wavefield-energy preconditioner: prec[x,y] = mean over shots
                # and time of u^2.  Dividing grad_c by this dampens updates in
                # well-illuminated regions and amplifies them where energy is low.
                if c_precond:
                    with torch.no_grad():
                        u_sq = u_inner_param.detach().pow(2).mean(dim=(0, 1))  # (NX, NY)
                        _prec = u_sq[grid.interior_slice]
                        _eps_prec = c_precond_stab * float(
                            _prec.max().clamp(min=1e-30)
                        )

                # u just changed — c's curvature history is stale, reset it.
                c_optimiser = make_c_optimiser()
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
                        #L_c.backward()
                        #if (
                        #    c_precond
                        #    and c_interior_param.grad is not None
                        #    and _prec is not None
                        #):
                        #    with torch.no_grad():
                        #        c_interior_param.grad.div_(_prec + _eps_prec)
                        #return L_c
                        L_c.backward()

                        # Debug c-gradient BEFORE preconditioning
                        if c_interior_param.grad is not None:
                            with torch.no_grad():
                                g = c_interior_param.grad

                                print("raw c grad norm:", g.norm().item())
                                print("raw c grad min/max:", g.min().item(), g.max().item())
                                print("raw c grad RMS:", g.pow(2).mean().sqrt().item())

                        if (
                            c_precond
                            and _prec is not None
                            and c_interior_param.grad is not None
                        ):
                            with torch.no_grad():
                                c_interior_param.grad.div_(_prec + _eps_prec)

                                # Debug c-gradient AFTER preconditioning
                                g = c_interior_param.grad
                                print("precond c grad norm:", g.norm().item())
                                print("precond c grad min/max:", g.min().item(), g.max().item())
                                print("precond c grad RMS:", g.pow(2).mean().sqrt().item())

                        return L_c
                    loss_value = c_optimiser.step(c_closure)
                    with torch.no_grad():
                        if c_min is not None or c_max is not None:
                            c_interior_param.clamp_(min=c_min, max=c_max)
                # c just changed — u's curvature history is stale, reset it.
                # u_optimiser = make_u_optimiser()

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
