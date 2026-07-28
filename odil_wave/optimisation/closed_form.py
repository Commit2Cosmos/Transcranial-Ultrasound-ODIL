"""Block-coordinate frequency-domain FWI with a closed-form ``c`` update.

Same u-block as :class:`~odil_wave.optimisation.base.LBFGSB` (complex L-BFGS on
the wavefield with ``c`` fixed), but the inner c-optimiser is replaced by a
single per-cell variable-projection solve
``c* = c0 √( Σ Re(ā b) / Σ |a|² )`` (see
:meth:`odil_wave.operator.utils.WaveEquation.c_closed_form`).
"""

from typing import List, Tuple

import torch

from odil_wave.loss import DiscreteLoss, ForwardLoss, InverseLoss
from odil_wave.loss.utils import LossTape
from odil_wave.models import VelocityModel
from odil_wave.wavefield import Wavefield

from .base import Optimiser


class LBFGSClosedForm(Optimiser):
    """Frequency-domain FWI: L-BFGS on ``u``, closed-form ``c``.

    Each outer iteration runs ``u_steps`` of complex u-L-BFGS with ``c`` fixed,
    then (every ``c_update_every`` outer iterations) replaces the whole c-L-BFGS
    block of :class:`LBFGSB` with one exact per-cell update::

        c* = c0 √( Σ_{shots,freq} Re(ā b) / Σ |a|² ),
        a = ∇'^2 u,   b = λ_tt u - f̂' + sponge(u),

    optionally relaxed as ``c ← (1-α) c + α c*`` (``c_relax=α``) and clamped to
    ``[c_min, c_max]``. If a regulariser is configured, a tiny illumination-
    weighted proximal solve applies it to ``c*`` after the projection.

    Because the data term does not depend on ``c``, ``c*`` is the argmin of the
    full (pde + data) loss over ``c`` for the current ``u`` — no c-step size,
    preconditioner, or line search to tune.
    """

    _DEFAULT_OPTS = {
        "n_iter": 100,
        "u_steps": 1,
        "max_iter": 4,
        "history_size": 10,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
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
        c_update_every: int = 1,
        c_relax: float = 1.0,
        illum_rel_floor: float = 1e-6,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        grid = wavefield.grid
        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.u_init = u_init
        self.c_update_every = int(c_update_every)
        self.c_relax = float(c_relax)
        self.illum_rel_floor = float(illum_rel_floor)
        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self):
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        u_steps = int(opts.pop("u_steps", 1))
        # Silently drop keys only the L-BFGS c-block understands.
        for legacy in (
            "c_steps",
            "c_lr",
            "c_max_iter",
            "c_history_size",
            "c_precond",
            "c_precond_type",
            "c_precond_sigma",
            "c_precond_stab",
            "reset_c_history",
            "u_precond",
            "c_update",
        ):
            opts.pop(legacy, None)
        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}
        return n_iter, u_steps, u_torch_opts

    def _seed_complex(self, n_shots, cdtype, device) -> torch.Tensor:
        """``(n_shots, nf, nx, ny)`` complex seed from ``u_init``/wavefield."""
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
        """Solve ``min_c 0.5 Σ w (c - c*)² + λ R(c)`` on the interior map,
        with ``w = illum / mean(illum)``. No-op without a regulariser.
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

    def minimise(
        self,
        on_iteration=None,
        **overrides,
    ) -> Tuple[List[Wavefield], LossTape]:
        """Run the u-L-BFGS / closed-form-c block-coordinate loop."""
        self.opts.update(overrides)
        n_iter, u_steps, u_torch_opts = self._split_opts()

        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        wave_eq = self.loss.config.wave_eq
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
            # Plain physical tensor: c is updated in closed form, never by L-BFGS.
            c_interior = vm_c_const[grid.interior_slice].clone()
        else:
            c_interior = None

        def make_u_optimiser():
            return torch.optim.LBFGS([u_real, u_imag], **u_torch_opts)

        u_optimiser = make_u_optimiser()
        c_min, c_max = self.c_min, self.c_max
        alpha = self.c_relax
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None

        def u_closure():
            u_optimiser.zero_grad()
            amps = pack_u()
            if is_inverse:
                c_full = vm_in.build_full_c(c_interior)
                L = self.loss.evaluate(amps, c_full)
            else:
                L = self.loss.evaluate(amps, vm_c_const)
            L.backward()
            return L

        for i in range(n_iter):
            for _ in range(u_steps):
                loss_value = u_optimiser.step(u_closure)

            did_c_update = (
                is_inverse
                and self.c_update_every > 0
                and (i + 1) % self.c_update_every == 0
            )
            if did_c_update:
                with torch.no_grad():
                    amps = pack_u_detached()
                    c_full_cur = vm_in.build_full_c(c_interior)
                    c_star_full, illum_full = wave_eq.c_closed_form(
                        amps,
                        self.loss.sources,
                        c_current=c_full_cur,
                        illum_rel_floor=self.illum_rel_floor,
                    )
                    c_star = c_star_full[grid.interior_slice]
                    c_star = self._prox_regularise(
                        c_star, illum_full[grid.interior_slice]
                    )
                    c_interior.mul_(1.0 - alpha).add_(alpha * c_star)
                    if c_min is not None or c_max is not None:
                        c_interior.clamp_(min=c_min, max=c_max)
                # New c ⇒ the u-L-BFGS curvature history is stale.
                u_optimiser = make_u_optimiser()

            should_log = (i % log_every == 0) or (i == n_iter - 1)

            # Refresh loss / residuals at the current (u, c) so logging reflects
            # the closed-form c-update, not the pre-update u-step.
            c_full_now = None
            if should_log or on_iteration is not None or is_inverse:
                with torch.no_grad():
                    if is_inverse:
                        c_full_now = vm_in.build_full_c(c_interior)
                        loss_value = self.loss.evaluate(
                            pack_u_detached(), c_full_now, c_interior
                        )
                    else:
                        loss_value = self.loss.evaluate(pack_u_detached(), vm_c_const)

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
                tag = " | closed-form c" if did_c_update else ""
                print(
                    f"Iteration: {i} | loss = {loss_scalar:.6e} | "
                    f"|r_pde|/|src| = {ratio:.3e}{tag}"
                )

        if isinstance(self.loss, ForwardLoss):
            vm_out = vm_in
        else:
            c_full_final = vm_in.build_full_c(c_interior)
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
