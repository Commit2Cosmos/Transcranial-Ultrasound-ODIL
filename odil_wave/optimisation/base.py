from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from odil_wave.loss import DiscreteLoss, ForwardLoss, InverseLoss
from odil_wave.loss.utils import LossTape
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
    """`torch.optim.LBFGS`-based optimiser with optional velocity guards.

    Hard IC and frozen-PML c:
      - per-shot amplitudes are `(NT-1, NX, NY)` parameters; a zero row is
        spliced at t=0 inside the closure (and on the returned `Wavefield`s).
      - the c parameter has interior shape `(interior_nx, interior_ny)`;
        `LossConfig.build_full_c` pads to `(NX, NY)` with `pml_c` on the
        PML ring.

    Velocity guards (inverse problems only):
      After every L-BFGS step the interior wavespeed parameter is projected
      back onto [c_min, c_max] by clamping.  This
      It guarantees physical velocities at every iteration.
    """

    _DEFAULT_OPTS = {
        "n_iter": 100,
        "max_iter": 20,
        "history_size": 100,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
    }

    def __init__(
        self,
        wavefield: Wavefield,
        loss: DiscreteLoss,
        c_min: Optional[float] = None,
        c_max: Optional[float] = None,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        self.c_min = c_min
        self.c_max = c_max
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

    def minimise(self, **overrides) -> Tuple[List[Wavefield], LossTape]:
        self.opts.update(overrides)
        n_iter, torch_opts = self._split_opts()

        grid = self.wavefield.grid
        dtype = grid.dtype
        device = grid.device
        Nx, Ny = grid.shape
        n_shots = self.loss.config.geometry.n_sources

        zero_row = torch.zeros(1, Nx, Ny, dtype=dtype, device=device)

        # per-shot amplitudes (NT-1, NX, NY), zero IC row spliced in closure
        amp_seed = (
            self.wavefield.amplitude[1:].detach().clone().to(dtype=dtype, device=device)
        )
        u_inner_params = [torch.nn.Parameter(amp_seed.clone()) for _ in range(n_shots)]

        if isinstance(self.loss, InverseLoss):
            c0 = self.wavefield.wavespeed.detach().to(dtype=dtype, device=device)
            c0_int = c0[grid.interior_slice].clone()
            c_interior_param = torch.nn.Parameter(c0_int)
            params = u_inner_params + [c_interior_param]
        else:
            c_interior_param = None
            params = list(u_inner_params)

        optimiser = torch.optim.LBFGS(params, **torch_opts)

        # Capture guard values in local vars for use inside closure.
        c_min, c_max = self.c_min, self.c_max

        def closure():
            optimiser.zero_grad()
            # Project c into [c_min, c_max] before every loss evaluation so
            # every step stays in bounds.
            if c_interior_param is not None and (c_min is not None or c_max is not None):
                with torch.no_grad():
                    c_interior_param.data.clamp_(min=c_min, max=c_max)
            amps = torch.stack(
                [
                    torch.cat([zero_row, u_inner_params[s]], dim=0)
                    for s in range(n_shots)
                ]
            )
            if isinstance(self.loss, InverseLoss):
                c_full = self.loss.config.build_full_c(c_interior_param)
                L = self.loss.evaluate(amps, c_full, c_interior_param)
            else:
                wsp = self.wavefield.wavespeed
                L = self.loss.evaluate(amps, wsp)
            L.backward()
            return L

        log_every = max(1, int(self.loss.callback.log_every))
        for i in range(n_iter):
            loss_value = optimiser.step(closure)
            if isinstance(self.loss, InverseLoss):
                # Final clamp after the full step.
                if c_min is not None or c_max is not None:
                    with torch.no_grad():
                        c_interior_param.data.clamp_(min=c_min, max=c_max)

                c_full_now = (
                    self.loss.config.build_full_c(c_interior_param)
                    .detach()
                    .cpu()
                    .numpy()
                )
                self.loss.callback.log_c(c_full_now)

            if i % log_every == 0 or i == n_iter - 1:
                self.loss.callback.log(
                    float(loss_value.detach().cpu()),
                    self.loss._last_residuals,
                )

            print(f"Iteration: {i}")

        # Build returned wavefields with hard zero IC row and reshaped c.
        outputs: List[Wavefield] = []
        if isinstance(self.loss, ForwardLoss):
            wsp_full = self.wavefield.wavespeed.detach()
            for s in range(n_shots):
                wf = Wavefield(grid=grid, init_wavespeed=wsp_full)
                amp_full = torch.cat([zero_row, u_inner_params[s].detach()], dim=0)
                wf.amplitude = amp_full
                outputs.append(wf)
        else:
            c_full_final = self.loss.config.build_full_c(c_interior_param).detach()
            for s in range(n_shots):
                wf = Wavefield(grid=grid, init_wavespeed=c_full_final)
                amp_full = torch.cat([zero_row, u_inner_params[s].detach()], dim=0)
                wf.amplitude = amp_full
                outputs.append(wf)

        self.loss.callback.result = {
            "loss": (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            ),
            "n_outer_iter": n_iter,
        }
        return outputs, self.loss.callback
    