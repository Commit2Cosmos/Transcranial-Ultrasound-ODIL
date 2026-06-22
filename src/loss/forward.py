from typing import Tuple
from .base import DiscreteLoss

import torch
import numpy as np


class ForwardLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def _eval_pde_loss(self, amp, wsp, shot_idx: int) -> torch.Tensor:
        return self.config.wave_eq.residual(
            amp, wsp, self.sources[shot_idx]
        )  # u_tt - c^2(u_xx + u_yy) - f

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int
    ) -> torch.Tensor:
        r_pde = self._eval_pde_loss(amp, wsp, shot_idx)
        return r_pde

    def _eval_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        return torch.sum(residuals**2)

    def evaluate(self, data: np.ndarray) -> Tuple[float, np.ndarray]:
        d = torch.tensor(
            data, requires_grad=True, dtype=self.config.dtype, device=self.config.device
        )

        # extract and reshape amplitude
        Nt = self.config.wavefield.grid.nt
        Nx, Ny = self.config.wavefield.grid.shape
        n_shots = self.config.geometry.n_sources
        amp = d.reshape(n_shots, Nt, Nx, Ny)

        #  get wavespeed separately, we need it to compute the PDE residuals
        wsp = self.config.wavefield.wavespeed

        residuals = [self._residuals(amp[s], wsp, s) for s in range(n_shots)]
        L = torch.stack([self._eval_loss(r) for r in residuals]).sum()

        # detach beofre .backward frees graph
        r_pde = torch.stack([r.detach() for r in residuals])

        L.backward()
        grad = d.grad if d.grad is not None else torch.zeros_like(d)

        self.evaluations += 1

        if self.evaluations % self.callback.log_every == 0:
            self.callback.log(
                L.item(), (r_pde,)
            )  # tuple for consistency with inverse loss

        return L.item(), grad.cpu().numpy()  # returns loss, grad together
