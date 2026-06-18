from typing import Tuple
from .base import DiscreteLoss

import torch
import numpy as np


class ForwardLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def evaluate(self, data: np.ndarray) -> Tuple[float, np.ndarray]:
        d = torch.tensor(data, requires_grad=True, dtype=torch.float64)

        # extract and reshape amplitude
        Nt = self.config.wavefield.grid.nt
        Nx, Ny = self.config.wavefield.grid.shape
        amp = d[: self.config.speed_offset].reshape(Nt, Nx, Ny)

        #  get wavespeed separately, we need it to compute the PDE residuals
        wsp = self.config.wavefield.wavespeed

        r = self._residuals(amp, wsp)
        L = self._eval_loss(r)
        L.backward()
        grad = d.grad if d.grad is not None else torch.zeros_like(d)

        self.evaluations += 1

        if self.evaluations % self.callback.log_every == 0:
            self.callback.log(L.item(), (r,))  # tuple for consistency with inverse loss
        return L.item(), grad.numpy()  # returns loss, grad together

    def _eval_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        return (residuals**2).mean()

    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        r_pde = self._eval_pde_loss(amp, wsp)
        return r_pde

    def _eval_pde_loss(self, amp, wsp):
        utt = self.time_op.apply(amp)
        lap = self.lap.apply(amp)
        return utt - (wsp**2) * lap  # u_tt - c^2(u_xx + u_yy)
