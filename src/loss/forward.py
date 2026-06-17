from typing import Tuple
from .base import DiscreteLoss

import torch
import numpy as np


class ForwardLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def evaluate(
        self, wavefield: torch.Tensor, wavespeed: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        wavefield.requires_grad_()
        r = self._residuals(wavefield, wavespeed)
        L = self._eval_loss(r)
        L.backward()
        grad = (
            wavefield.grad
            if wavefield.grad is not None
            else torch.zeros_like(wavefield)
        )

        self.callback.log(L.item(), r)
        return L.item(), grad.numpy()  # returns loss, grad together

    def _eval_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        return (residuals**2).sum()  # unnormalised for forward problem

    def _residuals(self, wavefield, wavespeed) -> torch.Tensor:
        r_pde = self._eval_pde_loss(wavefield, wavespeed)
        return r_pde

    def _eval_pde_loss(self, wavefield, wavespeed):
        utt = self.time_op.apply(wavefield)
        lap = self.lap.apply(wavefield)
        return utt - (wavespeed**2) * lap  # u_tt - c^2(u_xx + u_yy)
