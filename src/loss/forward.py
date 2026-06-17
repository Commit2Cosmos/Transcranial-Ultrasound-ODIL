from typing import Tuple
from .base import DiscreteLoss

import torch


class ForwardLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def evaluate(
        self, wavefield: torch.Tensor, wavespeed: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        wavefield.requires_grad_()
        r = self._residuals(wavefield, wavespeed)
        L = (r**2).sum()
        self.callback.log(L.item(), r)
        L.backward()
        grad = (
            wavefield.grad
            if wavefield.grad is not None
            else torch.zeros_like(wavefield)
        )
        return L.item(), grad

    def _residuals(self, wavefield, wavespeed) -> torch.Tensor:
        r_pde = self._eval_pde_loss(wavefield, wavespeed)
        return r_pde

    def _eval_pde_loss(self, wavefield, wavespeed):
        utt = self.time_op.apply(wavefield)
        lap = self.lap.apply(wavefield)
        return utt - (wavespeed**2) * lap  # u_tt - c^2(u_xx + u_yy)
