from typing import Tuple

from src.loss.utils import LossConfig, LossTape
from .base import DiscreteLoss

import torch


class InverseLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def __init__(
        self, observed_wavefield, config: LossConfig, callback: LossTape | None = None
    ):
        super().__init__(config, callback)
        self.d_obs = observed_wavefield

    def evaluate(
        self, wavefield: torch.Tensor, wavespeed: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:

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
        return L.item(), grad

    def _eval_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [(r**2).mean() for r in residuals]
        )  # normalise for inverse problem

    def _residuals(self, wavefield, wavespeed) -> torch.Tensor:
        r_pde = self._eval_pde_loss(wavefield, wavespeed)
        r_data = self._eval_data_loss(wavefield)
        return torch.cat([r_pde, r_data])

    def _eval_pde_loss(self, wavefield, wavespeed):
        utt = self.time_op.apply(wavefield)
        lap = self.lap.apply(wavefield)
        return utt - (wavespeed**2) * lap  # u_tt - c^2(u_xx + u_yy)

    def _eval_data_loss(self, wavefield):
        return wavefield - self.d_obs
