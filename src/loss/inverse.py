from typing import Tuple

from src.loss.utils import LossConfig, LossTape
from .base import DiscreteLoss

import torch
import numpy as np


class InverseLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def __init__(
        self, observed_wavefield, config: LossConfig, callback: LossTape | None = None
    ):
        super().__init__(config, callback)
        self.d_obs = observed_wavefield

    def evaluate(self, data: np.ndarray) -> Tuple[float, np.ndarray]:

        d = torch.tensor(data, requires_grad=True, dtype=torch.float64)
        r = self._residuals(d)
        L = self._eval_loss(r)
        L.backward()
        grad = d.grad if d.grad is not None else torch.zeros_like(d)

        self.callback.log(L.item(), r)
        return L.item(), grad.numpy()  # returns loss, grad together

    def _eval_loss(self, residuals: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r_pde, r_data = residuals
        return (r_pde**2).mean() + (r_data**2).mean()  # normalise for inverse problem

    def _residuals(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r_pde = self._eval_pde_loss(data)
        r_data = self._eval_data_loss(data)
        return r_pde, r_data

    def _eval_pde_loss(self, data: torch.Tensor):
        amp_idx = self.config.speed_offset  # idx to split amplitude and wave speed data
        utt = self.time_op.apply(data[:amp_idx])
        lap = self.lap.apply(data[:amp_idx])
        return utt - (data[amp_idx:] ** 2) * lap  # u_tt - c^2(u_xx + u_yy)

    def _eval_data_loss(self, wavefield):
        return wavefield - self.d_obs
