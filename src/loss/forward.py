from typing import Tuple
from .base import DiscreteLoss
from .utils import LossConfig, LossTape

import torch
import numpy as np


class ForwardLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def __init__(self, wavespeed, config: LossConfig, callback: LossTape | None = None):
        super().__init__(config, callback)
        self.c: torch.Tensor = wavespeed

    def evaluate(self, data: np.ndarray) -> Tuple[float, np.ndarray]:
        d = torch.tensor(data, requires_grad=True, dtype=torch.float64)
        r = self._residuals(d)
        L = self._eval_loss(r)
        L.backward()
        grad = d.grad if d.grad is not None else torch.zeros_like(d)

        self.callback.log(L.item(), r)
        return L.item(), grad.numpy()  # returns loss, grad together

    def _eval_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        return (residuals**2).mean()

    def _residuals(self, data: torch.Tensor) -> torch.Tensor:
        r_pde = self._eval_pde_loss(data)
        return r_pde

    def _eval_pde_loss(self, data: torch.Tensor):
        utt = self.time_op.apply(data)
        lap = self.lap.apply(data)
        return utt - (self.c**2) * lap  # u_tt - c^2(u_xx + u_yy)
