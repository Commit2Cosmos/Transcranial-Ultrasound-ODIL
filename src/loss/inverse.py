from typing import Tuple

from src.loss.utils import LossConfig, LossTape
from geometry import AcquisitionGeometry
from .base import DiscreteLoss

import torch
import numpy as np


class InverseLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def __init__(
        self,
        observed_wavefield,
        geometry: AcquisitionGeometry,
        config: LossConfig,
        callback: LossTape | None = None,
    ):
        super().__init__(config, callback)
        self.d_obs = observed_wavefield
        self.geometry = geometry

    def evaluate(self, data: np.ndarray) -> Tuple[float, np.ndarray]:
        # scipy wants a function that takes a flat np.ndarray and returns loss, jac
        # we need a torch.Tensor to compute the loss and gradient
        d = torch.tensor(data, requires_grad=True, dtype=torch.float64)

        # reshape out so operators can do their thing
        Nt = self.config.wavefield.grid.nt
        Nx, Ny = self.config.wavefield.grid.shape
        amp = d[: self.config.speed_offset].reshape(Nt, Nx, Ny)
        wsp = d[: self.config.speed_offset].reshape(Nx, Ny)

        # cmpute residuals and evaluate loss
        r = self._residuals(amp, wsp)
        L = self._eval_loss(r)

        # backprop and extract gradient
        L.backward()
        grad = d.grad if d.grad is not None else torch.zeros_like(d)

        # log
        self.evaluations += 1
        if self.evaluations % self.callback.log_every == 0:
            self.callback.log(L.item(), r)

        return L.item(), grad.numpy()  # return loss, grad together for scipy

    def _eval_pde_loss(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        utt = self.time_op.apply(amp)
        lap = self.lap.apply(amp)
        return utt - (wsp**2) * lap  # u_tt - c^2(u_xx + u_yy)

    def _eval_data_loss(self, d_syn: torch.Tensor):
        i, j = self.geometry.recv_ij[:, 0], self.geometry.recv_ij[:, 1]
        return d_syn[:, i, j] - self.d_obs[:, i, j]  # observed data at receivers

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r_pde = self._eval_pde_loss(amp, wsp)
        r_data = self._eval_data_loss(amp)
        return r_pde, r_data

    def _eval_loss(self, residuals: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r_pde, r_data = residuals
        return (r_pde**2).mean() + (r_data**2).mean()  # normalise for inverse problem
