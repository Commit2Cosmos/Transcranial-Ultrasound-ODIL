from typing import Tuple

from src.loss.utils import LossConfig, LossTape
from .base import DiscreteLoss

import torch
import numpy as np


class InverseLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def __init__(
        self,
        observed_wavefield,
        config: LossConfig,
        callback: LossTape | None = None,
    ):
        super().__init__(config, callback)
        self.d_obs = torch.as_tensor(
            observed_wavefield,
            dtype=torch.float64,
            device=self.config.device,
        )

    def _eval_pde_loss(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int
    ) -> torch.Tensor:
        return self.config.wave_eq.residual(
            amp, wsp, self.sources[shot_idx]
        )  # u_tt - c^2(u_xx + u_yy) - f

    def _eval_data_loss(self, d_syn: torch.Tensor, shot_idx: int) -> torch.Tensor:
        i, j = self.config.geometry.recv_ij[:, 0], self.config.geometry.recv_ij[:, 1]
        return (
            d_syn[:, i, j] - self.d_obs[shot_idx, :, i, j]
        )  # observed data for this source at receivers

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r_pde = self._eval_pde_loss(amp, wsp, shot_idx)
        r_data = self._eval_data_loss(amp, shot_idx)
        return r_pde, r_data

    def _eval_loss(self, residuals: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r_pde, r_data = residuals
        return torch.mean(r_pde**2) + torch.mean(
            r_data**2
        )  # normalise for inverse problem

    def evaluate(self, data: np.ndarray) -> Tuple[float, np.ndarray]:
        # scipy wants a function that takes a flat np.ndarray and returns loss, jac
        # we need a torch.Tensor to compute the loss and gradient
        d = torch.tensor(
            data, requires_grad=True, dtype=torch.float64, device=self.config.device
        )

        # reshape out so operators can do their thing
        Nt = self.config.wavefield.grid.nt
        Nx, Ny = self.config.wavefield.grid.shape
        n_shots = self.config.geometry.n_sources

        amp = d[: self.config.speed_offset].reshape(n_shots, Nt, Nx, Ny)
        wsp = d[self.config.speed_offset :].reshape(Nx, Ny)

        # cmpute residuals for each shot, then sum
        residuals = [self._residuals(amp[s], wsp, s) for s in range(n_shots)]
        L = torch.stack([self._eval_loss(r) for r in residuals]).sum()

        # detach before .backward frees graph
        r_pde = torch.stack([r[0].detach() for r in residuals])
        r_data = torch.stack([r[1].detach() for r in residuals])

        # backprop and extract gradient
        L.backward()
        grad = d.grad if d.grad is not None else torch.zeros_like(d)

        # log
        self.evaluations += 1
        if self.evaluations % self.callback.log_every == 0:
            self.callback.log(L.item(), (r_pde, r_data))

        return L.item(), grad.cpu().numpy()  # return loss, grad together for scipy
