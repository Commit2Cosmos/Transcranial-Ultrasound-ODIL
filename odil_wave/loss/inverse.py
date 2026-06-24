from typing import Tuple

import torch

from .base import DiscreteLoss
from .utils import LossConfig, LossTape


class InverseLoss(DiscreteLoss):
    """Joint (u, c) inverse-problem loss.

    L = w_pde  * sum_s mean(r_pde_s  ** 2)
      + w_data * sum_s mean(r_data_s ** 2)
      + w_reg  * R(c_interior)            (if a Regulariser is attached)
    """

    def __init__(
        self,
        observed_wavefield,
        config: LossConfig,
        callback: LossTape | None = None,
    ):
        super().__init__(config, callback)
        self.d_obs = torch.as_tensor(
            observed_wavefield, dtype=self.config.dtype, device=self.config.device
        )

    def _eval_pde_loss(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int
    ) -> torch.Tensor:
        return self.config.wave_eq.residual(amp, wsp, self.sources[shot_idx])

    def _eval_data_loss(self, d_syn: torch.Tensor, shot_idx: int) -> torch.Tensor:
        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        return d_syn[:, i, j] - self.d_obs[shot_idx, :, i, j]

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._eval_pde_loss(amp, wsp, shot_idx),
            self._eval_data_loss(amp, shot_idx),
        )

    def evaluate(
        self,
        amp: torch.Tensor,
        c_full: torch.Tensor,
        c_interior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_shots = amp.shape[0]
        residuals = [self._residuals(amp[s], c_full, s) for s in range(n_shots)]

        w = self.config.weights
        pde_terms = torch.stack([torch.mean(r[0] ** 2) for r in residuals]).sum()
        data_terms = torch.stack([torch.mean(r[1] ** 2) for r in residuals]).sum()
        L = w["pde"] * pde_terms + w["data"] * data_terms

        if self.config.regulariser is not None and c_interior is not None:
            L = L + w["reg"] * self.config.regulariser(c_interior)

        self.evaluations += 1
        if self.evaluations % self.callback.log_every == 0:
            r_pde = torch.stack([r[0].detach() for r in residuals])
            r_data = torch.stack([r[1].detach() for r in residuals])
            self.callback.log(L.item(), (r_pde, r_data))

        return L
