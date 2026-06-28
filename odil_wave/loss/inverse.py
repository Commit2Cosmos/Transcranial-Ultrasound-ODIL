from typing import Tuple

import torch

from odil_wave.wavefield import Wavefield

from .base import DiscreteLoss
from .utils import LossConfig, LossTape


class InverseLoss(DiscreteLoss):
    """Joint (u, c) inverse-problem loss.

    L = w_pde  * sum_s mean(r_pde_s  ** 2)
      + w_data * sum_s mean(r_data_s ** 2)
      + w_reg  * R(c_interior)            (if a Regulariser is attached)

    `normalize_data` rescales the receiver traces that enter the data
    residual (the only place the wavefield meets observations in a
    loss-relevant way). Modes:
      - None / "none":   off — raw amplitudes (default, unchanged).
      - "per_receiver":  each receiver's trace is divided by its own max-abs
                         in `d_obs`. The same per-receiver factor is applied
                         to `d_syn`, so the residual is `(d_syn - d_obs) /
                         scale` — a diagonal-Mahalanobis misfit that gives
                         every channel equal weight regardless of natural
                         amplitude.
      - "global":        per-shot global max-abs (one scalar per shot).
    Note: enabling normalisation changes the magnitude of the data term, so
    the `w_data` weight typically needs re-tuning relative to `w_pde`.
    """

    def __init__(
        self,
        observed_wavefield,
        config: LossConfig,
        callback: LossTape | None = None,
        normalize_data: str | None = None,
    ):
        super().__init__(config, callback)
        obs = self._stack_observations(observed_wavefield)
        self.d_obs = torch.as_tensor(
            obs, dtype=self.config.dtype, device=self.config.device
        )
        self.normalize_data = normalize_data
        self._trace_scale = self._compute_trace_scale()

    @staticmethod
    def _stack_observations(observed_wavefield):
        """Coerce input to a (n_shots, NT, NX, NY) tensor.

        Accepts a pre-stacked tensor/array (returned unchanged) or a list of
        per-shot `Wavefield`s / amplitude tensors, which get stacked here so
        callers no longer have to write `torch.stack([w.amplitude for w in ...])`.
        """
        if isinstance(observed_wavefield, (list, tuple)):
            amps = [
                w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                for w in observed_wavefield
            ]
            return torch.stack(amps)
        return observed_wavefield

    def _compute_trace_scale(self) -> torch.Tensor | None:
        """Per-shot/-receiver scale factor derived from `d_obs` (or None).

        Shape is broadcastable against an extracted trace tensor `(NT, n_rcv)`:
          - per_receiver -> (n_shots, 1, n_rcv)
          - global       -> (n_shots, 1, 1)
        Returned detached so the constant doesn't drag gradients.
        """
        if self.normalize_data is None or self.normalize_data == "none":
            return None
        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        obs_tr = self.d_obs[:, :, i, j]  # (n_shots, NT, n_rcv)
        if self.normalize_data == "per_receiver":
            scale = obs_tr.abs().amax(dim=1, keepdim=True)  # (n_shots, 1, n_rcv)
        elif self.normalize_data == "global":
            scale = obs_tr.abs().amax(dim=(1, 2), keepdim=True)  # (n_shots, 1, 1)
        else:
            raise ValueError(
                f"Invalid normalize_data {self.normalize_data!r}; "
                "expected one of None, 'per_receiver', 'global'."
            )
        return scale.clamp(min=1e-12).detach()

    def _eval_pde_loss(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int
    ) -> torch.Tensor:
        return self.config.wave_eq.residual(amp, wsp, self.sources[shot_idx])

    def _eval_data_loss(self, d_syn: torch.Tensor, shot_idx: int) -> torch.Tensor:
        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        syn_tr = d_syn[:, i, j]
        obs_tr = self.d_obs[shot_idx, :, i, j]
        if self._trace_scale is not None:
            scale = self._trace_scale[shot_idx]  # (1, n_rcv) or (1, 1)
            syn_tr = syn_tr / scale
            obs_tr = obs_tr / scale
        return syn_tr - obs_tr

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
        self._last_residuals = (
            torch.stack([r[0].detach() for r in residuals]),
            torch.stack([r[1].detach() for r in residuals]),
        )

        return L
