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

    ``normalize_data`` rescales the receiver traces that enter the data
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
        """Per-shot/-receiver scale factor derived from `d_obs` (or None)."""
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

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # amp, sources: (n_shots, NT, NX, NY)
        pde = self.config.wave_eq.residual(amp, wsp, self.sources)

        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        syn_tr = amp[:, :, i, j]  # (n_shots, NT, n_rcv)
        obs_tr = self.d_obs[:, :, i, j]  # (n_shots, NT, n_rcv)
        if self._trace_scale is not None:
            syn_tr = syn_tr / self._trace_scale
            obs_tr = obs_tr / self._trace_scale
        data = syn_tr - obs_tr
        return pde, data

    def evaluate(
        self,
        amp: torch.Tensor,
        c_full: torch.Tensor,
        c_interior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r_pde, r_data = self._residuals(amp, c_full)

        w = self.config.weights
        pde_terms = (r_pde**2).mean(dim=(1, 2, 3)).sum()
        data_terms = (r_data**2).mean(dim=(1, 2)).sum()
        L = w["pde"] * pde_terms + w["data"] * data_terms

        if self.config.regulariser is not None and c_interior is not None:
            L = L + w["reg"] * self.config.regulariser(c_interior)

        self.evaluations += 1
        self._last_residuals = (r_pde.detach(), r_data.detach())

        return L
