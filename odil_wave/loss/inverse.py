import math
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

    Observations may be supplied either as full per-shot wavefields
    (``observed_wavefield``) or as receiver traces only (``observed_traces``,
    shape ``(n_shots, NT, n_receivers)``). The latter is intended for data
    from external forward models such as Stride.

    ``normalize_data`` rescales the receiver traces that enter the data
    residual. Modes:
      - None / "none":   off — raw amplitudes (default).
      - "per_receiver":  each receiver's trace is divided by its own max-abs
                         in the observations. The same factor is applied to
                         synthetic traces.
      - "global":        per-shot global max-abs (one scalar per shot).

    Early-arrival muting
    --------------------
    Pass ``t_mask`` (same units as ``grid.t``) to taper out samples before
    that time in the data residual. Pass ``None`` (default) to disable.
    """

    def __init__(
        self,
        observed_wavefield=None,
        *,
        config: LossConfig,
        observed_traces=None,
        callback: LossTape | None = None,
        normalize_data: str | None = None,
        t_mask: float | None = None,
        mute_taper_steps: int = 4,
    ):
        super().__init__(config, callback)

        has_wf = observed_wavefield is not None
        has_tr = observed_traces is not None
        if has_wf == has_tr:
            raise ValueError(
                "Supply exactly one of observed_wavefield or observed_traces."
            )

        self._trace_mode = has_tr
        if has_tr:
            tr = torch.as_tensor(observed_traces, dtype=self.config.dtype)
            if tr.ndim != 3:
                raise ValueError(
                    f"observed_traces must be (n_shots, NT, n_receivers), "
                    f"got {tuple(tr.shape)}"
                )
            self.d_obs = tr.to(device=self.config.device)
        else:
            obs = self._stack_observations(observed_wavefield)
            self.d_obs = torch.as_tensor(
                obs, dtype=self.config.dtype, device=self.config.device
            )

        self.normalize_data = normalize_data
        self._mute_mask = self._build_mute_mask(t_mask, mute_taper_steps)
        self._trace_scale = self._compute_trace_scale()

    @staticmethod
    def _stack_observations(observed_wavefield):
        """Coerce input to a (n_shots, NT, NX, NY) tensor."""
        if isinstance(observed_wavefield, (list, tuple)):
            amps = [
                w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                for w in observed_wavefield
            ]
            return torch.stack(amps)
        return observed_wavefield

    def _obs_traces(self) -> torch.Tensor:
        """Observation traces as ``(n_shots, NT, n_receivers)``."""
        if self._trace_mode:
            return self.d_obs
        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        return self.d_obs[:, :, i, j]

    def _compute_trace_scale(self) -> torch.Tensor | None:
        if self.normalize_data is None or self.normalize_data == "none":
            return None
        obs_tr = self._obs_traces()
        if self._mute_mask is not None:
            obs_tr = obs_tr * self._mute_mask
        if self.normalize_data == "per_receiver":
            scale = obs_tr.abs().amax(dim=1, keepdim=True)
        elif self.normalize_data == "global":
            scale = obs_tr.abs().amax(dim=(1, 2), keepdim=True)
        else:
            raise ValueError(
                f"Invalid normalize_data {self.normalize_data!r}; "
                "expected one of None, 'per_receiver', 'global'."
            )
        return scale.clamp(min=1e-12).detach()

    def _build_mute_mask(
        self,
        t_mask: float | None,
        taper_steps: int,
    ) -> torch.Tensor | None:
        if t_mask is None:
            return None

        cfg = self.config
        grid = cfg.wavefield.grid
        t = grid.t.to(dtype=cfg.dtype, device=cfg.device)
        taper_width = max(1, int(taper_steps)) * float(grid.dt)

        delta = t - float(t_mask)
        ramp = 0.5 - 0.5 * torch.cos(
            torch.clamp(delta / taper_width, min=0.0, max=1.0) * math.pi
        )
        mask = torch.where(delta <= 0, torch.zeros_like(delta), ramp)
        return mask.view(1, -1, 1).detach()

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pde = self.config.wave_eq.residual(amp, wsp, self.sources)

        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        syn_tr = amp[:, :, i, j]
        obs_tr = self._obs_traces()
        if self._trace_scale is not None:
            syn_tr = syn_tr / self._trace_scale
            obs_tr = obs_tr / self._trace_scale
        data = syn_tr - obs_tr
        if self._mute_mask is not None:
            data = data * self._mute_mask
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
        with torch.no_grad():
            pde_rms = float(r_pde.pow(2).mean().sqrt().detach().cpu())
            data_rms = float(r_data.pow(2).mean().sqrt().detach().cpu())
            src_rms = float(self.sources.pow(2).mean().sqrt().detach().cpu())

        self._last_residuals = {
            "pde_rms": pde_rms,
            "data_rms": data_rms,
            "pde_loss": float(pde_terms.detach().cpu()),
            "data_loss": float(data_terms.detach().cpu()),
            "pde_src_ratio": pde_rms / max(src_rms, 1e-30),
        }

        return L
