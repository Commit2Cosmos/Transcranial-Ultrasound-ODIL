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

    Early-arrival muting
    --------------------
    Direct source-to-receiver arrivals dominate trace amplitude but carry
    almost no information about the bulk medium (they travel along the
    source-receiver chord, not through the region of interest). Pass a
    ``t_mask`` (in the same units as ``grid.t``) to multiply the data
    residual by a rectangular cosine-tapered mask that zeros samples with
    ``t < t_mask`` for every shot and every receiver. Pass ``None`` (default)
    to disable muting.
    """

    def __init__(
        self,
        observed_wavefield,
        config: LossConfig,
        callback: LossTape | None = None,
        normalize_data: str | None = None,
        t_mask: float | None = None,
        mute_taper_steps: int = 4,
    ):
        super().__init__(config, callback)
        obs = self._stack_observations(observed_wavefield)
        self.d_obs = torch.as_tensor(
            obs, dtype=self.config.dtype, device=self.config.device
        )
        self.normalize_data = normalize_data
        self._mute_mask = self._build_mute_mask(t_mask, mute_taper_steps)
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

        When an early-arrival mute is active, the scale is taken from the
        muted observations so the normaliser tracks late-arrival energy.
        """
        if self.normalize_data is None or self.normalize_data == "none":
            return None
        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        obs_tr = self.d_obs[:, :, i, j]  # (n_shots, NT, n_rcv)
        if self._mute_mask is not None:
            obs_tr = obs_tr * self._mute_mask
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

    def _build_mute_mask(
        self,
        t_mask: float | None,
        taper_steps: int,
    ) -> torch.Tensor | None:
        """Rectangular pre-``t_mask`` cutoff, receiver- and shot-independent.

        Returns None when ``t_mask`` is None. Otherwise returns a
        ``(1, NT, 1)`` tensor: 0 for ``t ≤ t_mask``, ramps 0→1 over
        ``taper_steps`` timesteps via a half-Hann edge, then stays at 1. The
        singleton axes broadcast against the ``(n_shots, NT, n_rcv)`` data
        residual so the same cutoff applies uniformly.
        """
        if t_mask is None:
            return None

        cfg = self.config
        grid = cfg.wavefield.grid
        t = grid.t.to(dtype=cfg.dtype, device=cfg.device)
        dt = float(grid.dt)
        taper_width = max(1, int(taper_steps)) * dt

        # Smooth gate: mask = 0 for t ≤ t_mask, ramps to 1 over taper_steps*dt,
        # then stays at 1. Half-cosine (Hann) edge — bounded, monotone, C¹.
        delta = t - float(t_mask)
        ramp = 0.5 - 0.5 * torch.cos(
            torch.clamp(delta / taper_width, min=0.0, max=1.0) * math.pi
        )
        mask = torch.where(delta <= 0, torch.zeros_like(delta), ramp)
        return mask.view(1, -1, 1).detach()

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
        self._last_residuals = (r_pde.detach(), r_data.detach())

        return L
