from abc import ABC, abstractmethod

import torch

from .utils import LossConfig, LossTape


class DiscreteLoss(ABC):
    """Base class for discrete loss functions."""

    def __init__(
        self,
        config: LossConfig,
        callback: LossTape | None = None,
    ):
        self.config = config
        self.callback = callback if callback is not None else LossTape()

        # Precompute source fields per shot, cast to config.dtype/device, and
        # apply the non-dimensionalisation scaling f' = t0**2 * f so that the
        # `WaveEquation.residual` stays dimensionally consistent (see
        # :class:`Grid` for the derivation).
        t0 = self.config.wave_eq.wavefield.grid.t0
        self.sources = (
            torch.stack(
                [
                    self.config.geometry.source_field(i).to(
                        dtype=self.config.dtype, device=self.config.device
                    )
                    for i in range(self.config.geometry.n_sources)
                ]
            )
            * t0**2
        )

        # RMS of the (pre-scaled) sources: reference scale for judging how
        # converged a solve is via |r_pde| / |src|.
        self.src_rms = float(self.sources.pow(2).mean().sqrt())

        self.evaluations = 0

    def pde_src_ratio(self) -> float:
        """|r_pde|_rms / |src|_rms of the last evaluation (1.0 ~ u = 0)."""
        r_pde = self._last_residuals[0]
        return float(r_pde.pow(2).mean().sqrt()) / max(self.src_rms, 1e-30)

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> torch.Tensor:
        """Return a torch scalar loss; autograd handles gradients."""
        raise NotImplementedError

    @abstractmethod
    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        """Vectorised residual on shot-batched inputs (``amp.shape[0] == n_shots``)."""
        raise NotImplementedError
