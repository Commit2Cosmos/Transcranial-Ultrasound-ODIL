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

        self.evaluations = 0

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> torch.Tensor:
        """Return a torch scalar loss; autograd handles gradients."""
        raise NotImplementedError

    @abstractmethod
    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        """Vectorised residual on shot-batched inputs (``amp.shape[0] == n_shots``)."""
        raise NotImplementedError
