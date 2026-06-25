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

        # precompute (cast to config.dtype/device) source fields for each shot
        self.sources = torch.stack(
            [
                self.config.geometry.source_field(i).to(
                    dtype=self.config.dtype, device=self.config.device
                )
                for i in range(self.config.geometry.n_sources)
            ]
        )

        self.evaluations = 0

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> torch.Tensor:
        """Return a torch scalar loss; autograd handles gradients."""
        raise NotImplementedError

    @abstractmethod
    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int
    ) -> torch.Tensor:
        raise NotImplementedError
