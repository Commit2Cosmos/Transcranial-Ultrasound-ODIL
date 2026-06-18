from abc import ABC, abstractmethod
from typing import Tuple
from .utils import LossConfig, LossTape
import torch
import numpy as np


class DiscreteLoss(ABC):
    """Base class for discrete loss functions."""

    def __init__(
        self,
        config: LossConfig,
        callback: LossTape | None = None,
    ):
        self.config = config  # loss configuration
        self.callback = (
            callback if callback is not None else LossTape()
        )  # loss history callback

        # extract operator config
        self.time_op = config.time_operator
        self.lap = config.laplacian_operator

        # precompute source fields for each shot
        self.sources = torch.stack(
            [
                self.config.geometry.source_field(i)
                for i in range(self.config.geometry.n_sources)
            ]
        )

        self.evaluations = 0  # counter for number of loss evaluations

    @abstractmethod
    def evaluate(self, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate the loss function given a wavefield."""
        pass

    @abstractmethod
    def _eval_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _residuals(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the residuals of the loss function given a wavefield."""
        pass

    @abstractmethod
    def _eval_pde_loss(self, data: torch.Tensor) -> torch.Tensor:
        pass
