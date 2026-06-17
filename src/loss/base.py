from abc import ABC, abstractmethod
from typing import Tuple
from .utils import LossConfig, LossTape
import torch


class DiscreteLoss(ABC):
    """Base class for discrete loss functions."""

    def __init__(self, config: LossConfig, callback: LossTape | None = None):
        self.config = config  # loss configuration
        self.callback = (
            callback if callback is not None else LossTape()
        )  # loss history callback

        # extract operator config
        self.time_op = config.time_operator
        self.lap = config.laplacian_operator

    @abstractmethod
    def evaluate(self, wavefield) -> Tuple[float, torch.Tensor]:
        """Evaluate the loss function given a wavefield."""
        pass

    @abstractmethod
    def _eval_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _residuals(self, wavefield) -> torch.Tensor:
        """Compute the residuals of the loss function given a wavefield."""
        pass

    @abstractmethod
    def _eval_pde_loss(self, wavefield, wavespeed) -> torch.Tensor:
        pass
