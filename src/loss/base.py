from abc import ABC, abstractmethod
from typing import Tuple
from .utils import LossConfig, LossTape
import torch


class DiscreteLoss(ABC):
    """Base class for discrete loss functions."""

    def __init__(
        self, config: LossConfig | None = None, callback: LossTape | None = None
    ):
        self.config = config  # loss configuration
        self.callback = (
            callback if callback is not None else LossTape()
        )  # loss history callback

    @abstractmethod
    def evaulate(self, wavefield) -> Tuple[float, torch.Tensor]:
        """Evaluate the loss function given a wavefield."""
        pass

    def _residuals(self, wavefield):
        """Compute the residuals of the loss function given a wavefield."""
        pass
