from abc import ABC, abstractmethod
from typing import List, Tuple

from odil_wave.loss import DiscreteLoss
from odil_wave.loss.utils import LossTape
from odil_wave.wavefield import Wavefield


class Optimiser(ABC):
    """Base optimiser class."""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        """Store the wavefield to optimise and the loss to minimise it against."""
        self.loss = loss
        self.wavefield = wavefield

    @abstractmethod
    def minimise(self, **kwargs) -> Tuple[List[Wavefield], LossTape]:
        """Run the optimiser and return the solved wavefields and loss tape."""
        raise NotImplementedError
