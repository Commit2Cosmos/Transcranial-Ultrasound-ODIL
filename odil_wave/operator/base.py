from abc import ABC, abstractmethod
import torch
from odil_wave.wavefield import Wavefield


class DenseOperator(ABC):
    """Matrix-free operator using local stencils (e.g. ``torch.roll``)."""

    def __init__(self, wavefield: Wavefield) -> None:
        """Bind the operator to the wavefield it acts on."""
        self.wavefield = wavefield

    @abstractmethod
    def apply(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the discrete operator to field u."""
        pass
