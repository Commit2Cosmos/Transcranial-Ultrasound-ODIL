"Boundary and Initial conditions for finite-difference stencils"

from abc import ABC, abstractmethod
from src.wavefield import Wavefield

import torch


class Conditions(ABC):
    """Base class for constraint helpers."""

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError


class InitialConditions(Conditions):
    """Hard displacement IC enforced as a residual row at t=0."""

    def __init__(self, wavefield: Wavefield, weight: float = 1.0) -> None:
        self.wavefield = wavefield
        self.weight = weight

    def apply_residual(self, fu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        fu = fu.clone()
        fu[0, :, :] = (u[0, :, :] - self.wavefield.amplitude[0, :, :]) * self.weight
        return fu

    def apply(self, fu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.apply_residual(fu, u)
