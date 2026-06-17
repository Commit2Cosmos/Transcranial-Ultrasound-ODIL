"Boundary and Initial conditions for finite-difference stencils"

from abc import ABC, abstractmethod

import torch


class Conditions(ABC):
    """Base class for constraint helpers."""

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError


class NeumannMirrorBC2nd(Conditions):
    """Mirror ghost neighbours for 2nd-order spatial stencils
    (zero normal derivative).
    """

    @abstractmethod
    def patch_spatial_neighbors(
        self,
        uxm: torch.Tensor,
        uxp: torch.Tensor,
        uym: torch.Tensor,
        uyp: torch.Tensor,
        utm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass


class NeumannMirrorBC4th(Conditions):
    """Mirror ghost neighbours for 4th-order spatial stencils (+/-1 and +/-2)."""

    @abstractmethod
    def patch_spatial_neighbors(
        self,
        uxm2: torch.Tensor,
        uxm: torch.Tensor,
        uxp: torch.Tensor,
        uxp2: torch.Tensor,
        uym2: torch.Tensor,
        uym: torch.Tensor,
        uyp: torch.Tensor,
        uyp2: torch.Tensor,
        utm: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass


class InitialConditions(Conditions):
    """Hard displacement IC enforced as a residual row at t=0."""

    pass
