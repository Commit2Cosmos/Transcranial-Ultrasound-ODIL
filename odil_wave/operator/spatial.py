"""Spatial finite-difference operators (Laplacian)."""

from abc import abstractmethod
from dataclasses import dataclass

import torch

from .base import DenseOperator
from .conditions import NeumannMirrorBC2nd, NeumannMirrorBC4th
from odil_wave.wavefield import Wavefield


# second-order 5 point Laplacian
def _laplacian_5pt(
    utm: torch.Tensor,
    uxm: torch.Tensor,
    uxp: torch.Tensor,
    uym: torch.Tensor,
    uyp: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    return (uxm - 2.0 * utm + uxp) / dx**2 + (uym - 2.0 * utm + uyp) / dy**2


# fourth-order uxx or uyy
def _fourth_derivative_1d(
    um2: torch.Tensor,
    um1: torch.Tensor,
    u: torch.Tensor,
    up1: torch.Tensor,
    up2: torch.Tensor,
) -> torch.Tensor:
    # 4th-order central u_xx: (-u_{-2} + 16*u_{-1} - 30*u + 16*u_{+1} - u_{+2}) / 12
    return (-um2 + 16.0 * um1 - 30.0 * u + 16.0 * up1 - up2) / 12.0


class SpatialOperator(DenseOperator):
    """Spatial Laplacian on the t-1 field, cell-centred grid."""

    @abstractmethod
    def gather_neighbors(self, utm: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, utm: torch.Tensor, bc=None, **kwargs) -> torch.Tensor:
        """Discrete Laplacian (u_xx + u_yy) on utm."""
        raise NotImplementedError


@dataclass
class Laplacian2ndOrder(SpatialOperator):
    """2nd-order 5-point Laplacian."""

    def __init__(self, wavefield: Wavefield):
        super().__init__(wavefield)

    # collects the four spatial neighbors of the t-1 field
    def gather_neighbors(
        self, utm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uxm = torch.roll(utm, 1, dims=1)  # neigbor to the left
        uxp = torch.roll(utm, -1, dims=1)  # neigbor to the right
        uym = torch.roll(utm, 1, dims=2)  # neigbor below
        uyp = torch.roll(utm, -1, dims=2)  # neigbor above
        return uxm, uxp, uym, uyp

    def apply(
        self, utm: torch.Tensor, bc: NeumannMirrorBC2nd | None = None
    ) -> torch.Tensor:
        bc = bc or NeumannMirrorBC2nd()
        uxm, uxp, uym, uyp = self.gather_neighbors(utm)  # collect the neigbors
        uxm, uxp, uym, uyp = bc.patch_spatial_neighbors(
            uxm, uxp, uym, uyp, utm
        )  # replace teh wrong periodic roll with the mirrored interior values
        dx = self.wavefield.grid.dx
        dy = self.wavefield.grid.dy
        return _laplacian_5pt(utm, uxm, uxp, uym, uyp, dx, dy)


@dataclass
class Laplacian4thOrder(SpatialOperator):
    """4th-order 9-point Laplacian."""

    def __init__(self, wavefield: Wavefield):
        super().__init__(wavefield)

    def gather_neighbors(self, utm: torch.Tensor) -> tuple[torch.Tensor, ...]:
        uxm2 = torch.roll(utm, 2, dims=1)  # two cells left(i-2,j)
        uxm = torch.roll(utm, 1, dims=1)  # one cell left(i-1,j)
        uxp = torch.roll(utm, -1, dims=1)  # one cell right(i+1,j)
        uxp2 = torch.roll(utm, -2, dims=1)  # two cells right(i+2,j)
        uym2 = torch.roll(utm, 2, dims=2)  # two cells down(i,j-2)
        uym = torch.roll(utm, 1, dims=2)  # one cell down(i,j-1)
        uyp = torch.roll(utm, -1, dims=2)  # one cell up(i,j+1)
        uyp2 = torch.roll(utm, -2, dims=2)
        return uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2

    def apply(
        self,
        utm: torch.Tensor,
        bc: NeumannMirrorBC4th | None = None,
    ) -> torch.Tensor:
        bc = bc or NeumannMirrorBC4th()
        uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2 = self.gather_neighbors(utm)
        neighbours = bc.patch_spatial_neighbors(
            uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2, utm
        )
        uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2 = neighbours
        dx = self.wavefield.grid.dx
        dy = self.wavefield.grid.dy
        u_xx = _fourth_derivative_1d(uxm2, uxm, utm, uxp, uxp2) / dx**2
        u_yy = _fourth_derivative_1d(uym2, uym, utm, uyp, uyp2) / dy**2
        return u_xx + u_yy
