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

    def patch_spatial_neighbors(
        self,
        uxm: torch.Tensor,
        uxp: torch.Tensor,
        uym: torch.Tensor,
        uyp: torch.Tensor,
        utm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uxm = uxm.clone()
        uxp = uxp.clone()
        uym = uym.clone()
        uyp = uyp.clone()
        uxm[:, 0, :] = utm[:, 1, :]  # left edge: ghost left  <- copy cell 1
        uxp[:, -1, :] = utm[:, -2, :] # right edge: ghost right <- copy cell nx-2
        uym[:, :, 0] = utm[:, :, 1] # bottom
        uyp[:, :, -1] = utm[:, :, -2] # top
        return uxm, uxp, uym, uyp

    def apply(self, *args, **kwargs):
        return self.patch_spatial_neighbors(*args, **kwargs)


class NeumannMirrorBC4th(Conditions):
    """Mirror ghost neighbours for 4th-order spatial stencils (+/-1 and +/-2)."""

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
        uxm2 = uxm2.clone()
        uxm = uxm.clone()
        uxp = uxp.clone()
        uxp2 = uxp2.clone()
        uym2 = uym2.clone()
        uym = uym.clone()
        uyp = uyp.clone()
        uyp2 = uyp2.clone()

        uxm[:, 0, :] = utm[:, 1, :] # fix at i=0 using u1
        uxp[:, -1, :] = utm[:, -2, :] # at the last cell, use u_{nx-2}
        uxm2[:, 0, :] = utm[:, 2, :] # fix at i=0 using u2
        uxm2[:, 1, :] = utm[:, 1, :] # fix at i=1 using u1
        uxp2[:, -1, :] = utm[:, -3, :] # at the last cell, use u_{nx-3}
        uxp2[:, -2, :] = utm[:, -2, :] # at the second last cell, use u_{nx-2}
        # same pattern for the y-direction
        uym[:, :, 0] = utm[:, :, 1] 
        uyp[:, :, -1] = utm[:, :, -2]
        uym2[:, :, 0] = utm[:, :, 2]
        uym2[:, :, 1] = utm[:, :, 1]
        uyp2[:, :, -1] = utm[:, :, -3]
        uyp2[:, :, -2] = utm[:, :, -2]

        return uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2

    def apply(self, *args, **kwargs):
        return self.patch_spatial_neighbors(*args, **kwargs)


class InitialConditions(Conditions):
    """Hard displacement IC enforced as a residual row at t=0."""

    def __init__(self, grid) -> None:
        self.grid = grid

    def apply_residual(self, fu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        fu = fu.clone()
        fu[0, :, :] = (u[0, :, :] - self.grid.u0) * self.grid.kimp
        return fu

    def apply(self, fu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.apply_residual(fu, u)
