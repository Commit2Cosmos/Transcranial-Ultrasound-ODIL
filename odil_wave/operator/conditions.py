"Boundary and Initial conditions for finite-difference stencils"

from abc import ABC, abstractmethod
from odil_wave.wavefield import Wavefield

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

    @staticmethod
    def _patch_first(neighbour: torch.Tensor, replacement: torch.Tensor, axis: int):
        # Replace the ``axis=0`` slice along the given spatial axis (1 or 2).
        if axis == 1:
            return torch.cat([replacement.unsqueeze(1), neighbour[:, 1:, :]], dim=1)
        return torch.cat([replacement.unsqueeze(2), neighbour[:, :, 1:]], dim=2)

    @staticmethod
    def _patch_last(neighbour: torch.Tensor, replacement: torch.Tensor, axis: int):
        # Replace the ``axis=-1`` slice along the given spatial axis (1 or 2).
        if axis == 1:
            return torch.cat([neighbour[:, :-1, :], replacement.unsqueeze(1)], dim=1)
        return torch.cat([neighbour[:, :, :-1], replacement.unsqueeze(2)], dim=2)

    def patch_spatial_neighbors(
        self,
        uxm: torch.Tensor,
        uxp: torch.Tensor,
        uym: torch.Tensor,
        uyp: torch.Tensor,
        utm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uxm = self._patch_first(uxm, utm[:, 1, :], axis=1)
        uxp = self._patch_last(uxp, utm[:, -2, :], axis=1)
        uym = self._patch_first(uym, utm[:, :, 1], axis=2)
        uyp = self._patch_last(uyp, utm[:, :, -2], axis=2)
        return uxm, uxp, uym, uyp

    def apply(self, *args, **kwargs):
        return self.patch_spatial_neighbors(*args, **kwargs)


class NeumannMirrorBC4th(Conditions):
    """Mirror ghost neighbours for 4th-order spatial stencils (+/-1 and +/-2).
    Has the effect of undoing the periodicity introduced by .roll,
    but is a valid BC
    """

    @staticmethod
    def _replace_x_front(n: torch.Tensor, r0: torch.Tensor, r1: torch.Tensor):
        # Override n[:, 0, :] with r0 and n[:, 1, :] with r1 (out-of-place).
        return torch.cat([r0.unsqueeze(1), r1.unsqueeze(1), n[:, 2:, :]], dim=1)

    @staticmethod
    def _replace_x_back(n: torch.Tensor, rm2: torch.Tensor, rm1: torch.Tensor):
        # Override n[:, -2, :] with rm2 and n[:, -1, :] with rm1.
        return torch.cat([n[:, :-2, :], rm2.unsqueeze(1), rm1.unsqueeze(1)], dim=1)

    @staticmethod
    def _replace_y_front(n: torch.Tensor, r0: torch.Tensor, r1: torch.Tensor):
        return torch.cat([r0.unsqueeze(2), r1.unsqueeze(2), n[:, :, 2:]], dim=2)

    @staticmethod
    def _replace_y_back(n: torch.Tensor, rm2: torch.Tensor, rm1: torch.Tensor):
        return torch.cat([n[:, :, :-2], rm2.unsqueeze(2), rm1.unsqueeze(2)], dim=2)

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
        # Mirror ghosts: same semantics as the original in-place patcher.
        uxm = torch.cat([utm[:, 1:2, :], uxm[:, 1:, :]], dim=1)  # uxm[0] <- u[1]
        uxp = torch.cat([uxp[:, :-1, :], utm[:, -2:-1, :]], dim=1)  # uxp[-1] <- u[-2]
        uxm2 = self._replace_x_front(uxm2, utm[:, 2, :], utm[:, 1, :])
        uxp2 = self._replace_x_back(uxp2, utm[:, -2, :], utm[:, -3, :])

        uym = torch.cat([utm[:, :, 1:2], uym[:, :, 1:]], dim=2)
        uyp = torch.cat([uyp[:, :, :-1], utm[:, :, -2:-1]], dim=2)
        uym2 = self._replace_y_front(uym2, utm[:, :, 2], utm[:, :, 1])
        uyp2 = self._replace_y_back(uyp2, utm[:, :, -2], utm[:, :, -3])

        return uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2

    def apply(self, *args, **kwargs):
        return self.patch_spatial_neighbors(*args, **kwargs)


class PML(Conditions):
    """Absorbing layer enforced as a damping term.

    Operates on the dimensionless residual: the precomputed
    ``sigma_sum`` and ``sigma_prod`` use ``grid.sigma_*_nd`` (i.e.
    physical sigma multiplied by ``t0``), and the time derivative is
    taken with respect to ``t' = t / t0``.
    """

    def __init__(self, wavefield: Wavefield, weight: float = 1.0) -> None:
        self.wavefield = wavefield
        self.weight = weight
        grid = wavefield.grid

        # precompute pml in non-dimensional form (sigma * t0)
        self.sigma_sum = grid.sigma_x_nd + grid.sigma_y_nd
        self.sigma_prod = grid.sigma_x_nd * grid.sigma_y_nd

        # Endpoint masks for the time derivative used in the damping term.
        from .temporal import _make_endpoint_masks  # avoid circular import

        self._mask_first, self._mask_last = _make_endpoint_masks(grid.nt, grid.device)

    def apply_residual(self, fu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        from .temporal import _first_time_derivative  # avoid circular import

        dt_nd = self.wavefield.grid.dt_nd
        u_t = _first_time_derivative(
            u,
            dt_nd,
            self.wavefield.init_ut_nd,
            self._mask_first,
            self._mask_last,
        )
        return fu + self.weight * (self.sigma_sum * u_t + self.sigma_prod * u)

    def apply(self, fu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.apply_residual(fu, u)
