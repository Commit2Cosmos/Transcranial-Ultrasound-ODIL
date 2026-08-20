from abc import ABC, abstractmethod

import torch

from odil_wave.wavefield import Wavefield


class Conditions(ABC):
    """Base class for constraint helpers."""

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Apply this constraint; subclasses define the specific transform."""
        raise NotImplementedError


class NeumannMirrorBC2nd(Conditions):
    """Mirror ghost neighbours for 2nd-order spatial stencils."""

    @staticmethod
    def _patch_first(neighbour: torch.Tensor, replacement: torch.Tensor, axis: int):
        """Replace the first slice along axis with a mirrored ghost value."""
        if axis == 1:
            return torch.cat([replacement.unsqueeze(1), neighbour[:, 1:, :]], dim=1)
        return torch.cat([replacement.unsqueeze(2), neighbour[:, :, 1:]], dim=2)

    @staticmethod
    def _patch_last(neighbour: torch.Tensor, replacement: torch.Tensor, axis: int):
        """Replace the last slice along axis with a mirrored ghost value."""
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
        """Patch the four spatial ghost neighbour tensors with mirrored boundary values from utm."""
        uxm = self._patch_first(uxm, utm[:, 1, :], axis=1)
        uxp = self._patch_last(uxp, utm[:, -2, :], axis=1)
        uym = self._patch_first(uym, utm[:, :, 1], axis=2)
        uyp = self._patch_last(uyp, utm[:, :, -2], axis=2)
        return uxm, uxp, uym, uyp

    def apply(self, *args, **kwargs):
        """Alias for patch_spatial_neighbors, satisfying the Conditions interface."""
        return self.patch_spatial_neighbors(*args, **kwargs)


class NeumannMirrorBC4th(Conditions):
    """Mirror ghost neighbours for 4th-order spatial stencils."""

    @staticmethod
    def _replace_x_front(n: torch.Tensor, r0: torch.Tensor, r1: torch.Tensor):
        """Replace the first two x-neighbour slices with mirrored ghost values."""
        return torch.cat([r0.unsqueeze(1), r1.unsqueeze(1), n[:, 2:, :]], dim=1)

    @staticmethod
    def _replace_x_back(n: torch.Tensor, rm2: torch.Tensor, rm1: torch.Tensor):
        """Replace the last two x-neighbour slices with mirrored ghost values."""
        return torch.cat([n[:, :-2, :], rm2.unsqueeze(1), rm1.unsqueeze(1)], dim=1)

    @staticmethod
    def _replace_y_front(n: torch.Tensor, r0: torch.Tensor, r1: torch.Tensor):
        """Replace the first two y-neighbour slices with mirrored ghost values."""
        return torch.cat([r0.unsqueeze(2), r1.unsqueeze(2), n[:, :, 2:]], dim=2)

    @staticmethod
    def _replace_y_back(n: torch.Tensor, rm2: torch.Tensor, rm1: torch.Tensor):
        """Replace the last two y-neighbour slices with mirrored ghost values."""
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
        """Patch the eight spatial ghost-neighbour tensors with mirrored boundary values from utm."""
        uxm = torch.cat([utm[:, 1:2, :], uxm[:, 1:, :]], dim=1)
        uxp = torch.cat([uxp[:, :-1, :], utm[:, -2:-1, :]], dim=1)
        uxm2 = self._replace_x_front(uxm2, utm[:, 2, :], utm[:, 1, :])
        uxp2 = self._replace_x_back(uxp2, utm[:, -2, :], utm[:, -3, :])

        uym = torch.cat([utm[:, :, 1:2], uym[:, :, 1:]], dim=2)
        uyp = torch.cat([uyp[:, :, :-1], utm[:, :, -2:-1]], dim=2)
        uym2 = self._replace_y_front(uym2, utm[:, :, 2], utm[:, :, 1])
        uyp2 = self._replace_y_back(uyp2, utm[:, :, -2], utm[:, :, -3])

        return uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2

    def apply(self, *args, **kwargs):
        """Alias for patch_spatial_neighbors, satisfying the Conditions interface."""
        return self.patch_spatial_neighbors(*args, **kwargs)


class Sponge(Conditions):
    """Simple absorbing sponge (not a full PML).

    Frequency-domain form of the time residual damping
    ``sigma_sum * D_t[u] + sigma_prod * u``, with discrete FD symbol ``lambda_t``::

        fu + weight * (sigma_sum_nd * lambda_t * u + sigma_prod_nd * u)

    ``lambda_t`` is broadcast ``(1, nf, 1, 1)`` from FrequencySelection.
    """

    def __init__(self, wavefield: Wavefield, weight: float = 1.0) -> None:
        """Precompute the sponge's damping coefficients from the wavefield's grid."""
        self.wavefield = wavefield
        self.weight = weight
        grid = wavefield.grid
        self.sigma_sum = grid.sigma_x_nd + grid.sigma_y_nd
        self.sigma_prod = grid.sigma_x_nd * grid.sigma_y_nd

    def apply_residual(
        self, fu: torch.Tensor, u: torch.Tensor, lambda_t: torch.Tensor
    ) -> torch.Tensor:
        """Add sigma-weighted damping to a PDE residual fu."""
        # sigma: (nx, ny); lambda_t: (1, nf, 1, 1); u: (..., nf, nx, ny)
        return fu + self.weight * (
            self.sigma_sum * lambda_t * u + self.sigma_prod * u
        )

    def apply(
        self, fu: torch.Tensor, u: torch.Tensor, lambda_t: torch.Tensor
    ) -> torch.Tensor:
        """Alias for apply_residual, satisfying the Conditions interface."""
        return self.apply_residual(fu, u, lambda_t)