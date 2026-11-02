from abc import ABC, abstractmethod

import torch

from odil_wave.wavefield import Wavefield


class Conditions(ABC):
    """Base class for constraint helpers."""

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Apply this constraint; subclasses define the specific transform."""
        raise NotImplementedError


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
        return fu + self.weight * (self.sigma_sum * lambda_t * u + self.sigma_prod * u)

    def apply(
        self, fu: torch.Tensor, u: torch.Tensor, lambda_t: torch.Tensor
    ) -> torch.Tensor:
        """Alias for apply_residual, satisfying the Conditions interface."""
        return self.apply_residual(fu, u, lambda_t)
