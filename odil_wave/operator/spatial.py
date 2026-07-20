"""Spatial finite-difference operators (Laplacian).
"""

from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import DenseOperator
from odil_wave.wavefield import Wavefield


# 10th-order 1D central FD coefficients for d²u/dx²
_C10 = [5.0 / 3.0, -5.0 / 21.0, 5.0 / 126.0, -5.0 / 1008.0, 1.0 / 3150.0]
_C10_CENTER = -5269.0 / 1800.0


def _fourth_derivative_1d(
    um2: torch.Tensor,
    um1: torch.Tensor,
    u: torch.Tensor,
    up1: torch.Tensor,
    up2: torch.Tensor,
) -> torch.Tensor:
    # 4th-order central u_xx: (-u_{-2} + 16*u_{-1} - 30*u + 16*u_{+1} - u_{+2}) / 12
    return (-um2 + 16.0 * um1 - 30.0 * u + 16.0 * up1 - up2) / 12.0


def _apply_conv_laplacian(
    utm: torch.Tensor, kernel: torch.Tensor, pad: int
) -> torch.Tensor:
    """Apply a fixed Laplacian kernel with reflect padding (Neumann mirror).

    Complex inputs are processed as real/imag separately (``F.conv2d`` is real).
    """
    if utm.is_complex():
        return torch.complex(
            _apply_conv_laplacian(utm.real, kernel, pad),
            _apply_conv_laplacian(utm.imag, kernel, pad),
        )
    spatial = utm.shape[-2:]
    leading = utm.shape[:-2]
    x = utm.reshape(-1, 1, *spatial)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(x, kernel)
    return out.reshape(*leading, *spatial)


class SpatialOperator(DenseOperator):
    """Spatial Laplacian on the t-1 field, cell-centred grid."""

    @abstractmethod
    def apply(self, utm: torch.Tensor, bc=None, **kwargs) -> torch.Tensor:
        """Discrete Laplacian (u_xx + u_yy) on utm."""
        raise NotImplementedError


@dataclass
class Laplacian2ndOrder(SpatialOperator):
    """2nd-order 5-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        super().__init__(wavefield)
        g = wavefield.grid
        dx, dy = g.dx_nd, g.dy_nd
        # Kernel laid out as (1, 1, H=NX-stencil, W=NY-stencil).
        K = torch.zeros(1, 1, 3, 3, dtype=g.dtype, device=g.device)
        K[0, 0, 0, 1] = 1.0 / dx**2  # uxm (h-1, w)
        K[0, 0, 2, 1] = 1.0 / dx**2  # uxp (h+1, w)
        K[0, 0, 1, 0] = 1.0 / dy**2  # uym (h, w-1)
        K[0, 0, 1, 2] = 1.0 / dy**2  # uyp (h, w+1)
        K[0, 0, 1, 1] = -2.0 / dx**2 - 2.0 / dy**2
        self._kernel = K

    def apply(self, utm: torch.Tensor, bc=None) -> torch.Tensor:
        return _apply_conv_laplacian(utm, self._kernel, pad=1)


@dataclass
class Laplacian4thOrder(SpatialOperator):
    """4th-order 9-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        super().__init__(wavefield)
        g = wavefield.grid
        dx, dy = g.dx_nd, g.dy_nd
        # 5x5 cross kernel matching the original 1D 4th-order central stencil.
        K = torch.zeros(1, 1, 5, 5, dtype=g.dtype, device=g.device)
        cx = 1.0 / dx**2
        cy = 1.0 / dy**2
        # u_xx column at w=2
        K[0, 0, 0, 2] = -1.0 / 12.0 * cx
        K[0, 0, 1, 2] = 16.0 / 12.0 * cx
        K[0, 0, 3, 2] = 16.0 / 12.0 * cx
        K[0, 0, 4, 2] = -1.0 / 12.0 * cx
        # u_yy row at h=2
        K[0, 0, 2, 0] = -1.0 / 12.0 * cy
        K[0, 0, 2, 1] = 16.0 / 12.0 * cy
        K[0, 0, 2, 3] = 16.0 / 12.0 * cy
        K[0, 0, 2, 4] = -1.0 / 12.0 * cy
        # center contributes both -30/12 / dx^2 and -30/12 / dy^2
        K[0, 0, 2, 2] = -30.0 / 12.0 * (cx + cy)
        self._kernel = K

    def apply(self, utm: torch.Tensor, bc=None) -> torch.Tensor:
        return _apply_conv_laplacian(utm, self._kernel, pad=2)


@dataclass
class Laplacian10thOrder(SpatialOperator):
    """10th-order 11-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        super().__init__(wavefield)
        g = wavefield.grid
        dx, dy = g.dx_nd, g.dy_nd
        cx = 1.0 / dx**2
        cy = 1.0 / dy**2
        K = torch.zeros(1, 1, 11, 11, dtype=g.dtype, device=g.device)
        # u_xx: column at w=5 (spatial-x axis varies along dim-2 of the field).
        for k, ck in enumerate(_C10):
            offset = k + 1
            K[0, 0, 5 - offset, 5] = ck * cx
            K[0, 0, 5 + offset, 5] = ck * cx
        K[0, 0, 5, 5] += _C10_CENTER * cx
        # u_yy: row at h=5 (spatial-y axis varies along dim-3 of the field).
        for k, ck in enumerate(_C10):
            offset = k + 1
            K[0, 0, 5, 5 - offset] = ck * cy
            K[0, 0, 5, 5 + offset] = ck * cy
        K[0, 0, 5, 5] += _C10_CENTER * cy
        self._kernel = K

    def apply(self, utm: torch.Tensor, bc=None) -> torch.Tensor:
        return _apply_conv_laplacian(utm, self._kernel, pad=5)
