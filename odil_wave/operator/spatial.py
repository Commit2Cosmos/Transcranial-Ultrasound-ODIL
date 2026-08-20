"""Spatial finite-difference operators (Laplacian)."""

from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import DenseOperator
from odil_wave.wavefield import Wavefield


# 1D central FD coefficients for d²u/dx² (offsets 1, 2, ... ; plus the centre).
# 6th-order 7-point stencil.
_C6 = [3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0]
_C6_CENTER = -49.0 / 18.0
# 8th-order 9-point stencil.
_C8 = [8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0]
_C8_CENTER = -205.0 / 72.0
# 10th-order 11-point stencil.
_C10 = [5.0 / 3.0, -5.0 / 21.0, 5.0 / 126.0, -5.0 / 1008.0, 1.0 / 3150.0]
_C10_CENTER = -5269.0 / 1800.0


def _cross_laplacian_kernel(coeffs, center, cx, cy, dtype, device) -> torch.Tensor:
    """Build a ``(1, 1, N, N)`` cross-stencil Laplacian kernel.

    Parameters
    ----------
    coeffs : sequence of float
        1D d²/dx² central coefficients for offsets ``1..P``.
    center : float
        Central stencil coefficient.
    cx, cy : float
        Inverse squared grid spacings ``1/dx**2`` and ``1/dy**2``.
    dtype : torch.dtype
        Kernel dtype.
    device : torch.device
        Kernel device.

    Returns
    -------
    torch.Tensor
        Cross-stencil kernel of shape ``(1, 1, N, N)`` with ``N = 2*P + 1``;
        ``u_xx`` on the centre column, ``u_yy`` on the centre row.
    """
    p = len(coeffs)
    n = 2 * p + 1
    K = torch.zeros(1, 1, n, n, dtype=dtype, device=device)
    for k, ck in enumerate(coeffs):
        offset = k + 1
        # u_xx: column at w=p
        K[0, 0, p - offset, p] = ck * cx
        K[0, 0, p + offset, p] = ck * cx
        # u_yy: row at h=p
        K[0, 0, p, p - offset] = ck * cy
        K[0, 0, p, p + offset] = ck * cy
    K[0, 0, p, p] = center * (cx + cy)
    return K


def _fourth_derivative_1d(
    um2: torch.Tensor,
    um1: torch.Tensor,
    u: torch.Tensor,
    up1: torch.Tensor,
    up2: torch.Tensor,
) -> torch.Tensor:
    """4th-order central second derivative on five stacked samples.

    Parameters
    ----------
    um2, um1, u, up1, up2 : torch.Tensor
        Field samples at offsets -2, -1, 0, +1, +2.

    Returns
    -------
    torch.Tensor
        Unscaled ``u_xx`` (caller divides by the squared spacing).
    """
    return (-um2 + 16.0 * um1 - 30.0 * u + 16.0 * up1 - up2) / 12.0


def _apply_conv_laplacian(
    utm: torch.Tensor, kernel: torch.Tensor, pad: int
) -> torch.Tensor:
    """Apply a fixed Laplacian kernel with reflect padding (Neumann mirror).

    Parameters
    ----------
    utm : torch.Tensor
        Field with spatial dims last; may be complex.
    kernel : torch.Tensor
        Laplacian convolution kernel of shape ``(1, 1, N, N)``.
    pad : int
        Reflect-padding width (stencil half-width).

    Returns
    -------
    torch.Tensor
        Laplacian of ``utm``, same shape and dtype.

    Notes
    -----
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
        """Discrete Laplacian ``u_xx + u_yy`` on ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.
        bc : optional
            Boundary condition (reflect padding is built in).

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        raise NotImplementedError


@dataclass
class Laplacian2ndOrder(SpatialOperator):
    """2nd-order 5-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 2nd-order 5-point Laplacian kernel from the grid spacing."""
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
        """Apply the 2nd-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.
        bc : optional
            Unused; reflect padding is built in.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, pad=1)


@dataclass
class Laplacian4thOrder(SpatialOperator):
    """4th-order 9-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 4th-order 5x5 cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        g = wavefield.grid
        dx, dy = g.dx_nd, g.dy_nd
        # 5x5 cross kernel for the 4th-order central stencil.
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
        """Apply the 4th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.
        bc : optional
            Unused; reflect padding is built in.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, pad=2)


@dataclass
class Laplacian6thOrder(SpatialOperator):
    """6th-order 13-point cross Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 6th-order cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        g = wavefield.grid
        self._kernel = _cross_laplacian_kernel(
            _C6, _C6_CENTER, 1.0 / g.dx_nd**2, 1.0 / g.dy_nd**2, g.dtype, g.device
        )

    def apply(self, utm: torch.Tensor, bc=None) -> torch.Tensor:
        """Apply the 6th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.
        bc : optional
            Unused; reflect padding is built in.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, pad=3)


@dataclass
class Laplacian8thOrder(SpatialOperator):
    """8th-order 17-point cross Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 8th-order cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        g = wavefield.grid
        self._kernel = _cross_laplacian_kernel(
            _C8, _C8_CENTER, 1.0 / g.dx_nd**2, 1.0 / g.dy_nd**2, g.dtype, g.device
        )

    def apply(self, utm: torch.Tensor, bc=None) -> torch.Tensor:
        """Apply the 8th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.
        bc : optional
            Unused; reflect padding is built in.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, pad=4)


@dataclass
class Laplacian10thOrder(SpatialOperator):
    """10th-order 11-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 10th-order cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        g = wavefield.grid
        self._kernel = _cross_laplacian_kernel(
            _C10, _C10_CENTER, 1.0 / g.dx_nd**2, 1.0 / g.dy_nd**2, g.dtype, g.device
        )

    def apply(self, utm: torch.Tensor, bc=None) -> torch.Tensor:
        """Apply the 10th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.
        bc : optional
            Unused; reflect padding is built in.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, pad=5)
