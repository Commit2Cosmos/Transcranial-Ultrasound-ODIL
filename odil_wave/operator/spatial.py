"""Spatial finite-difference operators (Laplacian)."""

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .base import DenseOperator
from odil_wave.wavefield import Wavefield


# 1D central FD coefficients for d²u/dx², by space order: (centre, offsets 1..P).
# The single source of truth for both the torch conv kernels below and the
# sparse Helmholtz assembly.
FD2_COEFFS: dict[int, tuple[float, tuple[float, ...]]] = {
    2: (-2.0, (1.0,)),
    4: (-30.0 / 12.0, (16.0 / 12.0, -1.0 / 12.0)),
    6: (-49.0 / 18.0, (3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0)),
    8: (-205.0 / 72.0, (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)),
    10: (
        -5269.0 / 1800.0,
        (5.0 / 3.0, -5.0 / 21.0, 5.0 / 126.0, -5.0 / 1008.0, 1.0 / 3150.0),
    ),
}


def laplacian_stencil(
    space_order: int, dx_nd: float, dy_nd: float
) -> tuple[np.ndarray, int]:
    """Cross-stencil Laplacian ``kernel[hx, hy]`` and its half-width.

    Parameters
    ----------
    space_order : int
        FD order, a key of :data:`FD2_COEFFS` (2 | 4 | 6 | 8 | 10).
    dx_nd, dy_nd : float
        Non-dimensional grid spacings.

    Returns
    -------
    tuple
        ``(kernel, pad)``: float64 kernel of shape ``(2*pad + 1, 2*pad + 1)``
        with ``u_xx`` on the centre column and ``u_yy`` on the centre row.

    Notes
    -----
    Raises ``ValueError`` for an unsupported ``space_order``.
    """
    if space_order not in FD2_COEFFS:
        raise ValueError(f"Unsupported space order: {space_order}")
    center, coeffs = FD2_COEFFS[space_order]
    cx = 1.0 / dx_nd**2
    cy = 1.0 / dy_nd**2
    pad = len(coeffs)
    K = np.zeros((2 * pad + 1, 2 * pad + 1), dtype=np.float64)
    for offset, ck in enumerate(coeffs, start=1):
        K[pad - offset, pad] = K[pad + offset, pad] = ck * cx
        K[pad, pad - offset] = K[pad, pad + offset] = ck * cy
    K[pad, pad] = center * (cx + cy)
    return K, pad


def _conv_kernel(space_order: int, wavefield: Wavefield) -> tuple[torch.Tensor, int]:
    """:func:`laplacian_stencil` as a ``(1, 1, N, N)`` conv kernel on the grid."""
    g = wavefield.grid
    K, pad = laplacian_stencil(space_order, g.dx_nd, g.dy_nd)
    return torch.as_tensor(K, dtype=g.dtype, device=g.device)[None, None], pad


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
    def apply(self, utm: torch.Tensor, **kwargs) -> torch.Tensor:
        """Discrete Laplacian ``u_xx + u_yy`` on ``utm`` (reflect padding).

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.

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
        self._kernel, self._pad = _conv_kernel(2, wavefield)

    def apply(self, utm: torch.Tensor) -> torch.Tensor:
        """Apply the 2nd-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, self._pad)


@dataclass
class Laplacian4thOrder(SpatialOperator):
    """4th-order 9-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 4th-order 5x5 cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        self._kernel, self._pad = _conv_kernel(4, wavefield)

    def apply(self, utm: torch.Tensor) -> torch.Tensor:
        """Apply the 4th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, self._pad)


@dataclass
class Laplacian6thOrder(SpatialOperator):
    """6th-order 13-point cross Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 6th-order cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        self._kernel, self._pad = _conv_kernel(6, wavefield)

    def apply(self, utm: torch.Tensor) -> torch.Tensor:
        """Apply the 6th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, self._pad)


@dataclass
class Laplacian8thOrder(SpatialOperator):
    """8th-order 17-point cross Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 8th-order cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        self._kernel, self._pad = _conv_kernel(8, wavefield)

    def apply(self, utm: torch.Tensor) -> torch.Tensor:
        """Apply the 8th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, self._pad)


@dataclass
class Laplacian10thOrder(SpatialOperator):
    """10th-order 11-point Laplacian via ``F.conv2d`` + reflect padding."""

    def __init__(self, wavefield: Wavefield):
        """Build the 10th-order cross Laplacian kernel from the grid spacing."""
        super().__init__(wavefield)
        self._kernel, self._pad = _conv_kernel(10, wavefield)

    def apply(self, utm: torch.Tensor) -> torch.Tensor:
        """Apply the 10th-order Laplacian kernel to ``utm``.

        Parameters
        ----------
        utm : torch.Tensor
            Field with spatial dims last.

        Returns
        -------
        torch.Tensor
            Discrete Laplacian of ``utm``.
        """
        return _apply_conv_laplacian(utm, self._kernel, self._pad)
