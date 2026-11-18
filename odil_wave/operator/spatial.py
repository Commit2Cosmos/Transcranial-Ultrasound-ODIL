"""Spatial finite-difference operators (Laplacian)."""

import numpy as np
import torch
import torch.nn.functional as F

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
    """``(kernel, pad)``: the float64 cross-stencil Laplacian ``(2*pad + 1,
    2*pad + 1)`` for ``space_order`` (a key of :data:`FD2_COEFFS`) and
    non-dimensional spacings."""
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


def _apply_conv_laplacian(
    utm: torch.Tensor, kernel: torch.Tensor, pad: int
) -> torch.Tensor:
    """Laplacian of ``utm`` (spatial dims last, may be complex) with reflect
    padding (Neumann mirror)."""
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


class Laplacian:
    """Discrete Laplacian ``u_xx + u_yy`` of FD order ``space_order`` with
    reflect padding, on the wavefield's grid."""

    def __init__(self, wavefield: Wavefield, *, space_order: int) -> None:

        g = wavefield.grid
        K, self._pad = laplacian_stencil(space_order, g.dx_nd, g.dy_nd)
        self._kernel = torch.as_tensor(K, dtype=g.dtype, device=g.device)[None, None]

    def apply(self, utm: torch.Tensor) -> torch.Tensor:
        """Discrete Laplacian of ``utm`` (spatial dims last; may be complex),
        same shape and dtype."""
        return _apply_conv_laplacian(utm, self._kernel, self._pad)
