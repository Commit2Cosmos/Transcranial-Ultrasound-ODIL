"""Spatial finite-difference operators: the order-P second differences along
each axis, and the staggered first-difference coefficients. The PML's stretch
of them lives in :mod:`.conditions`."""

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


# 1D staggered FD coefficients for du/dx between nodes, by space order: offsets
# 1/2 .. (P - 1)/2. The PML's correction (:mod:`.conditions`) uses them, in both
# the matrix-free and the sparse operator.
STAGGERED_D1_COEFFS: dict[int, tuple[float, ...]] = {
    2: (1.0,),
    4: (9.0 / 8.0, -1.0 / 24.0),
    6: (75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0),
    8: (1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0),
    10: (
        19845.0 / 16384.0,
        -735.0 / 8192.0,
        567.0 / 40960.0,
        -405.0 / 229376.0,
        35.0 / 294912.0,
    ),
}


def second_difference_stencil(space_order: int, h_nd: float) -> np.ndarray:
    """The float64 1D stencil ``(2*pad + 1,)`` of ``d²/dx²`` for
    ``space_order`` (a key of :data:`FD2_COEFFS`) at non-dimensional spacing
    ``h_nd``."""
    if space_order not in FD2_COEFFS:
        raise ValueError(f"Unsupported space order: {space_order}")
    center, coeffs = FD2_COEFFS[space_order]
    pad = len(coeffs)
    k = np.zeros(2 * pad + 1, dtype=np.float64)
    for offset, ck in enumerate(coeffs, start=1):
        k[pad - offset] = k[pad + offset] = ck
    k[pad] = center
    return k / h_nd**2


def _apply_conv(utm: torch.Tensor, kernel: torch.Tensor, pad: tuple) -> torch.Tensor:
    """``kernel`` convolved over ``utm``'s last two dims (may be complex), with
    reflect padding ``pad`` (Neumann mirror)."""
    if utm.is_complex():
        return torch.complex(
            _apply_conv(utm.real, kernel, pad), _apply_conv(utm.imag, kernel, pad)
        )
    spatial = utm.shape[-2:]
    leading = utm.shape[:-2]
    x = utm.reshape(-1, 1, *spatial)
    x = F.pad(x, pad, mode="reflect")
    out = F.conv2d(x, kernel)
    return out.reshape(*leading, *spatial)


class Laplacian:
    """Discrete ``u_xx`` and ``u_yy`` of FD order ``space_order`` with reflect
    padding, on the wavefield's grid."""

    def __init__(self, wavefield: Wavefield, *, space_order: int) -> None:
        g = wavefield.grid
        kx = second_difference_stencil(space_order, g.dx_nd)
        ky = second_difference_stencil(space_order, g.dy_nd)
        p = len(kx) // 2
        self._kx = torch.as_tensor(kx, dtype=g.dtype, device=g.device).view(1, 1, -1, 1)
        self._ky = torch.as_tensor(ky, dtype=g.dtype, device=g.device).view(1, 1, 1, -1)
        self._pad_x = (0, 0, p, p)
        self._pad_y = (p, p, 0, 0)

    def axes(self, utm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(u_xx, u_yy)`` of ``utm`` (spatial dims last; may be complex),
        each shaped and typed as ``utm``."""
        return (
            _apply_conv(utm, self._kx, self._pad_x),
            _apply_conv(utm, self._ky, self._pad_y),
        )
