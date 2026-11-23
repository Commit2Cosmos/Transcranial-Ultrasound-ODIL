"""The PML: a stretched-coordinate absorbing layer on the wave equation's
mass term and Laplacian. The profile ``sigma`` lives on
:class:`~odil_wave.grid.Grid`."""

import torch

from odil_wave.wavefield import Wavefield
from .spatial import STAGGERED_D1_COEFFS


def _reflect(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Map node indices to ``[0, n)`` like ``F.pad(..., mode='reflect')``."""
    period = 2 * n - 2
    m = torch.remainder(idx, period)
    return torch.where(m >= n, period - m, m)


class Pml:
    """Conservative stretched-coordinate PML, per frequency.

    With ``s_ξ = 1 + weight σ_ξ / (iω)`` along axis ``ξ`` (``weight = 0``
    disables it)::

        mass(λ_tt) u = λ_tt s_x s_y u
        stretched_laplacian(u) = s_y Λ̃_x u + s_x Λ̃_y u,
        Λ̃_ξ u = Λ_ξ u + D⁻_ξ[(1/s_ξ − 1) D⁺_ξ u]

    ``Λ_ξ`` is the order-P second difference and ``D±`` the staggered order-P
    first differences, ``1/s`` sampled on the faces between nodes. In the
    interior ``s = 1``, so the mass reduces to ``λ_tt u``, and the Laplacian to
    ``Λ_x u + Λ_y u`` beyond the ``P/2 - 1`` cells ``D⁻`` reaches from the PML.
    At order 2 ``Λ̃`` is the conservative form ``D⁻ (1/s) D⁺``.
    """

    def __init__(
        self, wavefield: Wavefield, weight: float, *, space_order: int
    ) -> None:
        """Precompute the stretch at every frequency of ``wavefield``; ``ValueError``
        for an unsupported ``space_order``, or a 0 Hz frequency with the PML on."""
        if space_order not in STAGGERED_D1_COEFFS:
            raise ValueError(f"Unsupported space order: {space_order}")
        self.d1_coeffs = STAGGERED_D1_COEFFS[space_order]
        grid = wavefield.grid
        freq = wavefield.frequency_selection
        cdtype = wavefield.cdtype
        self.dx_nd, self.dy_nd = grid.dx_nd, grid.dy_nd
        sig_x = grid.sigma_x_nd[:, 0].view(1, -1, 1)
        sig_y = grid.sigma_y_nd[0, :].view(1, 1, -1)
        sig_xf = grid.sigma_x_faces_nd.view(1, -1, 1)
        sig_yf = grid.sigma_y_faces_nd.view(1, 1, -1)
        nf = freq.n_frequencies
        if weight == 0.0 or grid.pml_width == 0:
            inv_iw = torch.zeros(nf, 1, 1, dtype=cdtype, device=grid.device)
        else:
            omega = freq.omega_nd.view(nf, 1, 1)
            if bool((omega == 0).any()):
                raise ValueError("the PML's stretch is undefined at 0 Hz")
            inv_iw = (1.0 / (1j * omega)).to(cdtype)
        w = float(weight)
        # (nf, nx, 1) and (nf, 1, ny): the stretch at the nodes.
        self.s_x = 1.0 + w * sig_x * inv_iw
        self.s_y = 1.0 + w * sig_y * inv_iw
        # (nf, nx + P - 1, 1) and (nf, 1, ny + P - 1): 1/s - 1 at the faces
        # k - 1/2 for k = 1 - P/2 .. n - 1 + P/2, the ghosts mirroring.
        half = space_order // 2
        self.a_x = (1.0 / (1.0 + w * sig_xf * inv_iw) - 1.0).index_select(
            1, _mirror_faces(grid.nx, half).to(grid.device)
        )
        self.a_y = (1.0 / (1.0 + w * sig_yf * inv_iw) - 1.0).index_select(
            2, _mirror_faces(grid.ny, half).to(grid.device)
        )

    def mass(self, lambda_tt: torch.Tensor) -> torch.Tensor:
        """``λ_tt s_x s_y``, ``(nf, nx, ny)``; ``lambda_tt`` broadcast ``(…, nf,
        1, 1)``."""
        return lambda_tt * self.s_x * self.s_y

    def stretched_laplacian(
        self, u: torch.Tensor, u_xx: torch.Tensor, u_yy: torch.Tensor
    ) -> torch.Tensor:
        """``s_y Λ̃_x u + s_x Λ̃_y u`` from ``u`` and its order-P ``u_xx``,
        ``u_yy``; all ``(…, nf, nx, ny)`` complex."""
        corr_x = _flux_difference(u, self.a_x, self.d1_coeffs, dim=-2)
        corr_y = _flux_difference(u, self.a_y, self.d1_coeffs, dim=-1)
        return self.s_y * (u_xx + corr_x / self.dx_nd**2) + self.s_x * (
            u_yy + corr_y / self.dy_nd**2
        )


def _mirror_faces(n: int, half: int) -> torch.Tensor:
    """Indices into the ``n - 1`` faces between nodes of the faces ``k - 1/2``,
    ``k = 1 - half .. n - 1 + half``: a ghost maps to its mirror image."""
    k = torch.arange(1 - half, n + half)
    return torch.maximum(_reflect(k - 1, n), _reflect(k, n)) - 1


def _staggered_difference(
    x: torch.Tensor, coeffs: tuple[float, ...], dim: int, length: int
) -> torch.Tensor:
    """``length`` staggered differences of ``x`` along ``dim`` (unit spacing),
    the first centred between ``x``'s entries ``P/2 - 1`` and ``P/2``."""
    half = len(coeffs)
    return sum(
        c * (x.narrow(dim, half - 1 + m, length) - x.narrow(dim, half - m, length))
        for m, c in enumerate(coeffs, start=1)
    )


def _flux_difference(
    u: torch.Tensor, a: torch.Tensor, coeffs: tuple[float, ...], dim: int
) -> torch.Tensor:
    """``D⁻[a D⁺ u]`` along ``dim`` (unit spacing, staggered ``coeffs``),
    reflect-padded; ``a`` holds the face coefficients, ghosts included."""
    n = u.shape[dim]
    p = 2 * len(coeffs)
    up = u.index_select(dim, _reflect(torch.arange(1 - p, n - 1 + p), n).to(u.device))
    flux = a * _staggered_difference(up, coeffs, dim, n + p - 1)
    return _staggered_difference(flux, coeffs, dim, n)
