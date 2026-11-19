"""The PML: a stretched-coordinate absorbing layer on the wave equation's
mass term and Laplacian. The profile ``sigma`` lives on
:class:`~odil_wave.grid.Grid`."""

import torch

from odil_wave.wavefield import Wavefield


def _mirror_ghost_faces(a: torch.Tensor) -> torch.Tensor:
    """Append a ghost face at each end of ``a``'s face axis (dim 1), mirroring
    the reflect padding of the second differences."""
    return torch.cat([a[:, :1], a, a[:, -1:]], dim=1)


class Pml:
    """Conservative stretched-coordinate PML, per frequency.

    With ``s_ξ = 1 + weight σ_ξ / (iω)`` along axis ``ξ`` (``weight = 0``
    disables it)::

        mass(λ_tt) u = λ_tt s_x s_y u
        stretched_laplacian(u) = s_y Λ̃_x u + s_x Λ̃_y u,
        Λ̃_ξ u = Λ_ξ u + D⁻_ξ[(1/s_ξ − 1) D⁺_ξ u]

    ``Λ_ξ`` is the order-P second difference and ``D±`` the order-2 one-sided
    differences, ``1/s`` sampled on the faces between nodes. In the interior
    ``s = 1``, so both reduce to ``λ_tt u`` and ``Λ_x u + Λ_y u``. At order 2
    ``Λ̃`` is the conservative form ``D⁻ (1/s) D⁺``; at higher orders the PML
    reflects more (#160 decides that form).
    """

    def __init__(self, wavefield: Wavefield, weight: float) -> None:
        """Precompute the stretch at every frequency of ``wavefield``; ``ValueError``
        for a 0 Hz frequency with the PML on."""
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
        # (nf, nx + 1, 1) and (nf, 1, ny + 1): 1/s - 1 at the faces, ghosts
        # included.
        self.a_x = _mirror_ghost_faces(1.0 / (1.0 + w * sig_xf * inv_iw) - 1.0)
        self.a_y = _mirror_ghost_faces(
            (1.0 / (1.0 + w * sig_yf * inv_iw) - 1.0).transpose(1, 2)
        ).transpose(1, 2)

    def mass(self, lambda_tt: torch.Tensor) -> torch.Tensor:
        """``λ_tt s_x s_y``, ``(nf, nx, ny)``; ``lambda_tt`` broadcast ``(…, nf,
        1, 1)``."""
        return lambda_tt * self.s_x * self.s_y

    def stretched_laplacian(
        self, u: torch.Tensor, u_xx: torch.Tensor, u_yy: torch.Tensor
    ) -> torch.Tensor:
        """``s_y Λ̃_x u + s_x Λ̃_y u`` from ``u`` and its order-P ``u_xx``,
        ``u_yy``; all ``(…, nf, nx, ny)`` complex."""
        corr_x = _flux_difference(u, self.a_x, dim=-2) / self.dx_nd**2
        corr_y = _flux_difference(u, self.a_y, dim=-1) / self.dy_nd**2
        return self.s_y * (u_xx + corr_x) + self.s_x * (u_yy + corr_y)


def _flux_difference(u: torch.Tensor, a: torch.Tensor, dim: int) -> torch.Tensor:
    """``D⁻[a D⁺ u]`` along ``dim`` (unit spacing), reflect-padded; ``a`` holds
    the face coefficients, ghost faces included."""
    n = u.shape[dim]
    idx = torch.cat(
        [
            torch.tensor([1]),
            torch.arange(n),
            torch.tensor([n - 2]),
        ]
    ).to(u.device)
    up = u.index_select(dim, idx)
    flux = a * torch.diff(up, dim=dim)
    return torch.diff(flux, dim=dim)
