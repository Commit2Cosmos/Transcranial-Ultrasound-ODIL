"""Spatial-smoothness / TV regularisers on the velocity field c(x, y).

Each regulariser returns the *unweighted* penalty value; the scalar weight
is supplied by the inverse loss via ``LossConfig.weights["reg"]``.
"""

from typing import Literal

import torch


RegKind = Literal["tikhonov", "tv_aniso", "tv_iso"]


class Regulariser:
    """Smoothness / TV prior on c(x, y) (interior cells only)."""

    def __init__(self, kind: RegKind = "tikhonov", eps: float = 1e-8) -> None:
        """Configure the regulariser kind.

        Parameters
        ----------
        kind : {"tikhonov", "tv_aniso", "tv_iso"}
            Penalty type on the spatial gradient of ``c``.
        eps : float
            Smoothing constant for the isotropic-TV square root.

        Notes
        -----
        Raises ``ValueError`` for an unknown ``kind``.
        """
        if kind not in ("tikhonov", "tv_aniso", "tv_iso"):
            raise ValueError(f"Unknown regulariser kind: {kind!r}")
        self.kind = kind
        self.eps = eps

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        """Evaluate the unweighted penalty on the interior velocity field.

        Parameters
        ----------
        c : torch.Tensor
            Interior velocity field ``c(x, y)``.

        Returns
        -------
        torch.Tensor
            Scalar penalty for the configured ``kind``.
        """
        dx_c = c[1:, :-1] - c[:-1, :-1]
        dy_c = c[:-1, 1:] - c[:-1, :-1]

        if self.kind == "tikhonov":
            return torch.sum(dx_c**2) + torch.sum(dy_c**2)

        if self.kind == "tv_aniso":
            return torch.sum(torch.abs(dx_c)) + torch.sum(torch.abs(dy_c))

        if self.kind == "tv_iso":
            return torch.sum(torch.sqrt(dx_c**2 + dy_c**2 + self.eps))
