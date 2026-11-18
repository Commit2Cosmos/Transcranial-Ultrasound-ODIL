"""Spatial-smoothness / TV regularisers on the velocity field c(x, y).

Each regulariser returns the *unweighted* penalty value; the optimiser objective
applies the weight.

The value is dimensionless and resolution-independent: with the grid's
scales ``c' = c / c0`` and ``grad' = L0 * grad``, it is the interior mean

    R = (1 / |Omega'|) * integral over Omega of |grad' c'|^p dA'

so a given physical model gives the same ``R`` at any resolution or spacing.
Only interior cells and edges count: the PML velocity is not a model unknown.
"""

import math
from typing import Literal

import torch

from odil_wave.grid import Grid

RegKind = Literal["tikhonov", "tv_aniso", "tv_iso"]


def _edge_mean(f: torch.Tensor, dim: int) -> torch.Tensor:
    """Area-weighted mean of ``f`` over one family of interior edges, ``dim``
    the axis across them. Boundary edges carry half weight, since half their
    dual cell lies outside the interior (trapezoid rule)."""
    ends = f.select(dim, 0) + f.select(dim, f.shape[dim] - 1)
    return (f.sum() - 0.5 * ends.sum()) / (f.numel() - ends.numel())


class Regulariser:
    """Smoothness / TV prior on c(x, y) (interior cells only)."""

    def __init__(
        self, *, grid: Grid, kind: RegKind = "tv_iso", eps: float = 1e-4
    ) -> None:
        """Defaults follow ADR 0001. ``kind`` is the mean of ``|grad' c'|^2``
        (``"tikhonov"``), of ``|d'_x c'| + |d'_y c'|`` (``"tv_aniso"``) or of
        ``sqrt(|grad' c'|^2 + eps) - sqrt(eps)`` (``"tv_iso"``). ``eps`` > 0 is a
        squared dimensionless gradient: ``sqrt(eps)`` is a contrast of
        ``sqrt(eps) * c0`` across ``L0``.
        """
        if kind not in ("tikhonov", "tv_aniso", "tv_iso"):
            raise ValueError(f"Unknown regulariser kind: {kind!r}")
        if not eps > 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.kind = kind
        self.eps = eps
        self.dx_nd = grid.dx_nd
        self.dy_nd = grid.dy_nd
        self.c0 = grid.c0

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        """The unweighted penalty of interior ``c`` ``(interior_nx,
        interior_ny)``, m/s."""
        c = c / self.c0
        gx = (c[1:, :] - c[:-1, :]) / self.dx_nd  # x-edges, (nx-1, ny)
        gy = (c[:, 1:] - c[:, :-1]) / self.dy_nd  # y-edges, (nx, ny-1)

        if self.kind == "tikhonov":
            return _edge_mean(gx**2, dim=1) + _edge_mean(gy**2, dim=0)

        if self.kind == "tv_aniso":
            return _edge_mean(torch.abs(gx), dim=1) + _edge_mean(torch.abs(gy), dim=0)

        # tv_iso: the gradient at cell centres, each component the average of
        # the cell's two parallel edges; subtracting sqrt(eps) zeroes a
        # constant field without changing the optimum.
        gx_c = 0.5 * (gx[:, 1:] + gx[:, :-1])
        gy_c = 0.5 * (gy[1:, :] + gy[:-1, :])
        tv = torch.mean(torch.sqrt(gx_c**2 + gy_c**2 + self.eps))
        return tv - math.sqrt(self.eps)
