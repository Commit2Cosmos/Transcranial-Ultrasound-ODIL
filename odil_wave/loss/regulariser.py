"""Spatial-smoothness / TV regularisers on the velocity field c(x, y).

Each regulariser returns the *unweighted* penalty value; the scalar weight
comes from ``loss.weights.reg`` (``LossConfig.weights["reg"]``) and is
applied by the optimiser objective.

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
    """Area-weighted mean of ``f`` over one family of interior edges.

    ``dim`` is the axis across the edges (``1`` for x-edges). Edges on the
    interior boundary along ``dim`` carry half weight, since half of their
    dual cell lies outside the interior (trapezoid rule); the weights then sum
    to the ``(nx-1)(ny-1)`` interior cells.
    """
    ends = f.select(dim, 0) + f.select(dim, f.shape[dim] - 1)
    return (f.sum() - 0.5 * ends.sum()) / (f.numel() - ends.numel())


class Regulariser:
    """Smoothness / TV prior on c(x, y) (interior cells only)."""

    def __init__(self, *, kind: RegKind, eps: float, grid: Grid) -> None:
        """Configure the regulariser kind.

        ``kind`` and ``eps`` are set by the run config's ``loss.regulariser``
        block and, with ``grid``, are required keywords (ADR 0001).

        Parameters
        ----------
        kind : {"tikhonov", "tv_aniso", "tv_iso"}
            Measure of the spatial gradient of ``c``: mean of
            ``|grad' c'|^2``, of ``|d'_x c'| + |d'_y c'|``, or of
            ``sqrt(|grad' c'|^2 + eps) - sqrt(eps)``.
        eps : float
            > 0 smoothing floor inside the isotropic-TV square root, a squared
            dimensionless gradient: ``sqrt(eps)`` is a contrast of
            ``sqrt(eps) * c0`` across ``L0``.
        grid : Grid
            Supplies the interior spacing and the scales ``L0`` and ``c0``.

        Notes
        -----
        Raises ``ValueError`` for an unknown ``kind`` or ``eps <= 0``.
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
        """Evaluate the unweighted penalty on the interior velocity field.

        Parameters
        ----------
        c : torch.Tensor
            Interior velocity field ``c(x, y)`` in m/s, shape
            ``(interior_nx, interior_ny)``.

        Returns
        -------
        torch.Tensor
            Scalar value for the configured ``kind``; ``0`` on a constant
            field.
        """
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
