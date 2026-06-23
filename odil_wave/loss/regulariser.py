"""Spatial-smoothness / TV regularisers on the velocity field c(x, y).

Mirrors the `c_regularizer` helper used in `sandbox/odil_2d_inv.ipynb`:
returns the *unweighted* penalty value; the scalar `lam` is supplied by
the inverse loss via `LossConfig.weights["reg"]`.
"""

from typing import Literal

import torch


RegKind = Literal["tikhonov", "tv_aniso", "tv_iso"]


class Regulariser:
    """Smoothness / TV prior on c(x, y) (interior cells only)."""

    def __init__(self, kind: RegKind = "tikhonov", eps: float = 1e-8) -> None:
        if kind not in ("tikhonov", "tv_aniso", "tv_iso"):
            raise ValueError(f"Unknown regulariser kind: {kind!r}")
        self.kind = kind
        self.eps = eps

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        dx_c = c[1:, :-1] - c[:-1, :-1]
        dy_c = c[:-1, 1:] - c[:-1, :-1]

        if self.kind == "tikhonov":
            return torch.sum(dx_c**2) + torch.sum(dy_c**2)

        if self.kind == "tv_aniso":
            return torch.sum(torch.abs(dx_c)) + torch.sum(torch.abs(dy_c))

        # TODO: implement the isotropic TV branch.
        if self.kind == "tv_iso":
            raise NotImplementedError("tv_iso branch isn't implementated!")
