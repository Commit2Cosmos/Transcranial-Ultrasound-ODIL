"""Geometry-aware grid-transfer (prolongation / resampling) operators.

Convention
----------
:class:`~odil_wave.grid.Grid` samples a field on a **vertex-centred**
(node-centred) lattice: ``x = linspace(x_min, x_max, nx)`` with the interior
spanning exactly ``interior_extent`` and *both* endpoints included
(``dx = (x_max - x_min) / (nx - 1)``). Two grids that share ``interior_extent``
therefore have coincident interior corner nodes, so bilinear resampling with
``align_corners=True`` is the geometrically correct transfer for this
convention: a linear function of the node index equals a linear function of the
physical coordinate, so the operator is exact for constant and affine physical
fields (up to floating-point round-off).

Only the vertex-centred convention is implemented. A mismatched ``staggering``
argument raises rather than silently applying the wrong geometry — the module
never guesses a convention the surrounding code does not actually use.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from odil_wave.grid import complex_dtype

#: Grid-sampling conventions this module knows how to transfer between.
SUPPORTED_STAGGERINGS: Tuple[str, ...] = ("vertex",)

# Aliases callers may reasonably pass for the vertex-centred convention.
_VERTEX_ALIASES = frozenset({"vertex", "node", "node_centered", "node-centred"})

_ExtentPair = Tuple[Tuple[float, float], Tuple[float, float]]


def _canonical_staggering(staggering: str) -> str:
    """Normalise a staggering string, raising for anything unsupported."""
    key = str(staggering).lower()
    if key in _VERTEX_ALIASES:
        return "vertex"
    raise ValueError(
        f"Unsupported staggering {staggering!r}; this codebase samples fields on "
        f"a vertex-centred grid. Supported: {SUPPORTED_STAGGERINGS}. A "
        f"cell-centred transfer would need a different sampling geometry "
        f"(align_corners=False with a half-cell offset) and is not implemented."
    )


def _as_extent_pair(interior_extent) -> _ExtentPair:
    """Validate + return ``((x0, x1), (y0, y1))`` for a 2-D interior extent."""
    try:
        (x0, x1), (y0, y1) = interior_extent
    except (TypeError, ValueError) as exc:  # not a pair-of-pairs
        raise ValueError(
            f"interior_extent must be ((x0, x1), (y0, y1)); got {interior_extent!r}"
        ) from exc
    return (float(x0), float(x1)), (float(y0), float(y1))


def _extents_close(a: _ExtentPair, b: _ExtentPair, tol: float) -> bool:
    (ax0, ax1), (ay0, ay1) = a
    (bx0, bx1), (by0, by1) = b
    scale = max(abs(ax1 - ax0), abs(ay1 - ay0), 1.0)
    pairs = ((ax0, bx0), (ax1, bx1), (ay0, by0), (ay1, by1))
    return all(abs(u - v) <= tol * scale for u, v in pairs)


def assert_compatible_interiors(
    src_grid,
    dst_grid,
    *,
    staggering: str = "vertex",
    tol: float = 1e-6,
) -> None:
    """Raise if ``src_grid``/``dst_grid`` interiors are not transfer-compatible.

    Compatibility requires, for the vertex-centred convention:

    * both grids are 2-D;
    * both share the same physical interior region (same origin/bounds/extent
      to a relative tolerance ``tol``);
    * a supported, matching staggering.

    The resolutions (node counts) may differ — that is the whole point of a
    transfer operator.
    """
    _canonical_staggering(staggering)
    src_ext = _as_extent_pair(src_grid.interior_extent)
    dst_ext = _as_extent_pair(dst_grid.interior_extent)

    if len(src_grid.interior_shape) != 2 or len(dst_grid.interior_shape) != 2:
        raise ValueError(
            "grid transfer is implemented for 2-D interiors only; got shapes "
            f"{src_grid.interior_shape} -> {dst_grid.interior_shape}"
        )
    if not _extents_close(src_ext, dst_ext, tol):
        raise ValueError(
            "source and destination interiors describe different physical "
            f"regions: {src_ext} vs {dst_ext} (rel-tol {tol:g}). A prolongation "
            "operator only transfers between grids over the same domain."
        )


def resample_interior_field(
    field: torch.Tensor,
    src_grid,
    dst_grid,
    *,
    staggering: str = "vertex",
    compute_dtype: Optional[torch.dtype] = None,
    align_corners: bool = True,
) -> torch.Tensor:
    """Resample a real interior field from ``src_grid`` to ``dst_grid``.

    Parameters
    ----------
    field : Tensor of shape ``(src_grid.interior_nx, src_grid.interior_ny)``
        Real-valued interior field (no PML ring).
    src_grid, dst_grid : Grid
        Must describe the same physical interior region (validated).
    staggering : str
        Grid-sampling convention; only ``"vertex"`` is supported.
    compute_dtype : torch.dtype, optional
        Precision at which the interpolation is carried out. Defaults to the
        input dtype (promoted to at least ``float32``). Pass ``torch.float64``
        for a maximally accurate one-off transfer (e.g. a warm-start hand-off).
    align_corners : bool
        Kept for completeness; must stay ``True`` for the vertex convention.

    Returns
    -------
    Tensor of shape ``(dst_grid.interior_nx, dst_grid.interior_ny)`` with
    ``dst_grid``'s dtype and device.

    Notes
    -----
    Differentiable with respect to ``field``: bilinear interpolation and the
    dtype/device casts all carry gradients, so this can sit inside an autograd
    graph (used by the mODIL parameterization).
    """
    _canonical_staggering(staggering)
    assert_compatible_interiors(src_grid, dst_grid, staggering=staggering)

    expected = (int(src_grid.interior_nx), int(src_grid.interior_ny))
    if field.ndim != 2 or tuple(field.shape) != expected:
        raise ValueError(
            f"field shape {tuple(field.shape)} does not match the source "
            f"interior {expected}."
        )

    if compute_dtype is None:
        work = (
            field.dtype
            if field.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
    else:
        work = compute_dtype

    dst_shape = (int(dst_grid.interior_nx), int(dst_grid.interior_ny))
    if dst_shape == expected and align_corners:
        # Same lattice: transfer is the identity (avoid a needless resample).
        out = field.to(work)
    else:
        src = field.to(work).unsqueeze(0).unsqueeze(0)
        out = (
            F.interpolate(
                src, size=dst_shape, mode="bilinear", align_corners=align_corners
            )
            .squeeze(0)
            .squeeze(0)
        )
    return out.to(dtype=dst_grid.dtype, device=dst_grid.device)


def prolongate_interior_velocity(
    full_velocity: torch.Tensor,
    src_grid,
    dst_grid,
    *,
    staggering: str = "vertex",
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Transfer the *interior* velocity of a full-grid field to a finer grid.

    Only the non-PML interior is transferred; the PML velocity is a property of
    each grid's absorbing layer and must be rebuilt per target grid (e.g. via
    :meth:`~odil_wave.models.VelocityModel.build_full_c`). Using bilinear (not
    nearest) resampling avoids injecting high-wavenumber content the fine grid
    would then have to unlearn.

    Parameters
    ----------
    full_velocity : Tensor of shape ``src_grid.shape`` (interior + PML).
    src_grid, dst_grid : Grid over the same physical interior.

    Returns
    -------
    Tensor of shape ``(dst_grid.interior_nx, dst_grid.interior_ny)`` (interior
    only), with ``dst_grid``'s dtype/device.
    """
    if tuple(full_velocity.shape) != tuple(src_grid.shape):
        raise ValueError(
            f"full_velocity shape {tuple(full_velocity.shape)} does not match "
            f"the source full grid {tuple(src_grid.shape)}."
        )
    interior = full_velocity[src_grid.interior_slice]
    return resample_interior_field(
        interior, src_grid, dst_grid, staggering=staggering, compute_dtype=compute_dtype
    )


def resample_full_wavefield(
    amp: torch.Tensor,
    src_grid,
    dst_grid,
    *,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Resample a complex full-grid wavefield ``(..., nx, ny)`` to ``dst_grid``.

    Real and imaginary parts are interpolated independently on the *full* grid
    (interior + PML). The returned complex dtype matches ``dst_grid`` (i.e.
    ``complex64`` for a ``float32`` grid, ``complex128`` for ``float64``) rather
    than always widening to ``complex128``.

    This is a raw geometric resample; it does **not** check that the two grids
    share a PML discretisation. Because a wavefield is a PML-affected discrete
    PDE state, transferring it across grids with different absorbing layers is
    unsafe — callers that use the result as an optimiser seed should validate
    compatibility first (see
    :meth:`~odil_wave.optimisation.coarse_to_fine.LevelHandoff.validated_wavefield_seed_on`).
    """
    if tuple(amp.shape[-2:]) != tuple(src_grid.shape):
        raise ValueError(
            f"wavefield trailing shape {tuple(amp.shape[-2:])} does not match "
            f"the source full grid {tuple(src_grid.shape)}."
        )
    if not amp.is_complex():
        raise ValueError("resample_full_wavefield expects a complex wavefield.")

    lead = amp.shape[:-2]
    work = compute_dtype or (
        torch.float64 if amp.dtype == torch.complex128 else torch.float32
    )

    def _resamp(x: torch.Tensor) -> torch.Tensor:
        x = x.to(work).reshape(-1, 1, src_grid.nx, src_grid.ny)
        y = F.interpolate(
            x, size=(dst_grid.nx, dst_grid.ny), mode="bilinear", align_corners=True
        )
        return y.reshape(*lead, dst_grid.nx, dst_grid.ny)

    re = _resamp(amp.real)
    im = _resamp(amp.imag)
    out = torch.complex(re, im)
    return out.to(dtype=complex_dtype(dst_grid.dtype), device=dst_grid.device)
