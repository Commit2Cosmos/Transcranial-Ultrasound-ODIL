from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as _skimage_ssim
import torch

from odil_wave.grid import Grid


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Coerce a torch tensor or numpy array to a float64 numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def _interior(arr: np.ndarray, grid: Grid) -> np.ndarray:
    """Slice the full-grid array down to the interior, excluding the absorbing boundary region."""
    return arr[grid.interior_slice]


def _interior_mask(mask: Union[np.ndarray, torch.Tensor], grid: Grid) -> np.ndarray:
    """Coerce a region mask to an interior-shaped boolean array.

    Accepts either an already interior-shaped mask or a full-grid mask (which
    is sliced down to the interior), as a torch tensor or numpy array.
    """
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy().astype(bool)
    else:
        m = np.asarray(mask).astype(bool)
    if m.shape == grid.shape:
        m = m[grid.interior_slice]
    return m


def mse(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
) -> float:
    """Mean squared error over the interior of (predicted - true).

    Evaluated on the interior region only, the absorbing boundary region is excluded.
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    return float(np.mean((p - t) ** 2))


def mae(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
) -> float:
    """Mean absolute error over the interior of (predicted - true).

    Evaluated on the interior region only, the absorbing boundary region is excluded.
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    return float(np.mean(np.abs(p - t)))


def ssim(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
    mask: Union[np.ndarray, torch.Tensor, None] = None,
    **kwargs,
) -> float:
    """Structural Similarity Index between predicted and true.

    Evaluated on the interior region only, the absorbing boundary region is
    excluded.

    Parameters
    ----------
    pred, true :
        Fields to compare, full-grid or already interior-shaped.
    grid :
        Grid providing the interior region.
    mask :
        Optional region (full-grid or interior-shaped) to restrict scoring
        to, e.g. the area inside the skull. When given, the per-pixel SSIM
        map is computed over the interior and averaged only over the mask.
    **kwargs :
        Forwarded to ``skimage.metrics.structural_similarity``, e.g.
        ``gaussian_weights=True``, ``sigma=1.5``, ``win_size=<int>``.
        ``data_range`` is inferred from the true field over the scored
        region when not provided explicitly.

    Returns
    -------
    float
        Structural similarity index, in [-1, 1] (1 = perfect match).
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    if mask is None:
        kwargs.setdefault("data_range", float(t.max() - t.min()))
        return float(_skimage_ssim(p, t, **kwargs))
    m = _interior_mask(mask, grid)
    kwargs.setdefault("data_range", float(t[m].max() - t[m].min()))
    _, S = _skimage_ssim(p, t, full=True, **kwargs)
    return float(S[m].mean())


def ssim_map(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
    title: str = "SSIM (1 = perfect recovery)",
    mask: Union[np.ndarray, torch.Tensor, None] = None,
    **kwargs,
) -> np.ndarray:
    """Per-pixel SSIM map between predicted and true. Plots and returns the map.

    Green = well recovered (near 1), red = poorly recovered (near -1).

    Parameters
    ----------
    pred, true :
        Fields to compare, full-grid or already interior-shaped.
    grid :
        Grid providing the interior region.
    title :
        Plot title.
    mask :
        Optional region (full-grid or interior-shaped) to restrict scoring
        to. When given, cells outside it are blanked (NaN) in the returned
        map.
    **kwargs :
        Forwarded to ``skimage.metrics.structural_similarity``.
        ``data_range`` is inferred from the true field over the scored
        region when not provided explicitly.

    Returns
    -------
    numpy.ndarray
        Per-pixel SSIM map over the interior region; ``NaN`` outside the
        mask when one is given.
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    m = _interior_mask(mask, grid) if mask is not None else None
    kwargs.setdefault(
        "data_range",
        float(t[m].max() - t[m].min()) if m is not None else float(t.max() - t.min()),
    )
    _, S = _skimage_ssim(p, t, full=True, **kwargs)
    if m is not None:
        S = np.where(m, S, np.nan)
    (ix0, ix1), (iy0, iy1) = grid.interior_extent
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        S.T, origin="lower", extent=(ix0, ix1, iy0, iy1), cmap="RdYlGn", vmin=-1, vmax=1
    )
    plt.colorbar(im, ax=ax, label="SSIM")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return S
