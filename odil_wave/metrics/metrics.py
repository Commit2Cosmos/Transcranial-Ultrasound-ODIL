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
    """``arr``, full-grid or already interior-shaped, over the interior;
    ``ValueError`` for any other shape."""
    interior_shape = tuple(grid.interior_shape)
    if arr.shape == interior_shape:
        return arr
    if arr.shape == grid.shape:
        return arr[grid.interior_slice]
    raise ValueError(
        f"array shape {arr.shape} matches neither the grid's full shape "
        f"{grid.shape} nor its interior shape {interior_shape}"
    )


def _interior_mask(mask: Union[np.ndarray, torch.Tensor], grid: Grid) -> np.ndarray:
    """A region mask, full-grid or interior-shaped, as an interior-shaped bool
    array."""
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy().astype(bool)
    else:
        m = np.asarray(mask).astype(bool)
    return _interior(m, grid)


def mse(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
) -> float:
    """Mean squared error of (predicted - true) over the interior."""
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    return float(np.mean((p - t) ** 2))


def mae(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
) -> float:
    """Mean absolute error of (predicted - true) over the interior."""
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
    """SSIM of ``pred`` against ``true`` (full-grid or interior-shaped) over
    the interior, its map averaged over ``mask`` when given. ``kwargs`` go to
    ``skimage.metrics.structural_similarity``; ``data_range`` defaults to
    ``true``'s over the scored region."""
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
    """Plot and return the interior SSIM map of ``pred`` against ``true``,
    ``NaN`` outside ``mask``; arguments as for :func:`ssim`."""
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
