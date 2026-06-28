from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as _skimage_ssim
import torch

from odil_wave.grid import Grid


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def _interior(arr: np.ndarray, grid: Grid) -> np.ndarray:
    """Slice the full-grid array down to the interior, excluding the PML ring."""
    return arr[grid.interior_slice]


def mse(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
) -> float:
    """Mean Squared Error over the interior of (predicted - true).
    Evaluated on the interior region only, excluding the PML sponge ring.
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    return float(np.mean((p - t) ** 2))


def mae(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
) -> float:
    """Mean Absolute Error over the interior of (predicted - true).
    Evaluated on the interior region only, excluding the PML sponge ring.
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    return float(np.mean(np.abs(p - t)))


def ssim(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
    **kwargs,
) -> float:
    """Structural Similarity Index between predicted and true.
    Evaluated on the interior region only, excluding the PML sponge ring.
    data_range is inferred from true when not supplied explicitly.

    Could be considered passing additional kwargs:
        gaussian_weights=True, sigma=1.5
        win_size=<int>                    (patch size, default 7)
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    kwargs.setdefault("data_range", float(t.max() - t.min()))
    return float(_skimage_ssim(p, t, **kwargs))


def ssim_map(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    grid: Grid,
    title: str = "Per-pixel SSIM (1 = perfect recovery)",
    **kwargs,
) -> np.ndarray:
    """Per-pixel SSIM map between predicted and true. Plots and returns the map.
    Green = well recovered (near 1), red = poorly recovered (near -1).
    """
    p = _interior(_to_numpy(pred), grid)
    t = _interior(_to_numpy(true), grid)
    kwargs.setdefault("data_range", float(t.max() - t.min()))
    _, S = _skimage_ssim(p, t, full=True, **kwargs)
    (ix0, ix1), (iy0, iy1) = grid.interior_extent
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(S.T, origin="lower", extent=(ix0, ix1, iy0, iy1),
                   cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="SSIM")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return S