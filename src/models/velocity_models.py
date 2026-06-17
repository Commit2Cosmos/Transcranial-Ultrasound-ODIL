from typing import Optional
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import numpy as np

from src.grid import Grid


class VelocityModel:
    """2D velocity field c(x, y) attached to a Grid (full extended grid)."""

    def __init__(
        self,
        grid: Grid,
        profile: str = "homogeneous",
        base: float = 1.0,
        contrast: float = 0.4,
        **profile_kwargs,
    ):
        self.grid = grid
        self.profile = profile
        self.base = base
        self.contrast = contrast
        self.profile_kwargs = profile_kwargs
        self.c = self._build()

    def _build(self) -> torch.Tensor:
        g = self.grid
        base_field = torch.full(g.shape, self.base, dtype=g.dtype, device=g.device)

        if self.profile == "homogeneous":
            return base_field

        if self.profile == "overdensity":
            r = self.profile_kwargs.get("radius", 0.3)
            cx, cy = self.profile_kwargs.get("center", (0.0, 0.0))
            mask = (g.X - cx) ** 2 + (g.Y - cy) ** 2 <= r**2
            return torch.where(
                mask, torch.full_like(base_field, self.base + self.contrast), base_field
            )

        if self.profile == "skull":
            raise NotImplementedError("TODO")

        if self.profile == "shepp_logan":
            scale = self.profile_kwargs.get("scale", 0.7)
            s_nx = max(2, int(g.interior_nx * scale))
            s_ny = max(2, int(g.interior_ny * scale))
            phantom = shepp_logan_phantom().astype(np.float32)
            # Rotate 90deg so the phantom's long axis aligns with the
            # AcquisitionGeometry ellipse's semi-major axis (y).
            phantom = np.rot90(phantom, k=1).copy()
            phantom = resize(phantom, (s_nx, s_ny), anti_aliasing=True, mode="reflect")
            phantom_t = torch.from_numpy(phantom).to(dtype=g.dtype, device=g.device)
            p = g.pml_width
            i0 = p + (g.interior_nx - s_nx) // 2
            j0 = p + (g.interior_ny - s_ny) // 2
            c = base_field.clone()
            c[i0 : i0 + s_nx, j0 : j0 + s_ny] = self.base + self.contrast * phantom_t
            return c

        raise ValueError(f"Unknown velocity profile: {self.profile!r}")

    @property
    def c_max(self) -> float:
        return float(self.c.max())

    @property
    def c_min(self) -> float:
        return float(self.c.min())

    def show(
        self,
        ax=None,
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_pml: bool = True,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        im = ax.imshow(
            self.c.cpu().numpy().T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_title(title or f"c(x, y) [{self.profile}]")
        plt.colorbar(im, ax=ax, shrink=0.85, label="c [m/s]")
        if show_pml:
            (ix0, ix1), (iy0, iy1) = self.grid.interior_extent
            ax.add_patch(
                Rectangle(
                    (ix0, iy0),
                    ix1 - ix0,
                    iy1 - iy0,
                    fill=False,
                    edgecolor="white",
                    linestyle="--",
                    linewidth=1.0,
                    label="non-PML interior",
                )
            )
        return ax
