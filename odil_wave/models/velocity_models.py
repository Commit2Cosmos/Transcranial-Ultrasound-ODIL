from typing import Optional
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage import morphology
from scipy import ndimage as ndi

import numpy as np

from odil_wave.grid import Grid


class VelocityModel:
    """2D velocity field c(x, y) attached to a Grid (full extended grid)."""

    def __init__(
        self,
        grid: Grid,
        profile: str = "homogeneous",
        base: float = 1.0,
        contrast: float = 0.4,
        pml_c: Optional[float] = None,
        **profile_kwargs,
    ):
        self.grid = grid
        self.profile = profile
        self.base = base
        self.contrast = contrast
        self.profile_kwargs = profile_kwargs
        self.c = self._build()
        self.pml_c = float(pml_c) if pml_c is not None else float(self.c.min())

    @classmethod
    def from_field(
        cls,
        grid: Grid,
        c: torch.Tensor,
        pml_c: Optional[float] = None,
    ) -> "VelocityModel":
        """Build a VelocityModel from an already-computed full-grid c tensor."""
        vm = cls.__new__(cls)
        vm.grid = grid
        vm.profile = "custom"
        vm.base = float("nan")
        vm.contrast = float("nan")
        vm.profile_kwargs = {}
        vm.c = c.to(dtype=grid.dtype, device=grid.device).reshape(grid.shape)
        vm.pml_c = float(pml_c) if pml_c is not None else float(vm.c.min())
        return vm

    def build_full_c(self, c_interior: torch.Tensor) -> torch.Tensor:
        """Pad `(interior_nx, interior_ny)` c with `pml_c` to full grid shape."""
        p = self.grid.pml_width
        return F.pad(c_interior, (p, p, p, p), mode="constant", value=self.pml_c)

    def _shepp_logan_embedded(self, scale: float) -> torch.Tensor:
        """Resized, rotated, PML-padded Shepp-Logan phantom, values in [0, 1]."""
        g = self.grid
        s_nx = max(2, int(g.interior_nx * scale))
        s_ny = max(2, int(g.interior_ny * scale))
        phantom = shepp_logan_phantom().astype(np.float32)
        # Rotate 90 deg so the long axis aligns with AcquisitionGeometry's y.
        phantom = np.rot90(phantom, k=1).copy()
        phantom = resize(phantom, (s_nx, s_ny), anti_aliasing=True, mode="reflect")
        phantom = phantom / phantom.max()  # normalize so contrast = true peak delta
        phantom_t = torch.from_numpy(phantom).to(dtype=g.dtype, device=g.device)
        p = g.pml_width
        i0 = p + (g.interior_nx - s_nx) // 2
        j0 = p + (g.interior_ny - s_ny) // 2
        field = torch.zeros(g.shape, dtype=g.dtype, device=g.device)
        field[i0 : i0 + s_nx, j0 : j0 + s_ny] = phantom_t
        return field

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

        if self.profile == "shepp_logan":
            scale = self.profile_kwargs.get("scale", 0.7)
            phantom_field = self._shepp_logan_embedded(scale)
            return base_field + self.contrast * phantom_field

        if self.profile == "shepp_logan_skull":
            scale = self.profile_kwargs.get("scale", 0.7)
            threshold = self.profile_kwargs.get("threshold", 0.05)
            interior_erosion = self.profile_kwargs.get("interior_erosion", 2)
            interior_value = self.profile_kwargs.get("interior_value", self.base)
            phantom_field = self._shepp_logan_embedded(scale)

            # Filled-head mask: threshold + fill holes.
            mask_np = (phantom_field > threshold).cpu().numpy()
            head_mask_np = ndi.binary_fill_holes(mask_np)

            # Three disjoint regions:
            #   background (~head_mask)            -> base
            #   brain interior (eroded head_mask)  -> interior_value
            #   skull rim (head ^ ~interior)       -> base + contrast * phantom
            interior_mask_np = morphology.erosion(
                head_mask_np, morphology.disk(interior_erosion)
            )
            rim_mask_np = head_mask_np & ~interior_mask_np
            interior_mask = torch.from_numpy(interior_mask_np).to(device=g.device)
            rim_mask = torch.from_numpy(rim_mask_np).to(device=g.device)

            c = torch.where(
                interior_mask,
                torch.full_like(base_field, interior_value),
                base_field,
            )
            c = torch.where(
                rim_mask,
                base_field + self.contrast * phantom_field,
                c,
            )
            return c

        if self.profile == "skull":
            raise NotImplementedError("Implement a realistic skull model!")

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
