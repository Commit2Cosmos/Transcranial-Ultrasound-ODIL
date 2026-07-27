from typing import Optional
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from scipy import ndimage as ndi

import numpy as np

from odil_wave.grid import Grid
from odil_wave.plot_utils import length_scale


# Speed-of-sound constants + phantom parameters mirroring the Stride reference
# ``stride/stride_forward_shepp.py::shepp_logan_sos`` so the Shepp-Logan model
# built here is numerically identical to the Stride forward ground truth.
SOS_WATER = 1500.0  # background / PML fill [m/s]
SOS_SKULL = 3000.0  # constant skull-rim speed [m/s]
SOFT_INTERCEPT = 1450.0  # interior = SOFT_INTERCEPT + SOFT_SLOPE * phantom
SOFT_SLOPE = 300.0
SHEPP_PHANTOM_SCALE = 0.90  # phantom occupies this fraction of the interior
SHEPP_THRESHOLD = 0.05  # head = fill_holes(phantom > threshold)


def velocity_norm(vmin: float, vcenter: float, vmax: float):
    """Non-linear colorbar normalisation for velocity fields.

    Maps half the colormap to [vmin, vcenter] and the other half to
    [vcenter, vmax]. (e.g. water at 1500 m/s vs skull at 3000 m/s), so
    low-velocity contrast is not visually crushed by the high-velocity range.

    """
    from matplotlib.colors import TwoSlopeNorm

    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


class VelocityModel:
    """2D velocity field c(x, y) attached to a Grid (full extended grid)."""

    def __init__(
        self,
        grid: Grid,
        profile: str = "homogeneous",
        base: float = SOS_WATER,
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
        if pml_c is not None:
            self.pml_c = float(pml_c)
        elif self._interior_bg is not None:
            # Skull profiles: match the PML to the surrounding brain background.
            self.pml_c = float(self._interior_bg)
        else:
            self.pml_c = float(self.c.min())

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
        vm._head_mask = None
        vm._interior_mask = None
        vm._rim_mask = None
        vm._interior_bg = None
        return vm

    def build_full_c(self, c_interior: torch.Tensor) -> torch.Tensor:
        """Pad `(interior_nx, interior_ny)` c with `pml_c` to full grid shape."""
        p = self.grid.pml_width
        return F.pad(c_interior, (p, p, p, p), mode="constant", value=self.pml_c)

    def _shepp_logan_interior_phantom(self, scale: float, threshold: float):
        """Normalised Shepp-Logan phantom + head masks on the *interior* grid.

        Mirrors ``stride/stride_forward_shepp.py::shepp_logan_sos`` exactly:
        same 90 deg rotation, resize with ``preserve_range``, min/max
        normalisation, ``binary_fill_holes`` head and
        ``binary_erosion(iterations=...)`` rim. All arrays are interior-shaped
        ``(interior_nx, interior_ny)`` so the morphology sees the same array
        borders Stride does; the PML padding is added later by embedding.

        Returns ``(phantom, head, inner, rim)`` — a float32 phantom in [0, 1]
        and three boolean masks (whole head, eroded interior, skull rim).
        """
        g = self.grid
        shape = (g.interior_nx, g.interior_ny)
        # Rotate 90 deg so the long axis aligns with AcquisitionGeometry's y.
        original = np.rot90(shepp_logan_phantom(), k=1).astype(np.float32)
        phantom_shape = (
            max(2, int(round(shape[0] * scale))),
            max(2, int(round(shape[1] * scale))),
        )
        smaller = resize(
            original,
            phantom_shape,
            anti_aliasing=True,
            mode="reflect",
            preserve_range=True,
        ).astype(np.float32)
        smaller -= smaller.min()
        if smaller.max() > 0:
            smaller /= smaller.max()

        phantom = np.zeros(shape, dtype=np.float32)
        ox = (shape[0] - phantom_shape[0]) // 2
        oy = (shape[1] - phantom_shape[1]) // 2
        phantom[ox : ox + phantom_shape[0], oy : oy + phantom_shape[1]] = smaller

        head = ndi.binary_fill_holes(phantom > threshold)
        # Skull thickness scales with the phantom size, not the full grid.
        erosion_pixels = max(2, int(round(0.02 * min(phantom_shape))))
        inner = ndi.binary_erosion(head, iterations=erosion_pixels)
        rim = head & ~inner
        return phantom, head, inner, rim

    def _embed_interior(self, interior_np: np.ndarray, fill: float) -> torch.Tensor:
        """Place an interior-shaped array into a full grid padded with ``fill``."""
        g = self.grid
        full = torch.full(g.shape, float(fill), dtype=g.dtype, device=g.device)
        full[g.interior_slice] = torch.from_numpy(np.ascontiguousarray(interior_np)).to(
            dtype=g.dtype, device=g.device
        )
        return full

    def _embed_mask(self, interior_mask: np.ndarray) -> torch.Tensor:
        """Place an interior-shaped boolean mask into a full-grid bool tensor."""
        g = self.grid
        full = torch.zeros(g.shape, dtype=torch.bool, device=g.device)
        full[g.interior_slice] = torch.from_numpy(
            np.ascontiguousarray(interior_mask)
        ).to(device=g.device)
        return full

    def _build(self) -> torch.Tensor:
        g = self.grid
        base_field = torch.full(g.shape, self.base, dtype=g.dtype, device=g.device)
        # Head / interior / rim masks + brain-background level, populated only
        # by skull-bearing profiles (None otherwise).
        self._head_mask = None
        self._interior_mask = None
        self._rim_mask = None
        self._interior_bg = None

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
            scale = self.profile_kwargs.get("scale", SHEPP_PHANTOM_SCALE)
            threshold = self.profile_kwargs.get("threshold", SHEPP_THRESHOLD)
            c_water = self.profile_kwargs.get("c_water", SOS_WATER)
            c_skull = self.profile_kwargs.get("c_skull", SOS_SKULL)
            soft_intercept = self.profile_kwargs.get("soft_intercept", SOFT_INTERCEPT)
            soft_slope = self.profile_kwargs.get("soft_slope", SOFT_SLOPE)

            phantom, head, inner, rim = self._shepp_logan_interior_phantom(
                scale, threshold
            )
            # Match stride shepp_logan_sos: water background; the whole head
            # takes soft-tissue speed carrying the phantom's fine features
            # (`soft_intercept + soft_slope * phantom`); the eroded rim is then
            # overwritten with a constant skull speed.
            model = np.full(phantom.shape, c_water, dtype=np.float32)
            interior_values = soft_intercept + soft_slope * phantom
            model[head] = interior_values[head]
            model[rim] = c_skull

            self._interior_bg = float(c_water)
            self._head_mask = self._embed_mask(head)
            self._interior_mask = self._embed_mask(inner)
            self._rim_mask = self._embed_mask(rim)
            return self._embed_interior(model, c_water)

        if self.profile == "shepp_logan_skull":
            scale = self.profile_kwargs.get("scale", SHEPP_PHANTOM_SCALE)
            threshold = self.profile_kwargs.get("threshold", SHEPP_THRESHOLD)
            c_water = self.profile_kwargs.get("c_water", SOS_WATER)
            c_skull = self.profile_kwargs.get("c_skull", SOS_SKULL)

            phantom, head, inner, rim = self._shepp_logan_interior_phantom(
                scale, threshold
            )
            # Perfect-skull start (cf. stride perfect_skull_start): water
            # everywhere except the constant skull rim; no soft-tissue features.
            interior_value = self.profile_kwargs.get("interior_value", c_water)
            model = np.full(phantom.shape, c_water, dtype=np.float32)
            if interior_value != c_water:
                model[inner] = interior_value
            model[rim] = c_skull

            self._interior_bg = float(c_water)
            self._head_mask = self._embed_mask(head)
            self._interior_mask = self._embed_mask(inner)
            self._rim_mask = self._embed_mask(rim)
            return self._embed_interior(model, c_water)

        if self.profile == "skull":
            raise NotImplementedError("Implement a realistic skull model!")

        raise ValueError(f"Unknown velocity profile: {self.profile!r}")

    @property
    def head_mask(self) -> Optional[torch.Tensor]:
        """Interior-restricted boolean mask of the region inside the skull.

        Covers the whole head (skull rim + everything it encloses), sliced to
        the non-PML interior so it aligns with metrics such as ``ssim``. Returns
        ``None`` for profiles that have no skull/head (e.g. homogeneous).
        """
        if getattr(self, "_head_mask", None) is None:
            return None
        return self._head_mask[self.grid.interior_slice]

    def skull_region_masks(self) -> tuple:
        """Return ``(head_mask, interior_mask, rim_mask)`` on the full grid.

        Available for ``shepp_logan`` / ``shepp_logan_skull``. Kept for
        notebooks that still call this helper.
        """
        if getattr(self, "_head_mask", None) is None:
            raise ValueError(
                "skull_region_masks() requires a skull-bearing profile "
                f"(shepp_logan / shepp_logan_skull), got {self.profile!r}"
            )
        return self._head_mask, self._interior_mask, self._rim_mask

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
        norm=None,
        vcenter: float = 1600.0,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        c_np = self.c.cpu().numpy()
        imshow_kw = dict(
            origin="lower",
            extent=(xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult),
            cmap="viridis",
        )
        # A `norm` (e.g. the two-slope velocity_norm) fully controls the colour
        # mapping, so vmin/vmax must not also be passed to imshow.
        # Defaults match AcquisitionGeometry.show / LossTape.show_velocity_recovery
        # (1400 / 1600 / 3000) so soft tissue is not crushed by the skull.
        if norm is not None:
            imshow_kw["norm"] = norm
        elif vmin is None and vmax is None:
            imshow_kw["norm"] = velocity_norm(
                vmin=1400.0, vcenter=vcenter, vmax=3000.0
            )
        else:
            lo = 1400.0 if vmin is None else float(vmin)
            hi = 3000.0 if vmax is None else float(vmax)
            if hi > lo:
                imshow_kw["norm"] = velocity_norm(
                    vmin=lo, vcenter=vcenter, vmax=hi
                )
            else:
                imshow_kw["vmin"] = lo
                imshow_kw["vmax"] = hi
        im = ax.imshow(c_np.T, **imshow_kw)
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"y [{x_unit}]")
        ax.set_aspect("equal")
        ax.set_title(title or f"c(x, y) [{self.profile}]")
        plt.colorbar(im, ax=ax, shrink=0.85, label="c [m/s]")
        if show_pml:
            (ix0, ix1), (iy0, iy1) = self.grid.interior_extent
            ax.add_patch(
                Rectangle(
                    (ix0 * x_mult, iy0 * x_mult),
                    (ix1 - ix0) * x_mult,
                    (iy1 - iy0) * x_mult,
                    fill=False,
                    edgecolor="white",
                    linestyle="--",
                    linewidth=1.0,
                    label="non-PML interior",
                )
            )
        return ax