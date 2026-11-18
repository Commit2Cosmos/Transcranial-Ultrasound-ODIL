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
SHEPP_THRESHOLD = 0.05  # head = fill_holes(phantom > threshold)


def velocity_norm(vmin: float, vcenter: float, vmax: float):
    """Two-slope colour normalisation, ``[vmin, vcenter]`` and ``[vcenter,
    vmax]`` (m/s) each half the bar, so a wide high-velocity range (skull)
    does not crush low-velocity contrast."""
    from matplotlib.colors import TwoSlopeNorm

    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


# Two-slope velocity colour bounds [m/s] for the Shepp-Logan profiles: tissue
# in the lower half of the bar, up to the skull in the upper half.
SHEPP_LOGAN_PROFILES = ("shepp_logan", "shepp_logan_skull")
VELOCITY_VMIN = 1400.0
VELOCITY_VCENTER = 1600.0
VELOCITY_VMAX = 3000.0


def profile_velocity_norm(
    profile: Optional[str],
    vmin: Optional[float] = None,
    vcenter: Optional[float] = None,
    vmax: Optional[float] = None,
    data=None,
):
    """Colour normalisation for a velocity field whose truth is ``profile``.

    Two-slope (:func:`velocity_norm`) for a Shepp-Logan truth or when
    ``vcenter`` is given, with unset bounds taken from
    :data:`VELOCITY_VMIN` / :data:`VELOCITY_VCENTER` / :data:`VELOCITY_VMAX`.
    Linear otherwise, with unset bounds from ``data`` (e.g. every field sharing
    the scale), or autoscaled to the plotted field without it.
    """
    from matplotlib.colors import Normalize

    if profile in SHEPP_LOGAN_PROFILES or vcenter is not None:
        return velocity_norm(
            vmin=VELOCITY_VMIN if vmin is None else vmin,
            vcenter=VELOCITY_VCENTER if vcenter is None else vcenter,
            vmax=VELOCITY_VMAX if vmax is None else vmax,
        )
    norm = Normalize(vmin, vmax)
    if data is not None:
        norm.autoscale_None(data)
    return norm


class VelocityModel:
    """2D velocity field c(x, y) attached to a Grid (full extended grid).

    ``**profile_kwargs``: ``radius`` (m) and ``center`` (``(x, y)`` m,
    default the interior centre) for ``overdensity``; ``c_water``,
    ``c_skull``, ``threshold``, and ``soft_intercept`` / ``soft_slope``
    (``shepp_logan``) or ``interior_value`` (``shepp_logan_skull``) for the
    Shepp-Logan profiles. Defaults follow ADR 0001.
    """

    def __init__(
        self,
        grid: Grid,
        *,
        profile: str = "shepp_logan",
        base: float = 1500.0,
        contrast: float = 0.4,
        pml_c: Optional[float] = None,
        scale: float = 0.85,
        skull_alpha: float = 1.0,
        skull_sigma: float = 0.0,
        **profile_kwargs,
    ):
        """Build ``profile``'s velocity field on the full grid.

        ``base`` and ``contrast`` are m/s. ``pml_c`` ``None`` is the interior
        background (skull profiles), else the field minimum. ``scale`` is the
        fraction of the interior the Shepp-Logan phantom occupies.
        ``shepp_logan_skull`` only: ``skull_alpha`` in [0, 1] scales the skull
        contrast, ``c = c_water + alpha * G_σ(c_perfect - c_water)``;
        ``skull_sigma`` >= 0 is ``σ`` in **grid cells** (``0`` a sharp rim).
        """
        self.grid = grid
        self.profile = profile
        self.base = base
        self.contrast = contrast
        self.scale = scale
        self.skull_alpha = skull_alpha
        self.skull_sigma = skull_sigma
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
        """A maskless ``"custom"`` model of a full-grid ``c``; ``pml_c``
        defaults to its minimum."""
        vm = cls.__new__(cls)
        vm.grid = grid
        vm.profile = "custom"
        vm.base = float("nan")
        vm.contrast = float("nan")
        vm.profile_kwargs = {}
        vm.c = c.to(dtype=grid.dtype, device=grid.device).reshape(grid.shape)
        vm.pml_c = float(pml_c) if pml_c is not None else float(vm.c.min())
        vm._head_mask = None  # noqa: SLF001 - alternate constructor
        vm._interior_mask = None  # noqa: SLF001 - alternate constructor
        vm._rim_mask = None  # noqa: SLF001 - alternate constructor
        vm._interior_bg = None  # noqa: SLF001 - alternate constructor
        return vm

    @classmethod
    def from_interior(
        cls, grid: Grid, c_interior: torch.Tensor, pml_c: float
    ) -> "VelocityModel":
        """A model from an interior ``c`` ``(interior_nx, interior_ny)``, its PML
        filled with ``pml_c`` by :meth:`build_full_c`: a saved band's model."""
        vm = cls.from_field(grid, torch.full(grid.shape, float(pml_c)), pml_c=pml_c)
        vm.c = vm.build_full_c(c_interior.to(dtype=grid.dtype, device=grid.device))
        return vm

    def build_full_c(self, c_interior: torch.Tensor) -> torch.Tensor:
        """The full-grid ``c`` of an interior ``c`` ``(interior_nx,
        interior_ny)``, its PML at the fixed ``pml_c``."""
        p = self.grid.pml_width
        if p == 0:
            return c_interior
        return F.pad(c_interior, (p, p, p, p), mode="constant", value=self.pml_c)

    def _shepp_logan_interior_phantom(self, scale: float, threshold: float):
        """``(phantom, head, inner, rim)`` on the interior: a float32 phantom in
        [0, 1] and interior-shaped bool masks. Built on the interior, as
        ``stride/stride_forward_shepp.py::shepp_logan_sos`` does, so the morphology
        sees the same array borders."""
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
        g = self.grid
        full = torch.full(g.shape, float(fill), dtype=g.dtype, device=g.device)
        full[g.interior_slice] = torch.from_numpy(np.ascontiguousarray(interior_np)).to(
            dtype=g.dtype, device=g.device
        )
        return full

    def _embed_mask(self, interior_mask: np.ndarray) -> torch.Tensor:

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
            cx, cy = self.profile_kwargs.get("center", g.interior_center)
            mask = (g.X - cx) ** 2 + (g.Y - cy) ** 2 <= r**2
            return torch.where(
                mask, torch.full_like(base_field, self.base + self.contrast), base_field
            )

        if self.profile == "shepp_logan":
            scale = self.scale
            threshold = self.profile_kwargs.get("threshold", SHEPP_THRESHOLD)
            c_water = self.profile_kwargs.get("c_water", SOS_WATER)
            c_skull = self.profile_kwargs.get("c_skull", SOS_SKULL)
            soft_intercept = self.profile_kwargs.get("soft_intercept", SOFT_INTERCEPT)
            soft_slope = self.profile_kwargs.get("soft_slope", SOFT_SLOPE)

            phantom, head, inner, rim = self._shepp_logan_interior_phantom(
                scale, threshold
            )
            # As stride/stride_forward_shepp.py::shepp_logan_sos.
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
            scale = self.scale
            threshold = self.profile_kwargs.get("threshold", SHEPP_THRESHOLD)
            c_water = self.profile_kwargs.get("c_water", SOS_WATER)
            c_skull = self.profile_kwargs.get("c_skull", SOS_SKULL)
            skull_alpha = float(self.skull_alpha)
            if skull_alpha < 0.0 or skull_alpha > 1.0:
                raise ValueError(f"skull_alpha must be in [0, 1]; got {skull_alpha}")
            skull_sigma = float(self.skull_sigma)
            if skull_sigma < 0.0:
                raise ValueError(f"skull_sigma must be >= 0; got {skull_sigma}")

            phantom, head, inner, rim = self._shepp_logan_interior_phantom(
                scale, threshold
            )
            # Perfect-skull geometry; no soft-tissue features.
            interior_value = self.profile_kwargs.get("interior_value", c_water)
            model = np.full(phantom.shape, c_water, dtype=np.float32)
            if interior_value != c_water:
                model[inner] = interior_value
            model[rim] = c_skull

            cw = np.float32(c_water)
            contrast = model - cw
            if skull_sigma > 0.0:
                contrast = ndi.gaussian_filter(contrast, sigma=skull_sigma)
            model = cw + np.float32(skull_alpha) * contrast

            self._interior_bg = float(c_water)
            self._head_mask = self._embed_mask(head)
            self._interior_mask = self._embed_mask(inner)
            self._rim_mask = self._embed_mask(rim)
            return self._embed_interior(model, c_water)

        raise ValueError(f"Unknown velocity profile: {self.profile!r}")

    @property
    def head_mask(self) -> Optional[torch.Tensor]:
        """The head region on the interior, or ``None`` without a skull."""
        if getattr(self, "_head_mask", None) is None:
            return None
        return self._head_mask[self.grid.interior_slice]

    def skull_region_masks(self) -> tuple:
        """``(head_mask, interior_mask, rim_mask)`` on the full grid, for a
        Shepp-Logan profile."""
        if getattr(self, "_head_mask", None) is None:
            raise ValueError(
                "skull_region_masks() requires a skull-bearing profile "
                f"(shepp_logan / shepp_logan_skull), got {self.profile!r}"
            )
        return self._head_mask, self._interior_mask, self._rim_mask

    @property
    def c_max(self) -> float:
        """Maximum velocity on the full grid, m/s."""
        return float(self.c.max())

    @property
    def c_min(self) -> float:
        """Minimum velocity on the full grid, m/s."""
        return float(self.c.min())

    def show(
        self,
        ax=None,
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_pml: bool = True,
        norm=None,
        vcenter: Optional[float] = None,
    ):
        """Plot the velocity field, its colour scale
        :func:`profile_velocity_norm` (bounds in m/s) unless ``norm`` is
        given."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        if norm is None:
            norm = profile_velocity_norm(
                self.profile, vmin=vmin, vcenter=vcenter, vmax=vmax
            )
        im = ax.imshow(
            self.c.cpu().numpy().T,
            origin="lower",
            extent=(xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult),
            cmap="viridis",
            norm=norm,
        )
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
