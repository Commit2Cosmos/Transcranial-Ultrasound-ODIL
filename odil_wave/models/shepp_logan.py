"""The analytic Shepp-Logan phantom: a continuous head in metres that any grid
samples by cell-averaging ``1/c²``, with its head, brain and skull regions
(ADR 0004).

The velocity profiles built on it (``shepp_logan``, ``shepp_logan_skull``) and
the PML fill live in :mod:`odil_wave.models.velocity_models`. Entry point:
:class:`SheppLogan`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

SOS_WATER = 1500.0  # background [m/s]
SOS_SKULL = 3000.0  # skull [m/s]
SOS_BRAIN = 1510.0  # brain tissue outside the features [m/s]

# Kak & Slaney Table 3.1 ellipses c-j, in units of the phantom's half-extent:
# (x0, y0, semi-axis a, semi-axis b, rotation of a from x [deg], m/s added to
# the brain speed). Overlaps add, as in the table; the speeds are those of the
# image phantom this replaced (ventricles 1450, features 1540, overlaps 1570).
# Table 3.1's head and brain ellipses (a, b) are replaced by HEAD_SEMI_AXES and
# the skull thickness.
FEATURES = (
    (0.22, 0.0, 0.31, 0.11, 72.0, -60.0),
    (-0.22, 0.0, 0.41, 0.16, 108.0, -60.0),
    (0.0, 0.35, 0.25, 0.21, 90.0, 30.0),
    (0.0, 0.1, 0.046, 0.046, 0.0, 30.0),
    (0.0, -0.1, 0.046, 0.046, 0.0, 30.0),
    (-0.08, -0.605, 0.046, 0.023, 0.0, 30.0),
    (0.0, -0.606, 0.023, 0.023, 0.0, 30.0),
    (0.06, -0.605, 0.046, 0.023, 90.0, 30.0),
)
# Table 3.1's head ellipse (x, y semi-axes), long axis along y.
HEAD_SEMI_AXES = (0.69, 0.92)


@dataclass(frozen=True)
class Raster:
    """A :class:`SheppLogan` sampled on a grid: ``c`` (m/s) and bool region
    masks, each ``(nx, ny)``."""

    c: np.ndarray
    head: np.ndarray
    brain: np.ndarray
    skull: np.ndarray


@dataclass(frozen=True)
class SheppLogan:
    """The Shepp-Logan head centred at ``center`` (m), Table 3.1's unit square
    spanning ``half_extent`` (m) either side, with a single-layer skull
    ``skull_thickness`` (m) thick. ``c_brain`` replaces the brain tissue and its
    features with one speed (m/s); ``None`` keeps them. ``ValueError`` if the
    skull leaves no brain."""

    center: Tuple[float, float]
    half_extent: Tuple[float, float]
    skull_thickness: float
    c_water: float = SOS_WATER
    c_skull: float = SOS_SKULL
    c_brain: Optional[float] = None

    def __post_init__(self):
        if self.skull_thickness <= 0.0:
            raise ValueError(
                f"skull_thickness must be > 0 m; got {self.skull_thickness}"
            )
        if self.skull_thickness >= min(self._head_axes):
            raise ValueError(
                f"skull_thickness={self.skull_thickness} m leaves no brain: the "
                f"head's semi-axes are {[float(a) for a in self._head_axes]} m"
            )

    @property
    def _head_axes(self) -> Tuple[float, float]:
        return (
            HEAD_SEMI_AXES[0] * self.half_extent[0],
            HEAD_SEMI_AXES[1] * self.half_extent[1],
        )

    @property
    def _brain_axes(self) -> Tuple[float, float]:
        ax, ay = self._head_axes
        return (ax - self.skull_thickness, ay - self.skull_thickness)

    def _inside(self, x, y, axes) -> np.ndarray:
        dx = (x - self.center[0]) / axes[0]
        dy = (y - self.center[1]) / axes[1]
        return dx**2 + dy**2 <= 1.0

    def speed(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The speed (m/s) at points ``(x, y)`` (m), any matching shapes."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        head = self._inside(x, y, self._head_axes)
        brain = self._inside(x, y, self._brain_axes)
        if self.c_brain is not None:
            tissue = np.full(x.shape, float(self.c_brain))
        else:
            tissue = np.full(x.shape, SOS_BRAIN)
            # Turned 180° from Table 3.1, as the image phantom this replaced lay.
            u = (self.center[0] - x) / self.half_extent[0]
            v = (self.center[1] - y) / self.half_extent[1]
            for x0, y0, a, b, phi, dc in FEATURES:
                cos, sin = np.cos(np.radians(phi)), np.sin(np.radians(phi))
                p = (u - x0) * cos + (v - y0) * sin
                q = -(u - x0) * sin + (v - y0) * cos
                tissue += dc * ((p / a) ** 2 + (q / b) ** 2 <= 1.0)
        return np.where(brain, tissue, np.where(head, self.c_skull, self.c_water))

    def rasterise(self, x: np.ndarray, y: np.ndarray, n_sub: int = 8) -> Raster:
        """Sample on the uniform node coordinates ``x`` ``(nx,)`` and ``y``
        ``(ny,)`` (m), each node the centre of an ``h × h`` cell.

        A cell's speed averages ``1/c²`` over ``n_sub × n_sub`` sub-samples. A
        cell is skull when at least half of it is skull, head when at least half
        is head, and brain only when it lies wholly inside the brain.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        hx, hy = x[1] - x[0], y[1] - y[0]
        sub = (np.arange(n_sub) + 0.5) / n_sub - 0.5
        xs = (x[:, None] + hx * sub[None, :])[:, None, :, None]
        ys = (y[:, None] + hy * sub[None, :])[None, :, None, :]
        c = self.speed(xs, ys)
        m = np.mean(c**-2.0, axis=(2, 3))
        head = self._inside(xs, ys, self._head_axes)
        brain = self._inside(xs, ys, self._brain_axes)
        skull_frac = np.mean(head & ~brain, axis=(2, 3))
        head_frac = np.mean(head, axis=(2, 3))
        # An ellipse is convex, so a cell lies inside it iff its corners do.
        corners = np.ones((x.size, y.size), dtype=bool)
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                corners &= self._inside(
                    x[:, None] + sx * hx, y[None, :] + sy * hy, self._brain_axes
                )
        return Raster(
            c=m**-0.5, head=head_frac >= 0.5, brain=corners, skull=skull_frac >= 0.5
        )
