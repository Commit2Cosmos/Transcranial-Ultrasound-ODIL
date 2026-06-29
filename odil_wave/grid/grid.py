import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

from odil_wave.plot_utils import length_scale, time_scale


@dataclass
class Grid:
    """2D space + time discretization with a PML sponge layer wrapping the interior.

    The user specifies the *interior* (non-PML) shape and physical extent.
    The PML adds `pml_width` extra cells per side, extending the total grid
    (and total physical box) by `pml_width * dx` per side. The wavefield
    optimization variable lives on the full extended grid `(NT, NX, NY)`.

    Non-dimensionalisation
    ----------------------
    Characteristic scales `L0` (length) and `c0` (velocity) define a time
    scale ``t0 = L0 / c0``. Dimensionless coordinates are
    ``x' = x / L0``, ``t' = t / t0``, ``c' = c / c0``. The PDE
    ``u_tt - c^2 (u_xx + u_yy) = f`` becomes (multiplying by ``t0**2``)::

        u_t't' - c'**2 (u_x'x' + u_y'y') = t0**2 * f

    Physical attributes (`dx`, `dy`, `dt`, `sigma_x`, `sigma_y`) are kept
    for plotting and indexing. The wave-equation operators consume the
    matching non-dimensional companions (`dx_nd`, `dy_nd`, `dt_nd`,
    `sigma_x_nd`, `sigma_y_nd`) so the discrete residual is dimensionless.
    Defaults: ``c0 = c_max`` and ``L0 = max(interior extents)``.
    """

    # TODO: enforce it is passed; no default
    interior_shape: Tuple[int, int] = (100, 100)
    interior_extent: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-1.0, 1.0),
        (-1.0, 1.0),
    )
    # TODO: enforce it is passed; no default
    c_min: Optional[float] = None
    c_max: float = 1.5
    pml_width: int = 10  # extra cells per side wrapping the interior
    pml_power: int = 3  # sigma(d) = sigma_max * (d / L_pml)^pml_power
    pml_R0: float = 1e-6  # target theoretical reflection coefficient
    # TODO: enforce it is passed; no default
    t_max: float = 1.0  # specify based on forward wavefield observations
    init_nt: Optional[int] = None
    # characteristic scales for non-dimensionalisation; None -> sensible defaults
    L0: Optional[float] = None
    c0: Optional[float] = None
    # TODO: make settable
    device: torch.device = field(init=False)
    dtype: torch.dtype = torch.float32

    # derived
    interior_nx: int = field(init=False)
    interior_ny: int = field(init=False)
    nx: int = field(init=False)
    ny: int = field(init=False)
    nt: int = field(init=False)
    extent: Tuple[Tuple[float, float], Tuple[float, float]] = field(init=False)
    dx: float = field(init=False)
    dy: float = field(init=False)
    dt: float = field(init=False)
    t0: float = field(init=False)  # = L0 / c0
    dx_nd: float = field(init=False)
    dy_nd: float = field(init=False)
    dt_nd: float = field(init=False)
    x: torch.Tensor = field(init=False, repr=False)
    y: torch.Tensor = field(init=False, repr=False)
    t: torch.Tensor = field(init=False, repr=False)
    X: torch.Tensor = field(init=False, repr=False)
    Y: torch.Tensor = field(init=False, repr=False)
    sigma_x: torch.Tensor = field(init=False, repr=False)
    sigma_y: torch.Tensor = field(init=False, repr=False)
    sigma_x_nd: torch.Tensor = field(init=False, repr=False)
    sigma_y_nd: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        self.interior_nx, self.interior_ny = self.interior_shape
        (ix_min, ix_max), (iy_min, iy_max) = self.interior_extent
        # TODO: add mps support∑
        # self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.dx = (ix_max - ix_min) / (self.interior_nx - 1)
        self.dy = (iy_max - iy_min) / (self.interior_ny - 1)

        p = self.pml_width
        self.nx = self.interior_nx + 2 * p
        self.ny = self.interior_ny + 2 * p
        x_min = ix_min - p * self.dx
        x_max = ix_max + p * self.dx
        y_min = iy_min - p * self.dy
        y_max = iy_max + p * self.dy
        self.extent = ((x_min, x_max), (y_min, y_max))

        if self.init_nt is None:
            dt_cfl = 1.0 / (self.c_max * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))
            self.nt = int(math.ceil(self.t_max / dt_cfl)) + 1
        else:
            self.nt = self.init_nt
        self.dt = self.t_max / (self.nt - 1)

        self.x = torch.linspace(
            x_min, x_max, self.nx, dtype=self.dtype, device=self.device
        )
        self.y = torch.linspace(
            y_min, y_max, self.ny, dtype=self.dtype, device=self.device
        )
        self.t = torch.linspace(
            0.0, self.t_max, self.nt, dtype=self.dtype, device=self.device
        )
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="ij")

        self.sigma_x, self.sigma_y = self._build_pml_profiles()

        # Characteristic scales for non-dimensionalisation.
        # Default L0 to the larger interior side, c0 to c_max.
        if self.L0 is None:
            self.L0 = float(max(ix_max - ix_min, iy_max - iy_min))
        else:
            self.L0 = float(self.L0)
        if self.c0 is None:
            self.c0 = float(self.c_max)
        else:
            self.c0 = float(self.c0)
        if self.L0 <= 0 or self.c0 <= 0:
            raise ValueError("L0 and c0 must be positive.")
        self.t0 = self.L0 / self.c0

        self.dx_nd = self.dx / self.L0
        self.dy_nd = self.dy / self.L0
        self.dt_nd = self.dt / self.t0
        # sigma has units of 1/s; sigma * t0 is dimensionless.
        self.sigma_x_nd = self.sigma_x * self.t0
        self.sigma_y_nd = self.sigma_y * self.t0

    @property
    def shape(self) -> Tuple[int, int]:
        """Total grid shape (interior + PML)."""
        return (self.nx, self.ny)

    @property
    def natural_source_amplitude(self) -> float:
        """Physical source amplitude whose non-dimensional form has unit peak."""
        return 1.0 / self.t0**2

    @property
    def interior_slice(self) -> Tuple[slice, slice]:
        """Slice into a full-grid tensor that picks out the interior."""
        p = self.pml_width
        return (slice(p, p + self.interior_nx), slice(p, p + self.interior_ny))

    def _sigma_max(self, L_pml_phys: float) -> float:
        """sigma_max from a target theoretical reflection coefficient."""
        if L_pml_phys <= 0:
            raise ValueError("L_pml_phys must be positive.")
        if self.c_max <= 0:
            raise ValueError("c_max must be positive.")
        if not (0.0 < self.pml_R0 < 1.0):
            raise ValueError("pml_R0 must lie strictly between 0 and 1.")

        return -((self.pml_power + 1) * self.c_max * math.log(self.pml_R0)) / (
            2.0 * L_pml_phys
        )

    def _build_pml_profiles(self):
        """sigma_x(i, j), sigma_y(i, j) on the full grid; zero in the interior."""
        p = self.pml_width
        if p == 0:
            zeros = torch.zeros(self.nx, self.ny, dtype=self.dtype, device=self.device)
            return zeros, zeros
        L_pml_x = p * self.dx
        L_pml_y = p * self.dy
        sigma_max_x = self._sigma_max(L_pml_x)
        sigma_max_y = self._sigma_max(L_pml_y)

        i = torch.arange(self.nx, dtype=self.dtype, device=self.device)
        j = torch.arange(self.ny, dtype=self.dtype, device=self.device)

        d_x = (
            torch.clamp(p - i, min=0.0) + torch.clamp(i - (self.nx - 1 - p), min=0.0)
        ) * self.dx
        d_y = (
            torch.clamp(p - j, min=0.0) + torch.clamp(j - (self.ny - 1 - p), min=0.0)
        ) * self.dy

        sigma_x_1d = sigma_max_x * (d_x / L_pml_x) ** self.pml_power
        sigma_y_1d = sigma_max_y * (d_y / L_pml_y) ** self.pml_power

        sigma_x = sigma_x_1d.view(-1, 1).expand(self.nx, self.ny).contiguous()
        sigma_y = sigma_y_1d.view(1, -1).expand(self.nx, self.ny).contiguous()
        return sigma_x, sigma_y

    # Alternative constructor
    # TODO: don't allow to set interior_shape
    @classmethod
    def from_frequency(
        cls,
        f_max: float,
        c_min: float,
        interior_extent: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (-1.0, 1.0),
            (-1.0, 1.0),
        ),
        n_ppw: int = 5,
        **kwargs,
    ) -> "Grid":
        """Construct a Grid with dx chosen from maximum source frequency.

        Uses the points-per-wavelength (PPW) criterion:
            dx = c_min / (f_max * n_ppw)

        """
        if "init_nt" in kwargs:
            raise ValueError(
                "init_nt cannot be passed to Grid.from_frequency: the number "
                "of timesteps must be determined by the CFL condition once dx "
                "is set from frequency. Pass t_max instead."
            )
        dx = c_min / (f_max * n_ppw)
        (ix_min, ix_max), (iy_min, iy_max) = interior_extent
        nx = round((ix_max - ix_min) / dx) + 1
        ny = round((iy_max - iy_min) / dx) + 1
        return cls(
            interior_shape=(nx, ny),
            interior_extent=interior_extent,
            c_min=c_min,
            **kwargs,
        )

    def cfl(self, c_max: float) -> float:
        return c_max * self.dt * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2)

    def plot_absorption_profile(self, ax=None):
        """Plot the 2D PML absorption (sigma_x + sigma_y) over the full grid."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        (xmin, xmax), (ymin, ymax) = self.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        im = ax.imshow(
            (self.sigma_x + self.sigma_y).cpu().numpy().T,
            origin="lower",
            extent=(xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult),
            cmap="magma",
            aspect="equal",
        )
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"y [{x_unit}]")
        ax.set_title(r"PML absorption $\sigma_x + \sigma_y$")
        (ix0, ix1), (iy0, iy1) = self.interior_extent
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
        plt.colorbar(im, ax=ax, shrink=0.85, label=r"$\sigma$ [1/s]")
        return ax

    @property
    def summary(self) -> str:
        (ix0, ix1), (iy0, iy1) = self.interior_extent
        (xmin, xmax), (ymin, ymax) = self.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        t_mult, t_unit = time_scale(self.t_max)
        L_mult, L_unit = length_scale(self.L0)
        t0_mult, t0_unit = time_scale(self.t0)
        return (
            f"Grid interior {self.interior_nx}x{self.interior_ny} -> "
            f"total {self.nx}x{self.ny} (PML p={self.pml_width}),"
            f"\nnt={self.nt}, dx={self.dx * x_mult:.3f} {x_unit}, "
            f"dy={self.dy * x_mult:.3f} {x_unit},"
            f"\ndt={self.dt * t_mult:.3f} {t_unit}, "
            f"cfl@c_max={self.cfl(self.c_max):.3f},"
            f"\ninterior x in [{ix0 * x_mult:.2f}, {ix1 * x_mult:.2f}] {x_unit},"
            f"\ny in [{iy0 * x_mult:.2f}, {iy1 * x_mult:.2f}] {x_unit},"
            f"\ntotal x in [{xmin * x_mult:.2f}, {xmax * x_mult:.2f}] {x_unit},"
            f"\ntotal y in [{ymin * x_mult:.2f}, {ymax * x_mult:.2f}] {x_unit},"
            f"\nL0={self.L0 * L_mult:.3f} {L_unit}, c0={self.c0:.1f} m/s, "
            f"t0={self.t0 * t0_mult:.3f} {t0_unit},"
            f"\ndx_nd={self.dx_nd:.4f}, dt_nd={self.dt_nd:.4f}, "
            f"t_max_nd={self.t_max / self.t0:.3f}"
        )
