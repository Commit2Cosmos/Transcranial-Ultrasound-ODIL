from typing import Optional, Tuple

import math
import torch
import numpy as np

import matplotlib.pyplot as plt

from odil_wave.grid import Grid
from odil_wave.models import VelocityModel
from odil_wave.source import SourceSignal


class AcquisitionGeometry:
    """Elliptical array of transducers around the interior region.

    `n_receivers` transducer positions record every shot. A subset
    (`n_sources`, evenly spaced) act as shot sources. The temporal
    source waveform is supplied by an injected `SourceSignal`.
    """

    def __init__(
        self,
        grid: Grid,
        source: SourceSignal,
        n_receivers: int = 16,
        n_sources: Optional[int] = None,
        sigma_s: Optional[float] = None,
        a_frac: float = 0.55,
        b_frac: float = 0.70,
        ring_center: Tuple[float, float] = (0.0, 0.0),
    ):
        self.grid = grid
        self.source = source
        # ensure same device and dtype as grid
        self.device = self.grid.device
        self.dtype = self.grid.dtype

        self.n_receivers = n_receivers
        self.n_sources = n_receivers if n_sources is None else n_sources
        self.sigma_s = 1.5 * max(grid.dx, grid.dy) if sigma_s is None else sigma_s
        self.a_frac = a_frac
        self.b_frac = b_frac
        self.ring_center = ring_center

        self.recv_ij = self._place_ellipse(self.n_receivers)
        step = max(1, self.n_receivers // self.n_sources)
        self.src_ij = self.recv_ij[::step][: self.n_sources]

    def _place_ellipse(self, n: int) -> torch.Tensor:
        """Return (n, 2) integer full-grid indices on an ellipse inside the interior."""
        (ix_min, ix_max), (iy_min, iy_max) = self.grid.interior_extent
        cx, cy = self.ring_center
        a = self.a_frac * (ix_max - ix_min) / 2.0
        b = self.b_frac * (iy_max - iy_min) / 2.0

        k = torch.arange(n, dtype=self.dtype, device=self.device)
        theta = 2.0 * math.pi * k / n
        x_k = cx + a * torch.cos(theta)
        y_k = cy + b * torch.sin(theta)

        (xmin, _), (ymin, _) = self.grid.extent
        i = torch.round((x_k - xmin) / self.grid.dx).long().clamp(0, self.grid.nx - 1)
        j = torch.round((y_k - ymin) / self.grid.dy).long().clamp(0, self.grid.ny - 1)
        return torch.stack([i, j], dim=-1)

    def src_position(self, src_idx: int) -> Tuple[float, float]:
        i, j = int(self.src_ij[src_idx, 0]), int(self.src_ij[src_idx, 1])
        return float(self.grid.x[i]), float(self.grid.y[j])

    def recv_position(self, rcv_idx: int) -> Tuple[float, float]:
        i, j = int(self.recv_ij[rcv_idx, 0]), int(self.recv_ij[rcv_idx, 1])
        return float(self.grid.x[i]), float(self.grid.y[j])

    def source_field(self, src_idx: int) -> torch.Tensor:
        """(NT, NX, NY) Gaussian-in-space, SourceSignal-in-time source field."""
        x_src, y_src = self.src_position(src_idx)
        spatial = torch.exp(
            -(
                ((self.grid.X - x_src) ** 2 + (self.grid.Y - y_src) ** 2)
                / self.sigma_s**2
            )
        )
        temporal = self.source.waveform(self.grid.t)
        return temporal.view(-1, 1, 1) * spatial.view(1, *self.grid.shape)

    def extract_observations(self, U: torch.Tensor) -> torch.Tensor:
        """Pull (NT, n_receivers) sensor data from a (NT, NX, NY) wavefield."""
        return U[:, self.recv_ij[:, 0], self.recv_ij[:, 1]]

    def plot_source_field(self, src_idx: int = 0, t_idx: Optional[int] = None, ax=None):
        """Plot a spatial snapshot of the source field s(x, y, t_idx).

        Defaults to the peak time of the source's temporal waveform.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        src = self.source_field(src_idx).cpu().numpy()
        if t_idx is None:
            t_idx = int(
                np.argmax(np.abs(self.source.waveform(self.grid.t).cpu().numpy()))
            )
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        vmax = float(np.max(np.abs(src))) * 1.05 + 1e-12
        im = ax.imshow(
            src[t_idx].T,
            origin="lower",
            aspect="equal",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"source field s(x, y, t={self.grid.t[t_idx].item():.3f} s)")
        plt.colorbar(im, ax=ax, shrink=0.85, label="amplitude [a.u.]")
        return ax

    def show(self, velocity_model: VelocityModel, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5))
        velocity_model.show(
            ax=ax, title=f"acquisition on {velocity_model.profile}", show_pml=True
        )

        rx = self.grid.x[self.recv_ij[:, 0]].cpu().numpy()
        ry = self.grid.y[self.recv_ij[:, 1]].cpu().numpy()
        sx = self.grid.x[self.src_ij[:, 0]].cpu().numpy()
        sy = self.grid.y[self.src_ij[:, 1]].cpu().numpy()
        ax.scatter(
            rx,
            ry,
            marker="v",
            c="lime",
            edgecolor="black",
            s=70,
            label=f"{self.n_receivers} receivers",
            zorder=5,
        )
        ax.scatter(
            sx,
            sy,
            marker="*",
            c="red",
            edgecolor="black",
            s=180,
            label=f"{self.n_sources} sources",
            zorder=6,
        )
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        return ax
