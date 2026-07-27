from typing import Optional, Tuple

import math
import torch
import numpy as np

import matplotlib.pyplot as plt

from odil_wave.grid import Grid, FrequencySelection
from odil_wave.models import VelocityModel, velocity_norm
from odil_wave.plot_utils import length_scale, time_scale
from odil_wave.source import SourceSignal


def _normalize_traces(traces: np.ndarray, mode: str | None) -> np.ndarray:
    """Rescale a (NT, n_receivers) trace gather for plotting.

    mode:
      - None / "none":    pass-through
      - "per_receiver":   divide each column (one receiver's trace) by its
                          own max-abs — the standard seismic balancing.
      - "global":         divide the whole gather by a single max-abs.
    """
    if mode is None or mode == "none":
        return traces
    if mode == "per_receiver":
        scale = np.max(np.abs(traces), axis=0, keepdims=True)
        scale = np.where(scale > 0, scale, 1.0)
        return traces / scale
    if mode == "global":
        scale = float(np.max(np.abs(traces)))
        return traces / scale if scale > 0 else traces
    raise ValueError(
        f"Invalid normalize mode {mode!r}; expected one of None, 'per_receiver', 'global'."
    )


class AcquisitionGeometry:
    """Elliptical array of transducers around the interior region.

    `source_spatial` selects the spatial injection profile:
      - ``"gaussian"`` (default): Gaussian blob of width `sigma_s`
      - ``"point"``: unit Kronecker delta at the source grid node
    """

    def __init__(
        self,
        grid: Grid,
        source: SourceSignal,
        frequency_selection: FrequencySelection,
        n_receivers: int = 16,
        n_sources: Optional[int] = None,
        sigma_s: Optional[float] = None,
        a_frac: float = 0.55,
        b_frac: float = 0.70,
        ring_center: Tuple[float, float] = (0.0, 0.0),
        source_spatial: str = "gaussian",
    ):
        self.grid = grid
        self.source = source
        self.frequency_selection = frequency_selection
        # ensure same device and dtype as grid
        self.device = self.grid.device
        self.dtype = self.grid.dtype

        self.n_receivers = n_receivers
        self.n_sources = n_receivers if n_sources is None else n_sources
        self.sigma_s = (
            self._default_sigma_s(grid, source) if sigma_s is None else sigma_s
        )
        self.a_frac = a_frac
        self.b_frac = b_frac
        self.ring_center = ring_center
        self.source_spatial = source_spatial

        self.recv_ij = self._place_ellipse(self.n_receivers)
        step = max(1, self.n_receivers // self.n_sources)
        self.src_ij = self.recv_ij[::step][: self.n_sources]

    @staticmethod
    def _default_sigma_s(grid: Grid, source: SourceSignal) -> float:
        """Gaussian source width."""
        h = max(grid.dx, grid.dy)
        c_ref = grid.c_min if grid.c_min is not None else grid.c_max
        f_max = 2.5 * source.f0
        lambda_min = c_ref / f_max
        return max(1.5 * h, lambda_min / (2.0 * math.pi))

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

    def _spatial_profile(self, src_idx: int) -> torch.Tensor:
        if self.source_spatial == "point":
            i = int(self.src_ij[src_idx, 0])
            j = int(self.src_ij[src_idx, 1])
            spatial = torch.zeros(self.grid.shape, dtype=self.dtype, device=self.device)
            spatial[i, j] = 1.0
            return spatial
        x_src, y_src = self.src_position(src_idx)
        return torch.exp(
            -(
                ((self.grid.X - x_src) ** 2 + (self.grid.Y - y_src) ** 2)
                / self.sigma_s**2
            )
        )

    def source_field_time(self, src_idx: int) -> torch.Tensor:
        """Real ``(nt, nx, ny)`` source for leapfrog / FFT reference."""
        spatial = self._spatial_profile(src_idx)
        temporal = self.source.waveform(self.grid.t)
        return temporal.view(-1, 1, 1) * spatial.view(1, *self.grid.shape)

    def source_field(self, src_idx: int) -> torch.Tensor:
        """Complex ``(n_frequencies, nx, ny)`` source on FrequencySelection bins."""
        spatial = self._spatial_profile(src_idx)
        spectrum = self.source.spectrum(self.frequency_selection)
        cdtype = self.frequency_selection.cdtype
        return spectrum.to(dtype=cdtype).view(-1, 1, 1) * spatial.to(dtype=cdtype).view(
            1, *self.grid.shape
        )

    def extract_observations(self, U: torch.Tensor) -> torch.Tensor:
        """Pull ``(..., n_receivers)`` from a field with trailing ``(nx, ny)``.

        Time-domain ``U``: ``(nt, nx, ny)`` → ``(nt, n_rcv)``.
        Frequency-domain ``U``: ``(nf, nx, ny)`` → ``(nf, n_rcv)``.
        """
        return U[..., self.recv_ij[:, 0], self.recv_ij[:, 1]]

    def plot_traces(
        self,
        wavefield,
        ax=None,
        normalize: Optional[str] = "per_receiver",
        cmap: str = "RdBu_r",
        title: Optional[str] = None,
    ):
        """Plot recorded traces: receiver id (y) vs time (x), amplitude as colour.

        Parameters
        ----------
        wavefield :
            A `Wavefield`, a `(NT, NX, NY)` amplitude tensor/array, or a list
            of either (one entry per shot). A list produces a subplot grid.
        normalize : {"per_receiver", "global", None}, default "per_receiver"
            Per-receiver max-abs balancing (default — emphasises weak
            channels), single global max-abs, or no rescaling.
        """
        # Coerce input to a list and remember whether it was originally one.
        is_list = isinstance(wavefield, (list, tuple))
        items = list(wavefield) if is_list else [wavefield]

        traces_per_shot = []
        for item in items:
            amp = (
                item.amplitude if hasattr(item, "amplitude") else torch.as_tensor(item)
            )
            tr = self.extract_observations(amp).detach().cpu().numpy()
            traces_per_shot.append(_normalize_traces(tr, normalize))

        t = self.grid.t.cpu().numpy()
        t_mult, t_unit = time_scale(float(t[-1]) if t.size else 1.0)
        extent = (
            float(t[0]) * t_mult,
            float(t[-1]) * t_mult,
            -0.5,
            self.n_receivers - 0.5,
        )
        norm_tag = "" if normalize is None else f" ({normalize})"

        if not is_list:
            traces = traces_per_shot[0]
            vmax = float(np.max(np.abs(traces))) or 1.0
            if ax is None:
                _, ax = plt.subplots(figsize=(7.5, 4.5))
            im = ax.imshow(
                traces.T,  # (n_rcv, NT) so time is along x
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_xlabel(f"time [{t_unit}]")
            ax.set_ylabel("receiver id")
            ax.set_title(title or f"trace gather{norm_tag}")
            plt.colorbar(im, ax=ax, shrink=0.85, label="amplitude")
            return ax

        n_shots = len(items)
        ncols = min(n_shots, 2)
        nrows = math.ceil(n_shots / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.5 * ncols, 3.5 * nrows), squeeze=False
        )
        global_vmax = max(float(np.max(np.abs(tr))) for tr in traces_per_shot) or 1.0
        for k, (ax_k, traces) in enumerate(zip(axes.flat, traces_per_shot)):
            vmax = (
                global_vmax
                if normalize == "global"
                else (float(np.max(np.abs(traces))) or 1.0)
            )
            im = ax_k.imshow(
                traces.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            ax_k.set_xlabel(f"time [{t_unit}]")
            ax_k.set_ylabel("receiver id")
            ax_k.set_title(f"shot {k}")
            plt.colorbar(im, ax=ax_k, shrink=0.85, label="amplitude")
        for ax_k in axes.flat[n_shots:]:
            ax_k.set_axis_off()
        fig.suptitle(title or f"trace gathers{norm_tag}")
        fig.tight_layout()
        return axes

    def plot_source_field(self, src_idx: int = 0, t_idx: Optional[int] = None, ax=None):
        """Plot a spatial snapshot of the source field s(x, y, t_idx).

        Defaults to the peak time of the source's temporal waveform.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        src = self.source_field_time(src_idx).cpu().numpy()
        if t_idx is None:
            t_idx = int(
                np.argmax(np.abs(self.source.waveform(self.grid.t).cpu().numpy()))
            )
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        t_val = float(self.grid.t[t_idx].item())
        t_mult, t_unit = time_scale(float(self.grid.t[-1].item()))

        vmax = float(np.max(np.abs(src))) * 1.05 + 1e-12
        im = ax.imshow(
            src[t_idx].T,
            origin="lower",
            aspect="equal",
            extent=(xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"y [{x_unit}]")
        ax.set_title(
            f"source field (shot {src_idx}, t = {t_val * t_mult:.2f} {t_unit})",
            pad=8,
        )
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04, label="amplitude [a.u.]")
        return ax

    def show(
        self,
        velocity_model: VelocityModel,
        ax=None,
        norm=None,
        vmin: float = 1400.0,
        vcenter: float = 1600.0,
        vmax: float = 3000.0,
    ):
        """Overlay the source/receiver ring on the velocity model.

        The velocity colourbar uses a two-slope normalisation so that
        ``[vmin, vcenter]`` and ``[vcenter, vmax]`` each fill half the bar
        (defaults 1400 / 1600 / 3000 m/s). Pass an explicit ``norm`` to
        override.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5))
        if norm is None and vmax > vmin:
            norm = velocity_norm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        velocity_model.show(
            ax=ax,
            title=f"acquisition on {velocity_model.profile}",
            show_pml=True,
            norm=norm,
        )

        # Match the axis units chosen by VelocityModel.show.
        (_, xmax), (_, ymax) = self.grid.extent
        x_mult, _ = length_scale(max(abs(xmax), abs(ymax)))
        rx = self.grid.x[self.recv_ij[:, 0]].cpu().numpy() * x_mult
        ry = self.grid.y[self.recv_ij[:, 1]].cpu().numpy() * x_mult
        sx = self.grid.x[self.src_ij[:, 0]].cpu().numpy() * x_mult
        sy = self.grid.y[self.src_ij[:, 1]].cpu().numpy() * x_mult
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
