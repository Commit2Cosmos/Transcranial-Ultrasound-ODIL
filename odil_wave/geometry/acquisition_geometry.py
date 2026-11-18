from typing import Optional, Tuple

import math
import torch
import numpy as np

import matplotlib.pyplot as plt

from odil_wave.grid import Grid, FrequencySelection
from odil_wave.models import VelocityModel
from odil_wave.plot_utils import length_scale, time_scale
from odil_wave.source import SourceSignal


def _normalize_traces(traces: np.ndarray, mode: str | None) -> np.ndarray:
    """Rescale a (NT, n_traces) trace gather for plotting: ``"per_receiver"``
    is per-column max-abs balancing, ``None`` / ``"none"`` none."""
    if mode is None or mode == "none":
        return traces
    if mode == "per_receiver":
        scale = np.max(np.abs(traces), axis=0, keepdims=True)
        scale = np.where(scale > 0, scale, 1.0)
        return traces / scale
    raise ValueError(
        f"Invalid normalize mode {mode!r}; expected one of None, 'per_receiver'."
    )


class AcquisitionGeometry:
    """Elliptical ring of transducers around the interior region.

    Transducers are evenly spaced on an ellipse inscribed in the grid's
    interior region. Shot ``s`` transmits from transducer ``s`` and is recorded
    by the other ``n_transducers - 1`` (`receivers`); a transducer never
    records its own shot. Provides the source injection fields, trace
    extraction and plotting helpers.
    """

    def __init__(
        self,
        grid: Grid,
        source: SourceSignal,
        frequency_selection: FrequencySelection,
        *,
        n_transducers: int,
        sigma_s: float = 2.5e-3,
        a_frac: float = 0.9,
        b_frac: float = 0.9,
        ring_center: Optional[Tuple[float, float]] = None,
        source_spatial: str = "gaussian",
    ):
        """Build the transducer ring on ``grid``; defaults follow ADR 0001.

        ``n_transducers`` >= 2. ``sigma_s`` is the width in metres of the
        Gaussian source density ``exp(-r**2 / sigma_s**2) / (pi * sigma_s**2)``
        (``source_spatial="gaussian"``; ``"point"`` is ``1 / (dx * dy)`` at one
        node). ``a_frac`` / ``b_frac`` are the ellipse semi-axes as fractions of
        the interior half-width / half-height; ``ring_center`` is in metres,
        ``None`` the interior centre.
        """
        self.grid = grid
        self.source = source
        self.frequency_selection = frequency_selection
        self.device = self.grid.device
        self.dtype = self.grid.dtype

        if n_transducers < 2:
            raise ValueError(
                f"n_transducers must be >= 2 so every shot is recorded, "
                f"got {n_transducers}."
            )
        self.n_transducers = int(n_transducers)
        self.sigma_s = float(sigma_s)
        self.a_frac = a_frac
        self.b_frac = b_frac
        self.ring_center = grid.interior_center if ring_center is None else ring_center
        self.source_spatial = source_spatial

        self.transducer_ij = self._place_ellipse(self.n_transducers)
        # receivers[s]: the transducers recording shot s, ascending, without s.
        ids = torch.arange(self.n_transducers, device=self.device)
        others = ids.view(1, -1) != ids.view(-1, 1)
        self.receivers = ids.expand(self.n_transducers, -1)[others].view(
            self.n_transducers, self.n_transducers - 1
        )
        # (n_shots, n_transducers - 1, 2) grid nodes of each shot's receivers.
        self.receiver_ij = self.transducer_ij[self.receivers]

    @property
    def n_shots(self) -> int:
        """One shot per transducer."""
        return self.n_transducers

    def _place_ellipse(self, n: int) -> torch.Tensor:
        """(n, 2) full-grid indices of n points evenly spaced on the ellipse."""
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
        """Physical (x, y) coordinates of source `src_idx`."""
        i, j = int(self.transducer_ij[src_idx, 0]), int(self.transducer_ij[src_idx, 1])
        return float(self.grid.x[i]), float(self.grid.y[j])

    def _spatial_profile(self, src_idx: int) -> torch.Tensor:
        """Spatial injection density (per m²) of source `src_idx`, full-grid.
        Its sum times ``dx * dy`` is 1 (the Gaussian's up to
        ``4 exp(-pi² sigma_s² / h²)``), so the injected amplitude does not
        depend on the grid spacing."""
        if self.source_spatial == "point":
            i = int(self.transducer_ij[src_idx, 0])
            j = int(self.transducer_ij[src_idx, 1])
            spatial = torch.zeros(self.grid.shape, dtype=self.dtype, device=self.device)
            spatial[i, j] = 1.0 / (self.grid.dx * self.grid.dy)
            return spatial
        x_src, y_src = self.src_position(src_idx)
        r2 = (self.grid.X - x_src) ** 2 + (self.grid.Y - y_src) ** 2
        return torch.exp(-r2 / self.sigma_s**2) / (math.pi * self.sigma_s**2)

    def source_field_time(self, src_idx: int) -> torch.Tensor:
        """Time-domain source field of shot `src_idx`, real ``(nt, nx, ny)``."""
        spatial = self._spatial_profile(src_idx)
        temporal = self.source.waveform()
        return temporal.view(-1, 1, 1) * spatial.view(1, *self.grid.shape)

    def source_field(self, src_idx: int) -> torch.Tensor:
        """Frequency-domain source field of shot `src_idx`, complex
        ``(n_frequencies, nx, ny)`` on `self.frequency_selection`'s bins."""
        spatial = self._spatial_profile(src_idx)
        spectrum = self.source.spectrum(self.frequency_selection)
        cdtype = self.frequency_selection.cdtype
        return spectrum.to(dtype=cdtype).view(-1, 1, 1) * spatial.to(dtype=cdtype).view(
            1, *self.grid.shape
        )

    def extract_observations(self, U: torch.Tensor) -> torch.Tensor:
        """Each shot's traces: ``U`` ``(n_shots, ..., nx, ny)`` sampled at
        `receivers[s]`, ``(n_shots, ..., n_transducers - 1)``."""
        ij = self.receiver_ij
        shot = torch.arange(self.n_shots, device=self.device).view(-1, 1)
        # Advanced indices split by `...` put their broadcast dims first.
        return U[shot, ..., ij[..., 0], ij[..., 1]].movedim(1, -1)

    def plot_traces(
        self,
        wavefield,
        ax=None,
        normalize: Optional[str] = "per_receiver",
        cmap: str = "RdBu_r",
        title: Optional[str] = None,
        shot: int = 0,
    ):
        """Plot recorded traces: transducer id (y) vs time (x), amplitude as colour.

        The transmitting transducer records nothing, so its row is left blank.
        ``wavefield`` is a `Wavefield` or `(NT, NX, NY)` array of shot `shot`,
        or a list of either indexed by shot, drawn as a subplot grid (`ax`
        ignored).
        """
        is_list = isinstance(wavefield, (list, tuple))
        items = list(wavefield) if is_list else [wavefield]
        shots = list(range(len(items))) if is_list else [shot]

        traces_per_shot = []
        for s, item in zip(shots, items):
            amp = (
                item.amplitude if hasattr(item, "amplitude") else torch.as_tensor(item)
            )
            ij = self.receiver_ij[s]
            tr = amp[..., ij[:, 0], ij[:, 1]].detach().cpu().numpy()
            gather = np.full((tr.shape[0], self.n_transducers), np.nan)
            gather[:, self.receivers[s].cpu().numpy()] = _normalize_traces(
                tr, normalize
            )
            traces_per_shot.append(gather)

        t = self.grid.t.cpu().numpy()
        t_mult, t_unit = time_scale(float(t[-1]) if t.size else 1.0)
        extent = (
            float(t[0]) * t_mult,
            float(t[-1]) * t_mult,
            -0.5,
            self.n_transducers - 0.5,
        )
        norm_tag = "" if normalize is None else f" ({normalize})"

        if not is_list:
            traces = traces_per_shot[0]
            vmax = float(np.nanmax(np.abs(traces))) or 1.0
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
            ax.set_ylabel("transducer id")
            ax.set_title(title or f"trace gather, shot {shot}{norm_tag}")
            plt.colorbar(im, ax=ax, shrink=0.85, label="amplitude")
            return ax

        n_shots = len(items)
        ncols = min(n_shots, 2)
        nrows = math.ceil(n_shots / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.5 * ncols, 3.5 * nrows), squeeze=False
        )
        for k, (ax_k, traces) in enumerate(zip(axes.flat, traces_per_shot)):
            vmax = float(np.nanmax(np.abs(traces))) or 1.0
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
            ax_k.set_ylabel("transducer id")
            ax_k.set_title(f"shot {k}")
            plt.colorbar(im, ax=ax_k, shrink=0.85, label="amplitude")
        for ax_k in axes.flat[n_shots:]:
            ax_k.set_axis_off()
        fig.suptitle(title or f"trace gathers{norm_tag}")
        fig.tight_layout()
        return axes

    def plot_source_field(self, src_idx: int = 0, t_idx: Optional[int] = None, ax=None):
        """Plot the source field s(x, y, t_idx); ``t_idx`` defaults to the
        waveform's peak."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        src = self.source_field_time(src_idx).cpu().numpy()
        if t_idx is None:
            t_idx = int(np.argmax(np.abs(self.source.waveform().cpu().numpy())))
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
        vmin: Optional[float] = None,
        vcenter: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Overlay the transducer ring on the velocity model, its colour scale
        :func:`~odil_wave.models.profile_velocity_norm` unless ``norm`` is
        given."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5))
        velocity_model.show(
            ax=ax,
            title=f"acquisition on {velocity_model.profile}",
            show_pml=True,
            norm=norm,
            vmin=vmin,
            vcenter=vcenter,
            vmax=vmax,
        )

        (_, xmax), (_, ymax) = self.grid.extent
        x_mult, _ = length_scale(max(abs(xmax), abs(ymax)))
        tx = self.grid.x[self.transducer_ij[:, 0]].cpu().numpy() * x_mult
        ty = self.grid.y[self.transducer_ij[:, 1]].cpu().numpy() * x_mult
        ax.scatter(
            tx,
            ty,
            marker="v",
            c="lime",
            edgecolor="black",
            s=70,
            label=f"{self.n_transducers} transducers",
            zorder=5,
        )
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        return ax
