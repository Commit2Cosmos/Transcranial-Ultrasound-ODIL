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
    """Rescale a (NT, n_traces) trace gather for plotting.

    mode:
      - None / "none":    pass-through
      - "per_receiver":   divide each column (one receiver's trace) by its
                          own max-abs, the standard seismic balancing.
    """
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
    records its own shot. Provides the source injection fields (time- and
    frequency-domain), trace extraction, and plotting helpers used by the
    forward solvers and loss functions.
    """

    def __init__(
        self,
        grid: Grid,
        source: SourceSignal,
        frequency_selection: FrequencySelection,
        *,
        n_transducers: int,
        sigma_s: Optional[float],
        a_frac: float,
        b_frac: float,
        ring_center: Optional[Tuple[float, float]],
        source_spatial: str,
    ):
        """Build the transducer ring for a given grid.

        Parameters
        ----------
        grid : Grid
            Spatial/temporal grid the geometry is placed on; also supplies
            the device and dtype used for the geometry's tensors.
        source : SourceSignal
            Temporal source wavelet, used both for its waveform/spectrum and
            (when `sigma_s` is not given) to pick a default Gaussian width
            from its centre frequency `f0`.
        frequency_selection : FrequencySelection
            Frequency bins used by `source_field` for the frequency-domain
            source, and by the complex dtype it produces.
        n_transducers : int
            Number of transducers placed on the ellipse (>= 2); also the
            number of shots.
        sigma_s : float or None
            Width in metres of the Gaussian source profile
            ``exp(-r**2 / sigma_s**2)`` (standard deviation
            ``sigma_s / sqrt(2)``; ignored when `source_spatial="point"`).
            ``None`` -> derived from the grid spacing and the source's
            dominant wavelength.
        a_frac, b_frac : float
            Semi-axes of the transducer ellipse as a fraction of the half-width
            / half-height of the grid's interior region.
        ring_center : (float, float) or None
            Physical-coordinate centre of the transducer ellipse; ``None`` ->
            the centre of the grid's interior region.
        source_spatial : {"gaussian", "point"}
            Spatial injection profile: ``"gaussian"`` is a Gaussian blob of
            width `sigma_s`; ``"point"`` is a unit Kronecker delta at the
            source grid node.
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
        self.sigma_s = (
            self._default_sigma_s(grid, source) if sigma_s is None else sigma_s
        )
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

    @staticmethod
    def _default_sigma_s(grid: Grid, source: SourceSignal) -> float:
        """Default Gaussian source width, derived from grid spacing and the
        source's shortest resolved wavelength."""
        h = max(grid.dx, grid.dy)
        c_ref = grid.c_min if grid.c_min is not None else grid.c_max
        f_max = 2.5 * source.f0
        lambda_min = c_ref / f_max
        return max(1.5 * h, lambda_min / (2.0 * math.pi))

    def _place_ellipse(self, n: int) -> torch.Tensor:
        """Return (n, 2) integer full-grid indices for n points evenly spaced
        on an ellipse inside the grid's interior region."""
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
        """Spatial injection profile for source `src_idx`, as a full-grid
        tensor of shape `grid.shape`. Selects between a Gaussian blob and a
        point (Kronecker delta) source based on `self.source_spatial`."""
        if self.source_spatial == "point":
            i = int(self.transducer_ij[src_idx, 0])
            j = int(self.transducer_ij[src_idx, 1])
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
        """Time-domain source field for `src_idx`, used by `plot_source_field`.

        Parameters
        ----------
        src_idx : int
            Shot index (0 .. `self.n_shots` - 1).

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(nt, nx, ny)``: the source's temporal
            waveform times its spatial profile.
        """
        spatial = self._spatial_profile(src_idx)
        temporal = self.source.waveform(self.grid.t)
        return temporal.view(-1, 1, 1) * spatial.view(1, *self.grid.shape)

    def source_field(self, src_idx: int) -> torch.Tensor:
        """Frequency-domain source field for `src_idx`, used by the Helmholtz
        solver and loss functions.

        Parameters
        ----------
        src_idx : int
            Shot index (0 .. `self.n_shots` - 1).

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(n_frequencies, nx, ny)``, on the bins
            of `self.frequency_selection`: the source's spectrum times its
            spatial profile.
        """
        spatial = self._spatial_profile(src_idx)
        spectrum = self.source.spectrum(self.frequency_selection)
        cdtype = self.frequency_selection.cdtype
        return spectrum.to(dtype=cdtype).view(-1, 1, 1) * spatial.to(dtype=cdtype).view(
            1, *self.grid.shape
        )

    def extract_observations(self, U: torch.Tensor) -> torch.Tensor:
        """Pull each shot's recorded traces out of its full-grid wavefield.

        Parameters
        ----------
        U : torch.Tensor
            Per-shot fields ``(n_shots, ..., nx, ny)`` — e.g. frequency-domain
            ``(n_shots, nf, nx, ny)`` or time-domain ``(n_shots, nt, nx, ny)``.

        Returns
        -------
        torch.Tensor
            Shot ``s`` sampled at its `receivers[s]` nodes, shape
            ``(n_shots, ..., n_transducers - 1)``; the transmitting
            transducer's own node is never sampled.
        """
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

        Parameters
        ----------
        wavefield :
            A `Wavefield` or `(NT, NX, NY)` amplitude tensor/array for shot
            `shot`, or a list of either with one entry per shot (index = shot),
            producing a subplot grid. `ax` is ignored when a list is given.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into for a single wavefield. Defaults to a new
            figure/axes.
        normalize : {"per_receiver", "none", None}, default "per_receiver"
            Per-receiver max-abs balancing, or no rescaling.
        shot : int, default 0
            Shot a single (non-list) `wavefield` belongs to.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of Axes
            Single Axes for one wavefield, array of Axes for a list.
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
        """Plot a spatial snapshot of the source field s(x, y, t_idx).

        Parameters
        ----------
        src_idx : int, default 0
            Shot index (0 .. `self.n_shots` - 1).
        t_idx : int, optional
            Time index to plot. Defaults to the peak time of the source's
            temporal waveform.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Defaults to a new figure/axes.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes drawn into.
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
        """Overlay the transducer ring on the velocity model.

        The velocity colourbar uses a two-slope normalisation so that
        ``[vmin, vcenter]`` and ``[vcenter, vmax]`` each fill half the bar
        (defaults 1400 / 1600 / 3000 m/s). Pass an explicit ``norm`` to
        override; `vmin`/`vcenter`/`vmax` are then ignored.

        Parameters
        ----------
        velocity_model : VelocityModel
            Background image; also sets the axis unit scaling that the
            transducer markers are plotted in.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Defaults to a new figure/axes.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes drawn into.
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
