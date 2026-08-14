from dataclasses import dataclass, field

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from odil_wave.grid import Grid, FrequencySelection, complex_dtype
from odil_wave.models import VelocityModel
from odil_wave.plot_utils import length_scale, frequency_scale


def _normalize_amplitude(amp: np.ndarray, mode: str | None) -> np.ndarray:
    """Rescale a wavefield array for plotting."""
    if mode is None or mode == "none":
        return amp
    if mode == "global":
        scale = float(np.max(np.abs(amp)))
        return amp / scale if scale > 0 else amp
    if mode == "per_frame":
        axes = tuple(range(1, amp.ndim))
        scale = np.max(np.abs(amp), axis=axes, keepdims=True)
        scale = np.where(scale > 0, scale, 1.0)
        return amp / scale
    raise ValueError(
        f"Invalid normalize mode {mode!r}; expected one of None, 'global', 'per_frame'."
    )


@dataclass
class Wavefield:
    """Complex frequency-domain wavefield u(omega, x, y) on a `Grid`.

    Amplitude shape is ``(n_frequencies, nx, ny)`` complex for a single shot.
    Multi-shot batches used by losses/optimisers are
    ``(n_shots, n_frequencies, nx, ny)``.

    ``init_ut`` is the initial condition ``u_t(x, y, t=0)`` and is retained
    only for the leapfrog time-domain reference generator used in FFT
    sanity tests; it is unused by the frequency ODIL path.
    """

    grid: Grid
    frequency_selection: FrequencySelection
    _amplitude: torch.Tensor = field(init=False)
    init_amplitude: torch.Tensor | np.ndarray | None = None
    velocity_model: VelocityModel | None = None
    _init_ut: torch.Tensor = field(init=False)
    init_velocity: torch.Tensor | np.ndarray | None = None
    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)
    cdtype: torch.dtype = field(init=False)

    def __post_init__(self) -> None:
        """Validate grid/frequency_selection compatibility and allocate tensors.

        Copies ``device``/``dtype`` from ``grid``, then initialises the
        complex amplitude tensor from ``init_amplitude`` (or zeros) and the
        real initial-velocity tensor from ``init_velocity`` (or zeros).
        """
        if self.frequency_selection.grid is not self.grid:
            # Allow equal grids constructed separately if metadata matches
            if (
                self.frequency_selection.n_time != self.grid.nt
                or abs(self.frequency_selection.dt - self.grid.dt) > 1e-15
            ):
                raise ValueError(
                    "frequency_selection must be built from the same Grid "
                    "(matching nt/dt) as this Wavefield."
                )

        Nx, Ny = self.grid.shape
        nf = self.frequency_selection.n_frequencies

        self.device = self.grid.device
        self.dtype = self.grid.dtype
        self.cdtype = complex_dtype(self.dtype)

        self._amplitude = (
            torch.zeros(size=(nf, Nx, Ny), dtype=self.cdtype, device=self.device)
            if self.init_amplitude is None
            else torch.as_tensor(
                self.init_amplitude, dtype=self.cdtype, device=self.device
            )
        )

        if self.velocity_model is None:
            self.velocity_model = VelocityModel(self.grid)

        self._init_ut = (
            torch.zeros(size=(Nx, Ny), dtype=self.dtype, device=self.device)
            if self.init_velocity is None
            else torch.as_tensor(
                self.init_velocity, dtype=self.dtype, device=self.device
            )
        )

    @property
    def init_ut(self) -> torch.Tensor:
        """Initial (dimensional) time-derivative field ``u_t(x, y, t=0)``."""
        return self._init_ut

    @property
    def init_ut_nd(self) -> torch.Tensor:
        """Initial time-derivative field non-dimensionalised by ``grid.t0``."""
        return self._init_ut * self.grid.t0

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies carried by ``frequency_selection``."""
        return self.frequency_selection.n_frequencies

    @property
    def amplitude(self) -> torch.Tensor:
        """Complex amplitude tensor, shape ``(n_frequencies, nx, ny)``."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: np.ndarray | torch.Tensor) -> None:
        """Set the amplitude tensor, casting to ``cdtype``/``device`` and
        reshaping to ``(n_frequencies, nx, ny)``.

        Args:
            value: New amplitude data, as a numpy array or torch tensor,
                with a number of elements matching
                ``n_frequencies * nx * ny``.
        """
        nf = self.n_frequencies
        Nx, Ny = self.grid.shape
        amp = (
            value.to(dtype=self.cdtype, device=self.device)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(
                np.asarray(value), dtype=self.cdtype, device=self.device
            )
        )
        self._amplitude = amp.reshape(nf, Nx, Ny)

    def show(
        self,
        idx: int,
        title: str = "Wavefield",
        normalize: str | None = None,
    ):
        """Show frequency slice ``idx`` as a 2x2 panel of the complex field.

        Panels are ``|u|`` (magnitude), ``Re(u)``, ``Im(u)`` and the phase
        ``arg(u)``.

        Args:
            idx: Frequency index to plot, in ``[0, n_frequencies)``.
            title: Figure suptitle prefix; the plotted frequency is appended.
            normalize: ``None``/``"none"`` to plot the raw field, or
                ``"global"``/``"per_frame"`` to rescale the complex field by
                its peak magnitude before the magnitude/real/imaginary panels
                are drawn (both modes are equivalent for a single slice); the
                phase panel is unaffected.

        Raises:
            ValueError: If ``idx`` is out of range or ``normalize`` is not
                one of ``None``, ``"none"``, ``"global"``, ``"per_frame"``.
        """
        if not (0 <= idx < self.n_frequencies):
            raise ValueError(f"idx should be in [0, {self.n_frequencies}), got {idx}.")
        amp = self.amplitude[idx].detach().cpu().numpy()
        # Normalise by this slice's peak magnitude.
        if normalize in ("global", "per_frame"):
            scale = float(np.max(np.abs(amp)))
            if scale > 0:
                amp = amp / scale
        elif normalize not in (None, "none"):
            raise ValueError(
                f"Invalid normalize mode {normalize!r}; expected one of "
                "None, 'global', 'per_frame'."
            )

        # (data, cmap, label, symmetric-diverging?)
        panels = [
            (np.abs(amp), "viridis", "|u|", False),
            (amp.real, "RdBu_r", "Re(u)", True),
            (amp.imag, "RdBu_r", "Im(u)", True),
            (np.angle(amp), "twilight", "phase [rad]", False),
        ]

        (xmin, xmax), (ymin, ymax) = self.grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        x_extent = (xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult)
        f = float(self.frequency_selection.frequencies[idx].item())
        f_mult, f_unit = frequency_scale(abs(f) if f != 0 else 1.0)

        fig, axs = plt.subplots(2, 2, figsize=(11, 9))
        for ax, (data, cmap, label, diverging) in zip(axs.ravel(), panels):
            imshow_kw = dict(origin="lower", extent=x_extent, cmap=cmap)
            if label.startswith("phase"):
                imshow_kw["vmin"], imshow_kw["vmax"] = -np.pi, np.pi
            elif diverging:
                m = float(np.max(np.abs(data))) or 1.0
                imshow_kw["vmin"], imshow_kw["vmax"] = -m, m
            im = ax.imshow(data.T, **imshow_kw)
            ax.set_xlabel(f"x [{x_unit}]")
            ax.set_ylabel(f"y [{x_unit}]")
            ax.set_title(label)
            plt.colorbar(im, ax=ax, label=label, shrink=0.85)

        fig.suptitle(f"{title} (f = {f * f_mult:.3g} {f_unit})")
        fig.tight_layout()
        plt.show()

    def animate(
        self,
        filename: str = "wavefield.gif",
        fps: int = 5,
        cmap: str = "viridis",
        title: str = "Wavefield |u|(f)",
        normalize: str | None = None,
    ) -> str:
        """Animate ``|u|`` over frequency index and save as a GIF.

        Args:
            filename: Output path for the saved GIF.
            fps: Frames per second.
            cmap: Colormap for the ``|u|`` panel.
            title: Title prefix; the frame's frequency is appended.
            normalize: See :func:`_normalize_amplitude`.

        Returns:
            ``filename``.
        """
        amp = np.abs(self.amplitude.detach().cpu().numpy())
        amp = _normalize_amplitude(amp, normalize)
        nf = self.n_frequencies
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        freqs = self.frequency_selection.frequencies.cpu().numpy()
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        f_mult, f_unit = frequency_scale(float(np.max(np.abs(freqs))) or 1.0)

        vmax = float(np.abs(amp).max()) or 1.0
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            amp[0].T,
            origin="lower",
            extent=(xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            animated=True,
        )
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"y [{x_unit}]")
        ttl = ax.set_title(f"{title}  (f = {freqs[0] * f_mult:.3g} {f_unit})")
        plt.colorbar(im, ax=ax, label="|u|", shrink=0.85)

        def update(frame: int):
            """Update the image and title for animation frame ``frame``."""
            im.set_data(amp[frame].T)
            ttl.set_text(f"{title}  (f = {freqs[frame] * f_mult:.3g} {f_unit})")
            return im, ttl

        anim = animation.FuncAnimation(
            fig, update, frames=nf, interval=1000 / fps, blit=False
        )
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return filename
