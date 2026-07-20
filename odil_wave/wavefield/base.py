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
    """Complex frequency-domain wavefield u(ω, x, y) on a `Grid`.

    Amplitude shape is ``(n_frequencies, nx, ny)`` complex for a single shot.
    Multi-shot batches used by losses/optimisers are
    ``(n_shots, n_frequencies, nx, ny)``.

    ``init_ut`` is retained only for the leapfrog *time-domain reference*
    generator used in FFT sanity tests; it is unused by the frequency ODIL path.
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
        return self._init_ut

    @property
    def init_ut_nd(self) -> torch.Tensor:
        return self._init_ut * self.grid.t0

    @property
    def n_frequencies(self) -> int:
        return self.frequency_selection.n_frequencies

    @property
    def amplitude(self) -> torch.Tensor:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: np.ndarray | torch.Tensor) -> None:
        nf = self.n_frequencies
        Nx, Ny = self.grid.shape
        amp = (
            value.to(dtype=self.cdtype, device=self.device)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(np.asarray(value), dtype=self.cdtype, device=self.device)
        )
        self._amplitude = amp.reshape(nf, Nx, Ny)

    @property
    def wavespeed(self) -> torch.Tensor:
        return self.velocity_model.c

    def show(
        self,
        idx: int,
        title="Wavefield and model",
        view: str = "abs",
        normalize: str | None = None,
        norm=None,
    ):
        """Show frequency slice ``idx``.

        ``view``: ``"abs"`` | ``"real"`` | ``"imag"`` | ``"phase"``.
        """
        if not (0 <= idx < self.n_frequencies):
            raise ValueError(
                f"idx should be in [0, {self.n_frequencies}), got {idx}."
            )
        amp = self.amplitude[idx].detach().cpu().numpy()
        if view == "abs":
            amp_data = np.abs(amp)
            cmap = "viridis"
            label = "|u|"
        elif view == "real":
            amp_data = amp.real
            cmap = "RdBu_r"
            label = "Re(u)"
        elif view == "imag":
            amp_data = amp.imag
            cmap = "RdBu_r"
            label = "Im(u)"
        elif view == "phase":
            amp_data = np.angle(amp)
            cmap = "twilight"
            label = "phase"
        else:
            raise ValueError(f"view must be abs/real/imag/phase, got {view!r}")

        amp_data = _normalize_amplitude(amp_data, normalize)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        x_extent = (xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult)

        im1 = axs[0].imshow(
            amp_data.T, origin="lower", extent=x_extent, cmap=cmap
        )

        im2_kw = dict(origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="viridis")
        wsp_np = self.wavespeed.cpu().numpy()
        if norm is None:
            im2_kw["vmin"] = float(wsp_np.min())
            im2_kw["vmax"] = float(wsp_np.max())
        else:
            im2_kw["norm"] = norm
        im2 = axs[1].imshow(wsp_np.T, **im2_kw)

        f = float(self.frequency_selection.frequencies[idx].item())
        f_mult, f_unit = frequency_scale(abs(f) if f != 0 else 1.0)
        for ax in axs:
            ax.set_xlabel(f"x [{x_unit}]")
            ax.set_ylabel(f"y [{x_unit}]")
        axs[0].set_title(f"{label} (f = {f * f_mult:.3g} {f_unit})")
        axs[1].set_title("Wave speed model")
        plt.colorbar(im1, ax=axs[0], label=label, shrink=0.85)
        plt.colorbar(im2, ax=axs[1], label=r"Wavespeed ($ms^{-1}$)", shrink=0.85)
        fig.suptitle(title)
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
        """Animate ``|u|`` over frequency index."""
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
            im.set_data(amp[frame].T)
            ttl.set_text(f"{title}  (f = {freqs[frame] * f_mult:.3g} {f_unit})")
            return im, ttl

        anim = animation.FuncAnimation(
            fig, update, frames=nf, interval=1000 / fps, blit=False
        )
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return filename