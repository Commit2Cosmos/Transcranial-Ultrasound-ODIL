from dataclasses import dataclass, field
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from odil_wave.grid import Grid


def _normalize_amplitude(amp: np.ndarray, mode: str | None) -> np.ndarray:
    """Rescale a wavefield array for plotting.

    mode:
      - None / "none": pass-through
      - "global":     divide by global max-abs (whole volume / slice)
      - "per_frame":  divide each leading-axis frame by its own max-abs
    """
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
    grid: Grid
    _amplitude: torch.Tensor = field(init=False)
    init_amplitude: torch.Tensor | np.ndarray | None = (
        None  # optionally initialise field
    )

    _wavespeed: torch.Tensor = field(init=False)
    init_wavespeed: torch.Tensor | np.ndarray | None = None  # optional initialise speed

    _init_ut: torch.Tensor = field(init=False)
    init_velocity: torch.Tensor | np.ndarray | None = (
        None  # optional initial velocity field
    )

    # ensure device and dataype are consistent
    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)

    def __post_init__(self) -> None:
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        # extract device and dtype from grid for consistency
        self.device = self.grid.device
        self.dtype = self.grid.dtype

        # cast inputs to torch Tensors to accept numpy arrays as well
        # initialise amplitude as Nx*Ny*Nt or provided values
        self._amplitude = (
            torch.zeros(size=(Nt, Nx, Ny), dtype=self.dtype, device=self.device)
            if self.init_amplitude is None
            else torch.as_tensor(
                self.init_amplitude, dtype=self.dtype, device=self.device
            )
        )

        # initialise wavespeed as Nx*Ny or provided values (flat 1D accepted, reshaped to (Nx, Ny))
        if self.init_wavespeed is None:
            self._wavespeed = torch.ones(
                size=(Nx, Ny), dtype=self.dtype, device=self.device
            )
        else:
            wsp = torch.as_tensor(
                self.init_wavespeed, dtype=self.dtype, device=self.device
            )
            self._wavespeed = wsp.reshape(Nx, Ny)

        self._init_ut = (
            torch.zeros(
                size=(Nx, Ny), dtype=self.dtype, device=self.device
            )  # default init is zeros
            if self.init_velocity is None
            else torch.as_tensor(
                self.init_velocity, dtype=self.dtype, device=self.device
            )
        )

    @property
    def init_ut(self) -> torch.Tensor:
        return self._init_ut

    @property
    def amplitude(self) -> torch.Tensor:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: np.ndarray | torch.Tensor) -> None:
        Nt = self.grid.nt
        Nx, Ny = self.grid.shape
        self._amplitude = torch.as_tensor(
            np.asarray(value).reshape(Nt, Nx, Ny), dtype=self.dtype, device=self.device
        )

    @property
    def wavespeed(self) -> torch.Tensor:
        return self._wavespeed

    @wavespeed.setter
    def wavespeed(self, value) -> None:
        Nx, Ny = self.grid.shape
        wsp = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self._wavespeed = wsp.reshape(Nx, Ny)

    def show(
        self,
        idx: int,
        title="Wavefield and model",
        view: str = "xy",
        normalize: str | None = None,
    ):
        assert view in [
            "xy",
            "ty",
            "tx",
        ], "View must be str and one of 'xy', 'ty', 'tx'"
        axis = {"xy": 0, "ty": 1, "tx": 2}[view]  # extract axis index

        # extrcat shapes
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        param = [Nt, Nx, Ny][axis]  # extract upper limit of axis

        if not (0 <= idx < param):
            raise ValueError(f"idx should be an int in range [0, {param}], got {idx}.")

        amp_data = np.take(
            self.amplitude.cpu().numpy(), idx, axis=axis
        )  # extract slice
        amp_data = _normalize_amplitude(amp_data, normalize)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        (xmin, xmax), (ymin, ymax) = self.grid.extent

        im1 = axs[0].imshow(
            amp_data.T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
        )

        im2 = axs[1].imshow(
            self.wavespeed.cpu().numpy().T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
        )

        i, j = view[0], view[1]  # extract letters for labelling
        for ax in axs:
            ax.set_xlabel(i)
            ax.set_ylabel(j)

        slice_plane = ["t", "x", "y"][axis]
        axs[0].set_title(f"Amplitude field ({view} plane, {slice_plane} = {idx})")
        axs[1].set_title("Wave speed model")

        plt.colorbar(im1, ax=axs[0], label="Amplitude", shrink=0.85)
        plt.colorbar(im2, ax=axs[1], label=r"Wavespeed ($ms^{-1}$)", shrink=0.85)

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    def animate(
        self,
        filename: str = "wavefield.gif",
        fps: int = 20,
        cmap: str = "RdBu_r",
        title: str = "Wavefield history",
        normalize: str | None = None,
    ) -> str:
        """Render the amplitude field over all time steps to an animated GIF.

        `normalize`:
          - None (default): a single global colour scale (±max-abs over the
            whole history) — physically faithful, but late-time wavefields
            can look washed out if the early signal is much stronger.
          - "global": same scale, but values rescaled to [-1, 1].
          - "per_frame": each frame is rescaled to its own ±max-abs, which
            keeps weak frames visible at the cost of comparability.
        """

        amp = self.amplitude.cpu().numpy()  # (Nt, Nx, Ny)
        amp = _normalize_amplitude(amp, normalize)
        Nt = self.grid.nt
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        t = self.grid.t.cpu().numpy()

        # fixed colour scale
        vmax = float(np.abs(amp).max()) or 1.0
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            amp[0].T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ttl = ax.set_title(f"{title}  (t = {t[0]:.3f} s)")
        plt.colorbar(im, ax=ax, label="Amplitude", shrink=0.85)

        # update for drawing frames
        def update(frame: int):
            im.set_data(amp[frame].T)
            ttl.set_text(f"{title}  (t = {t[frame]:.3f} s)")
            return im, ttl

        anim = animation.FuncAnimation(
            fig, update, frames=Nt, interval=1000 / fps, blit=False
        )
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return filename
