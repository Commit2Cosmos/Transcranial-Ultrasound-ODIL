from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt

from src.grid import Grid


@dataclass
class Wavefield:
    grid: Grid
    _amplitude: np.ndarray = field(init=False)
    init_amplitude: np.ndarray | None = None  # optionally initialise field

    _wavespeed: np.ndarray = field(init=False)
    init_wavespeed: np.ndarray | None = None  # optional initialise speed

    amp_extent: int = field(init=False)

    def __post_init__(self) -> None:
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        # data are stored as flattened arrays
        # initialise amplitude as Nx*Ny*Nt or provided values
        self._amplitude = (
            np.zeros(shape=(Nt, Nx, Ny), dtype=float)
            if self.init_amplitude is None
            else np.asarray(self.init_amplitude, dtype=float)
        )

        # initialise wavespeed as Nx*Ny or provided values
        self._wavespeed = (
            np.ones(shape=(Nx, Ny))
            if self.init_wavespeed is None
            else np.asarray(self.init_wavespeed, dtype=float)
        )

    @property
    def amplitude(self) -> np.ndarray:
        return self._amplitude

    @property
    def wavespeed(self) -> np.ndarray:
        return self._amplitude

    @property
    def data(self) -> np.ndarray:
        """Return flat parameter vector [amp (nt*nx*nx), wvsp (nx*ny)]"""
        return np.concatenate([self._amplitude.ravel(), self._wavespeed.ravel()])

    @data.setter
    def data(self, flat: np.ndarray) -> None:
        """Cast flattened data back into amplitude and wavespeed components"""
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        n_amp = Nx * Ny * Nt  # num amplitude entries
        n_wsp = Nx * Ny  # num wavespeed entries
        total = n_amp + n_wsp  # total

        if flat.shape != (total,):
            raise ValueError(
                f"expected flat vector of length {total}, got {flat.shape}"
            )

        self._amplitude = flat[:n_amp].reshape(Nt, Nx, Ny)
        self._wavespeed = flat[n_amp:].reshape(Nx, Ny)

    def show(self, title="Wavefield and model", view: str = "xy", idx=None):
        assert view in [
            "xy",
            "ty",
            "tx",
        ], "View must be str and one of 'xy', 'ty', 'tx'"
        axis = {"xy": 0, "ty": 1, "tx": 2}[view]  # extract axis index

        # extrcat shapes
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        # default to centre of chosen plane
        if idx is None:
            param = [Nt, Nx, Ny][axis]  # extract upper limit of axis
            idx = param // 2

        assert (
            idx >= 0 and idx < param
        ), f"idx should be an int in range [0, {param}], got {idx}."

        amp_data = np.take(self.amplitude, idx, axis=axis)  # extract slice

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        (xmin, xmax), (ymin, ymax) = self.grid.extent

        im1 = plt.imshow(
            amp_data,
            ax=axs[0],
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
            aspect="auto",
            title=f"Amplitude field ({view} plane, idx = {idx})",
        )

        im2 = plt.imshow(
            self.wavespeed,
            ax=axs[1],
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
            aspect="auto",
            title="Wave speed model",
        )

        i, j = view[0], view[1]  # extract letters for labelling
        for ax in axs:
            ax.set_xlabel(i)
            ax.set_ylabel(j)

        plt.colorbar(im1, cax=axs[0], title="Amplitude")
        plt.colorbar(im2, cax=axs[1], title=r"Wavespeed ($ms^{-1}$)")

        fig.suptitle(title)
        fig.tight_layout()
        fig.show()
