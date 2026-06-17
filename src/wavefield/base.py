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
        return self._wavespeed

    @property
    def data(self) -> np.ndarray:
        """Return flat parameter vector [amp (nt*nx*ny), wvsp (nx*ny)]"""
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

    def show(self, idx: int, title="Wavefield and model", view: str = "xy"):
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

        amp_data = np.take(self.amplitude, idx, axis=axis)  # extract slice

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        (xmin, xmax), (ymin, ymax) = self.grid.extent

        im1 = axs[0].imshow(
            amp_data,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
            aspect="auto",
        )

        im2 = axs[1].imshow(
            self.wavespeed,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
            aspect="auto",
        )

        i, j = view[0], view[1]  # extract letters for labelling
        for ax in axs:
            ax.set_xlabel(i)
            ax.set_ylabel(j)

        slice_plane = ["t", "x", "y"][axis]
        axs[0].set_title(f"Amplitude field ({view} plane, {slice_plane} = {idx})")
        axs[1].set_title("Wave speed model")

        plt.colorbar(im1, ax=axs[0], label="Amplitude")
        plt.colorbar(im2, ax=axs[1], label=r"Wavespeed ($ms^{-1}$)")

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    grid = Grid()
    amp = np.random.rand(grid.nt, *grid.shape)
    wsp = np.random.rand(*grid.shape)
    u = Wavefield(grid, init_amplitude=amp, init_wavespeed=wsp)
    u.show(title="Test plot", idx=100)
