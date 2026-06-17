from dataclasses import dataclass, field
import numpy as np

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
