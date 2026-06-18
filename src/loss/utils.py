from dataclasses import dataclass, field
from typing import Tuple
from src.operator import WaveEquation
from wavefield import Wavefield
from geometry import AcquisitionGeometry
import matplotlib.pyplot as plt
import torch


@dataclass
class LossConfig:
    """Configuration for the loss function."""

    wave_eq: WaveEquation
    geometry: AcquisitionGeometry
    speed_offset: int = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        wf = self.wave_eq.wavefield
        (Nx, Ny), Nt = wf.grid.shape, wf.grid.nt
        self.speed_offset = self.geometry.n_sources * Nx * Ny * Nt
        self.device = wf.grid.device
        self.dtype = wf.grid.dtype

    @property
    def wavefield(self) -> Wavefield:
        return self.wave_eq.wavefield


@dataclass
class LossTape:
    """Tape to store the loss history."""

    name: str = "Default LossTape"
    log_every: int = 5  # log interval
    history: dict = field(
        default_factory=lambda: {"loss": [], "pde_residuals": [], "data_residuals": []}
    )
    success: bool = False

    def log(self, loss: float, residuals: Tuple[torch.Tensor, ...]) -> None:
        """Log the loss and residuals."""
        self.history["loss"].append(loss)
        self.history["pde_residuals"].append(residuals[0].detach().cpu().numpy())

        # forward solver has no data loss
        if len(residuals) > 1:
            self.history["data_residuals"].append(residuals[1].detach().cpu().numpy())

    def show(self, title: str = "Loss History"):
        assert len(self.history["loss"]) > 0, "No loss history to show."
        ncols = 3 if len(self.history["data_residuals"]) > 0 else 2
        fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

        # compute residual norms
        pde_norms = [
            torch.norm(residuals) for residuals in self.history["pde_residuals"]
        ]

        axs[0].semilogy(self.history["loss"])
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss Value")

        axs[1].semilogy(pde_norms)
        axs[1].set_title("PDE Residual Norms")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Residual Norm")

        if ncols == 3:
            data_norms = [
                torch.norm(residuals) for residuals in self.history["data_residuals"]
            ]
            axs[2].semilogy(data_norms)
            axs[2].set_title("Data Residual Norms")
            axs[2].set_xlabel("Iteration")
            axs[2].set_ylabel("Residual Norm")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
