from dataclasses import dataclass, field
from typing import Tuple
from src.operator import DenseOperator, SparseOperator
from src.operator import TimeOperator2ndOrder, TimeOperator4thOrder
from src.operator import Laplacian2ndOrder, Laplacian4thOrder
from src.wavefield import Wavefield
from geometry import AcquisitionGeometry
import matplotlib.pyplot as plt
import torch


@dataclass
class LossConfig:
    """Configuration for the loss function."""

    wavefield: Wavefield
    geometry: AcquisitionGeometry
    time_order: int = 2  # order of the time-derivative method
    space_order: int = 2  # order of the Laplacian method
    time_operator: DenseOperator | SparseOperator = field(init=False)
    laplacian_operator: DenseOperator | SparseOperator = field(init=False)
    speed_offset: int = field(init=False)
    device: torch.Device = field(init=False)

    def __post_init__(self):
        if self.time_order == 2:
            self.time_operator = TimeOperator2ndOrder(self.wavefield)
        elif self.time_order == 4:
            self.time_operator = TimeOperator4thOrder(self.wavefield)
        else:
            raise ValueError(f"Invalid time order: {self.time_order}")

        if self.space_order == 2:
            self.laplacian_operator = Laplacian2ndOrder()
        elif self.space_order == 4:
            self.laplacian_operator = Laplacian4thOrder()
        else:
            raise ValueError(f"Invalid space order: {self.space_order}")

        (Nx, Ny), Nt = self.wavefield.grid.shape, self.wavefield.grid.nt
        n_shots = self.geometry.n_sources
        self.speed_offset = n_shots * Nx * Ny * Nt  # idx for accessing wavespeed

        self.device = self.wavefield.device
        self.dtype = self.wavefield.dtype


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
