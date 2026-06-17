from dataclasses import dataclass, field
from src.operator import DenseOperator, SparseOperator
from src.operator import TimeOperator2ndOrder, TimeOperator4thOrder
from src.operator import Laplacian2ndOrder, Laplacian4thOrder
import matplotlib.pyplot as plt
import torch


@dataclass
class LossConfig:
    """Configuration for the loss function."""

    time_order: int = 2  # order of the time-derivative method
    space_order: int = 2  # order of the Laplacian method
    time_operator: DenseOperator | SparseOperator = field(init=False)
    laplacian_operator: DenseOperator | SparseOperator = field(init=False)

    def __post_init__(self):
        if self.time_order == 2:
            self.time_operator = TimeOperator2ndOrder()
        elif self.time_order == 4:
            self.time_operator = TimeOperator4thOrder()
        else:
            raise ValueError(f"Invalid time order: {self.time_order}")

        if self.space_order == 2:
            self.laplacian_operator = Laplacian2ndOrder()
        elif self.space_order == 4:
            self.laplacian_operator = Laplacian4thOrder()
        else:
            raise ValueError(f"Invalid space order: {self.space_order}")


@dataclass
class LossTape:
    """Tape to store the loss history."""

    name: str = "Default LossTape"
    log_every: int = 5  # log interval
    history: dict = {"loss": [], "residuals": []}
    success: bool = False

    def log(self, loss: float, residuals: torch.Tensor):
        """Log the loss and residuals."""
        self.history["loss"].append(loss)
        self.history["residuals"].append(residuals.detach().cpu().numpy())

    def show(self, title: str = "Loss History"):
        assert len(self.history["loss"]) > 0, "No loss history to show."
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # compute residual norms
        r_norms = [torch.norm(residuals) for residuals in self.history["residuals"]]

        axs[0].plot(self.history["loss"])
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss Value")

        axs[1].plot(r_norms)
        axs[1].set_title("Residual Norms")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Residual Norm")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
