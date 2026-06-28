from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from odil_wave.operator import WaveEquation
from odil_wave.wavefield import Wavefield
from odil_wave.geometry import AcquisitionGeometry
from odil_wave.models import VelocityModel
from .regulariser import Regulariser


_DEFAULT_WEIGHTS = {"pde": 1.0, "data": 1.0, "reg": 1.0}


@dataclass
class LossConfig:
    """Configuration for the loss function.

    `weights` multiplies the (already mean-reduced) per-block losses:
    `w_pde * mean(r_pde**2) + w_data * mean(r_data**2) + w_reg * R(c_int)`.
    Missing keys default to 1.0.

    The PML ring of c is frozen at the attached `VelocityModel.pml_c`; only
    the interior of c is the optimisation variable. Pad with
    `wavefield.velocity_model.build_full_c(c_interior)`.
    """

    wave_eq: WaveEquation
    geometry: AcquisitionGeometry
    weights: Optional[dict] = None
    regulariser: Optional[Regulariser] = None

    speed_offset: int = field(init=False)
    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)

    def __post_init__(self):
        wf = self.wave_eq.wavefield
        Nx, Ny = wf.grid.shape
        Nt = wf.grid.nt

        merged = dict(_DEFAULT_WEIGHTS)
        if self.weights is not None:
            merged.update(self.weights)
        self.weights = merged

        # Hard zero IC at t=0 means amplitude has (NT-1, NX, NY) per shot.
        self.speed_offset = self.geometry.n_sources * (Nt - 1) * Nx * Ny
        self.device = wf.grid.device
        self.dtype = wf.grid.dtype

    @property
    def wavefield(self) -> Wavefield:
        return self.wave_eq.wavefield


@dataclass
class LossTape:
    """Tape to store the loss + residuals + c-history during optimisation."""

    name: str = "Default LossTape"
    log_every: int = 1
    history: dict = field(
        default_factory=lambda: {
            "loss": [],
            "pde_residuals": [],
            "data_residuals": [],
            "c_history": [],
        }
    )
    _norm_cache: dict = field(default_factory=lambda: {"pde": [], "data": []})
    _result: object = field(init=False, default=None)

    def _norms(self, key: str, cache_key: str) -> list:
        residuals = self.history[key]
        cache = self._norm_cache[cache_key]
        for r in residuals[len(cache) :]:
            cache.append(float(np.linalg.norm(r)))
        return cache

    def log(self, loss: float, residuals: Tuple[torch.Tensor, ...]) -> None:
        self.history["loss"].append(loss)
        self.history["pde_residuals"].append(residuals[0].detach().cpu().numpy())
        if len(residuals) > 1:
            self.history["data_residuals"].append(residuals[1].detach().cpu().numpy())

    def log_c(self, c_arr: np.ndarray) -> None:
        """Record a snapshot of the full-grid velocity field at one outer step."""
        self.history["c_history"].append(np.asarray(c_arr).copy())

    def show(self, title: str = "Loss History"):
        assert len(self.history["loss"]) > 0, "No loss history to show."
        ncols = 3 if len(self.history["data_residuals"]) > 0 else 2
        fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

        pde_norms = self._norms("pde_residuals", "pde")

        axs[0].semilogy(self.history["loss"])
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss Value")

        axs[1].semilogy(pde_norms)
        axs[1].set_title("PDE Residual Norms")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Residual Norm")

        if ncols == 3:
            data_norms = self._norms("data_residuals", "data")
            axs[2].semilogy(data_norms)
            axs[2].set_title("Data Residual Norms")
            axs[2].set_xlabel("Iteration")
            axs[2].set_ylabel("Residual Norm")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def animate_c(
        self,
        grid,
        filename: str = "c_history.gif",
        fps: int = 10,
        cmap: str = "viridis",
        title: str = "c(x, y) evolution",
    ) -> str:
        """Render the per-iteration c(x, y) snapshots to an animated GIF."""
        history = self.history["c_history"]
        if not history:
            raise RuntimeError("No c_history to animate. Run an inverse solve first.")

        stack = np.stack(history)  # (n_iter, NX, NY)
        (xmin, xmax), (ymin, ymax) = grid.extent
        vmin = float(stack.min())
        vmax = float(stack.max())

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            stack[0].T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ttl = ax.set_title(f"{title}  (iter 0)")
        plt.colorbar(im, ax=ax, label="c [m/s]", shrink=0.85)

        def update(frame: int):
            im.set_data(stack[frame].T)
            ttl.set_text(f"{title}  (iter {frame})")
            return im, ttl

        anim = animation.FuncAnimation(
            fig, update, frames=stack.shape[0], interval=1000 / fps, blit=False
        )
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return filename

    def show_velocity_recovery(
        self,
        truth: VelocityModel,
        recovered: VelocityModel,
        geom: AcquisitionGeometry,
        title: str = "Velocity recovery",
    ):
        """4-panel: truth | recovered | (recovered - truth) | ||c-c*||_rel vs iter."""
        grid = truth.grid
        c_true_np = truth.c.cpu().numpy()
        c_final_np = recovered.c.cpu().numpy()
        diff = c_final_np - c_true_np

        (xmin, xmax), (ymin, ymax) = grid.extent
        vmin = float(min(c_true_np.min(), c_final_np.min()))
        vmax = float(max(c_true_np.max(), c_final_np.max()))
        dmax = float(np.max(np.abs(diff))) * 1.05 + 1e-9

        rx = grid.x[geom.recv_ij[:, 0]].cpu().numpy()
        ry = grid.y[geom.recv_ij[:, 1]].cpu().numpy()
        sx = grid.x[geom.src_ij[:, 0]].cpu().numpy()
        sy = grid.y[geom.src_ij[:, 1]].cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        panels = [
            (axes[0, 0], c_true_np, "viridis", vmin, vmax, f"truth ({truth.profile})"),
            (axes[0, 1], c_final_np, "viridis", vmin, vmax, "recovered"),
            (axes[1, 0], diff, "RdBu_r", -dmax, dmax, "recovered - truth"),
        ]
        for ax, field_, cmap, lo, hi, t in panels:
            im = ax.imshow(
                field_.T,
                origin="lower",
                extent=[xmin, xmax, ymin, ymax],
                cmap=cmap,
                vmin=lo,
                vmax=hi,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_title(t)
            ax.scatter(rx, ry, marker="v", c="lime", edgecolor="black", s=25, zorder=5)
            ax.scatter(sx, sy, marker="*", c="red", edgecolor="black", s=80, zorder=6)
            plt.colorbar(im, ax=ax, shrink=0.85)

        ax_err = axes[1, 1]
        if self.history["c_history"]:
            denom = float(np.linalg.norm(c_true_np))
            err_hist = [
                float(np.linalg.norm(c - c_true_np) / denom)
                for c in self.history["c_history"]
            ]
            ax_err.semilogy(err_hist, color="tab:red")
            ax_err.set_xlabel("iteration")
            ax_err.set_ylabel(r"$\|c-c^*\|_\mathrm{rel}$")
            ax_err.set_title("c recovery error")
            ax_err.grid(alpha=0.3)
        else:
            ax_err.text(0.5, 0.5, "no c_history", ha="center", va="center")
            ax_err.set_axis_off()

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value
