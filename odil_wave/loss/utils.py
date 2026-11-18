from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize

from odil_wave.geometry import AcquisitionGeometry
from odil_wave.models import VelocityModel, profile_velocity_norm, velocity_norm
from odil_wave.plot_utils import length_scale


@dataclass
class LossTape:
    """Scalar loss history and optional ``c`` snapshots, for notebook plots.

    Filled live by passing :meth:`log` as ``JointFreqODIL.minimise(on_log=...)``,
    or from a saved run's tape rows by :meth:`from_records`. Full residual
    tensors are never stored: they can grow a Jupyter kernel until it crashes.
    """

    name: str = "Default LossTape"
    store_c_history: bool = True
    c_history_every: int = 1

    history: dict = field(
        default_factory=lambda: {
            "loss": [],
            "pde_rel_residual": [],
            "rel_data_residual": [],
            "c_history": [],
        }
    )

    def log(self, iterate) -> None:
        """Append one :class:`~odil_wave.optimisation.Iterate`'s scalars, and
        its ``c_full`` every ``c_history_every``-th call if ``store_c_history``."""
        self.history["loss"].append(float(iterate.loss))
        self.history["pde_rel_residual"].append(float(iterate.pde_rel_residual))
        self.history["rel_data_residual"].append(float(iterate.rel_data_residual))
        if (
            self.store_c_history
            and len(self.history["loss"]) % self.c_history_every == 0
        ):
            c = iterate.c_full.detach().cpu().numpy()
            self.history["c_history"].append(np.asarray(c, dtype=np.float32).copy())

    @classmethod
    def from_records(cls, records, name: str = "loaded run") -> "LossTape":
        """A tape's scalar history from ``tape.csv`` rows; ``c_history`` stays
        empty."""
        # (history key, record key) — records use ``loss_total`` for the loss.
        key_map = [
            ("loss", "loss_total"),
            ("pde_rel_residual", "pde_rel_residual"),
            ("rel_data_residual", "rel_data_residual"),
            ("rmse_brain", "rmse_brain"),
            ("ssim_brain", "ssim_brain"),
        ]
        tape = cls(name=name, store_c_history=False)
        for hist_key, rec_key in key_map:
            series = [
                (float(r[rec_key]) if r.get(rec_key) is not None else float("nan"))
                for r in records
            ]
            if any(np.isfinite(v) for v in series):
                tape.history[hist_key] = series
        return tape

    def show(self, title: str = "Loss History"):
        """Plot the loss and residual-ratio histories; the data panel only when
        a ``rel_data_residual`` series exists."""
        assert len(self.history["loss"]) > 0, "No loss history to show."

        has_data = len(self.history.get("rel_data_residual", [])) > 0
        ncols = 3 if has_data else 2
        fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

        axs[0].semilogy(self.history["loss"])
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Logged iteration")
        axs[0].set_ylabel("Loss Value")

        axs[1].semilogy(self.history["pde_rel_residual"])
        axs[1].set_title("PDE residual (relative)")
        axs[1].set_xlabel("Logged iteration")
        axs[1].set_ylabel("RMS(r_pde / pde_scale)")

        if has_data:
            axs[2].semilogy(self.history["rel_data_residual"])
            axs[2].set_title("Data residual (relative)")
            axs[2].set_xlabel("Logged iteration")
            axs[2].set_ylabel("||d_syn - d_obs|| / ||d_obs||")

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
        norm=None,
        frame_labels=None,
        profile: Optional[str] = None,
    ) -> str:
        """Render the ``c(x, y)`` snapshots in ``c_history`` to a GIF at
        ``filename`` and return it. Without ``norm``, the scale is
        :func:`profile_velocity_norm` of the truth's ``profile``."""
        history = self.history["c_history"]
        if not history:
            raise RuntimeError("No c_history to animate. Run an inverse solve first.")
        if frame_labels is not None and len(frame_labels) != len(history):
            raise ValueError(
                f"frame_labels has {len(frame_labels)} entries but there are "
                f"{len(history)} frames."
            )

        stack = np.stack(history)  # (n_iter, NX, NY)
        (xmin, xmax), (ymin, ymax) = grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))

        if norm is None and profile is not None:
            norm = profile_velocity_norm(profile, data=stack)
        if norm is None:
            _vmin = float(stack.min())
            _vmax = float(stack.max())
            if _vmax > _vmin:
                norm = velocity_norm(
                    vmin=_vmin,
                    vcenter=_vmin + 0.25 * (_vmax - _vmin),
                    vmax=_vmax,
                )
        else:
            _vmin = None
            _vmax = None

        imshow_kw = dict(
            origin="lower",
            extent=(xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult),
            cmap=cmap,
            animated=True,
        )

        if norm is not None:
            imshow_kw["norm"] = norm
        else:
            imshow_kw["vmin"] = _vmin
            imshow_kw["vmax"] = _vmax

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(stack[0].T, **imshow_kw)
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"y [{x_unit}]")

        def _frame_title(frame: int) -> str:
            tag = frame_labels[frame] if frame_labels is not None else f"frame {frame}"
            return f"{title}  ({tag})"

        ttl = ax.set_title(_frame_title(0))
        plt.colorbar(im, ax=ax, label="c [m/s]", shrink=0.85)

        def update(frame: int):
            im.set_data(stack[frame].T)
            ttl.set_text(_frame_title(frame))
            return im, ttl

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=stack.shape[0],
            interval=1000 / fps,
            blit=False,
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
        norm=None,
        vmin: Optional[float] = None,
        vcenter: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Plot truth, recovered, difference and c-recovery-error panels.
        Without ``norm``, the velocity scale is :func:`profile_velocity_norm`
        of ``truth.profile`` (bounds in m/s)."""
        grid = truth.grid
        c_true_np = truth.c.detach().cpu().numpy()
        c_final_np = recovered.c.detach().cpu().numpy()
        diff = c_final_np - c_true_np

        (xmin, xmax), (ymin, ymax) = grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        dmax = float(np.max(np.abs(diff))) * 1.05 + 1e-9

        tx = grid.x[geom.transducer_ij[:, 0]].detach().cpu().numpy() * x_mult
        ty = grid.y[geom.transducer_ij[:, 1]].detach().cpu().numpy() * x_mult

        if norm is None:
            # Truth and recovered share one scale: a linear one spans both.
            norm = profile_velocity_norm(
                truth.profile,
                vmin=vmin,
                vcenter=vcenter,
                vmax=vmax,
                data=np.stack([c_true_np, c_final_np]),
            )

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        panels = [
            (axes[0, 0], c_true_np, "viridis", norm, f"truth ({truth.profile})"),
            (axes[0, 1], c_final_np, "viridis", norm, "recovered"),
            (axes[1, 0], diff, "RdBu_r", Normalize(-dmax, dmax), "recovered - truth"),
        ]

        for ax, field_, cmap_, norm_, panel_title in panels:
            im = ax.imshow(
                field_.T,
                origin="lower",
                extent=[xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult],
                cmap=cmap_,
                norm=norm_,
            )
            ax.set_xlabel(f"x [{x_unit}]")
            ax.set_ylabel(f"y [{x_unit}]")
            ax.set_aspect("equal")
            ax.set_title(panel_title)
            ax.scatter(tx, ty, marker="v", c="lime", edgecolor="black", s=25, zorder=5)
            plt.colorbar(im, ax=ax, shrink=0.85)

        ax_err = axes[1, 1]
        rel_hist = [
            e
            for e in self.history.get("rel_c_error", [])
            if e is not None and np.isfinite(e)
        ]
        if rel_hist:
            err_hist = rel_hist
        elif self.history["c_history"]:
            denom = float(np.linalg.norm(c_true_np))
            err_hist = [
                float(np.linalg.norm(c - c_true_np) / denom)
                for c in self.history["c_history"]
            ]
        else:
            err_hist = None

        if err_hist:
            ax_err.semilogy(err_hist, color="tab:red")
            ax_err.set_xlabel("logged iteration")
            ax_err.set_ylabel(r"$\|c-c^*\|_\mathrm{rel}$")
            ax_err.set_title("c recovery error")
            ax_err.grid(alpha=0.3)
        else:
            ax_err.text(0.5, 0.5, "no error history", ha="center", va="center")
            ax_err.set_axis_off()

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
