from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from odil_wave.operator import WaveEquation
from odil_wave.wavefield import Wavefield
from odil_wave.geometry import AcquisitionGeometry
from odil_wave.models import VelocityModel, velocity_norm
from odil_wave.plot_utils import length_scale
from .regulariser import Regulariser


_DEFAULT_WEIGHTS = {"pde": 1.0, "data": 1.0, "reg": 1.0}


@dataclass
class LossConfig:
    """Configuration for the loss function.

    `weights` multiplies the (already mean-reduced) per-block losses:
    `pde_weight * mean(r_pde**2) + data_weight * mean(r_data**2) + w_reg * R(c_int)`.
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
        """Merge default weights and derive device, dtype and DOF offsets.

        Notes
        -----
        Fills missing ``weights`` keys with 1.0 and computes
        ``speed_offset``, the number of real wavefield DOFs preceding the
        velocity parameters in a packed optimisation vector.
        """
        wf = self.wave_eq.wavefield
        Nx, Ny = wf.grid.shape
        nf = wf.n_frequencies

        merged = dict(_DEFAULT_WEIGHTS)
        if self.weights is not None:
            merged.update(self.weights)
        self.weights = merged

        # Real/imag packing: 2 * n_sources * nf * nx * ny free u DOFs.
        self.speed_offset = self.geometry.n_sources * nf * Nx * Ny * 2
        self.device = wf.grid.device
        self.dtype = wf.grid.dtype

    @property
    def wavefield(self) -> Wavefield:
        """Wavefield attached to the wave equation.

        Returns
        -------
        Wavefield
            ``self.wave_eq.wavefield``.
        """
        return self.wave_eq.wavefield


@dataclass
class LossTape:
    """Tape to store scalar loss/residual diagnostics + optional c-history.

    Important: this does NOT store full PDE/data residual tensors, because those
    are huge and can make Jupyter memory grow until the kernel crashes.
    """

    name: str = "Default LossTape"
    log_every: int = 1
    store_c_history: bool = True
    c_history_every: int = 1

    history: dict = field(
        default_factory=lambda: {
            "loss": [],
            "pde_rms": [],
            "data_rms": [],
            "pde_loss": [],
            "data_loss": [],
            "pde_src_ratio": [],
            "c_history": [],
        }
    )
    _result: object = field(init=False, default=None)

    def log(
        self,
        loss: float,
        residuals,
        pde_src_ratio: float | None = None,
    ) -> None:
        """Log scalar loss and residual diagnostics for one iteration.

        Parameters
        ----------
        loss : float
            Scalar loss value.
        residuals : dict or tuple of torch.Tensor
            Either ``{"pde_rms": ..., "data_rms": ...}`` or a tuple
            ``(r_pde, r_data)`` from which RMS values are derived.
        pde_src_ratio : float, optional
            PDE-to-source RMS ratio to append.

        Notes
        -----
        Only floats are stored; full residual tensors are never retained.
        """
        self.history["loss"].append(float(loss))

        if isinstance(residuals, dict):
            for key, value in residuals.items():
                self.history.setdefault(key, []).append(float(value))

        else:
            r_pde = residuals[0]
            with torch.no_grad():
                if r_pde.is_complex():
                    pde_sq = r_pde.real.square() + r_pde.imag.square()
                else:
                    pde_sq = r_pde.detach().square()
                self.history["pde_rms"].append(float(pde_sq.mean().sqrt().cpu()))
                self.history["pde_loss"].append(float(pde_sq.mean().cpu()))

                if len(residuals) > 1:
                    r_data = residuals[1]
                    if r_data.is_complex():
                        data_sq = r_data.real.square() + r_data.imag.square()
                    else:
                        data_sq = r_data.detach().square()
                    self.history["data_rms"].append(float(data_sq.mean().sqrt().cpu()))
                    self.history["data_loss"].append(float(data_sq.mean().cpu()))

        if pde_src_ratio is not None:
            self.history["pde_src_ratio"].append(float(pde_src_ratio))

    def log_c(self, c_arr: np.ndarray) -> None:
        """Record a snapshot of the full-grid velocity field.

        Parameters
        ----------
        c_arr : np.ndarray
            Full-grid velocity field to snapshot (stored as float32).

        Notes
        -----
        No-op when ``store_c_history`` is False or the iteration count is not
        a multiple of ``c_history_every``.
        """
        if not self.store_c_history:
            return

        n_logged = len(self.history["loss"])
        if n_logged % self.c_history_every != 0:
            return

        self.history["c_history"].append(np.asarray(c_arr, dtype=np.float32).copy())

    @classmethod
    def from_records(cls, records, name: str = "loaded run") -> "LossTape":
        """Rebuild a tape's *scalar* history from saved metric records.

        Parameters
        ----------
        records : sequence of dict
            Per-iteration metric dicts (``metrics.jsonl`` rows) written by
            :class:`odil_wave.experiment.RunRecorder`.
        name : str
            Name for the reconstructed tape.

        Returns
        -------
        LossTape
            Tape carrying the scalar series present in ``records``.

        Notes
        -----
        Lets ``show`` / ``show_velocity_recovery`` run against a finished run
        without re-solving. Full-grid ``c_history`` is not stored per record,
        so it stays empty.
        """
        # (history key, record key) — records use ``loss_total`` for the loss.
        key_map = [
            ("loss", "loss_total"),
            ("pde_rms", "pde_rms"),
            ("data_rms", "data_rms"),
            ("pde_loss", "pde_loss"),
            ("data_loss", "data_loss"),
            ("pde_src_ratio", "pde_src_ratio"),
            ("rel_c_error", "rel_c_error"),
            ("ssim_head_roi", "ssim_head_roi"),
        ]
        tape = cls(name=name, store_c_history=False)
        for hist_key, rec_key in key_map:
            series = [
                (float(r[rec_key]) if r.get(rec_key) is not None else float("nan"))
                for r in records
            ]
            # Keep a scalar series only if at least one record carried it.
            if any(np.isfinite(v) for v in series):
                tape.history[hist_key] = series
        return tape

    def show(self, title: str = "Loss History"):
        """Plot loss and PDE/data RMS residual histories.

        Parameters
        ----------
        title : str
            Figure suptitle.

        Notes
        -----
        Asserts that a loss history exists; the data-RMS panel is shown only
        when a data-residual series was logged.
        """
        assert len(self.history["loss"]) > 0, "No loss history to show."

        has_data = len(self.history.get("data_rms", [])) > 0
        ncols = 3 if has_data else 2
        fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

        axs[0].semilogy(self.history["loss"])
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss Value")

        axs[1].semilogy(self.history["pde_rms"])
        axs[1].set_title("PDE RMS Residual")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("RMS")

        if has_data:
            axs[2].semilogy(self.history["data_rms"])
            axs[2].set_title("Data RMS Residual")
            axs[2].set_xlabel("Iteration")
            axs[2].set_ylabel("RMS")

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
    ) -> str:
        """Render the ``c(x, y)`` snapshots in ``c_history`` to an animated GIF.

        Parameters
        ----------
        grid : Grid
            Grid providing the physical extent for the axes.
        filename : str
            Output GIF path.
        fps : int
            Frames per second.
        cmap : str
            Matplotlib colormap name.
        title : str
            Base plot title.
        norm : matplotlib.colors.Normalize, optional
            Colour normalisation; derived from the data range if omitted.
        frame_labels : sequence of str, optional
            Per-frame labels; the frame index is shown when omitted.

        Returns
        -------
        str
            The written ``filename``.

        Notes
        -----
        Raises ``RuntimeError`` if ``c_history`` is empty and ``ValueError``
        if ``frame_labels`` length does not match the number of frames.
        """
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
        vmin: float = 1400.0,
        vcenter: float = 1600.0,
        vmax: float = 3000.0,
    ):
        """Plot truth, recovered, difference and c-recovery-error panels.

        Parameters
        ----------
        truth : VelocityModel
            Ground-truth velocity model.
        recovered : VelocityModel
            Recovered velocity model.
        geom : AcquisitionGeometry
            Geometry providing source/receiver positions to overlay.
        title : str
            Figure suptitle.
        norm : matplotlib.colors.Normalize, optional
            Colour normalisation for the velocity panels.
        vmin, vcenter, vmax : float
            Two-slope normalisation bounds in m/s used when ``norm`` is None.

        Notes
        -----
        The velocity colourbar splits ``[vmin, vcenter]`` and
        ``[vcenter, vmax]`` across its two halves. The error panel prefers the
        recorded ``rel_c_error`` series and falls back to ``c_history``.
        """
        grid = truth.grid
        c_true_np = truth.c.detach().cpu().numpy()
        c_final_np = recovered.c.detach().cpu().numpy()
        diff = c_final_np - c_true_np

        (xmin, xmax), (ymin, ymax) = grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        dmax = float(np.max(np.abs(diff))) * 1.05 + 1e-9

        rx = grid.x[geom.recv_ij[:, 0]].detach().cpu().numpy() * x_mult
        ry = grid.y[geom.recv_ij[:, 1]].detach().cpu().numpy() * x_mult
        sx = grid.x[geom.src_ij[:, 0]].detach().cpu().numpy() * x_mult
        sy = grid.y[geom.src_ij[:, 1]].detach().cpu().numpy() * x_mult

        if norm is None and vmax > vmin:
            norm = velocity_norm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        panels = [
            (axes[0, 0], c_true_np, "viridis", vmin, vmax, f"truth ({truth.profile})"),
            (axes[0, 1], c_final_np, "viridis", vmin, vmax, "recovered"),
            (axes[1, 0], diff, "RdBu_r", -dmax, dmax, "recovered - truth"),
        ]

        for ax, field_, cmap_, lo, hi, panel_title in panels:
            imshow_kw = dict(
                origin="lower",
                extent=[xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult],
                cmap=cmap_,
            )

            if norm is not None and cmap_ == "viridis":
                imshow_kw["norm"] = norm
            else:
                imshow_kw["vmin"] = lo
                imshow_kw["vmax"] = hi

            im = ax.imshow(field_.T, **imshow_kw)
            ax.set_xlabel(f"x [{x_unit}]")
            ax.set_ylabel(f"y [{x_unit}]")
            ax.set_aspect("equal")
            ax.set_title(panel_title)
            ax.scatter(rx, ry, marker="v", c="lime", edgecolor="black", s=25, zorder=5)
            ax.scatter(sx, sy, marker="*", c="red", edgecolor="black", s=80, zorder=6)
            plt.colorbar(im, ax=ax, shrink=0.85)

        ax_err = axes[1, 1]
        # Prefer the recorded per-iteration relative-error series; otherwise
        # recompute it from any in-memory c_history.
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

    @property
    def result(self):
        """Optional result object attached to this tape.

        Returns
        -------
        object or None
            The stored result, or None if unset.
        """
        return self._result

    @result.setter
    def result(self, value):
        """Attach a result object to this tape.

        Parameters
        ----------
        value : object
            Result to store.
        """
        self._result = value
