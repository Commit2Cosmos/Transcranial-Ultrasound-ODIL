from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from odil_wave.grid import Grid
from odil_wave.loss import LossTape
from odil_wave.models import VelocityModel, velocity_norm
from odil_wave.plot_utils import length_scale

from .config import RunConfig, load_config_file
from .postprocess import load_tape as load_metrics
from .problem import Problem, build_problem


RunDirOrRecords = Union[str, Path, List[dict]]


# --------------------------------------------------------------------------- #
# Rebuilding the physical problem from a run's saved config
# --------------------------------------------------------------------------- #
def load_config(run_dir: Union[str, Path]) -> RunConfig:
    """Reconstruct the resolved :class:`RunConfig` saved alongside a run."""
    path = Path(run_dir) / "config_resolved.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no config_resolved.yaml under {run_dir!r} ({path}); "
            "cannot rebuild the problem from this run directory"
        )
    return RunConfig.from_dict(load_config_file(path))


def rebuild_problem(run_dir: Union[str, Path]) -> Problem:
    """Rebuild the physical :class:`Problem` from a run's saved config."""
    return build_problem(load_config(run_dir))


def _resolve_grid_pml(
    run_dir: Union[str, Path], grid: Optional[Grid], pml_c: Optional[float]
) -> Tuple[Grid, Optional[float]]:
    """Fill a missing ``grid`` from the run's rebuilt problem and a missing
    ``pml_c`` with the run's frozen PML speed (in ``problem.npz``).

    Rebuilds the problem only when the grid is missing.
    """
    if pml_c is None:
        with np.load(Path(run_dir) / "problem.npz") as saved:
            pml_c = float(saved["pml_c"])
    if grid is None:
        grid = rebuild_problem(run_dir).grid
    return grid, pml_c


# --------------------------------------------------------------------------- #
# Loading saved artifacts
# --------------------------------------------------------------------------- #
def _records(source: RunDirOrRecords) -> List[dict]:
    """Accept either a run directory or an already-loaded list of records."""
    if isinstance(source, (str, Path)):
        return load_metrics(source)
    return list(source)


def load_final_velocity(
    run_dir: Union[str, Path],
    grid: Optional[Grid] = None,
    pml_c: Optional[float] = None,
) -> VelocityModel:
    """Load the last completed band's final model (the run's final model) into
    a :class:`VelocityModel`.

    ``grid`` and ``pml_c`` default to the ones rebuilt from the run's saved
    config, so only the run directory is required; pass them to reuse a
    ``Problem`` you already hold.
    """
    bands = load_band_velocities(run_dir, grid, pml_c)
    if not bands:
        raise FileNotFoundError(f"no completed band under {run_dir!r}")
    return bands[-1][2]


def load_band_velocities(
    run_dir: Union[str, Path],
    grid: Optional[Grid] = None,
    pml_c: Optional[float] = None,
) -> List[Tuple[int, str, VelocityModel]]:
    """Load every completed band's final model, in band order.

    Each band's ``arrays.npz`` holds the interior ``c``; the full-grid model is
    it padded with ``pml_c``. Returns ``[(band_index, label, VelocityModel),
    ...]`` sorted by band index. ``grid`` defaults to the one rebuilt from the
    run's saved config and ``pml_c`` to the run's frozen PML speed (in
    ``problem.npz``), so only the run directory is required.
    """
    grid, pml_c = _resolve_grid_pml(run_dir, grid, pml_c)
    bands_dir = Path(run_dir) / "bands"
    if not bands_dir.exists():
        raise FileNotFoundError(f"no bands/ directory under {run_dir!r}")
    out: List[Tuple[int, str, VelocityModel]] = []
    for summary_path in sorted(bands_dir.glob("band_*/band_summary.json")):
        summary = json.loads(summary_path.read_text())
        with np.load(summary_path.with_name("arrays.npz")) as saved:
            c = torch.from_numpy(saved["c"]).to(grid.dtype)
        p = grid.pml_width
        c_full = torch.nn.functional.pad(c, (p, p, p, p), value=float(pml_c))
        out.append(
            (
                int(summary["band_index"]),
                str(summary["label"]),
                VelocityModel.from_field(grid, c_full, pml_c=pml_c),
            )
        )
    out.sort(key=lambda t: t[0])
    return out


def load_loss_tape(source: RunDirOrRecords, name: str = "loaded run") -> LossTape:
    """Reconstruct a :class:`LossTape` (scalar history) from a saved run.

    The returned tape supports ``tape.show()`` and
    ``tape.show_velocity_recovery(...)`` exactly as during a live run.
    """
    return LossTape.from_records(_records(source), name=name)


# --------------------------------------------------------------------------- #
# Live per-band monitor (callback for run_inverse(on_band_end=...))
# --------------------------------------------------------------------------- #
class LiveVelocityView:
    """Redraw the recovered velocity after each band so a run can be stopped early.

    Pass an instance as ``run_inverse(..., on_band_end=view)``. After every
    completed band it clears the current cell output (in Jupyter) and draws two
    panels: the newly recovered velocity, and the brain-region SSIM / RMSE
    trend across the bands finished so far — so a stalled inversion is
    obvious and the kernel can be interrupted.

    Parameters
    ----------
    truth:
        Optional ground-truth model; only its ``grid`` is used when ``grid`` is
        not given. The recovered field is always drawn from the band result.
    grid:
        Grid for the spatial axes (defaults to ``truth.grid`` or the band
        model's grid).
    vmin, vcenter, vmax:
        Two-slope velocity colour scaling (defaults 1400 / 1600 / 3000 m/s).
        Set ``vmax <= vmin`` to disable and use matplotlib's autoscaling.
    """

    def __init__(
        self,
        truth: Optional[VelocityModel] = None,
        grid: Optional[Grid] = None,
        *,
        vmin: float = 1400.0,
        vcenter: float = 1600.0,
        vmax: float = 3000.0,
        figsize: Tuple[float, float] = (12, 4.5),
    ) -> None:
        """Configure the live per-band velocity monitor.

        Parameters
        ----------
        truth : VelocityModel, optional
            Ground-truth model; only its ``grid`` is used when ``grid`` is not
            given.
        grid : Grid, optional
            Grid for the spatial axes (defaults to ``truth.grid`` or the band
            model's grid).
        vmin, vcenter, vmax : float, optional
            Two-slope velocity colour scaling; ``vmax <= vmin`` disables it and
            uses matplotlib autoscaling.
        figsize : tuple of float, optional
            Figure size for the two-panel view.
        """
        self.truth = truth
        self.grid = (
            grid if grid is not None else (truth.grid if truth is not None else None)
        )
        self.norm = velocity_norm(vmin, vcenter, vmax) if vmax > vmin else None
        self.figsize = figsize
        self._labels: List[str] = []
        self._ssim: List[Optional[float]] = []
        self._rmse: List[Optional[float]] = []

    def __call__(self, band_result, run_result: Any = None) -> None:
        """Redraw the recovered velocity and progress trend after a band.

        Parameters
        ----------
        band_result : BandResult
            The freshly completed band's result.
        run_result : Any, optional
            The overall run result (unused; accepted for the callback signature).

        Notes
        -----
        Intended as the ``run_inverse(..., on_band_end=...)`` callback; clears
        the current cell output in Jupyter before drawing.
        """
        vm = band_result.velocity_model
        grid = self.grid if self.grid is not None else vm.grid
        c = vm.c.detach().cpu().numpy()

        metrics = band_result.final_metrics or {}
        self._labels.append(f"{band_result.band_index}:{band_result.label}")
        self._ssim.append(metrics.get("ssim_brain"))
        self._rmse.append(metrics.get("rmse_brain"))

        try:  # only meaningful inside a notebook; harmless otherwise
            from IPython.display import clear_output

            clear_output(wait=True)
        except Exception:
            pass

        fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=self.figsize)
        self._draw_velocity(ax_c, c, grid, band_result)
        self._draw_trend(ax_m)
        fig.suptitle("Live inversion progress — interrupt the kernel to stop early")
        fig.tight_layout()
        plt.show()

    def _draw_velocity(self, ax, c, grid, band_result) -> None:
        """Draw the recovered velocity map for one band onto ``ax``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on.
        c : numpy.ndarray
            Recovered velocity field.
        grid : Grid
            Grid providing the physical extent.
        band_result : BandResult
            Band result supplying the index / label / SSIM for the title.

        """
        (xmin, xmax), (ymin, ymax) = grid.extent
        x_mult, x_unit = length_scale(max(abs(xmax), abs(ymax)))
        imshow_kw = dict(
            origin="lower",
            extent=(xmin * x_mult, xmax * x_mult, ymin * x_mult, ymax * x_mult),
            cmap="viridis",
        )
        if self.norm is not None:
            imshow_kw["norm"] = self.norm
        im = ax.imshow(c.T, **imshow_kw)
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"y [{x_unit}]")
        ax.set_aspect("equal")
        ssim = (band_result.final_metrics or {}).get("ssim_brain")
        ssim_s = f"  SSIM={ssim:.3f}" if isinstance(ssim, (int, float)) else ""
        ax.set_title(
            f"recovered c — band {band_result.band_index} ({band_result.label}){ssim_s}"
        )
        plt.colorbar(im, ax=ax, shrink=0.85, label="c [m/s]")

    def _draw_trend(self, ax) -> None:
        """Draw the per-band ssim_brain and rmse_brain trends onto ``ax``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on (a twin y-axis is added for the RMSE series).

        Notes
        -----
        Shows a placeholder message when no ground-truth metrics are available.
        """
        x = list(range(len(self._labels)))
        drew = False
        if any(s is not None for s in self._ssim):
            ax.plot(
                x,
                [np.nan if s is None else s for s in self._ssim],
                "o-",
                color="tab:green",
                label="ssim_brain",
            )
            ax.set_ylabel("ssim_brain", color="tab:green")
            ax.tick_params(axis="y", labelcolor="tab:green")
            drew = True
        if any(e is not None for e in self._rmse):
            ax_r = ax.twinx()
            ax_r.plot(
                x,
                [np.nan if e is None else e for e in self._rmse],
                "s--",
                color="tab:red",
                label="rmse_brain [m/s]",
            )
            ax_r.set_ylabel("rmse_brain [m/s]", color="tab:red")
            ax_r.tick_params(axis="y", labelcolor="tab:red")
            drew = True
        ax.set_xticks(x)
        ax.set_xticklabels(self._labels, rotation=45, ha="right")
        ax.set_xlabel("completed band")
        ax.set_title("progress per band")
        ax.grid(alpha=0.3)
        if not drew:
            ax.text(
                0.5,
                0.5,
                "no ground-truth metrics",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
