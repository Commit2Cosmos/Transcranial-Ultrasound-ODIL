"""Load-and-plot helpers over a saved run directory (the ``RunRecorder`` scheme).

Everything here reads the artifacts written by
:func:`odil_wave.experiment.run_inverse` — ``metrics.jsonl``,
``bands/band_XX/c_final.npy`` and ``final/c_final.npy`` — so the standard
diagnostics can be reproduced from disk without re-running the solve. It reuses
the library plotting already defined on :class:`odil_wave.loss.LossTape` and in
:mod:`odil_wave.metrics`, rather than re-deriving matplotlib code per notebook.

Live monitoring *during* a run is provided by :class:`LiveVelocityView`, which
is passed to ``run_inverse(..., on_band_end=...)`` and redraws the recovered
velocity after every completed frequency band.
"""

from __future__ import annotations

import collections
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


RunDirOrRecords = Union[str, Path, List[dict]]


# --------------------------------------------------------------------------- #
# Loading saved artifacts
# --------------------------------------------------------------------------- #
def load_metrics(run_dir: Union[str, Path]) -> List[dict]:
    """Read ``metrics.jsonl`` (one dict per logged iteration) from a run dir."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.jsonl under {run_dir!r} ({path})")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _records(source: RunDirOrRecords) -> List[dict]:
    """Accept either a run directory or an already-loaded list of records."""
    if isinstance(source, (str, Path)):
        return load_metrics(source)
    return list(source)


def load_final_velocity(
    run_dir: Union[str, Path], grid: Grid, pml_c: Optional[float] = None
) -> VelocityModel:
    """Load ``final/c_final.npy`` into a :class:`VelocityModel` on ``grid``."""
    path = Path(run_dir) / "final" / "c_final.npy"
    if not path.exists():
        raise FileNotFoundError(f"no final velocity at {path}")
    c = torch.from_numpy(np.load(path))
    return VelocityModel.from_field(grid, c, pml_c=pml_c)


def load_band_velocities(
    run_dir: Union[str, Path], grid: Grid, pml_c: Optional[float] = None
) -> List[Tuple[int, str, VelocityModel]]:
    """Load every ``bands/band_XX/c_final.npy`` in band order.

    Returns ``[(band_index, label, VelocityModel), ...]`` sorted by band index.
    """
    bands_dir = Path(run_dir) / "bands"
    if not bands_dir.exists():
        raise FileNotFoundError(f"no bands/ directory under {run_dir!r}")
    out: List[Tuple[int, str, VelocityModel]] = []
    for c_path in sorted(bands_dir.glob("band_*/c_final.npy")):
        meta_path = c_path.with_name("c_final_metadata.json")
        band_index, label = _band_meta(meta_path, fallback_dir=c_path.parent.name)
        c = torch.from_numpy(np.load(c_path))
        out.append((band_index, label, VelocityModel.from_field(grid, c, pml_c=pml_c)))
    out.sort(key=lambda t: t[0])
    return out


def _band_meta(meta_path: Path, fallback_dir: str) -> Tuple[int, str]:
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return int(meta.get("band_index", 0)), str(meta.get("label", fallback_dir))
    # Fall back to parsing "band_03_60khz".
    parts = fallback_dir.split("_", 2)
    band_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    label = parts[2] if len(parts) > 2 else fallback_dir
    return band_index, label


def load_loss_tape(source: RunDirOrRecords, name: str = "loaded run") -> LossTape:
    """Reconstruct a :class:`LossTape` (scalar history) from a saved run.

    The returned tape supports ``tape.show()`` and
    ``tape.show_velocity_recovery(...)`` exactly as during a live run.
    """
    return LossTape.from_records(_records(source), name=name)


# --------------------------------------------------------------------------- #
# Static diagnostics
# --------------------------------------------------------------------------- #
def plot_run_history(
    source: RunDirOrRecords,
    *,
    title: str = "Run history",
    x: str = "global_iter",
):
    """Per-band loss and head-ROI SSIM vs iteration (from ``metrics.jsonl``).

    ``x`` selects the horizontal axis: ``"global_iter"`` (default, a continuous
    run-wide view) or ``"solver_iter"`` (each band starts at 0, overlaid).
    """
    records = _records(source)
    if not records:
        raise ValueError("no records to plot")
    by_band = collections.defaultdict(list)
    for r in records:
        by_band[r.get("band_index", 0)].append(r)

    fig, (ax_loss, ax_ssim) = plt.subplots(1, 2, figsize=(12, 4))
    any_ssim = False
    for bi, recs in sorted(by_band.items()):
        it = [r.get(x) for r in recs]
        ax_loss.semilogy(
            it, [r.get("loss_total") for r in recs], "o-", ms=3, label=f"band {bi}"
        )
        ssim_vals = [r.get("ssim_head_roi") for r in recs]
        if any(s is not None for s in ssim_vals):
            any_ssim = True
            ax_ssim.plot(
                it,
                [np.nan if s is None else s for s in ssim_vals],
                "o-",
                ms=3,
                label=f"band {bi}",
            )

    ax_loss.set_xlabel(x)
    ax_loss.set_ylabel("loss_total")
    ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_ssim.set_xlabel(x)
    ax_ssim.set_ylabel("ssim_head_roi")
    ax_ssim.set_title("Head-ROI SSIM")
    ax_ssim.grid(alpha=0.3)
    if any_ssim:
        ax_ssim.legend()
    else:
        ax_ssim.text(
            0.5,
            0.5,
            "no SSIM (no ground truth)",
            ha="center",
            va="center",
            transform=ax_ssim.transAxes,
        )

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return fig


def plot_velocity_recovery(
    run_dir: Union[str, Path],
    truth: VelocityModel,
    geom,
    *,
    grid: Optional[Grid] = None,
    title: str = "Velocity recovery",
    **kwargs,
):
    """Truth | recovered | difference | recovery-error, from saved artifacts.

    Loads ``final/c_final.npy`` and the ``rel_c_error`` series and delegates to
    :meth:`odil_wave.loss.LossTape.show_velocity_recovery`. Extra ``kwargs``
    (``vmin`` / ``vcenter`` / ``vmax`` / ``norm``) are forwarded to it.
    """
    grid = grid if grid is not None else truth.grid
    recovered = load_final_velocity(run_dir, grid, pml_c=truth.pml_c)
    tape = load_loss_tape(run_dir)
    tape.show_velocity_recovery(truth, recovered, geom, title=title, **kwargs)
    return recovered


def animate_bands(
    run_dir: Union[str, Path],
    grid: Grid,
    *,
    filename: str = "c_bands.gif",
    pml_c: Optional[float] = None,
    fps: int = 2,
    title: str = "Recovered c(x, y) across bands",
    norm=None,
) -> str:
    """Animate the per-band recovered velocity (one frame per completed band).

    Reuses :meth:`odil_wave.loss.LossTape.animate_c` with the saved
    ``bands/band_XX/c_final.npy`` snapshots as frames.
    """
    bands = load_band_velocities(run_dir, grid, pml_c=pml_c)
    if not bands:
        raise RuntimeError(f"no per-band velocities found under {run_dir!r}")
    tape = LossTape(store_c_history=True)
    labels: List[str] = []
    for band_index, label, vm in bands:
        tape.history["c_history"].append(vm.c.detach().cpu().numpy())
        labels.append(f"band {band_index}: {label}")
    return tape.animate_c(
        grid,
        filename=filename,
        fps=fps,
        title=title,
        norm=norm,
        frame_labels=labels,
    )


# --------------------------------------------------------------------------- #
# Live per-band monitor (callback for run_inverse(on_band_end=...))
# --------------------------------------------------------------------------- #
class LiveVelocityView:
    """Redraw the recovered velocity after each band so a run can be stopped early.

    Pass an instance as ``run_inverse(..., on_band_end=view)``. After every
    completed band it clears the current cell output (in Jupyter) and draws two
    panels: the newly recovered velocity, and the head-ROI SSIM / relative
    c-error trend across the bands finished so far — so a stalled inversion is
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
        self.truth = truth
        self.grid = (
            grid if grid is not None else (truth.grid if truth is not None else None)
        )
        self.norm = velocity_norm(vmin, vcenter, vmax) if vmax > vmin else None
        self.figsize = figsize
        self._labels: List[str] = []
        self._ssim: List[Optional[float]] = []
        self._rel_err: List[Optional[float]] = []

    def __call__(self, band_result, run_result: Any = None) -> None:
        vm = band_result.velocity_model
        grid = self.grid if self.grid is not None else vm.grid
        c = vm.c.detach().cpu().numpy()

        metrics = band_result.final_metrics or {}
        self._labels.append(f"{band_result.band_index}:{band_result.label}")
        self._ssim.append(metrics.get("ssim_head_roi"))
        self._rel_err.append(metrics.get("rel_c_error"))

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
        ssim = (band_result.final_metrics or {}).get("ssim_head_roi")
        ssim_s = f"  SSIM={ssim:.3f}" if isinstance(ssim, (int, float)) else ""
        ax.set_title(
            f"recovered c — band {band_result.band_index} ({band_result.label}){ssim_s}"
        )
        plt.colorbar(im, ax=ax, shrink=0.85, label="c [m/s]")

    def _draw_trend(self, ax) -> None:
        x = list(range(len(self._labels)))
        drew = False
        if any(s is not None for s in self._ssim):
            ax.plot(
                x,
                [np.nan if s is None else s for s in self._ssim],
                "o-",
                color="tab:green",
                label="SSIM (head ROI)",
            )
            ax.set_ylabel("SSIM (head ROI)", color="tab:green")
            ax.tick_params(axis="y", labelcolor="tab:green")
            drew = True
        if any(e is not None for e in self._rel_err):
            ax_r = ax.twinx()
            ax_r.plot(
                x,
                [np.nan if e is None else e for e in self._rel_err],
                "s--",
                color="tab:red",
                label="rel c-error",
            )
            ax_r.set_ylabel("rel c-error", color="tab:red")
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
