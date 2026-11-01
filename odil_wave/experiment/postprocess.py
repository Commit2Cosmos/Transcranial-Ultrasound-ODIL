"""Post-processing: the fixed set of figures and CSVs drawn from a saved run.

:func:`postprocess_run` reads only a run directory (``config_resolved.yaml``,
``tape.csv``, ``problem.npz``, ``bands/*/arrays.npz`` and
``band_summary.json``) and writes PNGs at :data:`DPI` into
``<run_dir>/figures/``, with a CSV beside a figure for every derived number
that is not on the tape. It never builds the problem, solves or optimises. It
runs at the end of every run (:func:`~odil_wave.experiment.runner.run_inverse`,
including failed and interrupted runs, drawing what exists) and by
``main.py --postprocess <run_dir>``.

The set is fixed in code (:data:`RUN_PLOTS`); there is no config section. A
figure is a function ``(run, scales) -> Plot | None`` that returns ``None``
when the run lacks what it needs (e.g. a per-band figure of a one-band run).
Colour scales come from the truth model only (:class:`ColourScales`), so they
are identical across the runs of one experiment.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .config import load_config_file

# Resolution of every saved PNG.
DPI = 150

# The difference scale's half-width, as a fraction of c*'s peak-to-peak in the
# brain region.
DIFF_FRACTION = 0.5

VELOCITY_CMAP = "viridis"
DIFF_CMAP = "RdBu_r"

# Tape columns read as ints; an empty cell is ``None``, the rest floats.
_TAPE_INTS = {"band_index", "iter", "iter_total", "n_evaluations"}

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Reading a run directory
# --------------------------------------------------------------------------- #
def load_tape(run_dir: PathLike) -> List[Dict[str, Any]]:
    """Read ``tape.csv``: one dict per row, column -> value.

    ``run_id`` is a string, ``band_index``, ``iter``, ``iter_total`` and
    ``n_evaluations`` ints, the rest floats; an empty cell is ``None``.
    """
    path = Path(run_dir) / "tape.csv"
    if not path.exists():
        raise FileNotFoundError(f"no tape.csv under {run_dir!r} ({path})")
    with open(path, newline="") as f:
        return [
            {
                k: v
                if k == "run_id"
                else None
                if v == ""
                else (int if k in _TAPE_INTS else float)(v)
                for k, v in row.items()
            }
            for row in csv.DictReader(f)
        ]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@dataclass
class SavedBand:
    """A completed band of a saved run: its summary and (lazily) its arrays."""

    index: int
    label: str
    path: Path
    summary: Dict[str, Any]

    @cached_property
    def arrays(self) -> Dict[str, np.ndarray]:
        """The band's ``arrays.npz``."""
        return _load_npz(self.path / "arrays.npz")

    @property
    def c(self) -> np.ndarray:
        """The interior velocity at the band's final iterate."""
        return self.arrays["c"]


@dataclass
class SavedRun:
    """A read-only view of a run directory, as post-processing sees it.

    ``problem`` is ``problem.npz`` (``None`` if the run stopped before writing
    it) and ``bands`` the completed bands in band order.
    """

    path: Path
    config: Dict[str, Any]
    tape: List[Dict[str, Any]]
    problem: Optional[Dict[str, np.ndarray]]
    bands: List[SavedBand]

    @classmethod
    def load(cls, run_dir: PathLike) -> "SavedRun":
        """Read the run directory ``run_dir``.

        Raises
        ------
        FileNotFoundError
            ``run_dir`` holds no ``config_resolved.yaml``, so it is no run
            directory.
        """
        path = Path(run_dir)
        config_path = path / "config_resolved.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{str(path)!r} is not a run directory (no config_resolved.yaml)"
            )
        problem_path = path / "problem.npz"
        bands = []
        for summary_path in (path / "bands").glob("band_*/band_summary.json"):
            summary = json.loads(summary_path.read_text())
            bands.append(
                SavedBand(
                    index=int(summary["band_index"]),
                    label=str(summary["label"]),
                    path=summary_path.parent,
                    summary=summary,
                )
            )
        return cls(
            path=path,
            config=load_config_file(config_path),
            tape=load_tape(path) if (path / "tape.csv").exists() else [],
            problem=_load_npz(problem_path) if problem_path.exists() else None,
            bands=sorted(bands, key=lambda b: b.index),
        )

    @property
    def interior_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Interior node coordinates ``(x, y)`` in metres."""
        nx, ny = self.config["grid"]["interior_shape"]
        (x0, x1), (y0, y1) = self.config["grid"]["interior_extent"]
        return np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)


# --------------------------------------------------------------------------- #
# Colour scales
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColourScales:
    """Colour scales fixed by the truth model alone.

    Velocity spans ``[min, max]`` of ``c*`` over the interior; the difference
    ``c - c*`` is symmetric at :data:`DIFF_FRACTION` of ``c*``'s
    peak-to-peak in the brain region. Runs of one experiment share a truth, so
    they share these scales.
    """

    c_min: float
    c_max: float
    diff: float

    @classmethod
    def from_truth(cls, c_star: np.ndarray, brain_mask: np.ndarray) -> "ColourScales":
        """The scales of the truth ``c_star`` (interior) and its brain region."""
        return cls(
            c_min=float(np.min(c_star)),
            c_max=float(np.max(c_star)),
            diff=DIFF_FRACTION * float(np.ptp(c_star[brain_mask])),
        )

    @classmethod
    def of_run(cls, run: SavedRun) -> Optional["ColourScales"]:
        """The scales of ``run``'s truth, or ``None`` without ``problem.npz``."""
        if run.problem is None:
            return None
        return cls.from_truth(run.problem["c_star"], run.problem["brain_mask"])

    @property
    def velocity_norm(self) -> Normalize:
        return Normalize(self.c_min, self.c_max)

    @property
    def difference_norm(self) -> Normalize:
        return Normalize(-self.diff, self.diff)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
@dataclass
class Plot:
    """One post-processing output: ``<name>.png`` and, with a ``table``, the
    derived numbers beside it as ``<name>.csv`` (one dict per row)."""

    name: str
    figure: Figure
    table: Optional[List[Dict[str, Any]]] = None


PlotFn = Callable[[SavedRun, Optional[ColourScales]], Optional[Plot]]


def _value(v: Optional[float], log: bool = False) -> float:
    """A tape value for plotting: ``None`` (and non-positive on a log axis) -> nan."""
    if v is None or (log and v <= 0):
        return np.nan
    return v


# Convergence panels: (title, [(column, colour, linestyle)], log y).
_CONVERGENCE_PANELS = [
    (
        "Objective terms",
        [("pde_term", "C0", "-"), ("data_term", "C1", "-"), ("reg_term", "C2", "-")],
        True,
    ),
    (
        "Normalised residuals",
        [
            ("rel_data_residual", "C0", "-"),
            ("pde_src_ratio", "C1", "-"),
            ("rel_wavefield_error", "C2", "-"),
        ],
        True,
    ),
    (
        "Model error [m/s]",
        [
            ("rmse_brain", "C0", "-"),
            ("rmse_skull", "C1", "-"),
            ("bias_brain", "C0", "--"),
            ("bias_skull", "C1", "--"),
        ],
        False,
    ),
    ("Brain-region SSIM", [("ssim_brain", "C0", "-")], False),
]


def convergence(run: SavedRun, scales: Optional[ColourScales] = None) -> Optional[Plot]:
    """Four panels of tape columns against ``iter_total``, with each band's
    start marked: objective terms, normalised residuals (both log), region
    RMSE (solid) and bias (dashed) in m/s, and ``ssim_brain``."""
    if not run.tape:
        return None
    by_band: Dict[int, List[Dict[str, Any]]] = {}
    for row in run.tape:
        by_band.setdefault(row["band_index"], []).append(row)
    boundaries = [rows[0]["iter_total"] for b, rows in sorted(by_band.items()) if b]

    fig = Figure(figsize=(11, 8), layout="constrained")
    for ax, (title, columns, log) in zip(fig.subplots(2, 2).flat, _CONVERGENCE_PANELS):
        for column, colour, style in columns:
            for i, (_, rows) in enumerate(sorted(by_band.items())):
                ax.plot(
                    [r["iter_total"] for r in rows],
                    [_value(r[column], log) for r in rows],
                    style,
                    color=colour,
                    marker=".",
                    label=column if i == 0 else None,
                )
        for x in boundaries:
            ax.axvline(x, color="0.5", linestyle=":", linewidth=1)
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("iter_total")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle(f"Convergence: {run.path.name}")
    return Plot("convergence", fig)


def _draw_map(ax, run: SavedRun, field: np.ndarray, norm: Normalize, cmap: str):
    """``field`` (interior) on ``ax`` in mm, with brain and skull outlines."""
    x, y = (a * 1e3 for a in run.interior_axes)
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        norm=norm,
        cmap=cmap,
    )
    brain = run.problem["brain_mask"]
    head = brain | run.problem["skull_mask"]
    for mask in (head, brain):
        ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors="k", linewidths=0.7)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    return im


def velocity(run: SavedRun, scales: Optional[ColourScales]) -> Optional[Plot]:
    """Truth | recovered (the last completed band) | recovered − truth over the
    interior, with brain and skull outlines."""
    if run.problem is None or not run.bands or scales is None:
        return None
    c_star = run.problem["c_star"]
    final = run.bands[-1]
    fig = Figure(figsize=(15, 4.5), layout="constrained")
    axes = fig.subplots(1, 3)
    panels = [
        ("truth c*", c_star, scales.velocity_norm, VELOCITY_CMAP, "c [m/s]"),
        (
            f"recovered c (band {final.index}: {final.label})",
            final.c,
            scales.velocity_norm,
            VELOCITY_CMAP,
            "c [m/s]",
        ),
        ("c − c*", final.c - c_star, scales.difference_norm, DIFF_CMAP, "Δc [m/s]"),
    ]
    for ax, (title, field, norm, cmap, unit) in zip(axes, panels):
        im = _draw_map(ax, run, field, norm, cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
    return Plot("velocity", fig)


def per_band_error(run: SavedRun, scales: Optional[ColourScales]) -> Optional[Plot]:
    """``c_band − c*`` after each completed band on the shared difference
    scale; only with more than one band."""
    if run.problem is None or len(run.bands) < 2 or scales is None:
        return None
    c_star = run.problem["c_star"]
    n = len(run.bands)
    fig = Figure(figsize=(4.5 * n + 1, 4.5), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, n))
    for ax, band in zip(axes, run.bands):
        im = _draw_map(ax, run, band.c - c_star, scales.difference_norm, DIFF_CMAP)
        ax.set_title(f"band {band.index}: {band.label}")
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="c − c* [m/s]")
    return Plot("per_band_error", fig)


def head_centre(problem: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """Interior index of the head region's centroid (brain plus skull)."""
    head = problem["brain_mask"] | problem["skull_mask"]
    i, j = np.nonzero(head)
    return int(round(i.mean())), int(round(j.mean()))


def profiles(run: SavedRun, scales: Optional[ColourScales] = None) -> Optional[Plot]:
    """Truth, start and recovered ``c`` along the horizontal and vertical lines
    through the head region's centre; the samples go to ``profiles.csv``."""
    if run.problem is None or not run.bands:
        return None
    ic, jc = head_centre(run.problem)
    x, y = run.interior_axes
    fields = {
        "c_star": run.problem["c_star"],
        "c0": run.problem["c0"],
        "c_final": run.bands[-1].c,
    }
    # line -> (slice, x and y of each sample, position along it, axis, title)
    lines = {
        "horizontal": (
            np.s_[:, jc],
            x,
            np.full_like(x, y[jc]),
            x,
            "x [mm]",
            f"y = {y[jc] * 1e3:.1f} mm",
        ),
        "vertical": (
            np.s_[ic, :],
            np.full_like(y, x[ic]),
            y,
            y,
            "y [mm]",
            f"x = {x[ic] * 1e3:.1f} mm",
        ),
    }
    table: List[Dict[str, Any]] = []
    fig = Figure(figsize=(12, 4.5), layout="constrained")
    for ax, (line, spec) in zip(fig.subplots(1, 2), lines.items()):
        sl, xs, ys, pos, xlabel, where = spec
        for name, style in (("c_star", "k-"), ("c0", "C7--"), ("c_final", "C0-")):
            ax.plot(pos * 1e3, fields[name][sl], style, label=name)
        for k in range(len(pos)):
            table.append(
                {
                    "line": line,
                    "x_m": float(xs[k]),
                    "y_m": float(ys[k]),
                    **{name: float(f[sl][k]) for name, f in fields.items()},
                }
            )
        ax.set_title(f"{line} profile through the head centre ({where})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("c [m/s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    return Plot("profiles", fig, table)


# Band contribution metrics: the change from a band's start row to its final row.
CONTRIBUTION_METRICS = ["rmse_brain", "rmse_skull", "ssim_brain", "rel_data_residual"]


def band_contribution(
    run: SavedRun, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """Per completed band, the change in :data:`CONTRIBUTION_METRICS` from its
    start row to its final row, and its cost (``n_evaluations``, ``wall_s``);
    only with more than one band."""
    if len(run.bands) < 2:
        return None
    table = []
    for band in run.bands:
        start, final = band.summary["start_metrics"], band.summary["final_metrics"]
        table.append(
            {
                "band_index": band.index,
                "label": band.label,
                **{f"delta_{m}": final[m] - start[m] for m in CONTRIBUTION_METRICS},
                "n_evaluations": band.summary["optimiser_result"]["n_evaluations"],
                "wall_s": band.summary["wall_s"],
            }
        )
    idx = [b.index for b in run.bands]
    fig = Figure(figsize=(13, 7), layout="constrained")
    axes = list(fig.subplots(2, 3).flat)
    for ax, metric in zip(axes, CONTRIBUTION_METRICS):
        for i, band in zip(idx, run.bands):
            start = band.summary["start_metrics"][metric]
            final = band.summary["final_metrics"][metric]
            ax.annotate(
                "",
                xy=(i, final),
                xytext=(i, start),
                arrowprops=dict(arrowstyle="->", color="C0"),
            )
        ax.plot(
            idx,
            [b.summary["start_metrics"][metric] for b in run.bands],
            "o",
            mfc="none",
            color="C0",
            label="start row",
        )
        ax.plot(
            idx,
            [b.summary["final_metrics"][metric] for b in run.bands],
            "o",
            color="C0",
            label="final row",
        )
        ax.set_title(metric)
        ax.legend(fontsize="small")
    for ax, cost in zip(axes[len(CONTRIBUTION_METRICS) :], ["n_evaluations", "wall_s"]):
        values = [row[cost] for row in table]
        ax.plot(idx, values, "s", color="C1")
        ax.set_title(f"cost: {cost}")
        ax.set_ylim(0, 1.15 * max(values) or 1.0)
    for ax in fig.axes:
        ax.set_xticks(idx, [f"{b.index}: {b.label}" for b in run.bands])
        ax.set_xlim(idx[0] - 0.5, idx[-1] + 0.5)
        ax.set_xlabel("band")
        ax.grid(alpha=0.3)
    fig.suptitle("Band contribution: start row → final row, and cost")
    return Plot("band_contribution", fig, table)


# The per-run set, in drawing order.
RUN_PLOTS: List[PlotFn] = [
    convergence,
    velocity,
    per_band_error,
    profiles,
    band_contribution,
]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def save_plot(plot: Plot, out_dir: Path) -> List[Path]:
    """Write ``plot`` as ``<name>.png`` (and ``<name>.csv``) into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{plot.name}.png"
    plot.figure.savefig(png, dpi=DPI)
    written = [png]
    if plot.table:
        path = out_dir / f"{plot.name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(plot.table[0]))
            writer.writeheader()
            writer.writerows(plot.table)
        written.append(path)
    return written


def postprocess_run(run_dir: PathLike) -> List[Path]:
    """Draw the per-run figure set of ``run_dir`` into ``<run_dir>/figures/``.

    Reads only the run directory and draws what it holds: a run that stopped
    early gets the figures its saved bands support. Returns the written paths.
    """
    run = SavedRun.load(run_dir)
    scales = ColourScales.of_run(run)
    written: List[Path] = []
    for draw in RUN_PLOTS:
        plot = draw(run, scales)
        if plot is not None:
            written += save_plot(plot, run.path / "figures")
    return written
