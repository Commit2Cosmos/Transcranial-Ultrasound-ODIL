"""Experiment figures: the completed runs of an experiment compared.

:func:`postprocess_experiment_figures` reads only an experiment directory
(``experiment.json`` and each completed run's directory) and writes PNGs at
:data:`~odil_wave.experiment.postprocess.DPI` into ``<experiment_dir>/figures/``,
with a CSV beside a figure for every derived number that is not on a tape. It
draws only with more than one completed run: a failed, interrupted or pending
run has no final iterate to compare. Launch post-processing calls it after
``summary.csv`` (``launch.postprocess_experiment_level``).

As with the per-run set (:mod:`~odil_wave.experiment.postprocess`), the set is
fixed in code (:data:`EXPERIMENT_PLOTS`), each figure a function
``(experiment, scales) -> Plot | None``, and colour scales come from the truth
alone, which every run of an experiment shares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from matplotlib.figure import Figure

from .postprocess import (
    DIFF_CMAP,
    ColourScales,
    PathLike,
    Plot,
    SavedRun,
    _draw_map,
    _value,
    head_centre,
    save_plot,
)


@dataclass
class ComparedRun:
    """A completed run of an experiment, as its ``experiment.json`` entry
    names it, with its saved run directory."""

    label: str
    value: Any
    is_baseline: bool
    run: SavedRun

    @property
    def final(self) -> Dict[str, Any]:
        """The run's final tape row."""
        return self.run.tape[-1]

    @property
    def c(self) -> np.ndarray:
        """The interior velocity at the run's final iterate."""
        return self.run.bands[-1].c


@dataclass
class SavedExperiment:
    """A read-only view of an experiment directory: its record and its
    completed runs in ``values`` order."""

    path: Path
    record: Dict[str, Any]
    runs: List[ComparedRun]

    @classmethod
    def load(cls, exp_dir: PathLike) -> "SavedExperiment":
        """Read the experiment directory ``exp_dir``."""
        path = Path(exp_dir)
        record = json.loads((path / "experiment.json").read_text())
        runs = [
            ComparedRun(
                label=entry["label"],
                value=entry["value"],
                is_baseline=bool(entry["is_baseline"]),
                run=SavedRun.load(path / entry["dir"]),
            )
            for entry in record["runs"]
            if entry["status"] == "completed"
        ]
        return cls(path=path, record=record, runs=runs)

    @property
    def sweep_key(self) -> str:
        return self.record["sweep_key"]

    @property
    def start_row(self) -> Dict[str, Any]:
        """The reference start row: band 0's start row of the baseline, or of
        the first run if the baseline didn't complete."""
        reference = next((r for r in self.runs if r.is_baseline), self.runs[0])
        return reference.run.tape[0]

    @property
    def problem(self) -> Optional[Dict[str, np.ndarray]]:
        """``problem.npz`` of the first run (every run shares the truth)."""
        return self.runs[0].run.problem


ExperimentPlotFn = Callable[[SavedExperiment, Optional[ColourScales]], Optional[Plot]]


def _numeric(values: List[Any]) -> bool:
    """Whether the swept values are numbers (``bool`` and ``None`` are not)."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _value_axis(ax, values: List[float]) -> None:
    """A log-like x axis when the nonzero values span a decade or more:
    ``log`` when all are positive, else ``symlog`` linear up to the smallest
    nonzero magnitude (so a swept ``0`` shows)."""
    nonzero = [abs(v) for v in values if v != 0]
    if len(nonzero) < 2 or max(nonzero) < 10 * min(nonzero):
        return
    if all(v > 0 for v in values):
        ax.set_xscale("log")
    else:
        ax.set_xscale("symlog", linthresh=min(nonzero))


# Final metrics against the swept value: tape column and log y.
FINAL_METRICS = [
    ("rmse_brain", False),
    ("rmse_skull", False),
    ("ssim_brain", False),
    ("rel_data_residual", True),
    ("wall_s", False),
]


def final_metrics(
    exp: SavedExperiment, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """Final :data:`FINAL_METRICS` against the swept value, the reference
    start row a dashed line; a dot plot over the run labels when the value is
    categorical."""
    values = [r.value for r in exp.runs]
    numeric = _numeric(values)
    if numeric:
        order = sorted(range(len(values)), key=lambda k: values[k])
        xs = [float(values[k]) for k in order]
    else:
        order = list(range(len(values)))
        xs = [float(k) for k in order]
    fig = Figure(figsize=(15, 7.5), layout="constrained")
    axes = list(fig.subplots(2, 3).flat)
    for ax, (column, log) in zip(axes, FINAL_METRICS):
        ys = [_value(exp.runs[k].final[column], log) for k in order]
        ax.plot(xs, ys, "o-" if numeric else "o", color="C0", label="final")
        start = exp.start_row[column]
        if start is not None:
            ax.axhline(
                _value(start, log), color="0.4", linestyle="--", label="start row"
            )
        if log:
            ax.set_yscale("log")
        if numeric:
            _value_axis(ax, xs)
        else:
            ax.set_xticks(xs, [exp.runs[k].label for k in order], rotation=30)
            ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_title(column)
        ax.set_xlabel(exp.sweep_key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    axes[-1].remove()
    fig.suptitle(f"Final metrics against {exp.sweep_key}")
    return Plot("final_metrics", fig)


# Convergence overlay panels: tape column and log y.
OVERLAY_METRICS = [
    ("rmse_brain", False),
    ("rmse_skull", False),
    ("rel_data_residual", True),
]


def convergence_overlay(
    exp: SavedExperiment, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """:data:`OVERLAY_METRICS` against ``n_evaluations`` (the fair cost
    axis), one line per run."""
    fig = Figure(figsize=(15, 4.5), layout="constrained")
    for ax, (column, log) in zip(fig.subplots(1, 3), OVERLAY_METRICS):
        for k, r in enumerate(exp.runs):
            ax.plot(
                [_value(row["n_evaluations"]) for row in r.run.tape],
                [_value(row[column], log) for row in r.run.tape],
                marker=".",
                color=f"C{k % 10}",
                label=r.label,
            )
        if log:
            ax.set_yscale("log")
        ax.set_title(column)
        ax.set_xlabel("n_evaluations")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle("Convergence against n_evaluations")
    return Plot("convergence_overlay", fig)


def difference_maps(
    exp: SavedExperiment, scales: Optional[ColourScales]
) -> Optional[Plot]:
    """``c − c*`` at each run's final iterate, one panel per run, on the shared
    difference scale."""
    if exp.problem is None or scales is None:
        return None
    c_star = exp.problem["c_star"]
    n = len(exp.runs)
    cols = min(n, 4)
    rows = -(-n // cols)
    fig = Figure(figsize=(4.5 * cols + 1, 4.5 * rows + 0.8), layout="constrained")
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax, r in zip(axes, exp.runs):
        im = _draw_map(ax, r.run, r.c - c_star, scales.difference_norm, DIFF_CMAP)
        ax.set_title(r.label)
    for ax in axes[n:]:
        ax.remove()
    fig.colorbar(im, ax=list(axes[:n]), shrink=0.85, label="c − c* [m/s]")
    fig.suptitle("Recovered − truth")
    return Plot("difference_maps", fig)


def profile_overlay(
    exp: SavedExperiment, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """Truth and each run's final ``c`` along the horizontal line through the
    head region's centre; the samples go to ``profile_overlay.csv``."""
    if exp.problem is None:
        return None
    _, jc = head_centre(exp.problem)
    x, y = exp.runs[0].run.interior_axes
    c_star = exp.problem["c_star"][:, jc]
    fig = Figure(figsize=(9, 5), layout="constrained")
    ax = fig.subplots()
    ax.plot(x * 1e3, c_star, "k-", linewidth=2, label="c*")
    table: List[Dict[str, Any]] = []
    for k, r in enumerate(exp.runs):
        c = r.c[:, jc]
        ax.plot(x * 1e3, c, color=f"C{k % 10}", label=r.label)
        table += [
            {
                "label": r.label,
                "x_m": float(x[i]),
                "y_m": float(y[jc]),
                "c_star": float(c_star[i]),
                "c": float(c[i]),
            }
            for i in range(len(x))
        ]
    ax.set_title(
        f"Horizontal profile through the head centre (y = {y[jc] * 1e3:.1f} mm)"
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("c [m/s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    return Plot("profile_overlay", fig, table)


def accuracy_cost(
    exp: SavedExperiment, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """Final ``rmse_brain`` against total ``n_evaluations``, one labelled point
    per run: the best runs sit on the lower-left front."""
    fig = Figure(figsize=(7, 5.5), layout="constrained")
    ax = fig.subplots()
    xs = [_value(r.final["n_evaluations"]) for r in exp.runs]
    ys = [_value(r.final["rmse_brain"]) for r in exp.runs]
    ax.plot(xs, ys, "o", color="C0")
    ax.margins(0.12)  # room for the labels
    for x, y, r in zip(xs, ys, exp.runs):
        ax.annotate(r.label, (x, y), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("n_evaluations")
    ax.set_ylabel("final rmse_brain [m/s]")
    ax.set_title("Accuracy against cost")
    ax.grid(alpha=0.3)
    return Plot("accuracy_cost", fig)


# The per-experiment set, in drawing order.
EXPERIMENT_PLOTS: List[ExperimentPlotFn] = [
    final_metrics,
    convergence_overlay,
    difference_maps,
    profile_overlay,
    accuracy_cost,
]


def postprocess_experiment_figures(exp_dir: PathLike) -> List[Path]:
    """Draw the experiment figure set of ``exp_dir`` into
    ``<exp_dir>/figures/``, only with more than one completed run. Returns the
    written paths."""
    exp = SavedExperiment.load(exp_dir)
    if len(exp.runs) < 2:
        return []
    scales = ColourScales.of_run(exp.runs[0].run)
    written: List[Path] = []
    for draw in EXPERIMENT_PLOTS:
        plot = draw(exp, scales)
        if plot is not None:
            written += save_plot(plot, exp.path / "figures")
    return written
