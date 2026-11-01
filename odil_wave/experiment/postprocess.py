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
from .rundir import load_versioned_json

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


def load_run_config(run_dir: PathLike) -> Dict[str, Any]:
    """Read ``config_resolved.yaml``, the run's Resolved config, as a dict.

    Raises
    ------
    FileNotFoundError
        ``run_dir`` holds no ``config_resolved.yaml``, so it is no run
        directory.
    """
    path = Path(run_dir) / "config_resolved.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{str(run_dir)!r} is not a run directory (no config_resolved.yaml)"
        )
    return load_config_file(path)


def load_bands(run_dir: PathLike) -> List["SavedBand"]:
    """The run's completed bands (each with a ``band_summary.json``), in band
    order; their arrays are read lazily."""
    bands = []
    for summary_path in Path(run_dir).glob("bands/band_*/band_summary.json"):
        summary = json.loads(summary_path.read_text())
        bands.append(
            SavedBand(
                index=int(summary["band_index"]),
                label=str(summary["label"]),
                path=summary_path.parent,
                summary=summary,
            )
        )
    return sorted(bands, key=lambda b: b.index)


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
        SchemaVersionError
            Its ``metadata.json`` is of a format this code can't read.
        """
        path = Path(run_dir)
        config = load_run_config(path)
        if (path / "metadata.json").exists():  # a run may stop before writing it
            load_versioned_json(path / "metadata.json")
        problem_path = path / "problem.npz"
        return cls(
            path=path,
            config=config,
            tape=load_tape(path) if (path / "tape.csv").exists() else [],
            problem=_load_npz(problem_path) if problem_path.exists() else None,
            bands=load_bands(path),
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


def tape_value(v: Optional[float], log: bool = False) -> float:
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
                    [tape_value(r[column], log) for r in rows],
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


def draw_map(ax, run: SavedRun, field: np.ndarray, norm: Normalize, cmap: str):
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
        im = draw_map(ax, run, field, norm, cmap)
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
        im = draw_map(ax, run, band.c - c_star, scales.difference_norm, DIFF_CMAP)
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


# --------------------------------------------------------------------------- #
# Data fit
# --------------------------------------------------------------------------- #
def _receiver_ids(n_transducers: int) -> np.ndarray:
    """``(n, n - 1)``: shot ``s``'s receivers in saved order (every other
    transducer, ascending), as the last axis of ``d_obs`` / ``d_syn_*``."""
    ids = np.arange(n_transducers)
    return np.stack([np.delete(ids, s) for s in ids])


def offsets_deg(angles_deg: np.ndarray) -> np.ndarray:
    """``(n, n - 1)``: the **Offset** of each (transmitter, receiver) pair as a
    ring angle in ``[0, 180]`` degrees, from each transducer's ring angle."""
    n = len(angles_deg)
    d = angles_deg[_receiver_ids(n)] - angles_deg[:, None]
    return np.abs((d + 180.0) % 360.0 - 180.0)


def _rel_residual(d_syn: np.ndarray, d_obs: np.ndarray) -> float:
    """``||d_syn - d_obs|| / ||d_obs||`` over every sample given."""
    return float(np.linalg.norm(d_syn - d_obs) / np.linalg.norm(d_obs))


def _band_name(band: SavedBand) -> str:
    return f"band {band.index}: {band.label}"


def _per_band_figure(title: str, xlabel: str, ylabel: str, log: bool = True):
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    return fig, ax


def residual_by_offset(
    run: SavedRun, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """``rel_data_residual`` at each band's final iterate against **Offset**
    (0-180 degrees), one point per distinct offset, pooling every shot,
    frequency and receiver pair at that offset."""
    if run.problem is None or not run.bands:
        return None
    offsets = offsets_deg(run.problem["transducer_angles_deg"].astype(np.float64))
    # Distinct offsets up to float32 rounding of the ring angles.
    keys = np.round(offsets, 3)
    distinct = np.unique(keys)
    fig, ax = _per_band_figure(
        "Data residual by offset (final iterate)", "offset [deg]", "rel_data_residual"
    )
    table: List[Dict[str, Any]] = []
    for band in run.bands:
        # (shot, receiver, frequency), so a pair mask picks every frequency.
        d_syn = band.arrays["d_syn_final"].transpose(0, 2, 1)
        d_obs = band.arrays["d_obs"].transpose(0, 2, 1)
        values = []
        for offset in distinct:
            pairs = keys == offset
            values.append(_rel_residual(d_syn[pairs], d_obs[pairs]))
            table.append(
                {
                    "band_index": band.index,
                    "label": band.label,
                    "offset_deg": float(offset),
                    "n_pairs": int(pairs.sum()),
                    "rel_data_residual": values[-1],
                }
            )
        ax.plot(distinct, values, "o-", label=_band_name(band))
    ax.set_xlim(-5, 185)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.legend(fontsize="small")
    return Plot("residual_by_offset", fig, table)


def residual_by_frequency(
    run: SavedRun, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """``rel_data_residual`` at each band's final iterate per solved frequency
    (over every shot and receiver); only with more than one frequency in the
    run."""
    freqs = {float(f) for band in run.bands for f in band.arrays["freqs"]}
    if len(freqs) < 2:
        return None
    fig, ax = _per_band_figure(
        "Data residual by frequency (final iterate)",
        "frequency [kHz]",
        "rel_data_residual",
    )
    table: List[Dict[str, Any]] = []
    for band in run.bands:
        d_syn, d_obs = band.arrays["d_syn_final"], band.arrays["d_obs"]
        band_freqs = band.arrays["freqs"]
        values = [
            _rel_residual(d_syn[:, f], d_obs[:, f]) for f in range(len(band_freqs))
        ]
        table += [
            {
                "band_index": band.index,
                "label": band.label,
                "frequency_hz": float(freq),
                "rel_data_residual": value,
            }
            for freq, value in zip(band_freqs, values)
        ]
        ax.plot(band_freqs / 1e3, values, "o-", label=_band_name(band))
    ax.legend(fontsize="small")
    return Plot("residual_by_frequency", fig, table)


def residual_by_transmitter(
    run: SavedRun, scales: Optional[ColourScales] = None
) -> Optional[Plot]:
    """``rel_data_residual`` of each shot at each band's final iterate (over its
    frequencies and receivers) against the transmitting transducer's ring angle."""
    if run.problem is None or not run.bands:
        return None
    angles = run.problem["transducer_angles_deg"]
    fig, ax = _per_band_figure(
        "Data residual by transmitter (final iterate)",
        "transmitter angle [deg]",
        "rel_data_residual",
    )
    table: List[Dict[str, Any]] = []
    for band in run.bands:
        d_syn, d_obs = band.arrays["d_syn_final"], band.arrays["d_obs"]
        values = [_rel_residual(d_syn[s], d_obs[s]) for s in range(len(d_obs))]
        table += [
            {
                "band_index": band.index,
                "label": band.label,
                "shot": s,
                "transmitter_angle_deg": float(angles[s]),
                "rel_data_residual": value,
            }
            for s, value in enumerate(values)
        ]
        ax.plot(angles, values, "o-", label=_band_name(band))
    ax.set_xlim(-10, 370)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.legend(fontsize="small")
    return Plot("residual_by_transmitter", fig, table)


def _arcs(n_transducers: int) -> Tuple[np.ndarray, np.ndarray]:
    """Receiver columns of each shot along the two arcs of the ring, outwards
    from the transmitter: ``(n, n // 2)`` ascending ids (ending at the opposite
    transducer) and ``(n, n - 1 - n // 2)`` descending ids. Transducer ids run
    in ring order."""
    n = n_transducers
    shots = np.arange(n)[:, None]

    def columns(direction: int, n_steps: int) -> np.ndarray:
        ids = (shots + direction * np.arange(1, n_steps + 1)) % n
        return np.where(ids < shots, ids, ids - 1)  # the shot's own id is skipped

    return columns(1, n // 2), columns(-1, n - 1 - n // 2)


def phase_gap_per_pair(d_syn: np.ndarray, d_obs: np.ndarray) -> np.ndarray:
    """``(n_shots, nf, n - 1)``: the phase gap ``phase(d_syn) - phase(d_obs)``
    of each transducer pair, unwrapped along the ring.

    A wrapped gap never exceeds pi, so the gap is followed outwards from the
    transmitter along both arcs: the nearest receiver's gap is taken within
    half a cycle, and each next receiver's within half a cycle of its
    neighbour's. ``|gap| > pi`` then marks a skipped cycle (it assumes the
    ring samples the gap finely enough that neighbours differ by under pi).
    """
    wrapped = np.angle(d_syn * np.conj(d_obs))
    gap = np.empty_like(wrapped, dtype=np.float64)
    shots = np.arange(len(d_obs))[:, None]
    for arc in _arcs(d_obs.shape[0]):
        along = wrapped[shots, :, arc]  # (n_shots, n_steps, nf)
        gap[shots, :, arc] = np.unwrap(along, axis=1)
    return gap


def phase_gap(run: SavedRun, scales: Optional[ColourScales] = None) -> Optional[Plot]:
    """At each band's start row, per frequency, the fraction of transducer
    pairs whose phase gap (:func:`phase_gap_per_pair`) exceeds pi: the
    cycle-skipping check for whether a band needs a lower band before it."""
    if not run.bands:
        return None
    fig, ax = _per_band_figure(
        "Phase gap at each band's start: pairs more than π out",
        "frequency [kHz]",
        "fraction of pairs with |Δphase| > π",
        log=False,
    )
    table: List[Dict[str, Any]] = []
    for band in run.bands:
        d_obs = band.arrays["d_obs"]
        over = np.abs(phase_gap_per_pair(band.arrays["d_syn_start"], d_obs)) > np.pi
        freqs = band.arrays["freqs"]
        fractions = [float(over[:, f].mean()) for f in range(len(freqs))]
        table += [
            {
                "band_index": band.index,
                "label": band.label,
                "frequency_hz": float(freq),
                "n_pairs": int(over[:, f].size),
                "fraction_over_pi": fraction,
            }
            for f, (freq, fraction) in enumerate(zip(freqs, fractions))
        ]
        ax.plot(freqs / 1e3, fractions, "o", label=_band_name(band))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize="small")
    return Plot("phase_gap", fig, table)


# The per-run set, in drawing order.
RUN_PLOTS: List[PlotFn] = [
    convergence,
    velocity,
    per_band_error,
    profiles,
    band_contribution,
    residual_by_offset,
    residual_by_frequency,
    residual_by_transmitter,
    phase_gap,
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


def draw_reported(
    draw: Callable[[Path], List[Path]], directory: Path, *, interruptible: bool = True
) -> bool:
    """Call ``draw(directory)`` and print where its figures went; return
    whether it drew.

    An error from it is printed as a warning naming the redraw command, never
    raised, so figures never fail a run or an experiment. With
    ``interruptible=False`` a ``KeyboardInterrupt`` is reported likewise.
    """
    caught = Exception if interruptible else (Exception, KeyboardInterrupt)
    try:
        written = draw(directory)
    except caught as exc:  # noqa: BLE001 - reported, never raised
        print(
            f"  [post-processing warning] {type(exc).__name__}: {exc}; "
            f"redraw with: main.py --postprocess {directory}"
        )
        return False
    if written:
        print(f"figures: {directory / 'figures'} ({len(written)} files)")
    return True
