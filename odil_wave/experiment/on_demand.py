"""On-demand plots: one-line library calls that take a run directory and return
a matplotlib :class:`~matplotlib.figure.Figure`, for digging deeper from a
notebook. There is no CLI, and none of them is part of post-processing's fixed
set (:mod:`.postprocess`).

Like post-processing, they never re-run a solve or an optimisation: they read
the run directory and, where a figure needs the problem's geometry (the source
spectrum, the PDE residual map), rebuild it from ``config_resolved.yaml`` on the
CPU and evaluate fixed formulas or the operator on the saved arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from odil_wave.grid import FrequencySelection
from odil_wave.operator import WaveEquation
from odil_wave.wavefield import Wavefield

from .config import RunConfig, deep_merge, load_config_file
from .postprocess import DIFF_CMAP, SavedBand, SavedRun, _draw_map
from .problem import Problem, build_problem
from .runner import _band_label

PathLike = Union[str, Path]

MAGNITUDE_CMAP = "magma"

# The PDE residual map's log colour scale spans this many decades below its peak.
RESIDUAL_DECADES = 6


def _problem(run_dir: PathLike) -> Problem:
    """The run's problem rebuilt on the CPU from its ``config_resolved.yaml``
    (grid, source, models; no solve)."""
    path = Path(run_dir) / "config_resolved.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{str(run_dir)!r} is not a run directory (no config_resolved.yaml)"
        )
    config = deep_merge(load_config_file(path), {"runtime": {"device": "cpu"}})
    return build_problem(RunConfig.from_dict(config))


def source_spectrum(run_dir: PathLike) -> Figure:
    """The source's normalised magnitude spectrum ``|S(f)|`` with each band's
    inversion bins (the FFT bins solved at) marked. Reads the config only."""
    problem = _problem(run_dir)
    grid, source = problem.grid, problem.source
    s = source.waveform(grid.t).cpu().numpy()
    magnitude = np.abs(np.fft.rfft(s))
    f_khz = np.fft.rfftfreq(grid.nt, d=float(grid.dt)) / 1e3

    fig = Figure(figsize=(8, 3.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(f_khz, magnitude / magnitude.max(), "k-", label="|S(f)|")
    f_max = 0.0
    for b, band in enumerate(problem.cfg.continuation.bands):
        bins = FrequencySelection.from_frequencies(grid, band.frequencies_hz)
        khz = bins.frequencies.abs().cpu().numpy() / 1e3
        for k, f in enumerate(khz):
            ax.axvline(
                f,
                color=f"C{b}",
                linestyle="--",
                label=f"band {b}: {_band_label(band.frequencies_hz)}"
                if k == 0
                else None,
            )
        f_max = max(f_max, float(khz.max()))
    ax.set_xlim(0, 2.2 * f_max if f_max else f_khz[-1])
    ax.set_xlabel("f [kHz]")
    ax.set_ylabel("normalised |S(f)|")
    ax.set_title(f"Source spectrum and inversion bins: {Path(run_dir).name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    return fig


def _plot_by_band(ax, rows: List[Dict[str, Any]], column: str, style: str, colour: str):
    """``column`` against ``iter_total``, one line per band (``gid`` = column,
    labelled once), with each later band's start marked."""
    by_band: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_band.setdefault(row["band_index"], []).append(row)
    for i, (_, band_rows) in enumerate(sorted(by_band.items())):
        ax.plot(
            [r["iter_total"] for r in band_rows],
            [np.nan if r[column] is None else r[column] for r in band_rows],
            style,
            color=colour,
            marker=".",
            label=column if i == 0 else None,
            gid=column,
        )
        if i:
            ax.axvline(band_rows[0]["iter_total"], color="0.5", ls=":", lw=1)


def optimiser_health(run_dir: PathLike) -> Figure:
    """The tape's optimiser-health columns against ``iter_total``: ``c_min`` /
    ``c_max`` against the speed bounds (``grid.c_min`` / ``grid.c_max``),
    ``sat_frac``, and ``grad_u_norm`` / ``grad_z_norm`` (log)."""
    run = SavedRun.load(run_dir)
    if not run.tape:
        raise FileNotFoundError(f"no tape.csv rows under {str(run_dir)!r}")
    fig = Figure(figsize=(14, 4), layout="constrained")
    speed, saturation, gradients = fig.subplots(1, 3)
    for bound in ("c_min", "c_max"):
        speed.axhline(
            run.config["grid"][bound],
            color="crimson",
            ls="--",
            lw=1,
            label=f"grid.{bound}",
        )
    _plot_by_band(speed, run.tape, "c_min", "-", "C0")
    _plot_by_band(speed, run.tape, "c_max", "-", "C1")
    speed.set_title("Speed extremes over the interior")
    speed.set_ylabel("c [m/s]")
    _plot_by_band(saturation, run.tape, "sat_frac", "-", "C0")
    saturation.set_title("Fraction of cells pinned at a speed bound")
    _plot_by_band(gradients, run.tape, "grad_u_norm", "-", "C0")
    _plot_by_band(gradients, run.tape, "grad_z_norm", "-", "C1")
    gradients.set_yscale("log")
    gradients.set_title("Gradient norms")
    for ax in fig.axes:
        ax.set_xlabel("iter_total")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle(f"Optimiser health: {run.path.name}")
    return fig


# --------------------------------------------------------------------------- #
# One saved shot and frequency of a band
# --------------------------------------------------------------------------- #
def _band(run: SavedRun, band: int) -> SavedBand:
    """The completed band at position ``band`` (``-1``: the last)."""
    try:
        return run.bands[band]
    except IndexError:
        raise ValueError(
            f"no completed band {band} in {str(run.path)!r} "
            f"({len(run.bands)} completed)"
        ) from None


def _frequency(saved: SavedBand, frequency: int) -> float:
    """The solved frequency (Hz) at index ``frequency`` of ``saved``'s bins."""
    freqs = saved.arrays["freqs"]
    if not -len(freqs) <= frequency < len(freqs):
        raise ValueError(
            f"no frequency {frequency} in band {saved.index} "
            f"({len(freqs)} solved: {[float(f) for f in freqs]} Hz)"
        )
    return float(freqs[frequency])


def _saved_shot(saved: SavedBand, shot: Optional[int]) -> Tuple[int, int]:
    """``(shot, position in arrays.npz's shots)``; ``shot`` defaults to the
    band's worst-fit shot, which is always saved."""
    shots = [int(s) for s in saved.arrays["shots"]]
    if shot is None:
        shot = int(saved.summary["worst_fit_shot"])
    if shot not in shots:
        raise ValueError(
            f"shot {shot} has no saved wavefield in band {saved.index}; "
            f"saved shots: {shots}"
        )
    return shot, shots.index(shot)


def _interior(run: SavedRun) -> Tuple[slice, slice]:
    (x0, x1), (y0, y1) = run.problem["interior_slice"]
    return slice(int(x0), int(x1)), slice(int(y0), int(y1))


def wavefield_comparison(
    run_dir: PathLike, *, band: int = -1, shot: Optional[int] = None, frequency: int = 0
) -> Figure:
    """u | u* | u − u* over the interior for one saved shot and frequency of a
    band's final iterate: real part (top) and magnitude (bottom).

    ``band`` is a completed band's position (``-1``: the last), ``shot`` a saved
    shot (default the band's worst-fit shot), ``frequency`` an index into the
    band's solved frequencies. Each row shares one colour scale, set by u*.
    """
    run = SavedRun.load(run_dir)
    saved = _band(run, band)
    f_hz = _frequency(saved, frequency)
    shot, k = _saved_shot(saved, shot)
    u = saved.arrays["u"][k, frequency][_interior(run)]
    u_star = saved.arrays["u_star"][k, frequency]
    peak = float(np.abs(u_star).max()) or 1.0

    fig = Figure(figsize=(15, 9), layout="constrained")
    axes = fig.subplots(2, 3)
    rows = [
        ("Re", np.real, Normalize(-peak, peak), DIFF_CMAP),
        ("|·|", np.abs, Normalize(0.0, peak), MAGNITUDE_CMAP),
    ]
    for row, (part, take, norm, cmap) in zip(axes, rows):
        for ax, (name, field) in zip(
            row, [("u", u), ("u*", u_star), ("u − u*", u - u_star)]
        ):
            im = _draw_map(ax, run, take(field), norm, cmap)
            ax.set_title(f"{part}({name})")
        fig.colorbar(im, ax=list(row), shrink=0.85)
    fig.suptitle(
        f"Wavefield: band {saved.index} ({saved.label}), shot {shot}, "
        f"{f_hz / 1e3:.2f} kHz: {run.path.name}"
    )
    return fig


def shot_data_fit(
    run_dir: PathLike, *, band: int = -1, shot: Optional[int] = None, frequency: int = 0
) -> Figure:
    """Amplitude and unwrapped phase of ``d_obs`` and ``d_syn`` (the band's final
    iterate; its start row dotted) against receiver angle, for one shot and
    frequency.

    Every shot has data, so ``shot`` is any transducer (default the band's
    worst-fit shot); its receivers are the other transducers, at their ring
    angles. ``band`` and ``frequency`` are as for :func:`wavefield_comparison`.
    """
    run = SavedRun.load(run_dir)
    saved = _band(run, band)
    f_hz = _frequency(saved, frequency)
    d_obs = saved.arrays["d_obs"]
    if shot is None:
        shot = int(saved.summary["worst_fit_shot"])
    if not 0 <= shot < len(d_obs):
        raise ValueError(f"no shot {shot}: the run has {len(d_obs)} shots")
    angles = np.delete(run.problem["transducer_angles_deg"], shot)
    traces = [
        ("d_obs", d_obs, "k-"),
        ("d_syn", saved.arrays["d_syn_final"], "C0-"),
        ("d_syn (start row)", saved.arrays["d_syn_start"], "C7:"),
    ]

    fig = Figure(figsize=(12, 4.5), layout="constrained")
    amplitude, phase = fig.subplots(1, 2)
    for name, data, style in traces:
        trace = data[shot, frequency]
        amplitude.plot(angles, np.abs(trace), style, marker=".", label=name)
        phase.plot(angles, np.unwrap(np.angle(trace)), style, marker=".", label=name)
    amplitude.set_ylabel("|d|")
    amplitude.set_title("Amplitude")
    phase.set_ylabel("unwrapped phase [rad]")
    phase.set_title("Phase")
    for ax in (amplitude, phase):
        ax.axvline(
            run.problem["transducer_angles_deg"][shot],
            color="crimson",
            ls="--",
            lw=1,
            label="source",
        )
        ax.set_xlabel("receiver angle [deg]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle(
        f"Data fit: band {saved.index} ({saved.label}), shot {shot}, "
        f"{f_hz / 1e3:.2f} kHz: {run.path.name}"
    )
    return fig


def pde_residual_map(
    run_dir: PathLike, *, band: int = -1, shot: Optional[int] = None, frequency: int = 0
) -> Figure:
    """``|r_pde| / RMS(q)`` over the full grid (the PML too) for one saved shot
    and frequency of a band's final iterate: the per-cell form of the tape's
    ``pde_src_ratio``, whose RMS over the grid is that shot and frequency's
    ratio.

    The residual is the inversion's operator (``WaveEquation.residual``)
    evaluated on the saved ``u`` and ``c`` (padded with ``pml_c``), on the problem
    rebuilt from the run's config; nothing is solved. ``band``, ``shot`` and
    ``frequency`` are as for :func:`wavefield_comparison`.
    """
    run = SavedRun.load(run_dir)
    saved = _band(run, band)
    f_hz = _frequency(saved, frequency)
    shot, k = _saved_shot(saved, shot)
    problem = _problem(run_dir)
    grid = problem.grid
    freq = FrequencySelection.from_frequencies(grid, saved.summary["frequencies_hz"])
    geom = problem._geometry(grid, problem.source, freq)
    wave_eq = WaveEquation(
        Wavefield(grid, freq),
        space_order=problem.space_order,
        damping_scale=problem.damping_scale,
    )
    cdtype = wave_eq.wavefield.cdtype
    c_full = np.pad(saved.c, grid.pml_width, constant_values=run.problem["pml_c"])
    with torch.no_grad():
        u = torch.as_tensor(saved.arrays["u"][k], dtype=cdtype)
        q = geom.source_field(shot).to(cdtype) * grid.t0**2
        r = wave_eq.residual(
            u[None], torch.as_tensor(c_full, dtype=grid.dtype), q[None]
        )[0, frequency]
        q_rms = float(q[frequency].abs().square().mean().sqrt())
    ratio = r.abs().cpu().numpy() / max(q_rms, 1e-30)

    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.subplots()
    (gx0, gx1), (gy0, gy1) = (tuple(v * 1e3 for v in e) for e in grid.extent)
    peak = float(ratio.max()) or 1.0
    im = ax.imshow(
        ratio.T,
        origin="lower",
        extent=(gx0, gx1, gy0, gy1),
        norm=LogNorm(peak * 10.0**-RESIDUAL_DECADES, peak),
        cmap=MAGNITUDE_CMAP,
    )
    x, y = (a * 1e3 for a in run.interior_axes)
    ax.plot(
        [x[0], x[-1], x[-1], x[0], x[0]],
        [y[0], y[0], y[-1], y[-1], y[0]],
        "w--",
        lw=0.8,
        label="interior (PML outside)",
    )
    head = run.problem["brain_mask"] | run.problem["skull_mask"]
    for mask in (head, run.problem["brain_mask"]):
        ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors="w", linewidths=0.6)
    fig.colorbar(im, ax=ax, shrink=0.85, label="|r_pde| / RMS(q)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.legend(fontsize="small", loc="upper right")
    ax.set_title(
        f"PDE residual: band {saved.index} ({saved.label}), shot {shot}, "
        f"{f_hz / 1e3:.2f} kHz\nRMS = {np.sqrt(np.mean(ratio**2)):.3g} "
        f"(this shot and frequency's pde_src_ratio)"
    )
    return fig
