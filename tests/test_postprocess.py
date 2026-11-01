"""Post-processing: the figures and CSVs drawn from a saved run directory alone,
at the end of every run and by ``main.py --postprocess <run_dir>``."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main
from odil_wave.experiment import (
    deep_merge,
    new_run_directory,
    postprocess,
    resolve_config,
    run_inverse,
)
from odil_wave.experiment.postprocess import ColourScales, SavedRun

from conftest import TINY_RUN_CONFIG
from test_saved_run import band_dirs, load_npz
from test_tape import band_rows, read_tape, run_cli

QUICK = {"optimiser": {"n_iter": 1}}

# Every PNG is saved at this resolution.
DPI = 150

# The data-fit figures and CSVs of every run with a completed band; the
# breakdown by frequency only with more than one solved frequency (the tiny
# runs' 30 and 40 kHz bands both solve the 33.2 kHz bin).
DATA_FIT_OUTPUTS = {
    "residual_by_offset.png",
    "residual_by_offset.csv",
    "residual_by_transmitter.png",
    "residual_by_transmitter.csv",
    "phase_gap.png",
    "phase_gap.csv",
}
# The figures of a multi-band run, and the CSVs beside them.
MULTI_BAND_OUTPUTS = {
    "convergence.png",
    "velocity.png",
    "per_band_error.png",
    "profiles.png",
    "profiles.csv",
    "band_contribution.png",
    "band_contribution.csv",
    *DATA_FIT_OUTPUTS,
}
# Per-band error and band contribution are drawn only with more than one band.
SINGLE_BAND_OUTPUTS = {
    "convergence.png",
    "velocity.png",
    "profiles.png",
    "profiles.csv",
    *DATA_FIT_OUTPUTS,
}


def copy_run(run_dir: Path, tmp_path: Path) -> Path:
    """A copy of ``run_dir`` without its figures."""
    copy = tmp_path / run_dir.name
    shutil.copytree(run_dir, copy)
    shutil.rmtree(copy / "figures", ignore_errors=True)
    return copy


def outputs(run_dir: Path) -> set:
    return {p.name for p in (run_dir / "figures").iterdir()}


def test_postprocess_writes_the_figures_and_csvs(tiny_two_band_run, tmp_path):
    run_dir = copy_run(tiny_two_band_run.run_dir, tmp_path)

    assert main.main(["--postprocess", str(run_dir)]) == 0

    assert outputs(run_dir) == MULTI_BAND_OUTPUTS
    for png in (run_dir / "figures").glob("*.png"):
        with Image.open(png) as img:
            assert [round(d) for d in img.info["dpi"]] == [DPI, DPI], png.name


def test_a_directory_that_is_no_run_cannot_be_postprocessed(tmp_path, capsys):
    assert main.main(["--postprocess", str(tmp_path)]) == 2
    assert "not a run directory" in capsys.readouterr().err


def test_a_run_of_a_newer_schema_version_is_not_postprocessed(
    tiny_two_band_run, tmp_path, capsys
):
    """Readers take every ``schema_version`` from 1 to this code's, no later."""
    run_dir = copy_run(tiny_two_band_run.run_dir, tmp_path)
    meta = json.loads((run_dir / "metadata.json").read_text())
    (run_dir / "metadata.json").write_text(json.dumps({**meta, "schema_version": 2}))

    assert main.main(["--postprocess", str(run_dir)]) == 2

    assert "schema_version 2" in capsys.readouterr().err
    assert not (run_dir / "figures").exists()


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #
CONVERGENCE_PANELS = [
    ({"pde_term", "data_term", "reg_term"}, "log"),
    ({"rel_data_residual", "pde_rel_residual", "rel_wavefield_error"}, "log"),
    ({"rmse_brain", "rmse_skull", "bias_brain", "bias_skull"}, "linear"),
    ({"ssim_brain"}, "linear"),
]


def test_convergence_plots_four_panels_against_iter_total(tiny_two_band_run):
    run = SavedRun.load(tiny_two_band_run.run_dir)
    _, rows = read_tape(tiny_two_band_run.run_dir)

    fig = postprocess.convergence(run).figure

    assert len(fig.axes) == 4
    for ax, (columns, scale) in zip(fig.axes, CONVERGENCE_PANELS):
        assert ax.get_xlabel() == "iter_total"
        assert ax.get_yscale() == scale
        assert {t.get_text() for t in ax.get_legend().get_texts()} == columns
        # Band 1 starts after band 0's 3 outer iterations.
        boundaries = [
            line for line in ax.get_lines() if list(line.get_xdata()) == [3, 3]
        ]
        assert boundaries, "band boundary not marked"
    # The solid C0 series of the model-error panel is the tape's rmse_brain.
    xs, ys = [], []
    for line in fig.axes[2].get_lines():
        if line.get_color() == "C0" and line.get_linestyle() == "-":
            xs += list(line.get_xdata())
            ys += list(line.get_ydata())
    assert xs == [r["iter_total"] for r in rows]
    np.testing.assert_allclose(ys, [r["rmse_brain"] for r in rows])


# --------------------------------------------------------------------------- #
# Velocity and colour scales
# --------------------------------------------------------------------------- #
def truth_scales(run_dir: Path) -> tuple:
    """The spec's truth-only scales: velocity [min, max] of c* over the interior,
    difference symmetric at half c*'s peak-to-peak over the brain region."""
    problem = load_npz(run_dir / "problem.npz")
    c_star, brain = problem["c_star"], problem["brain_mask"]
    half = 0.5 * float(np.ptp(c_star[brain]))
    return (float(c_star.min()), float(c_star.max())), (-half, half)


def image_axes(fig) -> list:
    return [ax for ax in fig.axes if ax.get_images()]


def test_velocity_shows_truth_recovered_and_difference_with_outlines(
    tiny_two_band_run,
):
    run_dir = tiny_two_band_run.run_dir
    run = SavedRun.load(run_dir)
    problem = load_npz(run_dir / "problem.npz")
    final_c = load_npz(band_dirs(run_dir)[-1] / "arrays.npz")["c"]

    fig = postprocess.velocity(run, ColourScales.of_run(run)).figure

    axes = image_axes(fig)
    assert len(axes) == 3
    shown = [np.asarray(ax.get_images()[0].get_array()).T for ax in axes]
    np.testing.assert_allclose(shown[0], problem["c_star"])
    np.testing.assert_allclose(shown[1], final_c)
    np.testing.assert_allclose(shown[2], final_c - problem["c_star"])
    velocity_scale, diff_scale = truth_scales(run_dir)
    for ax, scale in zip(axes, [velocity_scale, velocity_scale, diff_scale]):
        np.testing.assert_allclose(ax.get_images()[0].get_clim(), scale, rtol=1e-6)
        # Head (outer skull) and brain outlines.
        assert len(ax.collections) == 2


def test_colour_scales_come_from_the_truth_alone(tiny_two_band_run, tmp_path):
    """Two runs of one truth with different recovered models share every scale."""
    other = copy_run(tiny_two_band_run.run_dir, tmp_path)
    for band in band_dirs(other):
        arrays = load_npz(band / "arrays.npz")
        arrays["c"] = arrays["c"] * 1.3 + 50.0
        np.savez(band / "arrays.npz", **arrays)

    clims = []
    for run_dir in (tiny_two_band_run.run_dir, other):
        run = SavedRun.load(run_dir)
        scales = ColourScales.of_run(run)
        plots = [
            postprocess.velocity(run, scales),
            postprocess.per_band_error(run, scales),
        ]
        clims.append(
            [
                ax.get_images()[0].get_clim()
                for plot in plots
                for ax in image_axes(plot.figure)
            ]
        )
    assert clims[0] == clims[1]
    _, diff_scale = truth_scales(other)
    # The two per-band error maps use the shared difference scale.
    np.testing.assert_allclose(clims[1][3:], [diff_scale] * 2, rtol=1e-6)


# --------------------------------------------------------------------------- #
# CSVs beside the figures
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_band_contribution_is_each_bands_change_and_cost(tiny_two_band_run):
    run_dir = tiny_two_band_run.run_dir
    _, rows = read_tape(run_dir)
    finals = [band_rows(rows, b)[-1] for b in (0, 1)]
    starts = [band_rows(rows, b)[0] for b in (0, 1)]

    table = read_csv(run_dir / "figures" / "band_contribution.csv")

    assert [int(r["band_index"]) for r in table] == [0, 1]
    for b, row in enumerate(table):
        for metric in ("rmse_brain", "rmse_skull", "ssim_brain", "rel_data_residual"):
            assert float(row[f"delta_{metric}"]) == pytest.approx(
                finals[b][metric] - starts[b][metric]
            )
        summary = json.loads((band_dirs(run_dir)[b] / "band_summary.json").read_text())
        assert float(row["wall_s"]) == pytest.approx(summary["wall_s"])
    # The tape's n_evaluations is run-wide cumulative; a band's cost is its share.
    assert int(table[0]["n_evaluations"]) == finals[0]["n_evaluations"]
    assert int(table[1]["n_evaluations"]) == (
        finals[1]["n_evaluations"] - finals[0]["n_evaluations"]
    )


def test_profiles_run_through_the_head(tiny_two_band_run):
    run_dir = tiny_two_band_run.run_dir
    problem = load_npz(run_dir / "problem.npz")
    final_c = load_npz(band_dirs(run_dir)[-1] / "arrays.npz")["c"]
    x = np.linspace(0.0, 0.05, 20)  # the tiny interior's nodes

    table = read_csv(run_dir / "figures" / "profiles.csv")

    for line, along in (("horizontal", "x_m"), ("vertical", "y_m")):
        samples = [r for r in table if r["line"] == line]
        np.testing.assert_allclose([float(r[along]) for r in samples], x)
        across = "y_m" if along == "x_m" else "x_m"
        (fixed,) = {float(r[across]) for r in samples}
        k = int(np.argmin(np.abs(x - fixed)))
        sl = np.s_[:, k] if line == "horizontal" else np.s_[k, :]
        for column, field in (
            ("c_star", problem["c_star"]),
            ("c0", problem["c0"]),
            ("c_final", final_c),
        ):
            np.testing.assert_allclose(
                [float(r[column]) for r in samples], field[sl], rtol=1e-6
            )
        # The line crosses the brain and both sides of the skull.
        head = (problem["brain_mask"] | problem["skull_mask"])[sl]
        assert problem["brain_mask"][sl].any()
        assert not head[0] and not head[-1] and problem["skull_mask"][sl].sum() >= 2


# --------------------------------------------------------------------------- #
# At the end of every run
# --------------------------------------------------------------------------- #
def test_every_run_ends_with_its_figures(tiny_two_band_run):
    assert outputs(tiny_two_band_run.run_dir) == MULTI_BAND_OUTPUTS


def test_a_one_band_run_has_no_per_band_figures(tmp_path, git_repo):
    run_dir = run_cli(tmp_path, git_repo, deep_merge(TINY_RUN_CONFIG, QUICK))

    assert outputs(run_dir) == SINGLE_BAND_OUTPUTS


def test_an_interrupted_run_is_postprocessed(tmp_path, git_repo):
    """Ctrl-C after the first of two bands: the figures of what was saved."""
    two_bands = {"continuation": {"bands": [{"frequencies_hz": [30e3]}] * 2}}
    cfg = resolve_config(deep_merge(deep_merge(TINY_RUN_CONFIG, QUICK), two_bands))
    out = tmp_path / "run"

    def ctrl_c(band_result, run_result):
        raise KeyboardInterrupt

    result = run_inverse(
        cfg, new_run_directory(out=out, repo_root=git_repo), on_band_end=ctrl_c
    )

    assert result.status == "interrupted"
    assert outputs(out) == SINGLE_BAND_OUTPUTS


# --------------------------------------------------------------------------- #
# Reads only the run directory
# --------------------------------------------------------------------------- #
def test_postprocess_never_builds_or_runs_the_optimiser(
    tiny_two_band_run, tmp_path, monkeypatch
):
    from odil_wave import HelmholtzSolver
    from odil_wave.optimisation import JointFreqODIL

    def forbidden(*args, **kwargs):
        raise AssertionError("post-processing solved or optimised")

    monkeypatch.setattr(JointFreqODIL, "__init__", forbidden)
    monkeypatch.setattr(JointFreqODIL, "minimise", forbidden)
    monkeypatch.setattr(HelmholtzSolver, "solve", forbidden)
    run_dir = copy_run(tiny_two_band_run.run_dir, tmp_path)

    assert main.main(["--postprocess", str(run_dir)]) == 0
    assert outputs(run_dir) == MULTI_BAND_OUTPUTS


@pytest.mark.parametrize(
    "name", ["plot_run_history", "plot_velocity_recovery", "animate_bands"]
)
def test_the_old_plot_functions_are_gone(name):
    import odil_wave.experiment as experiment
    from odil_wave.experiment import plots

    assert not hasattr(experiment, name)
    assert not hasattr(plots, name)


# --------------------------------------------------------------------------- #
# Data fit
# --------------------------------------------------------------------------- #
N_TRANSDUCERS = 8  # the tiny ring: transducer k at 360 k / 8 degrees


def receiver_ids(shot: int) -> list:
    """Shot ``shot``'s receivers in saved order: every other transducer, ascending."""
    return [r for r in range(N_TRANSDUCERS) if r != shot]


def ring_steps(shot: int, receiver: int) -> int:
    """Transducers between transmitter and receiver along the shorter arc."""
    d = (receiver - shot) % N_TRANSDUCERS
    return min(d, N_TRANSDUCERS - d)


def rel_residual(pairs: list) -> float:
    """``||d_syn - d_obs|| / ||d_obs||`` over ``(d_syn, d_obs)`` samples."""
    syn, obs = (np.array(x) for x in zip(*pairs))
    return float(np.linalg.norm(syn - obs) / np.linalg.norm(obs))


def test_residual_by_offset_is_each_bands_final_fit_per_offset(tiny_two_band_run):
    run_dir = tiny_two_band_run.run_dir

    table = read_csv(run_dir / "figures" / "residual_by_offset.csv")

    for b, band_dir in enumerate(band_dirs(run_dir)):
        arrays = load_npz(band_dir / "arrays.npz")
        by_step: dict = {}
        for s in range(N_TRANSDUCERS):
            for j, r in enumerate(receiver_ids(s)):
                for f in range(len(arrays["freqs"])):
                    by_step.setdefault(ring_steps(s, r), []).append(
                        (arrays["d_syn_final"][s, f, j], arrays["d_obs"][s, f, j])
                    )
        rows = [r for r in table if int(r["band_index"]) == b]
        # One point per distinct offset, 0-180 degrees.
        assert [float(r["offset_deg"]) for r in rows] == [45.0, 90.0, 135.0, 180.0]
        for row, step in zip(rows, [1, 2, 3, 4]):
            assert float(row["rel_data_residual"]) == pytest.approx(
                rel_residual(by_step[step]), rel=1e-5
            )


def test_residual_by_transmitter_is_each_shots_final_fit(tiny_two_band_run):
    run_dir = tiny_two_band_run.run_dir

    table = read_csv(run_dir / "figures" / "residual_by_transmitter.csv")

    for b, band_dir in enumerate(band_dirs(run_dir)):
        arrays = load_npz(band_dir / "arrays.npz")
        rows = [r for r in table if int(r["band_index"]) == b]
        assert [int(r["shot"]) for r in rows] == list(range(N_TRANSDUCERS))
        assert [float(r["transmitter_angle_deg"]) for r in rows] == [
            0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0
        ]  # fmt: skip
        for s, row in enumerate(rows):
            pairs = list(
                zip(arrays["d_syn_final"][s].ravel(), arrays["d_obs"][s].ravel())
            )
            assert float(row["rel_data_residual"]) == pytest.approx(
                rel_residual(pairs), rel=1e-5
            )


# One band of two frequencies the tiny time grid resolves to distinct bins.
TWO_FREQUENCIES = {"continuation": {"bands": [{"frequencies_hz": [20e3, 50e3]}]}}


def test_residual_by_frequency_is_the_final_fit_per_frequency(tmp_path, git_repo):
    config = deep_merge(deep_merge(TINY_RUN_CONFIG, QUICK), TWO_FREQUENCIES)
    run_dir = run_cli(tmp_path, git_repo, config)
    arrays = load_npz(band_dirs(run_dir)[0] / "arrays.npz")
    assert len(set(arrays["freqs"])) == 2

    table = read_csv(run_dir / "figures" / "residual_by_frequency.csv")

    assert outputs(run_dir) == SINGLE_BAND_OUTPUTS | {
        "residual_by_frequency.png",
        "residual_by_frequency.csv",
    }
    assert [float(r["frequency_hz"]) for r in table] == pytest.approx(
        list(arrays["freqs"])
    )
    for f, row in enumerate(table):
        pairs = list(
            zip(arrays["d_syn_final"][:, f].ravel(), arrays["d_obs"][:, f].ravel())
        )
        assert float(row["rel_data_residual"]) == pytest.approx(
            rel_residual(pairs), rel=1e-5
        )


def wrap(phase: float) -> float:
    """``phase`` into ``(-pi, pi]``."""
    return float(np.angle(np.exp(1j * phase)))


def phase_gaps(d_syn: np.ndarray, d_obs: np.ndarray, shot: int) -> list:
    """Shot ``shot``'s phase gap ``phase(d_syn) - phase(d_obs)`` at each receiver,
    followed outwards from the transmitter along both arcs of the ring: the
    nearest receiver's gap is taken within half a cycle, and each next one
    differs from its neighbour's by less than half a cycle."""
    column = {r: j for j, r in enumerate(receiver_ids(shot))}
    gaps = []
    for direction, n_steps in ((1, N_TRANSDUCERS // 2), (-1, N_TRANSDUCERS // 2 - 1)):
        gap = 0.0
        for step in range(1, n_steps + 1):
            j = column[(shot + direction * step) % N_TRANSDUCERS]
            raw = float(np.angle(d_syn[j]) - np.angle(d_obs[j]))
            gap += wrap(raw - gap)
            gaps.append(gap)
    return gaps


def test_a_phase_ramp_skips_a_cycle_only_past_half_a_cycle(tiny_two_band_run, tmp_path):
    """d_syn lagging d_obs by 1 rad per ring step: of each shot's 7 pairs only
    the opposite one (4 rad) is more than pi out."""
    run_dir = copy_run(tiny_two_band_run.run_dir, tmp_path)
    for band in band_dirs(run_dir):
        arrays = load_npz(band / "arrays.npz")
        steps = np.array(
            [[ring_steps(s, r) for r in receiver_ids(s)] for s in range(N_TRANSDUCERS)]
        )
        arrays["d_syn_start"] = arrays["d_obs"] * np.exp(-1j * steps)[:, None, :]
        np.savez(band / "arrays.npz", **arrays)

    postprocess.postprocess_run(run_dir)

    table = read_csv(run_dir / "figures" / "phase_gap.csv")
    assert [int(r["band_index"]) for r in table] == [0, 1]
    for row in table:
        assert int(row["n_pairs"]) == N_TRANSDUCERS * (N_TRANSDUCERS - 1)
        assert float(row["fraction_over_pi"]) == pytest.approx(1 / 7)


def test_phase_gap_is_each_bands_start_fit_per_frequency(tiny_two_band_run):
    run_dir = tiny_two_band_run.run_dir

    table = read_csv(run_dir / "figures" / "phase_gap.csv")

    for row, band_dir in zip(table, band_dirs(run_dir)):
        arrays = load_npz(band_dir / "arrays.npz")
        (freq,) = arrays["freqs"]
        assert float(row["frequency_hz"]) == pytest.approx(float(freq))
        gaps = [
            g
            for s in range(N_TRANSDUCERS)
            for g in phase_gaps(arrays["d_syn_start"][s, 0], arrays["d_obs"][s, 0], s)
        ]
        assert float(row["fraction_over_pi"]) == pytest.approx(
            np.mean(np.abs(gaps) > np.pi)
        )
