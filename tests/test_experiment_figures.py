"""Experiment figures: the runs of an experiment compared, drawn into the
experiment directory's ``figures/`` from what it holds, only with more than one
completed run.

The experiment directories here are copies of the session's tiny two-band run,
each shifted so the runs differ, with a hand-written ``experiment.json``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main
from odil_wave.experiment import experiment_figures
from odil_wave.experiment.experiment_figures import SavedExperiment
from odil_wave.experiment.launch import postprocess_experiment
from odil_wave.experiment.postprocess import ColourScales, load_tape
from test_saved_run import band_dirs, load_npz
from test_tape import read_tape

DPI = 150

EXPERIMENT_OUTPUTS = {
    "final_metrics.png",
    "convergence_overlay.png",
    "difference_maps.png",
    "profile_overlay.png",
    "profile_overlay.csv",
    "accuracy_cost.png",
}

# Tape columns a run copy scales by (1 + shift) at every row but the first.
SHIFTED_COLUMNS = ["rmse_brain", "rmse_skull", "rel_data_residual", "wall_s"]


def shifted_copy(run_dir: Path, dest: Path, shift: float) -> None:
    """Copy ``run_dir`` to ``dest`` with its metrics after the start row scaled
    by ``1 + shift``, its evaluations by ``1 + 2·shift``, and every band's
    ``c`` raised by ``100·shift`` m/s."""
    shutil.copytree(run_dir, dest)
    shutil.rmtree(dest / "figures", ignore_errors=True)
    with open(dest / "tape.csv", newline="") as f:
        header = csv.DictReader(f).fieldnames
    rows = load_tape(dest)
    with open(dest / "tape.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for k, row in enumerate(rows):
            if k:
                for column in SHIFTED_COLUMNS:
                    row[column] *= 1 + shift
                row["n_evaluations"] = round(row["n_evaluations"] * (1 + 2 * shift))
            writer.writerow({c: "" if row[c] is None else row[c] for c in header})
    for band in band_dirs(dest):
        arrays = load_npz(band / "arrays.npz")
        arrays["c"] = arrays["c"] + np.float32(100 * shift)
        np.savez(band / "arrays.npz", **arrays)


def make_experiment(tmp_path: Path, source: Path, runs: list) -> Path:
    """An experiment directory of ``runs``: ``(label, value, status, shift)``,
    the first the baseline. A run that isn't ``completed`` has no directory."""
    exp = tmp_path / "experiment"
    entries = []
    for index, (label, value, status, shift) in enumerate(runs):
        name = f"{index:02d}_{label}"
        if status == "completed":
            shifted_copy(source, exp / "runs" / name, shift)
        entries.append(
            {
                "index": index,
                "label": label,
                "dir": f"runs/{name}",
                "value": value,
                "is_baseline": index == 0,
                "status": status,
            }
        )
    record = {
        "schema_version": 1,
        "id": "0001-reg",
        "sweep_key": "loss.weights.reg",
        "status": "completed",
        "runs": entries,
    }
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "experiment.json").write_text(json.dumps(record))
    return exp


NUMERIC = [
    ("reg-0", 0.0, "completed", 0.0),
    ("reg-1e-7", 1e-7, "completed", 0.5),
    ("reg-1e-6", 1e-6, "completed", 1.0),
]


@pytest.fixture
def numeric_experiment(tiny_two_band_run, tmp_path) -> Path:
    return make_experiment(tmp_path, tiny_two_band_run.run_dir, NUMERIC)


def outputs(exp: Path) -> set:
    figures = exp / "figures"
    return {p.name for p in figures.iterdir()} if figures.is_dir() else set()


def test_an_experiment_draws_its_figures(numeric_experiment):
    postprocess_experiment(numeric_experiment)

    assert outputs(numeric_experiment) == EXPERIMENT_OUTPUTS
    for png in (numeric_experiment / "figures").glob("*.png"):
        with Image.open(png) as img:
            assert [round(d) for d in img.info["dpi"]] == [DPI, DPI], png.name


def test_an_experiment_of_a_newer_schema_version_is_not_postprocessed(
    numeric_experiment, capsys
):
    path = numeric_experiment / "experiment.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "schema_version": 2}))

    assert main.main(["--postprocess", str(numeric_experiment)]) == 2

    assert "schema_version 2" in capsys.readouterr().err
    assert outputs(numeric_experiment) == set()


def test_an_experiment_with_one_completed_run_draws_no_figures(
    tiny_two_band_run, tmp_path
):
    runs = [("reg-0", 0.0, "completed", 0.0), ("reg-1e-6", 1e-6, "failed", 0.0)]
    exp = make_experiment(tmp_path, tiny_two_band_run.run_dir, runs)

    postprocess_experiment(exp)

    assert outputs(exp) == set()
    assert (exp / "summary.csv").is_file()


# --------------------------------------------------------------------------- #
# The figures
# --------------------------------------------------------------------------- #
def run_dirs(exp: Path) -> list:
    return sorted((exp / "runs").iterdir())


def tape_rows(run_dir: Path) -> list:
    _, rows = read_tape(run_dir)
    return rows


def labelled(ax, label: str):
    (line,) = [line for line in ax.get_lines() if line.get_label() == label]
    return line


def test_final_metrics_against_a_numeric_swept_value(numeric_experiment):
    exp = SavedExperiment.load(numeric_experiment)
    finals = [tape_rows(d)[-1] for d in run_dirs(numeric_experiment)]
    start = tape_rows(run_dirs(numeric_experiment)[0])[0]

    fig = experiment_figures.final_metrics(exp).figure

    columns = ["rmse_brain", "rmse_skull", "ssim_brain", "rel_data_residual", "wall_s"]
    assert [ax.get_title() for ax in fig.axes] == columns
    for ax, column in zip(fig.axes, columns):
        assert ax.get_xlabel() == "loss.weights.reg"
        final = labelled(ax, "final")
        np.testing.assert_allclose(final.get_xdata(), [0.0, 1e-7, 1e-6])
        np.testing.assert_allclose(final.get_ydata(), [r[column] for r in finals])
        np.testing.assert_allclose(labelled(ax, "start row").get_ydata(), start[column])
        # 0 and a decade of positive weights: a symmetric log axis.
        assert ax.get_xscale() == "symlog"


TWO_BANDS = [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]


def test_final_metrics_is_a_dot_plot_for_a_categorical_value(
    tiny_two_band_run, tmp_path
):
    runs = [
        ("two-bands", TWO_BANDS, "completed", 0.0),
        ("one-band", TWO_BANDS[1:], "completed", 0.5),
    ]
    exp_dir = make_experiment(tmp_path, tiny_two_band_run.run_dir, runs)
    finals = [tape_rows(d)[-1] for d in run_dirs(exp_dir)]

    fig = experiment_figures.final_metrics(SavedExperiment.load(exp_dir)).figure

    ax = fig.axes[0]
    final = labelled(ax, "final")
    assert final.get_linestyle() == "None"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["two-bands", "one-band"]
    np.testing.assert_allclose(final.get_ydata(), [r["rmse_brain"] for r in finals])


def test_convergence_overlay_is_one_line_per_run_against_n_evaluations(
    numeric_experiment,
):
    exp = SavedExperiment.load(numeric_experiment)
    tapes = [tape_rows(d) for d in run_dirs(numeric_experiment)]

    fig = experiment_figures.convergence_overlay(exp).figure

    columns = ["rmse_brain", "rmse_skull", "rel_data_residual"]
    assert [ax.get_title() for ax in fig.axes] == columns
    for ax, column in zip(fig.axes, columns):
        assert ax.get_xlabel() == "n_evaluations"
        for (label, *_), rows in zip(NUMERIC, tapes):
            line = labelled(ax, label)
            np.testing.assert_allclose(
                line.get_xdata(), [r["n_evaluations"] for r in rows]
            )
            np.testing.assert_allclose(line.get_ydata(), [r[column] for r in rows])
    assert fig.axes[2].get_yscale() == "log"


def test_difference_maps_share_the_truth_scale(numeric_experiment):
    exp = SavedExperiment.load(numeric_experiment)
    problem = load_npz(run_dirs(numeric_experiment)[0] / "problem.npz")
    c_star = problem["c_star"]
    half = 0.5 * float(np.ptp(c_star[problem["brain_mask"]]))
    scales = ColourScales.of_run(exp.runs[0].run)

    plot = experiment_figures.difference_maps(exp, scales)

    axes = [ax for ax in plot.figure.axes if ax.get_images()]
    assert [ax.get_title() for ax in axes] == [label for label, *_ in NUMERIC]
    for ax, run_dir in zip(axes, run_dirs(numeric_experiment)):
        final_c = load_npz(band_dirs(run_dir)[-1] / "arrays.npz")["c"]
        image = ax.get_images()[0]
        np.testing.assert_allclose(np.asarray(image.get_array()).T, final_c - c_star)
        np.testing.assert_allclose(image.get_clim(), (-half, half), rtol=1e-6)


def test_profile_overlay_is_truth_and_each_runs_horizontal_profile(
    numeric_experiment,
):
    exp = SavedExperiment.load(numeric_experiment)
    problem = load_npz(run_dirs(numeric_experiment)[0] / "problem.npz")
    head = problem["brain_mask"] | problem["skull_mask"]
    jc = int(round(np.nonzero(head)[1].mean()))  # the head centroid's row
    x = np.linspace(0.0, 0.05, 20)  # the tiny interior's nodes

    plot = experiment_figures.profile_overlay(exp)

    ax = plot.figure.axes[0]
    c_star = problem["c_star"][:, jc]
    np.testing.assert_allclose(labelled(ax, "c*").get_ydata(), c_star)
    for (label, *_), run_dir in zip(NUMERIC, run_dirs(numeric_experiment)):
        final_c = load_npz(band_dirs(run_dir)[-1] / "arrays.npz")["c"][:, jc]
        np.testing.assert_allclose(labelled(ax, label).get_ydata(), final_c)
        rows = [r for r in plot.table if r["label"] == label]
        np.testing.assert_allclose([r["x_m"] for r in rows], x)
        np.testing.assert_allclose([r["c"] for r in rows], final_c, rtol=1e-6)
        np.testing.assert_allclose([r["c_star"] for r in rows], c_star, rtol=1e-6)


def test_accuracy_against_cost_labels_each_runs_final_point(numeric_experiment):
    exp = SavedExperiment.load(numeric_experiment)
    finals = [tape_rows(d)[-1] for d in run_dirs(numeric_experiment)]

    ax = experiment_figures.accuracy_cost(exp).figure.axes[0]

    (points,) = ax.get_lines()
    np.testing.assert_allclose(points.get_xdata(), [r["n_evaluations"] for r in finals])
    np.testing.assert_allclose(points.get_ydata(), [r["rmse_brain"] for r in finals])
    assert [t.get_text() for t in ax.texts] == [label for label, *_ in NUMERIC]
