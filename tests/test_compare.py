"""The run comparison (``python -m odil_wave.experiment.compare``): two run
directories have identical results, or each difference is named."""

import csv
import json
import shutil

import numpy as np
import pytest

from odil_wave.experiment.compare import compare_run_dirs


@pytest.fixture
def two_copies(tmp_path, tiny_two_band_run):
    """Two copies of one saved run: identical results, distinct directories."""
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(tiny_two_band_run.run_dir, a)
    shutil.copytree(tiny_two_band_run.run_dir, b)
    return a, b


def _edit_tape(run_dir, column, value):
    """Set ``column`` of the last ``tape.csv`` row to ``value``."""
    path = run_dir / "tape.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows[-1][column] = value
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _edit_json(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def test_identical_runs_have_no_differences(two_copies):
    assert compare_run_dirs(*two_copies) == []


def test_run_identity_and_timings_are_ignored(two_copies):
    """``run_id`` and wall-clock values differ between any two runs."""
    a, b = two_copies
    _edit_tape(b, "run_id", "other")
    _edit_tape(b, "wall_s", "123.0")
    summary = next(b.glob("bands/*/band_summary.json"))

    def retime(d):
        d["wall_s"] = 9.0
        d["t_start_epoch_s"] = 1.0
        d["final_metrics"]["wall_s"] = 9.0
        d["final_metrics"]["run_id"] = "other"

    _edit_json(summary, retime)
    assert compare_run_dirs(a, b) == []


def test_a_changed_tape_value_is_named(two_copies):
    a, b = two_copies
    _edit_tape(b, "loss_total", "0.5")
    (diff,) = compare_run_dirs(a, b)
    assert diff.startswith("tape.csv: loss_total")


def test_a_changed_array_is_named(two_copies):
    a, b = two_copies
    path = next(b.glob("bands/*/arrays.npz"))
    with np.load(path) as z:
        arrays = dict(z)
    arrays["c"] = arrays["c"] + 1e-3
    np.savez(path, **arrays)
    (diff,) = compare_run_dirs(a, b)
    assert diff.startswith(f"{path.relative_to(b)}: c")


def test_a_changed_summary_value_is_named(two_copies):
    a, b = two_copies
    summary = next(b.glob("bands/*/band_summary.json"))
    _edit_json(summary, lambda d: d["final_metrics"].update(rmse_brain=1.0))
    (diff,) = compare_run_dirs(a, b)
    assert diff.startswith(f"{summary.relative_to(b)}: final_metrics.rmse_brain")


def test_a_missing_result_file_is_named(two_copies):
    a, b = two_copies
    path = next(b.glob("bands/*/arrays.npz"))
    path.unlink()
    assert compare_run_dirs(a, b) == [f"{path.relative_to(b)}: only in the first run"]
