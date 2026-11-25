"""The run comparison (``python -m odil_wave.experiment.compare``): two run
directories have identical results, or each difference is named; what only
the tree run has is named as new, not as a difference."""

import csv
import json
import shutil

import numpy as np
import pytest

from odil_wave.experiment.compare import (
    Comparison,
    compare_run_dirs,
    report,
)


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
    assert compare_run_dirs(*two_copies).differences == []


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
    assert compare_run_dirs(a, b) == Comparison([], [])


def test_a_changed_tape_value_is_named(two_copies):
    a, b = two_copies
    _edit_tape(b, "loss_total", "0.5")
    (diff,) = compare_run_dirs(a, b).differences
    assert diff.startswith("tape.csv: loss_total")


def test_a_changed_array_is_named(two_copies):
    a, b = two_copies
    path = next(b.glob("bands/*/arrays.npz"))
    with np.load(path) as z:
        arrays = dict(z)
    arrays["c"] = arrays["c"] + 1e-3
    np.savez(path, **arrays)
    (diff,) = compare_run_dirs(a, b).differences
    assert diff.startswith(f"{path.relative_to(b)}: c")


def test_a_changed_summary_value_is_named(two_copies):
    a, b = two_copies
    summary = next(b.glob("bands/*/band_summary.json"))
    _edit_json(summary, lambda d: d["final_metrics"].update(rmse_brain=1.0))
    (diff,) = compare_run_dirs(a, b).differences
    assert diff.startswith(f"{summary.relative_to(b)}: final_metrics.rmse_brain")


def test_a_missing_result_file_is_named(two_copies):
    a, b = two_copies
    path = next(b.glob("bands/*/arrays.npz"))
    path.unlink()
    assert compare_run_dirs(a, b).differences == [
        f"{path.relative_to(b)}: only in the ref run"
    ]


def test_a_new_summary_key_is_new_not_a_difference(two_copies):
    a, b = two_copies
    summary = next(b.glob("bands/*/band_summary.json"))
    _edit_json(summary, lambda d: d.update(band_extra=1.0))
    assert compare_run_dirs(a, b) == Comparison(
        [], [f"{summary.relative_to(b)}: band_extra"]
    )


def _add_tape_column(run_dir, column, value):
    path = run_dir / "tape.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row[column] = value
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_new_tape_column_is_new_not_a_difference(two_copies):
    a, b = two_copies
    _add_tape_column(b, "extra", "1.0")
    assert compare_run_dirs(a, b) == Comparison([], ["tape.csv: extra"])


def test_a_missing_tape_column_is_a_difference(two_copies):
    a, b = two_copies
    _add_tape_column(a, "extra", "1.0")
    assert compare_run_dirs(a, b) == Comparison(
        ["tape.csv: extra: only in the ref run"], []
    )


def _add_array(path, name):
    with np.load(path) as z:
        arrays = dict(z)
    arrays[name] = np.zeros(3)
    np.savez(path, **arrays)


def test_a_new_array_is_new_not_a_difference(two_copies):
    a, b = two_copies
    path = next(b.glob("bands/*/arrays.npz"))
    _add_array(path, "extra")
    assert compare_run_dirs(a, b) == Comparison([], [f"{path.relative_to(b)}: extra"])


def test_a_missing_array_is_a_difference(two_copies):
    a, b = two_copies
    path = next(a.glob("bands/*/arrays.npz"))
    _add_array(path, "extra")
    assert compare_run_dirs(a, b) == Comparison(
        [f"{path.relative_to(a)}: extra: only in the ref run"], []
    )


def test_a_new_result_file_is_new_not_a_difference(two_copies):
    a, b = two_copies
    path = next(a.glob("bands/*/arrays.npz"))
    path.unlink()
    assert compare_run_dirs(a, b) == Comparison([], [str(path.relative_to(a))])


def test_identical_results_exit_0(capsys):
    assert report(Comparison([], []), "main") == 0
    assert "identical results" in capsys.readouterr().out


def test_only_new_results_exit_0_and_name_each(capsys):
    new = ["tape.csv: extra", "bands/a/band_summary.json: band_extra"]
    assert report(Comparison([], new), "main") == 0
    out = capsys.readouterr().out
    assert "differ" not in out
    assert all(n in out for n in new)


def test_differences_exit_1_and_name_each(capsys):
    comparison = Comparison(["tape.csv: x: only in the ref run"], ["tape.csv: y"])
    assert report(comparison, "main") == 1
    out = capsys.readouterr().out
    assert "results differ" in out
    assert "tape.csv: x" in out and "tape.csv: y" in out


def test_a_missing_summary_key_is_a_difference(two_copies):
    a, b = two_copies
    summary = next(a.glob("bands/*/band_summary.json"))
    _edit_json(summary, lambda d: d.update(band_extra=1.0))
    assert compare_run_dirs(a, b) == Comparison(
        [f"{summary.relative_to(a)}: band_extra: only in the ref run"], []
    )
