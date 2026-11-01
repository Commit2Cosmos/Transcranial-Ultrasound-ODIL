"""The live view: ``--live`` (CLI) and ``LiveVelocityView`` (notebook)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import yaml

from conftest import TINY_RUN_CONFIG, make_git_repo

import main


def _tape_rows(run_dir):
    with open(run_dir / "tape.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_live_without_a_display_warns_once_and_completes(tmp_path, capsys):
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY_RUN_CONFIG))
    run_dir = tmp_path / "run"
    argv = ["--config", str(config), "--out", str(run_dir), "--live"]

    code = main.main(argv, repo_root=make_git_repo(tmp_path / "repo"))

    assert code == 0
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["status"] == "completed"
    # Every logged iterate is still on the tape: start row plus 10 iterations.
    assert len(_tape_rows(run_dir)) == 11
    err = capsys.readouterr().err
    assert err.count("live view") == 1
    assert "no display" in err


def test_an_on_log_exception_never_aborts_the_run(tiny_run_config, new_run, capsys):
    from odil_wave.experiment import resolve_config, run_inverse

    calls = []

    def broken_view(logged):
        calls.append(logged.row["iter"])
        raise RuntimeError("view broke")

    result = run_inverse(resolve_config(tiny_run_config), new_run(), on_log=broken_view)

    assert result.status == "completed"
    assert len(_tape_rows(result.run_dir)) == 11
    assert calls == [0]  # reported once, then left alone
    assert capsys.readouterr().out.count("view broke") == 1


def test_ctrl_c_in_on_log_interrupts_the_run(tiny_run_config, new_run):
    from odil_wave.experiment import resolve_config, run_inverse

    def ctrl_c_at_iter_3(logged):
        if logged.row["iter"] == 3:
            raise KeyboardInterrupt

    result = run_inverse(
        resolve_config(tiny_run_config), new_run(), on_log=ctrl_c_at_iter_3
    )

    assert result.status == "interrupted"
    assert [int(r["iter"]) for r in _tape_rows(result.run_dir)] == [0, 1, 2, 3]


ROW_KEYS = ("rel_data_residual", "pde_src_ratio", "rmse_brain", "ssim_brain")


@pytest.fixture
def display(monkeypatch):
    """Pretend a display is attached, so the view draws under the Agg backend."""
    from odil_wave.experiment import live

    monkeypatch.setattr(live, "has_display", lambda: True)


def _logged(n, *, band_index=0, band_final=False):
    """Synthetic logged row number ``n``: every metric equals ``n + 1``."""
    from odil_wave.experiment.runner import LoggedRow

    row = {"band_index": band_index, "iter": n, "iter_total": n}
    row.update({k: float(n + 1) for k in ROW_KEYS})
    c_star = np.full((4, 5), 1500.0)
    return LoggedRow(row=row, c=c_star + n, c_star=c_star, band_final=band_final)


def _shown(view, column):
    """The y values a panel shows for ``column``."""
    for ax in view.figure.axes:
        for line in ax.get_lines():
            if line.get_label() == column:
                return list(line.get_ydata())
    raise AssertionError(f"no {column} line")


def test_redraws_at_most_every_5_s_and_at_each_band_final_row(display, capsys):
    from odil_wave.experiment import LiveVelocityView

    times = [0.0, 1.0, 2.0, 6.0, 7.0, 7.5, 8.0]
    view = LiveVelocityView(clock=iter(times).__next__)
    rows = [
        _logged(0),
        _logged(1),
        _logged(2),
        _logged(3),
        _logged(4, band_final=True),
        _logged(5, band_index=1),
        _logged(6, band_index=1),
    ]
    n_shown = []
    for logged in rows:
        view(logged)
        n_shown.append(len(_shown(view, "rel_data_residual")))

    # Drawn at 0 s (first row), 6 s (5 s on) and 7 s (band 0's final row).
    assert n_shown == [1, 1, 1, 4, 5, 5, 5]
    for column in ROW_KEYS:
        assert _shown(view, column) == [1.0, 2.0, 3.0, 4.0, 5.0]
    # c - c* of the last drawn row.
    (image,) = [im for ax in view.figure.axes for im in ax.get_images()]
    np.testing.assert_array_equal(np.asarray(image.get_array()).T, np.full((4, 5), 4.0))
    assert capsys.readouterr() == ("", "")


def test_the_last_redraw_shows_the_whole_run(display, tiny_run_config, new_run):
    """In a real run the window ends on the run's final row: every tape row,
    and c - c* from the saved final model and truth."""
    from odil_wave.experiment import (
        LiveVelocityView,
        deep_merge,
        resolve_config,
        run_inverse,
    )

    two_bands = {
        "continuation": {
            "bands": [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]
        },
        "optimiser": {"n_iter": 3, "log_every": 2},
    }
    view = LiveVelocityView()
    cfg = resolve_config(deep_merge(tiny_run_config, two_bands))
    result = run_inverse(cfg, new_run(), on_log=view)

    assert result.status == "completed"
    tape = _tape_rows(result.run_dir)
    assert len(tape) == 6  # per band: start row, iter 2, final iter 3
    for column in ROW_KEYS:
        np.testing.assert_allclose(
            _shown(view, column), [float(r[column]) for r in tape]
        )
    with np.load(result.run_dir / "problem.npz") as problem:
        c_star = problem["c_star"]
    with np.load(result.band_results[-1].band_dir / "arrays.npz") as arrays:
        c_final = arrays["c"]
    (image,) = [im for ax in view.figure.axes for im in ax.get_images()]
    # arrays.npz stores float32.
    np.testing.assert_allclose(
        np.asarray(image.get_array()).T, c_final - c_star, atol=1e-3
    )
