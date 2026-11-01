"""The arrays a run saves (``problem.npz``, each band's ``arrays.npz``) and its
band summaries, as laid out in the run-layer README's "Where each artifact is
written", checked against values recomputed independently of the run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from odil_wave.experiment import build_problem, deep_merge

from conftest import TINY_RUN_CONFIG
from test_tape import band_rows, read_tape, run_cli


def band_dirs(run_dir: Path) -> list:
    return sorted(p for p in (run_dir / "bands").iterdir() if p.is_dir())


def load_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# problem.npz
# --------------------------------------------------------------------------- #
def test_problem_npz_holds_the_truth_start_model_and_layout(tiny_two_band_run):
    problem = build_problem(tiny_two_band_run.config)
    saved = load_npz(tiny_two_band_run.run_dir / "problem.npz")

    # 20 x 20 interior inside a 6-cell PML.
    assert saved["interior_slice"].tolist() == [[6, 26], [6, 26]]
    sl = problem.grid.interior_slice
    np.testing.assert_allclose(
        saved["c_star"], problem.truth_velocity.c[sl].numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        saved["c0"], problem.init_velocity.c[sl].numpy(), rtol=1e-6
    )
    assert saved["c_star"].shape == (20, 20)
    assert float(saved["pml_c"]) == pytest.approx(problem.init_velocity.pml_c)
    _, brain, skull = problem.truth_velocity.skull_region_masks()
    assert np.array_equal(saved["brain_mask"], brain[sl].numpy())
    assert np.array_equal(saved["skull_mask"], skull[sl].numpy())
    assert saved["brain_mask"].any() and saved["skull_mask"].any()
    # 8 transducers evenly spaced around the ring.
    assert saved["transducer_angles_deg"].tolist() == [
        0,
        45,
        90,
        135,
        180,
        225,
        270,
        315,
    ]


def test_saved_arrays_are_single_precision_and_masks_bool(tiny_two_band_run):
    """The tiny run is float64; its arrays are still saved as float32/complex64."""
    assert tiny_two_band_run.config.runtime.dtype == "float64"
    paths = [tiny_two_band_run.run_dir / "problem.npz"] + [
        d / "arrays.npz" for d in band_dirs(tiny_two_band_run.run_dir)
    ]
    for path in paths:
        for key, arr in load_npz(path).items():
            if key.endswith("_mask"):
                assert arr.dtype == bool, (path.name, key)
            elif key in ("interior_slice", "shots"):
                assert np.issubdtype(arr.dtype, np.integer), (path.name, key)
            elif np.iscomplexobj(arr):
                assert arr.dtype == np.complex64, (path.name, key)
            else:
                assert arr.dtype == np.float32, (path.name, key)


# --------------------------------------------------------------------------- #
# Band arrays.npz
# --------------------------------------------------------------------------- #
def per_shot_residual(d_syn: np.ndarray, d_obs: np.ndarray) -> np.ndarray:
    """Each shot's ``||d_syn - d_obs|| / ||d_obs||`` over frequencies x receivers."""
    diff = np.linalg.norm((d_syn - d_obs).reshape(len(d_obs), -1), axis=1)
    return diff / np.linalg.norm(d_obs.reshape(len(d_obs), -1), axis=1)


def test_band_arrays_hold_the_data_wavefields_and_model(tiny_two_band_run):
    """Each band saves its model, all-shot data and the wavefields of the
    transducers nearest 0/90/180/270 degrees plus the worst-fit shot."""
    run_dir = tiny_two_band_run.run_dir
    problem = build_problem(tiny_two_band_run.config)
    bands = tiny_two_band_run.config.continuation.bands
    dirs = band_dirs(run_dir)
    assert len(dirs) == 2
    nx, ny = problem.grid.shape
    sl = problem.grid.interior_slice
    for band_cfg, band_dir in zip(bands, dirs):
        band = problem.make_band(band_cfg.frequencies_hz)
        saved = load_npz(band_dir / "arrays.npz")
        assert set(saved) == {
            "freqs",
            "c",
            "d_obs",
            "d_syn_start",
            "d_syn_final",
            "pde_scale",
            "shots",
            "u",
            "u_star",
        }
        assert saved["pde_scale"].shape == (8, 1)
        assert (saved["pde_scale"] > 0).all()
        # The frequencies solved at: the FFT bins nearest the requested ones.
        np.testing.assert_allclose(saved["freqs"], band.freq.frequencies, rtol=1e-6)
        assert saved["c"].shape == (20, 20)

        u_star = torch.stack([wf.amplitude for wf in band.observed_wfs])
        d_obs = band.geom.extract_observations(u_star).numpy()
        assert saved["d_obs"].shape == (8, 1, 7)
        np.testing.assert_allclose(saved["d_obs"], d_obs, rtol=1e-5, atol=1e-12)
        for key in ("d_syn_start", "d_syn_final"):
            assert saved[key].shape == saved["d_obs"].shape

        # 8 transducers: 0, 90, 180, 270 degrees are transducers 0, 2, 4, 6.
        worst = int(np.argmax(per_shot_residual(saved["d_syn_final"], saved["d_obs"])))
        expected = [0, 2, 4, 6] + ([] if worst in (0, 2, 4, 6) else [worst])
        assert saved["shots"].tolist() == expected
        n = len(expected)
        assert saved["u"].shape == (n, 1, nx, ny)
        assert saved["u_star"].shape == (n, 1, 20, 20)
        np.testing.assert_allclose(
            saved["u_star"],
            u_star.numpy()[expected][:, :, sl[0], sl[1]],
            rtol=1e-5,
            atol=1e-12,
        )
        # u is the full-grid final wavefield: sampled at each saved shot's
        # receivers it gives that shot's final synthetic data.
        ij = band.geom.receiver_ij.numpy()
        for i, s in enumerate(expected):
            recv = saved["u"][i][:, ij[s, :, 0], ij[s, :, 1]]
            np.testing.assert_allclose(recv, saved["d_syn_final"][s], rtol=1e-6)

        summary = json.loads((band_dir / "band_summary.json").read_text())
        assert summary["worst_fit_shot"] == worst


def test_tape_metrics_recompute_from_the_saved_arrays(tiny_two_band_run):
    """Each band's start and final rows, recomputed from problem.npz and its
    arrays.npz alone."""
    run_dir = tiny_two_band_run.run_dir
    _, rows = read_tape(run_dir)
    prob = load_npz(run_dir / "problem.npz")
    brain, skull = prob["brain_mask"], prob["skull_mask"]
    truth = prob["c_star"].astype(np.float64)
    for b, band_dir in enumerate(band_dirs(run_dir)):
        saved = load_npz(band_dir / "arrays.npz")
        d_obs = saved["d_obs"].astype(np.complex128)
        start, final = band_rows(rows, b)[0], band_rows(rows, b)[-1]
        for row, key in ((start, "d_syn_start"), (final, "d_syn_final")):
            d_syn = saved[key].astype(np.complex128)
            expected = np.linalg.norm(d_syn - d_obs) / np.linalg.norm(d_obs)
            assert row["rel_data_residual"] == pytest.approx(expected, rel=1e-5)
        diff = saved["c"].astype(np.float64) - truth
        assert final["rmse_brain"] == pytest.approx(
            np.sqrt(np.mean(diff[brain] ** 2)), rel=1e-4
        )
        assert final["bias_skull"] == pytest.approx(
            np.mean(diff[skull]), rel=1e-4, abs=1e-3
        )
        assert final["c_min"] == pytest.approx(saved["c"].min(), rel=1e-6)
        assert final["c_max"] == pytest.approx(saved["c"].max(), rel=1e-6)


# --------------------------------------------------------------------------- #
# Health warnings and partial bands
# --------------------------------------------------------------------------- #
def test_health_warnings_are_printed_and_summarised(tmp_path, git_repo, capsys):
    """A lower speed bound just under the 1500 m/s starting model puts c within
    1% of it and pins the water cells' sigmoid at the bound."""
    user = deep_merge(
        TINY_RUN_CONFIG, {"grid": {"c_min": 1499.9}, "optimiser": {"n_iter": 1}}
    )
    run_dir = run_cli(tmp_path, git_repo, user)
    out = capsys.readouterr().out

    (band_dir,) = band_dirs(run_dir)
    warnings = json.loads((band_dir / "band_summary.json").read_text())["warnings"]
    assert any("c_min" in w and "speed bound" in w for w in warnings)
    assert any("sat_frac" in w for w in warnings)
    run_warnings = json.loads((run_dir / "run_summary.json").read_text())["warnings"]
    assert set(warnings) <= set(run_warnings)
    for w in warnings:
        assert w in out


def interrupt_when(monkeypatch, target, name: str, fires) -> None:
    """Replace ``target.name`` with a wrapper that raises ``KeyboardInterrupt``
    (Ctrl-C) when ``fires(*args)`` is true, else calls through."""
    original = getattr(target, name)

    def wrapper(*args, **kwargs):
        if fires(*args):
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapper)


@pytest.mark.parametrize("moment", ["at the final row", "while saving the band"])
def test_an_interrupted_band_saves_no_arrays_or_summary(
    tmp_path, git_repo, monkeypatch, moment
):
    from odil_wave.experiment import runner, tape

    if moment == "at the final row":
        # Ctrl-C as the band's final iterate is logged (n_iter = 1).
        interrupt_when(
            monkeypatch, tape.BandTape, "record", lambda bt, it: it.iter == 1
        )
    else:
        # Ctrl-C after arrays.npz is written, before band_summary.json.
        interrupt_when(
            monkeypatch,
            runner,
            "write_json",
            lambda path, *_: Path(path).name == "band_summary.json",
        )
    user = deep_merge(TINY_RUN_CONFIG, {"optimiser": {"n_iter": 1}})
    run_dir = run_cli(tmp_path, git_repo, user, expect=1)

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert list((run_dir / "bands").iterdir()) == []
    _, rows = read_tape(run_dir)
    assert len(rows) >= 1
    assert (run_dir / "problem.npz").exists()
