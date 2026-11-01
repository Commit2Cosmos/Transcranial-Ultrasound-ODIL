"""On-demand plots: library functions that take a run directory and return a
figure, for digging deeper from a notebook. Each reads the saved run and solves
nothing."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from odil_wave import HelmholtzSolver, VelocityModel, Wavefield
from odil_wave.experiment import build_problem, on_demand

from test_saved_run import band_dirs, load_npz
from test_tape import band_rows, read_tape, saved_final_model


def test_source_spectrum_marks_each_bands_inversion_bins(tiny_two_band_run, tmp_path):
    """Reads the config only: a directory holding just the resolved config."""
    config_only = tmp_path / "config_only"
    config_only.mkdir()
    shutil.copy(tiny_two_band_run.run_dir / "config_resolved.yaml", config_only)
    solved_khz = [
        load_npz(band / "arrays.npz")["freqs"] / 1e3
        for band in band_dirs(tiny_two_band_run.run_dir)
    ]

    (ax,) = on_demand.source_spectrum(config_only).axes

    spectrum, *markers = ax.get_lines()
    assert np.max(spectrum.get_ydata()) == 1.0  # normalised |S(f)|
    marked = [list(line.get_xdata())[:1] for line in markers]
    np.testing.assert_allclose(marked, solved_khz, rtol=1e-6)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend[1:] == ["band 0: 30khz", "band 1: 40khz"]


def series(ax, label: str) -> tuple:
    """The x and y data of the line labelled ``label`` on ``ax``, joined over bands."""
    xs, ys = [], []
    for line in ax.get_lines():
        if line.get_label() == label or line.get_gid() == label:
            xs += list(line.get_xdata())
            ys += list(line.get_ydata())
    return xs, ys


def test_optimiser_health_plots_the_tape_against_the_speed_bounds(tiny_two_band_run):
    _, rows = read_tape(tiny_two_band_run.run_dir)
    iter_total = [r["iter_total"] for r in rows]

    speed, saturation, gradients = on_demand.optimiser_health(
        tiny_two_band_run.run_dir
    ).axes

    for ax, columns in (
        (speed, ["c_min", "c_max"]),
        (saturation, ["sat_frac"]),
        (gradients, ["grad_u_norm", "grad_z_norm"]),
    ):
        assert ax.get_xlabel() == "iter_total"
        for column in columns:
            xs, ys = series(ax, column)
            assert xs == iter_total
            np.testing.assert_allclose(ys, [r[column] for r in rows])
    # The tiny run config's grid bounds.
    bounds = [
        line.get_ydata()[0]
        for line in speed.get_lines()
        if list(line.get_ydata()) in ([1400.0, 1400.0], [1750.0, 1750.0])
    ]
    assert sorted(bounds) == [1400.0, 1750.0]
    assert gradients.get_yscale() == "log"


def shown(fig) -> list:
    """The image of every image panel, in (x, y) index order."""
    return [
        np.asarray(ax.get_images()[0].get_array()).T
        for ax in fig.axes
        if ax.get_images()
    ]


def test_wavefield_comparison_shows_a_saved_shot_against_the_true_wavefield(
    tiny_two_band_run,
):
    run_dir = tiny_two_band_run.run_dir
    arrays = load_npz(band_dirs(run_dir)[0] / "arrays.npz")
    (x0, x1), (y0, y1) = load_npz(run_dir / "problem.npz")["interior_slice"]
    k = list(arrays["shots"]).index(2)  # transducer 2 sits at 90 degrees
    u = arrays["u"][k, 0, x0:x1, y0:y1]
    u_star = arrays["u_star"][k, 0]

    fig = on_demand.wavefield_comparison(run_dir, band=0, shot=2, frequency=0)

    expected = [u.real, u_star.real, (u - u_star).real]
    expected += [np.abs(u), np.abs(u_star), np.abs(u - u_star)]
    for image, field in zip(shown(fig), expected, strict=True):
        np.testing.assert_allclose(image, field, rtol=1e-6)


def test_a_wavefield_is_shown_only_for_a_saved_shot(tiny_two_band_run):
    arrays = load_npz(band_dirs(tiny_two_band_run.run_dir)[-1] / "arrays.npz")
    saved = [int(s) for s in arrays["shots"]]
    unsaved = min(set(range(8)) - set(saved))

    with pytest.raises(ValueError, match=re.escape(f"saved shots: {saved}")):
        on_demand.wavefield_comparison(tiny_two_band_run.run_dir, shot=unsaved)


def test_shot_data_fit_is_amplitude_and_unwrapped_phase_by_receiver_angle(
    tiny_two_band_run,
):
    """Any shot has data, saved wavefield or not; its receivers are the other
    seven transducers, at 45 degree steps around the ring."""
    run_dir = tiny_two_band_run.run_dir
    arrays = load_npz(band_dirs(run_dir)[-1] / "arrays.npz")
    angles = [0.0, 45.0, 90.0, 180.0, 225.0, 270.0, 315.0]  # shot 3 at 135 degrees

    amplitude, phase = on_demand.shot_data_fit(run_dir, shot=3, frequency=0).axes

    for name, data in (("d_obs", arrays["d_obs"]), ("d_syn", arrays["d_syn_final"])):
        trace = data[3, 0]
        xs, ys = series(amplitude, name)
        np.testing.assert_allclose(xs, angles)
        np.testing.assert_allclose(ys, np.abs(trace), rtol=1e-6)
        xs, ys = series(phase, name)
        np.testing.assert_allclose(xs, angles)
        np.testing.assert_allclose(ys, np.unwrap(np.angle(trace)), rtol=1e-6)
    assert amplitude.get_xlabel() == phase.get_xlabel() == "receiver angle [deg]"


def with_helmholtz_wavefields(run_dir: Path, tmp_path: Path, config) -> Path:
    """A copy of ``run_dir`` whose last band's saved ``u`` is a Helmholtz solve
    on its saved final model: a wavefield that satisfies the PDE."""
    copy = tmp_path / "helmholtz"
    shutil.copytree(run_dir, copy)
    band_dir = band_dirs(copy)[-1]
    arrays = load_npz(band_dir / "arrays.npz")
    problem = build_problem(config)
    band = problem.make_band(config.continuation.bands[-1].frequencies_hz)
    model = VelocityModel.from_field(
        problem.grid,
        torch.as_tensor(saved_final_model(band_dir, problem), dtype=problem.dtype),
        pml_c=problem.init_velocity.pml_c,
    )
    wf = Wavefield(problem.grid, band.freq, velocity_model=model)
    solver = HelmholtzSolver(
        wf,
        band.geom,
        space_order=problem.space_order,
        damping_scale=problem.damping_scale,
    )
    u = np.stack([s.amplitude.numpy() for s in solver.solve(verbose=False)])
    arrays["u"] = u[arrays["shots"]].astype(np.complex64)
    np.savez(band_dir / "arrays.npz", **arrays)
    return copy


def test_pde_residual_map_vanishes_for_a_wavefield_that_solves_the_pde(
    tiny_two_band_run, tmp_path
):
    """|r_pde| / RMS(q) over the full grid (the PML too), the per-cell form of
    the tape's pde_src_ratio. The optimiser's iterate trades PDE residual for
    data fit; a Helmholtz solve on the same model has none, up to single
    precision."""
    run_dir = tiny_two_band_run.run_dir
    solved = with_helmholtz_wavefields(run_dir, tmp_path, tiny_two_band_run.config)
    final_row = band_rows(read_tape(run_dir)[1], 1)[-1]

    (iterate,) = shown(on_demand.pde_residual_map(run_dir, shot=2))
    (exact,) = shown(on_demand.pde_residual_map(solved, shot=2))

    assert iterate.shape == exact.shape == (20 + 2 * 6, 20 + 2 * 6)
    assert np.sqrt(np.mean(iterate**2)) > 0.1 * final_row["pde_src_ratio"]
    assert np.sqrt(np.mean(exact**2)) < 1e-3 * final_row["pde_src_ratio"]


ON_DEMAND_PLOTS = [
    on_demand.source_spectrum,
    on_demand.optimiser_health,
    on_demand.wavefield_comparison,
    on_demand.shot_data_fit,
    on_demand.pde_residual_map,
]


@pytest.mark.parametrize("plot", ON_DEMAND_PLOTS, ids=lambda f: f.__name__)
def test_an_on_demand_plot_returns_a_figure_and_solves_nothing(
    plot, tiny_two_band_run, monkeypatch
):
    from matplotlib.figure import Figure

    from odil_wave.experiment.problem import Problem
    from odil_wave.optimisation import JointFreqODIL

    def forbidden(*args, **kwargs):
        raise AssertionError("an on-demand plot solved or optimised")

    for owner, name in [
        (JointFreqODIL, "__init__"),
        (JointFreqODIL, "minimise"),
        (HelmholtzSolver, "solve"),
        (HelmholtzSolver, "factorize"),
        (Problem, "make_band"),
        (Problem, "warm_start"),
    ]:
        monkeypatch.setattr(owner, name, forbidden)

    assert isinstance(plot(tiny_two_band_run.run_dir), Figure)
