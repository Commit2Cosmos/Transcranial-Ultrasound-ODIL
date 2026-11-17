"""ADR 0001: a configured parameter is a required keyword-only argument iff no
one value serves most calls; otherwise the library signature owns its default
and the run config reads it with ``library_default``."""

import pytest
import torch

from odil_wave import (
    AcquisitionGeometry,
    Band,
    Grid,
    HelmholtzSolver,
    Regulariser,
    SourceSignal,
    VelocityModel,
    WaveEquation,
)
from odil_wave.defaults import library_default
from odil_wave.experiment import ConfigError, resolve_config, validate_config
from odil_wave.experiment.problem import build_problem
from odil_wave.optimisation import JointFreqODIL

from conftest import EXTENT, INTERIOR

GRID_REQUIRED = dict(
    interior_shape=INTERIOR,
    interior_extent=EXTENT,
    c_min=1400.0,
    c_max=1700.0,
    t_max=6e-5,
)
SOURCE_REQUIRED = dict(kind="tone_burst", f0=40e3)
GEOMETRY_REQUIRED = dict(n_transducers=8)
JOINT_REQUIRED = dict(
    n_iter=1, pde_weight=1.0, data_weight=1.0, reg_weight=0.0, regulariser=None
)


def _without(kw: dict, name: str) -> dict:
    return {k: v for k, v in kw.items() if k != name}


@pytest.mark.parametrize("name", list(GRID_REQUIRED))
def test_grid_requires_problem_params(name):
    with pytest.raises(TypeError, match=name):
        Grid(**_without(GRID_REQUIRED, name))


def test_grid_builds_from_required_params_and_takes_device_dtype_names():
    grid = Grid(**GRID_REQUIRED, dtype="float64")
    assert grid.device == torch.device("cpu")
    assert grid.dtype == torch.float64
    assert Grid(**GRID_REQUIRED).dtype == torch.float32


def test_grid_rejects_an_unknown_dtype_name():
    with pytest.raises(ValueError, match="float16"):
        Grid(**GRID_REQUIRED, dtype="float16")


@pytest.mark.parametrize("name", list(SOURCE_REQUIRED))
def test_source_requires_problem_params(name, grid):
    with pytest.raises(TypeError, match=name):
        SourceSignal(grid, **_without(SOURCE_REQUIRED, name))


def test_geometry_requires_n_transducers(grid, source, freq_selection):
    AcquisitionGeometry(grid, source, freq_selection, **GEOMETRY_REQUIRED)
    with pytest.raises(TypeError, match="n_transducers"):
        AcquisitionGeometry(grid, source, freq_selection)


def test_velocity_model_and_regulariser_build_from_defaults(grid):
    assert VelocityModel(grid).profile == library_default(VelocityModel, "profile")
    assert Regulariser(grid=grid).kind == library_default(Regulariser, "kind")


def test_operators_require_space_order(wave_stack):
    wf, geom = wave_stack.wavefield, wave_stack.geometry
    WaveEquation(wf, space_order=2)
    HelmholtzSolver(wf, geom, space_order=2)
    with pytest.raises(TypeError, match="space_order"):
        WaveEquation(wf)
    with pytest.raises(TypeError, match="space_order"):
        HelmholtzSolver(wf, geom)


@pytest.fixture(scope="module")
def band(wave_stack) -> Band:
    return Band(
        wave_stack.geometry,
        wave_stack.model,
        wave_stack.obs_traces,
        space_order=2,
        keep_factors=False,
    )


def test_band_requires_space_order(wave_stack):
    args = (wave_stack.geometry, wave_stack.model, wave_stack.obs_traces)
    with pytest.raises(TypeError, match="space_order"):
        Band(*args, keep_factors=False)


@pytest.mark.parametrize("name", list(JOINT_REQUIRED))
def test_joint_optimiser_requires_problem_params(name, band):
    JointFreqODIL(band, **JOINT_REQUIRED)
    with pytest.raises(TypeError, match=name):
        JointFreqODIL(band, **_without(JOINT_REQUIRED, name))


def test_library_default_reads_the_signature():
    assert library_default(Grid, "pml_width") == 40
    assert library_default(HelmholtzSolver.solve, "verbose") is False


@pytest.mark.parametrize("owner, name", [(Grid, "c_max"), (JointFreqODIL, "n_iter")])
def test_library_default_of_a_required_param_raises(owner, name):
    with pytest.raises(TypeError, match=name):
        library_default(owner, name)


def test_library_default_of_an_unknown_param_raises():
    with pytest.raises(TypeError, match="not_a_param"):
        library_default(Grid, "not_a_param")


def test_null_ring_center_centres_the_ring_on_the_interior(tiny_run_config):
    cfg = resolve_config(tiny_run_config)
    assert cfg.acquisition.ring_center is None
    problem = build_problem(cfg)
    assert problem.center == problem.grid.interior_center


@pytest.mark.parametrize(
    "bad",
    [
        {"acquisition": {"ring_center": "grid_center"}},
        {"optimiser": {"line_search_fn": "none"}},
        {"runtime": {"dtype": "f64"}},
    ],
)
def test_old_config_representations_are_rejected(bad):
    with pytest.raises(ConfigError):
        validate_config(resolve_config(bad))


def test_null_line_search_validates():
    validate_config(resolve_config({"optimiser": {"line_search_fn": None}}))
