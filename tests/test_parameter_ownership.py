"""ADR 0001: every parameter the run config sets is a required keyword-only
argument of the library constructor it feeds, so omitting one is a TypeError
(the library holds no competing default)."""

import pytest
import torch

from odil_wave import (
    AcquisitionGeometry,
    Grid,
    HelmholtzSolver,
    InverseLoss,
    LossConfig,
    Regulariser,
    SourceSignal,
    VelocityModel,
    WaveEquation,
    Wavefield,
)
from odil_wave.optimisation import JointFreqODIL

from conftest import EXTENT, INTERIOR

GRID_KW = dict(
    interior_shape=INTERIOR,
    interior_extent=EXTENT,
    c_min=1400.0,
    c_max=1700.0,
    pml_width=4,
    pml_power=3,
    pml_R0=1e-6,
    t_max=6e-5,
    init_nt=120,
    cfl_safety=0.9,
    device=torch.device("cpu"),
    dtype=torch.float64,
)
SOURCE_KW = dict(
    kind="tone_burst",
    f0=40e3,
    amplitude=1.0,
    n_cycles=2.0,
    envelope="gaussian",
    dimensionless=True,
)
GEOMETRY_KW = dict(
    n_receivers=8,
    n_sources=2,
    sigma_s=None,
    a_frac=0.9,
    b_frac=0.9,
    ring_center=None,
    source_spatial="point",
)
MODEL_KW = dict(profile="homogeneous", base=1500.0, contrast=0.0, pml_c=None)
OPERATOR_KW = dict(space_order=2, damping_scale=1.0)
JOINT_KW = dict(
    n_iter=1,
    inner_max_iter=2,
    history_size=5,
    line_search_fn="strong_wolfe",
    tolerance_grad=1e-12,
    tolerance_change=1e-14,
    lbfgs_lr=1.0,
    pde_weight=1.0,
    data_weight=1.0,
    reg_weight=0.0,
    log_every=1,
    verbose=False,
)


def _without(kw: dict, name: str) -> dict:
    return {k: v for k, v in kw.items() if k != name}


@pytest.mark.parametrize("name", list(GRID_KW))
def test_grid_requires_configured_params(name):
    Grid(**GRID_KW)  # complete kwargs construct
    with pytest.raises(TypeError, match=name):
        Grid(**_without(GRID_KW, name))


@pytest.mark.parametrize("name", list(SOURCE_KW))
def test_source_requires_configured_params(name, grid):
    SourceSignal(grid, **SOURCE_KW)
    with pytest.raises(TypeError, match=name):
        SourceSignal(grid, **_without(SOURCE_KW, name))


@pytest.mark.parametrize("name", list(GEOMETRY_KW))
def test_geometry_requires_configured_params(name, grid, source, freq_selection):
    AcquisitionGeometry(grid, source, freq_selection, **GEOMETRY_KW)
    with pytest.raises(TypeError, match=name):
        AcquisitionGeometry(grid, source, freq_selection, **_without(GEOMETRY_KW, name))


@pytest.mark.parametrize("name", list(MODEL_KW))
def test_velocity_model_requires_configured_params(name, grid):
    VelocityModel(grid, **MODEL_KW)
    with pytest.raises(TypeError, match=name):
        VelocityModel(grid, **_without(MODEL_KW, name))


@pytest.mark.parametrize(
    "profile, name",
    [
        ("shepp_logan", "scale"),
        ("shepp_logan_skull", "scale"),
        ("shepp_logan_skull", "skull_alpha"),
        ("shepp_logan_skull", "skull_sigma"),
    ],
)
def test_phantom_profiles_require_their_configured_kwargs(profile, name, grid):
    kw = dict(MODEL_KW, profile=profile, scale=0.85, skull_alpha=1.0, skull_sigma=0.0)
    VelocityModel(grid, **kw)
    with pytest.raises(TypeError, match=name):
        VelocityModel(grid, **_without(kw, name))


@pytest.mark.parametrize("name", list(OPERATOR_KW))
def test_operators_require_configured_params(name, wave_stack):
    wf, geom = wave_stack.wavefield, wave_stack.geometry
    with pytest.raises(TypeError, match=name):
        WaveEquation(wf, **_without(OPERATOR_KW, name))
    with pytest.raises(TypeError, match=name):
        HelmholtzSolver(wf, geom, **_without(OPERATOR_KW, name))


@pytest.mark.parametrize("name", ["kind", "eps"])
def test_regulariser_requires_configured_params(name):
    kw = dict(kind="tv_iso", eps=1e-8)
    with pytest.raises(TypeError, match=name):
        Regulariser(**_without(kw, name))


@pytest.fixture
def inverse_loss(wave_stack) -> InverseLoss:
    cfg = LossConfig(
        wave_stack.wave_eq,
        wave_stack.geometry,
        weights={"pde": 1.0, "data": 1.0},
        regulariser=None,
    )
    return InverseLoss(observed_wavefield=wave_stack.sols, config=cfg)


@pytest.mark.parametrize("name", ["weights", "regulariser"])
def test_loss_config_requires_configured_params(name, wave_stack):
    kw = dict(weights={"pde": 1.0, "data": 1.0}, regulariser=None)
    with pytest.raises(TypeError, match=name):
        LossConfig(wave_stack.wave_eq, wave_stack.geometry, **_without(kw, name))


@pytest.mark.parametrize("name", list(JOINT_KW))
def test_joint_optimiser_requires_configured_params(name, wave_stack, inverse_loss):
    wf = Wavefield(wave_stack.grid, wave_stack.freq, velocity_model=wave_stack.model)
    JointFreqODIL(wf, inverse_loss, u_init=wave_stack.sols, **JOINT_KW)
    with pytest.raises(TypeError, match=name):
        JointFreqODIL(
            wf, inverse_loss, u_init=wave_stack.sols, **_without(JOINT_KW, name)
        )


def test_joint_optimiser_requires_a_warm_start(wave_stack, inverse_loss):
    """No silent zero seed: a u = 0 start has an exactly zero model gradient."""
    wf = Wavefield(wave_stack.grid, wave_stack.freq, velocity_model=wave_stack.model)
    with pytest.raises(TypeError, match="u_init"):
        JointFreqODIL(wf, inverse_loss, **JOINT_KW)
    with pytest.raises(ValueError, match="warm-start"):
        JointFreqODIL(wf, inverse_loss, u_init=None, **JOINT_KW)
