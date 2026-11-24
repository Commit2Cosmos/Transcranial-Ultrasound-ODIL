"""config.py's enum constants mirror the library's
accepted values by hand. These tests fail when the two drift apart."""

from typing import get_args

import pytest

from odil_wave import Regulariser, SourceSignal, WaveEquation
from odil_wave.experiment import config
from odil_wave.loss.regulariser import RegKind
from odil_wave.optimisation.preconditioners import (
    MODEL_PRECONDITIONERS,
    WAVEFIELD_PRECONDITIONERS,
)

from conftest import make_model


def test_regulariser_kinds_match_library(grid):
    assert set(config._REGULARISERS) == set(get_args(RegKind))
    for kind in config._REGULARISERS:
        Regulariser(kind=kind, eps=1e-4, grid=grid)


def test_model_preconditioners_match_library():
    assert config._MODEL_PRECONDITIONERS == MODEL_PRECONDITIONERS


def test_wavefield_preconditioners_match_library():
    assert config._WAVEFIELD_PRECONDITIONERS == WAVEFIELD_PRECONDITIONERS


@pytest.mark.parametrize("order", range(1, 13))
def test_space_orders_match_wave_equation(order, wave_stack):
    if order in config._SPACE_ORDERS:
        WaveEquation(wave_stack.wavefield, space_order=order, damping_scale=1.0)
    else:
        with pytest.raises(ValueError):
            WaveEquation(wave_stack.wavefield, space_order=order, damping_scale=1.0)


@pytest.mark.parametrize("profile", config._PROFILES)
def test_profiles_build_a_velocity_model(profile, grid):
    make_model(grid, profile=profile, scale=0.85, skull_alpha=1.0, skull_sigma=0.0)


def test_unknown_profile_is_rejected(grid):
    with pytest.raises(ValueError):
        make_model(grid, profile="not_a_profile")


@pytest.mark.parametrize("kind", config._SOURCE_KINDS)
@pytest.mark.parametrize("envelope", config._ENVELOPES)
def test_source_kinds_and_envelopes_build_a_source(kind, envelope, grid):
    SourceSignal(grid, kind=kind, f0=40e3, n_cycles=2.0, envelope=envelope)


def test_unknown_source_kind_is_rejected(grid):
    with pytest.raises(ValueError):
        SourceSignal(grid, kind="chirp", f0=40e3)
