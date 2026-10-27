"""config.py stays torch-free, so its enum constants mirror the library's
accepted values by hand. These tests fail when the two drift apart."""

from typing import get_args

import pytest

from odil_wave import Regulariser, WaveEquation
from odil_wave.experiment import config
from odil_wave.loss.regulariser import RegKind

from conftest import make_model


def test_regulariser_kinds_match_library():
    assert set(config._REGULARISERS) == set(get_args(RegKind))
    for kind in config._REGULARISERS:
        Regulariser(kind=kind, eps=1e-8)


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
