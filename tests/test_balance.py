"""Tests for the balance check (issue #32, ADR-0002)."""

import pytest

from odil_wave.experiment import ConfigError, resolve_config
from odil_wave.experiment.balance import GridCase, balance_check, scaled_config

# A small physical problem: an overdensity truth (sized in metres, unlike the
# Shepp-Logan skull rim, which bottoms out at 2 cells on small grids) and a
# Gaussian source resolved on the coarse grid (σ = 1.74h).
SMALL = {
    "runtime": {"dtype": "float64"},
    "grid": {
        "interior_shape": [32, 32],
        "interior_extent": [[0.0, 0.08], [0.0, 0.08]],
        "c_min": 1400.0,
        "c_max": 1750.0,
    },
    "pml": {"width": 12},
    "time": {"t_max": 1.5e-4, "init_nt": 300},
    "source": {"f0": 40e3, "n_cycles": 2.0},
    "acquisition": {
        "n_transducers": 8,
        "a_frac": 0.6,
        "b_frac": 0.6,
        "sigma_s": 4.5e-3,
    },
    "truth": {"profile": "overdensity", "contrast": 150.0, "extra": {"radius": 0.015}},
    "init": {"profile": "homogeneous"},
    "continuation": {"bands": [{"frequencies_hz": [40e3]}]},
}


@pytest.mark.parametrize(
    "source_spatial, cases",
    [("gaussian", ["1", "2", "1:2"]), ("point", ["1", "2"])],
)
def test_balance_is_grid_independent(source_spatial, cases):
    """The PDE/data ratio agrees across resolution and PML thickness."""
    cfg = resolve_config(SMALL, [f"acquisition.source_spatial={source_spatial}"])
    rows = balance_check(cfg, [GridCase.parse(c) for c in cases])
    first = rows[0].ratio
    for row in rows[1:]:
        assert row.ratio / first == pytest.approx(1.0, abs=0.2), str(row.case)


def test_scaled_config_keeps_the_physical_problem():
    cfg = resolve_config(SMALL)
    same = scaled_config(cfg, GridCase.parse("2"))
    thick = scaled_config(cfg, GridCase.parse("1:2"))
    assert same.grid.interior_shape == [64, 64] and same.pml.width == 24
    assert thick.grid.interior_shape == [32, 32] and thick.pml.width == 24
    assert same.grid.interior_extent == cfg.grid.interior_extent
    assert same.acquisition.sigma_s == cfg.acquisition.sigma_s


def test_balance_check_rejects_observation_pml():
    cfg = resolve_config(SMALL, ["pml.observation_width=20"])
    with pytest.raises(ConfigError, match="observation_width"):
        balance_check(cfg, [GridCase(1)])


@pytest.mark.parametrize("text", ["0", "a", "1:0", "1:x"])
def test_grid_case_rejects_bad_text(text):
    with pytest.raises(ConfigError):
        GridCase.parse(text)
