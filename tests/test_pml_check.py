"""Tests for the PML check (#37)."""

import pytest

from odil_wave.experiment import ConfigError, resolve_config
from odil_wave.experiment.pml_check import format_table, pml_check

SMALL = {
    "grid": {
        "interior_shape": [41, 41],
        "interior_extent": [[0.0, 0.08], [0.0, 0.08]],
        "c_min": 1400.0,
        "c_max": 1750.0,
    },
    "time": {"t_max": 1.5e-4, "init_nt": 300},
    "acquisition": {"n_transducers": 8, "a_frac": 0.6, "b_frac": 0.6},
}


def test_pml_reflection_falls_with_width():
    """Each extra few cells cut the reflection, at both ends of the band, and
    10 cells reach the 1e-4 level of #134's pilot."""
    rows = pml_check(resolve_config(SMALL), [4, 7, 10], [20e3, 100e3])
    for case in ("edge", "corner"):
        for f in (20e3, 100e3):
            errs = [
                r.interior_error for r in rows if r.case == case and r.frequency_hz == f
            ]
            assert errs == sorted(errs, reverse=True)
            assert errs[-1] < 3e-4
    assert "| 10 |" in format_table(rows)


def test_pml_check_rejects_empty_layers():
    with pytest.raises(ConfigError, match=">= 1"):
        pml_check(resolve_config(SMALL), [0], [20e3])


def _worst_error(space_order):
    cfg = resolve_config(
        {**SMALL, "grid": {**SMALL["grid"], "space_order": space_order}}
    )
    return max(r.interior_error for r in pml_check(cfg, [10], [20e3, 100e3]))


def test_order_4_reflects_no_more_than_twice_order_2():
    """The staggered order-P correction (#169) keeps the PML's reflection
    within 2x order 2's, at both ends of the band. Orders 6 and 8 are too slow
    to factor the reference here."""
    assert _worst_error(4) <= 2 * _worst_error(2)
