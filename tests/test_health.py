"""Optimiser-health warnings: the fixed rules of "Standard post-processing and
plot set" (#53), fed one band's tape rows at a time."""

from __future__ import annotations

import pytest

from odil_wave.experiment.health import HealthCheck

# The speed bounds of the rows below.
C_MIN, C_MAX = 1400.0, 1750.0


def row(it: int = 0, **changes) -> dict:
    """A healthy tape row at outer iteration ``it``."""
    r = {
        "band_index": 0,
        "iter": it,
        "c_min": 1480.0,
        "c_max": 1650.0,
        "sat_frac": 0.0,
        "grad_u_norm": 1.0,
        "grad_z_norm": 1.0,
    }
    r.update(changes)
    return r


def feed(rows: list) -> HealthCheck:
    check = HealthCheck(c_min=C_MIN, c_max=C_MAX)
    for r in rows:
        check(r)
    return check


def test_a_healthy_band_warns_nothing():
    rows = [row(i, grad_u_norm=1.0 / (i + 1)) for i in range(6)]
    assert feed(rows).warnings == []


@pytest.mark.parametrize(
    "changes, fires",
    [
        ({"c_min": 1413.9}, True),  # within 1% of 1400
        ({"c_min": 1414.1}, False),
        ({"c_max": 1732.6}, True),  # within 1% of 1750
        ({"c_max": 1732.4}, False),
        ({"sat_frac": 0.011}, True),
        ({"sat_frac": 0.01}, False),
    ],
)
def test_bound_and_saturation_thresholds(changes, fires):
    warnings = feed([row(0), row(1, **changes)]).warnings
    assert len(warnings) == (1 if fires else 0)
    if fires:
        key = next(iter(changes))
        assert key in warnings[0] and "iter 1" in warnings[0]


@pytest.mark.parametrize("norm", ["grad_u_norm", "grad_z_norm"])
def test_a_gradient_norm_rising_over_three_rows_warns(norm):
    # Three consecutive rises (1 -> 2 -> 3 -> 4) fire at the fourth row.
    rising = [row(i, **{norm: float(i + 1)}) for i in range(4)]
    warnings = feed(rising).warnings
    assert len(warnings) == 1 and norm in warnings[0] and "iter 3" in warnings[0]
    # Two rises then a fall do not.
    assert feed(rising[:3] + [row(3, **{norm: 0.5})]).warnings == []


def test_each_rule_warns_once_per_band_and_returns_new_warnings():
    check = HealthCheck(c_min=C_MIN, c_max=C_MAX)
    assert len(check(row(0, sat_frac=0.5))) == 1
    assert check(row(1, sat_frac=0.5)) == []
    assert len(check.warnings) == 1
