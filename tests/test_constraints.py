import math

import pytest

from odil_wave.experiment import check_constraints, resolve_config


def _check(name, overrides=()):
    """The check called ``name`` on the default config with ``overrides``."""
    [check] = [
        c
        for c in check_constraints(resolve_config({}, list(overrides)))
        if c.name == name
    ]
    return check


# Expected values: research/discretisation-constraints.md §2-3 (#135) and #124.
def test_default_dispersion_fails_at_order_2():
    check = _check("dispersion")
    assert check.value == pytest.approx(1.51, abs=0.01)
    assert check.threshold == pytest.approx(math.pi / 4)
    assert check.severity == "warn" and not check.ok


def test_order_4_passes_dispersion_on_the_default_grid():
    check = _check("dispersion", ["grid.space_order=4"])
    assert check.value == pytest.approx(0.084, abs=0.002)
    assert check.ok


@pytest.mark.parametrize("n, ok", [(64, False), (75, False), (76, True)])
def test_ring_sampling_needs_more_than_2kR_transducers(n, ok):
    """2kR = 75.4 at 80 kHz in water on the default ring (Simonetti 2007)."""
    check = _check("ring_sampling", [f"acquisition.n_transducers={n}"])
    assert check.value == n
    assert check.threshold == pytest.approx(75.4, abs=0.05)
    assert check.severity == "warn" and check.ok is ok


@pytest.mark.parametrize("nt, ok", [(1300, True), (50, False)])
def test_temporal_nyquist_is_an_error_check(nt, ok):
    """dt = t_max / (nt - 1): 1/(2 dt) = 1.299 MHz at 1300 samples, 49 kHz at 50."""
    check = _check("temporal_nyquist", [f"time.init_nt={nt}"])
    assert check.value == 80e3
    assert check.threshold == pytest.approx((nt - 1) / (2 * 500e-6))
    assert check.severity == "error" and check.ok is ok


def test_time_step_follows_the_cfl_when_init_nt_is_null():
    """dt = 0.9 h / (3100 sqrt 2) = 0.4138 us -> nt = 1210, 1/(2 dt) = 1.209 MHz."""
    check = _check("temporal_nyquist", ["time.init_nt=null"])
    assert check.threshold == pytest.approx(1209 / (2 * 500e-6))


def test_default_time_symbol_error_passes():
    """omega dt = 0.194 -> (omega dt)^2 / 24 = 1.57e-3 over 12 wavelengths."""
    check = _check("time_symbol")
    assert check.value == pytest.approx(0.118, abs=0.002)
    assert check.threshold == pytest.approx(math.pi / 4)
    assert check.severity == "warn" and check.ok


@pytest.mark.parametrize("width, ok", [(15, True), (12, True), (11, False)])
def test_pml_needs_12_cells(width, ok):
    check = _check("pml_width", [f"pml.width={width}"])
    assert check.value == width and check.threshold == 12
    assert check.severity == "warn" and check.ok is ok


def _bands(*freqs):
    return f'continuation.bands=[{{"frequencies_hz": {list(freqs)}}}]'


@pytest.mark.parametrize(
    "freqs, value",
    [
        ([80e3], 1.0),
        # Gaussian tone burst, sigma_t = 3 / (6 * 80 kHz):
        # exp(-(2 pi 20 kHz sigma_t)^2 / 2)
        ([100e3], 0.735),
        # checked at the top band frequency only
        ([40e3, 80e3], 1.0),
    ],
)
def test_source_spectrum_at_the_top_band_frequency(freqs, value):
    check = _check("source_spectrum", [_bands(*freqs)])
    assert check.value == pytest.approx(value, abs=0.002)
    assert check.threshold == 0.5 and check.severity == "warn"
    assert check.ok is (value >= 0.5)


def test_ricker_spectrum_relative_to_its_peak():
    """(f / f0)^2 exp(1 - (f / f0)^2) at f = 2 f0: 4 e^-3."""
    check = _check("source_spectrum", ["source.kind=ricker", _bands(160e3)])
    assert check.value == pytest.approx(4 * math.exp(-3))


def test_rectangular_tone_burst_spectrum():
    """sinc((f - f0) n_cycles / f0) at 100 kHz, 3 cycles of 80 kHz: sinc(0.75)."""
    check = _check("source_spectrum", ["source.envelope=rectangular", _bands(100e3)])
    assert check.value == pytest.approx(math.sin(0.75 * math.pi) / (0.75 * math.pi))


SOURCE_WIDTH_CHECKS = ("source_resolved", "source_low_pass", "ring_clearance")


def test_default_source_width_checks_pass():
    """#124's defaults: sigma_s = 1.24 h, k_max sigma_s = 0.97, clearance 5 sigma_s."""
    values = {n: _check(n).value for n in SOURCE_WIDTH_CHECKS}
    assert values == pytest.approx(
        {"source_resolved": 1.24, "source_low_pass": 0.97, "ring_clearance": 5.0},
        abs=0.01,
    )
    assert all(
        _check(n).ok and _check(n).severity == "warn" for n in SOURCE_WIDTH_CHECKS
    )


def test_a_point_source_has_no_source_width_checks():
    cfg = resolve_config({}, ["acquisition.source_spatial=point"])
    names = {c.name for c in check_constraints(cfg)}
    assert names.isdisjoint(SOURCE_WIDTH_CHECKS)
