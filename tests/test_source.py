import pytest
import torch

from odil_wave import FrequencySelection, HelmholtzSolver, Wavefield

# conftest's tone burst: n_cycles / f0 = 2 / 40 kHz = 5e-5 s < the tiny grid's
# t_max (6e-5 s), so the burst fits.
from conftest import (
    F0,
    N_CYCLES,
    T_MAX,
    make_geometry,
    make_grid,
    make_model,
    make_source,
)


def test_tone_burst_shape_and_dtype(grid):
    """The tone burst samples onto grid.t with the grid's shape and dtype."""
    src = make_source(grid, t_start=0.0)
    w = src.waveform()
    assert w.shape == (grid.nt,)
    assert w.dtype == grid.dtype
    assert torch.isfinite(w).all()
    assert float(w.abs().max()) > 0


def test_amplitude_is_waveform_peak(grid):
    """``amplitude`` is the peak of s(t), with no hidden scaling."""
    t_centre = float(grid.t[grid.nt // 2])
    src = make_source(grid, kind="ricker", amplitude=2.5, t_centre=t_centre)
    assert float(src.waveform().abs().max()) == pytest.approx(2.5, rel=1e-12)


def test_tone_burst_is_zero_outside_its_window(grid):
    """The tone burst is zero before t_start and after n_cycles / f0."""
    t_start = 10 * grid.dt
    w = make_source(grid, t_start=t_start).waveform()
    outside = (grid.t < t_start) | (grid.t >= t_start + N_CYCLES / F0)
    torch.testing.assert_close(w[outside], torch.zeros_like(w[outside]), rtol=0, atol=0)
    assert float(w.abs().max()) > 0


def test_tone_burst_rectangular_envelope(grid):
    """The rectangular envelope is supported and produces a non-trivial burst."""
    tb = make_source(grid, t_start=0.0, envelope="rectangular")
    w = tb.waveform()
    assert w.shape == (grid.nt,)
    assert float(w.abs().max()) > 0


def test_tone_burst_bad_envelope_raises(grid):
    """An unsupported envelope is rejected at construction."""
    with pytest.raises(ValueError):
        make_source(grid, envelope="triangular")


@pytest.mark.parametrize("t_start", [-1e-6, 1.0])
def test_tone_burst_out_of_range_start_raises(grid, t_start):
    """A burst that starts before t=0 or runs past the time axis is rejected."""
    with pytest.raises(ValueError, match="t_start"):
        make_source(grid, t_start=t_start)


def test_tone_burst_too_long_for_grid_raises(grid):
    """A burst longer than the time axis is rejected, even at t_start = 0."""
    with pytest.raises(ValueError, match="t_start"):
        make_source(grid, t_start=0.0, n_cycles=100)


def test_ricker_centre_time(grid):
    """The Ricker wavelet peaks at t_centre, which defaults to 1 / f0."""
    for t_centre, expected in [(None, 1.0 / F0), (2.0 / F0, 2.0 / F0)]:
        src = make_source(grid, kind="ricker", t_centre=t_centre)
        peak = grid.t[src.waveform().abs().argmax()]
        assert float(peak) == pytest.approx(expected, abs=float(grid.dt))


def test_unknown_kind_raises(grid):
    """An unknown source kind is rejected at construction."""
    with pytest.raises(ValueError, match="kind"):
        make_source(grid, kind="bogus")


def test_spectrum_matches_fft_of_waveform(grid, freq_selection):
    """spectrum(fs) equals the FFT of the sampled waveform gathered on fs bins."""
    src = make_source(grid, t_start=0.0)
    spec = src.spectrum(freq_selection)
    assert spec.shape == (freq_selection.n_frequencies,)
    assert spec.is_complex()
    expected = freq_selection.fft_time_series(src.waveform(), dim=0)
    assert torch.allclose(spec, expected)


def test_spectrum_uses_selection_bins(grid):
    """Different bin selections gather different spectral samples."""
    src = make_source(grid, t_start=0.0)
    fs2 = FrequencySelection.from_bins(grid, [2, 4, 6])
    spec = src.spectrum(fs2)
    assert spec.shape == (3,)
    full = torch.fft.fft(src.waveform(), dim=0) * grid.dt
    assert torch.allclose(spec, full[torch.tensor([2, 4, 6])])


def _source_at_nt(nt: int, **source_kw):
    """A source on a grid of ``nt`` samples with ``nt * dt`` fixed, so the
    selected bin frequency does not move with ``nt``."""
    grid = make_grid(t_max=T_MAX * (nt - 1) / nt, init_nt=nt)
    freq = FrequencySelection.from_frequencies(grid, [F0])
    return grid, freq, make_source(grid, **source_kw)


def _ricker_at_nt(nt: int):
    return _source_at_nt(nt, kind="ricker")


@pytest.mark.parametrize(
    "source_kw",
    [
        dict(kind="ricker"),
        dict(envelope="gaussian"),
        dict(envelope="rectangular"),
    ],
)
def test_spectrum_independent_of_nt(source_kw):
    """At a fixed bin frequency, doubling nt leaves the source spectrum unchanged
    (an unnormalised DFT would double it; a burst built from samples would move
    by ~3%)."""
    (_, fs1, src1), (_, fs2, src2) = (
        _source_at_nt(120, **source_kw),
        _source_at_nt(240, **source_kw),
    )
    assert float(fs1.frequencies) == pytest.approx(float(fs2.frequencies))
    torch.testing.assert_close(
        src1.spectrum(fs1), src2.spectrum(fs2), rtol=1e-4, atol=0
    )


def test_true_traces_independent_of_nt():
    """Doubling nt at a fixed bin frequency leaves the traces unchanged: the exact
    time symbols (#116) do not depend on dt, so only the source spectrum's
    quadrature (1e-4, above) moves them."""
    traces = []
    for nt in (120, 240):
        grid, freq, src = _ricker_at_nt(nt)
        geom = make_geometry(grid, src, freq)
        wf = Wavefield(grid, freq, velocity_model=make_model(grid))
        solved = HelmholtzSolver(wf, geom, space_order=2, damping_scale=1.0).solve()
        traces.append(
            geom.extract_observations(torch.stack([w.amplitude for w in solved]))
        )
    rel = torch.linalg.vector_norm(traces[1] - traces[0]) / torch.linalg.vector_norm(
        traces[0]
    )
    assert float(rel) < 1e-4
