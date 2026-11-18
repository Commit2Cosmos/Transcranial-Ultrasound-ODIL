import pytest
import torch

from odil_wave import FrequencySelection

# conftest's tone burst: n_cycles / f0 = 2 / 40 kHz = 5e-5 s < the tiny grid's
# t_max (6e-5 s), so the burst fits.
from conftest import F0, make_source


def test_tone_burst_shape_and_dtype(grid):
    """The tone burst samples onto grid.t with the grid's shape and dtype."""
    src = make_source(grid, offset=0)
    w = src.waveform()
    assert w.shape == (grid.nt,)
    assert w.dtype == grid.dtype
    assert torch.isfinite(w).all()
    assert float(w.abs().max()) > 0


def test_dimensionless_scaling(grid):
    """dimensionless=True multiplies the wavelet by the grid's natural amplitude."""
    phys = make_source(grid, offset=0, amplitude=1.0, dimensionless=False)
    nd = make_source(grid, offset=0, amplitude=1.0, dimensionless=True)
    # The natural amplitude scales the whole signal, so the peak ratio is exact.
    ratio = nd.waveform().abs().max() / phys.waveform().abs().max()
    assert float(ratio) == pytest.approx(grid.natural_source_amplitude(F0), rel=1e-6)


def test_tone_burst_length_and_offset(grid):
    """Tone burst fills the whole time axis and stays zero before its offset."""
    offset = 10
    tb = make_source(grid, offset=offset)
    w = tb.waveform()
    assert w.shape == (grid.nt,)
    assert torch.all(w[:offset] == 0)
    assert float(w.abs().max()) > 0


def test_tone_burst_rectangular_envelope(grid):
    """The rectangular envelope is supported and produces a non-trivial burst."""
    tb = make_source(grid, offset=0, envelope="rectangular")
    w = tb.waveform()
    assert w.shape == (grid.nt,)
    assert float(w.abs().max()) > 0


def test_tone_burst_bad_envelope_raises(grid):
    """An unsupported envelope is rejected at construction."""
    with pytest.raises(ValueError):
        make_source(grid, envelope="triangular")


@pytest.mark.parametrize("offset", [-1, 10**6])
def test_tone_burst_out_of_range_offset_raises(grid, offset):
    """A burst that starts before t=0 or runs past the time axis is rejected."""
    with pytest.raises(ValueError, match="offset"):
        make_source(grid, offset=offset)


def test_tone_burst_too_long_for_grid_raises(grid):
    """A burst longer than the time axis is rejected, even at offset 0."""
    with pytest.raises(ValueError, match="offset"):
        make_source(grid, offset=0, n_cycles=100)


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
    src = make_source(grid, offset=0)
    spec = src.spectrum(freq_selection)
    assert spec.shape == (freq_selection.n_frequencies,)
    assert spec.is_complex()
    expected = freq_selection.fft_time_series(src.waveform(), dim=0)
    assert torch.allclose(spec, expected)


def test_spectrum_uses_selection_bins(grid):
    """Different bin selections gather different spectral samples."""
    src = make_source(grid, offset=0)
    fs2 = FrequencySelection.from_bins(grid, [2, 4, 6])
    spec = src.spectrum(fs2)
    assert spec.shape == (3,)
    full = torch.fft.fft(src.waveform(), dim=0)
    assert torch.allclose(spec, full[torch.tensor([2, 4, 6])])
