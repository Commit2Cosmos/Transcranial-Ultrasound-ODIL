import pytest
import torch

from odil_wave import FrequencySelection, SourceSignal

F0 = 40e3
# n_cycles / f0 = 5e-5 s < the tiny grid's t_max (6e-5 s) so the burst fits.
N_CYCLES = 2.0


def test_tone_burst_shape_and_dtype(grid):
    """The tone burst samples onto grid.t with the grid's shape and dtype."""
    src = SourceSignal(grid, kind="tone_burst", f0=F0, n_cycles=N_CYCLES)
    w = src.waveform(grid.t)
    assert w.shape == (grid.nt,)
    assert w.dtype == grid.dtype
    assert torch.isfinite(w).all()
    assert float(w.abs().max()) > 0


def test_dimensionless_scaling(grid):
    """dimensionless=True multiplies the wavelet by the grid's natural amplitude."""
    phys = SourceSignal(
        grid,
        kind="tone_burst",
        f0=F0,
        n_cycles=N_CYCLES,
        amplitude=1.0,
        dimensionless=False,
    )
    nd = SourceSignal(
        grid,
        kind="tone_burst",
        f0=F0,
        n_cycles=N_CYCLES,
        amplitude=1.0,
        dimensionless=True,
    )
    # The natural amplitude scales the whole signal, so the peak ratio is exact.
    ratio = nd.waveform(grid.t).abs().max() / phys.waveform(grid.t).abs().max()
    assert float(ratio) == pytest.approx(grid.natural_source_amplitude(F0), rel=1e-6)


def test_call_matches_waveform(grid, source):
    """__call__ is an alias for waveform."""
    assert torch.allclose(source(grid.t), source.waveform(grid.t))


def test_tone_burst_length_and_offset(grid):
    """Tone burst fills the whole time axis and stays zero before its offset."""
    offset = 10
    tb = SourceSignal(grid, kind="tone_burst", f0=F0, n_cycles=N_CYCLES, offset=offset)
    w = tb.waveform(grid.t)
    assert w.shape == (grid.nt,)
    # The leading offset samples are exactly zero.
    assert torch.all(w[:offset] == 0)
    # ...and the burst itself is non-trivial.
    assert float(w.abs().max()) > 0


def test_tone_burst_rectangular_envelope(grid):
    """The rectangular envelope is supported and produces a non-trivial burst."""
    tb = SourceSignal(
        grid, kind="tone_burst", f0=F0, n_cycles=N_CYCLES, envelope="rectangular"
    )
    w = tb.waveform(grid.t)
    assert w.shape == (grid.nt,)
    assert float(w.abs().max()) > 0


def test_tone_burst_bad_envelope_raises(grid):
    """An unsupported envelope is rejected at construction."""
    with pytest.raises(ValueError):
        SourceSignal(grid, kind="tone_burst", f0=F0, envelope="triangular")


def test_unknown_kind_raises(grid):
    """waveform() rejects an unknown source kind."""
    src = SourceSignal(grid, kind="bogus", f0=F0)
    with pytest.raises(ValueError):
        src.waveform(grid.t)


def test_spectrum_matches_fft_of_waveform(grid, freq_selection):
    """spectrum(fs) equals the FFT of the sampled waveform gathered on fs bins."""
    src = SourceSignal(grid, kind="tone_burst", f0=F0, n_cycles=N_CYCLES)
    spec = src.spectrum(freq_selection)
    assert spec.shape == (freq_selection.n_frequencies,)
    assert spec.is_complex()
    expected = freq_selection.fft_time_series(src.waveform(grid.t), dim=0)
    assert torch.allclose(spec, expected)


def test_spectrum_uses_selection_bins(grid):
    """Different bin selections gather different spectral samples."""
    src = SourceSignal(grid, kind="tone_burst", f0=F0, n_cycles=N_CYCLES)
    fs2 = FrequencySelection.from_bins(grid, [2, 4, 6])
    spec = src.spectrum(fs2)
    assert spec.shape == (3,)
    full = torch.fft.fft(src.waveform(grid.t), dim=0)
    assert torch.allclose(spec, full[torch.tensor([2, 4, 6])])
