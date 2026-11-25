"""Tests for :class:`odil_wave.Band`: its operator, sources, observed data and
warm start agree, and it validates its inputs."""

import pytest
import torch

from odil_wave import Band, FrequencySelection

from conftest import make_band, make_geometry, make_grid, make_model, make_source

OPERATOR_KW = dict(space_order=2, damping_scale=1.0)


def test_sources_satisfy_the_pde_at_the_true_wavefield(wave_stack):
    """The band's t0**2-scaled sources make the PDE residual ~0 at the
    Helmholtz solution of the true model."""
    band = make_band(wave_stack.geometry, wave_stack.model, wave_stack.sols)
    r = band.wave_eq.residual(wave_stack.amp, wave_stack.model.c, band.sources)
    rms = lambda x: float(x.abs().square().mean().sqrt())  # noqa: E731
    assert rms(r) < 1e-8 * rms(band.sources)


def test_warm_start_on_the_true_model_is_the_true_wavefield(wave_stack):
    """Started from the true model, the warm start is the true wavefield."""
    band = make_band(wave_stack.geometry, wave_stack.model, wave_stack.sols)
    torch.testing.assert_close(band.u_init, wave_stack.amp, rtol=0, atol=1e-12)


@pytest.mark.parametrize("keep_factors", [True, False])
def test_factors_are_kept_only_on_request(wave_stack, keep_factors):
    band = make_band(
        wave_stack.geometry,
        wave_stack.model,
        wave_stack.sols,
        keep_factors=keep_factors,
    )
    if keep_factors:
        assert len(band.factors) == wave_stack.freq.n_frequencies
    else:
        assert band.factors is None


def test_true_wavefield_interior_crops_to_the_interior(wave_stack):
    band = make_band(wave_stack.geometry, wave_stack.model, wave_stack.sols)
    sl = wave_stack.grid.interior_slice
    torch.testing.assert_close(
        band.true_wavefield_interior(),
        wave_stack.amp[:, :, sl[0], sl[1]],
        rtol=0,
        atol=0,
    )


def test_true_wavefield_interior_needs_true_wavefields(wave_stack):
    band = Band(
        wave_stack.geometry,
        wave_stack.model,
        wave_stack.obs_traces,
        keep_factors=False,
        **OPERATOR_KW,
    )
    with pytest.raises(ValueError, match="true_wavefields"):
        band.true_wavefield_interior()


def test_observed_traces_must_match_the_geometry(wave_stack):
    """Observed data must be (n_shots, nf, n_transducers - 1)."""
    with pytest.raises(ValueError, match="observed_traces"):
        Band(
            wave_stack.geometry,
            wave_stack.model,
            wave_stack.obs_traces[0],
            keep_factors=False,
            **OPERATOR_KW,
        )


def test_initial_model_must_share_the_geometry_grid(wave_stack):
    other = make_grid()
    fs = FrequencySelection.from_frequencies(other, wave_stack.freq.frequencies)
    geom = make_geometry(other, make_source(other), fs)
    with pytest.raises(ValueError, match="one grid"):
        Band(
            geom,
            wave_stack.model,
            wave_stack.obs_traces,
            keep_factors=False,
            **OPERATOR_KW,
        )


def test_observed_traces_skip_the_transmitting_transducer(wave_stack):
    """Perturbing u only at each shot's own transducer node leaves its
    receiver traces equal to the band's observed data."""
    band = make_band(wave_stack.geometry, wave_stack.model, wave_stack.sols)
    geom = wave_stack.geometry
    amp = wave_stack.amp.clone()
    for s in range(geom.n_shots):
        i, j = geom.transducer_ij[s]
        amp[s, :, i, j] += 10.0
    torch.testing.assert_close(
        geom.extract_observations(amp), band.observed_traces, rtol=0, atol=0
    )


def test_band_wavefield_holds_the_initial_model(wave_stack):
    model = make_model(wave_stack.grid, base=1450.0)
    band = make_band(wave_stack.geometry, model, wave_stack.sols)
    assert band.wavefield.velocity_model is model is band.initial_model
