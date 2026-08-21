"""Shared fixtures for the odil_wave test-suite.
"""

from types import SimpleNamespace

import matplotlib

import pytest
import torch

from odil_wave import (
    AcquisitionGeometry,
    FrequencySelection,
    Grid,
    HelmholtzSolver,
    SourceSignal,
    VelocityModel,
    WaveEquation,
    Wavefield,
)

matplotlib.use("Agg")

# Small, shared problem parameters.
INTERIOR = (20, 20)
EXTENT = ((0.0, 0.05), (0.0, 0.05))
C_MIN, C_MAX = 1400.0, 1700.0
T_MAX, INIT_NT = 6e-5, 120
PML = 4
F0 = 40e3
N_CYCLES = 2.0
OFFSET = 5
DTYPE = torch.float64


def make_grid(**overrides) -> Grid:
    """Build a tiny float64 grid; keyword overrides win over the defaults."""
    kw = dict(
        interior_shape=INTERIOR,
        interior_extent=EXTENT,
        c_min=C_MIN,
        c_max=C_MAX,
        t_max=T_MAX,
        init_nt=INIT_NT,
        pml_width=PML,
        dtype=DTYPE,
    )
    kw.update(overrides)
    return Grid(**kw)


@pytest.fixture
def grid() -> Grid:
    """A fresh tiny grid for a single test."""
    return make_grid()


@pytest.fixture
def freq_selection(grid) -> FrequencySelection:
    """Single-frequency selection at F0 on the tiny grid."""
    return FrequencySelection.from_frequencies(grid, [F0])


@pytest.fixture
def source(grid) -> SourceSignal:
    """Tone-burst wavelet at F0."""
    return SourceSignal(
        grid, kind="tone_burst", f0=F0, n_cycles=N_CYCLES, offset=OFFSET
    )


@pytest.fixture
def geometry(grid, source, freq_selection) -> AcquisitionGeometry:
    """8-receiver / 2-source point-injection ring."""
    return AcquisitionGeometry(
        grid,
        source,
        freq_selection,
        n_receivers=8,
        n_sources=2,
        source_spatial="point",
    )


@pytest.fixture
def homogeneous_model(grid) -> VelocityModel:
    """Uniform 1500 m/s medium on the tiny grid."""
    return VelocityModel(grid, profile="homogeneous", base=1500.0)


@pytest.fixture(scope="module")
def wave_stack() -> SimpleNamespace:
    """A fully-solved frequency-domain forward problem, shared per test module."""
    g = make_grid()
    fs = FrequencySelection.from_frequencies(g, [F0])
    src = SourceSignal(g, kind="tone_burst", f0=F0, n_cycles=N_CYCLES, offset=OFFSET)
    geom = AcquisitionGeometry(
        g, src, fs, n_receivers=8, n_sources=2, source_spatial="point"
    )
    model = VelocityModel(g, profile="homogeneous", base=1500.0)
    wf = Wavefield(g, fs, velocity_model=model)
    solver = HelmholtzSolver(wf, geom, space_order=2)
    sols = solver.solve(verbose=False)

    amp = torch.stack([s.amplitude for s in sols])
    wave_eq = WaveEquation(wf, space_order=2)
    sources = (
        torch.stack(
            [geom.source_field(i).to(dtype=wf.cdtype) for i in range(geom.n_sources)]
        )
        * g.t0**2
    )
    obs_traces = torch.stack(
        [geom.extract_observations(s.amplitude) for s in sols], dim=0
    )
    return SimpleNamespace(
        grid=g,
        freq=fs,
        source=src,
        geometry=geom,
        model=model,
        wavefield=wf,
        wave_eq=wave_eq,
        sols=sols,
        amp=amp,
        sources=sources,
        obs_traces=obs_traces,
    )
