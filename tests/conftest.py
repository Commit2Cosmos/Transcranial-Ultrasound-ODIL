"""Shared fixtures for the odil_wave test-suite."""

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

import pytest
import torch
import yaml

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
from odil_wave.experiment import deep_merge, resolve_config

matplotlib.use("Agg")

# The repo-root ``main.py`` CLI, importable as ``main``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
        pml_width=PML,
        pml_power=3,
        pml_R0=1e-6,
        t_max=T_MAX,
        init_nt=INIT_NT,
        cfl_safety=0.9,
        device=torch.device("cpu"),
        dtype=DTYPE,
    )
    kw.update(overrides)
    return Grid(**kw)


def make_source(grid, **overrides) -> SourceSignal:
    """Tone-burst wavelet at F0 on ``grid``; keyword overrides win."""
    kw = dict(
        kind="tone_burst",
        f0=F0,
        amplitude=1.0,
        n_cycles=N_CYCLES,
        envelope="gaussian",
        dimensionless=True,
        offset=OFFSET,
    )
    kw.update(overrides)
    return SourceSignal(grid, **kw)


def make_geometry(grid, source, freq, **overrides) -> AcquisitionGeometry:
    """8-transducer point-injection ring; keyword overrides win."""
    kw = dict(
        n_transducers=8,
        sigma_s=None,
        a_frac=0.55,
        b_frac=0.70,
        ring_center=None,
        source_spatial="point",
    )
    kw.update(overrides)
    return AcquisitionGeometry(grid, source, freq, **kw)


def make_model(grid, profile="homogeneous", **overrides) -> VelocityModel:
    """A velocity model on ``grid`` (1500 m/s background); overrides win."""
    kw = dict(base=1500.0, contrast=0.0, pml_c=None)
    kw.update(overrides)
    return VelocityModel(grid, profile=profile, **kw)


# Run config for a tiny single-band joint inversion: names only the
# values that differ from the default config (the ``RunConfig`` defaults).
TINY_RUN_CONFIG = {
    "runtime": {"dtype": "float64"},
    "grid": {
        "interior_shape": [20, 20],
        "interior_extent": [[0.0, 0.05], [0.0, 0.05]],
        "c_min": 1400.0,
        "c_max": 1750.0,
    },
    "pml": {"width": 6},
    "time": {"t_max": 6e-5, "init_nt": 200},
    "source": {"f0": 40e3, "n_cycles": 2.0},
    "acquisition": {"n_transducers": 8, "source_spatial": "point"},
    # Shepp-Logan has the brain and skull regions the tape scores. A slow skull
    # keeps the truth inside the grid's velocity bounds; a small head keeps the
    # transducers in water and is recovered within the tiny budget.
    "truth": {"profile": "shepp_logan", "scale": 0.6, "extra": {"c_skull": 1700.0}},
    "init": {"profile": "homogeneous"},
    "continuation": {"bands": [{"frequencies_hz": [40e3]}]},
    "loss": {"weights": {"data": 10.0}},
    "optimiser": {"n_iter": 10, "log_every": 1},
    # A 5-cell SSIM window leaves part of the tiny brain region after erosion.
    "metrics": {"ssim": {"sigma": 0.5}},
}


@pytest.fixture
def tiny_run_config(tmp_path) -> dict:
    """The tiny run config, writing its run under ``tmp_path``.

    Tests needing a variant ``deep_merge`` their changes onto it.
    """
    return deep_merge(
        TINY_RUN_CONFIG, {"run": {"output_root": str(tmp_path), "run_id": "tiny"}}
    )


# Two bands of 3 outer iterations logged every 2: ``log_every`` doesn't divide
# ``n_iter``, so each band's final iterate is logged off the cadence.
TINY_TWO_BAND_RUN = {
    "continuation": {"bands": [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]},
    "optimiser": {"n_iter": 3, "log_every": 2},
}


@pytest.fixture(scope="session")
def tiny_two_band_run(tmp_path_factory) -> SimpleNamespace:
    """One tiny two-band run through the CLI, shared by tests that only read it.

    Returns ``run_dir`` and the resolved ``config``.
    """
    import main

    root = tmp_path_factory.mktemp("tiny_two_band_run")
    user = deep_merge(
        deep_merge(TINY_RUN_CONFIG, TINY_TWO_BAND_RUN),
        {"run": {"output_root": str(root), "run_id": "two_band"}},
    )
    config_path = root / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(user))
    assert main.main(["--config", str(config_path)]) == 0
    config = resolve_config(user)
    return SimpleNamespace(run_dir=config.run_dir(), config=config)


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
    return make_source(grid)


@pytest.fixture
def geometry(grid, source, freq_selection) -> AcquisitionGeometry:
    """8-transducer point-injection ring."""
    return make_geometry(grid, source, freq_selection)


@pytest.fixture
def homogeneous_model(grid) -> VelocityModel:
    """Uniform 1500 m/s medium on the tiny grid."""
    return make_model(grid)


@pytest.fixture(scope="module")
def wave_stack() -> SimpleNamespace:
    """A fully-solved frequency-domain forward problem, shared per test module."""
    g = make_grid()
    fs = FrequencySelection.from_frequencies(g, [F0])
    src = make_source(g)
    geom = make_geometry(g, src, fs)
    model = make_model(g)
    wf = Wavefield(g, fs, velocity_model=model)
    solver = HelmholtzSolver(wf, geom, space_order=2, damping_scale=1.0)
    sols = solver.solve(verbose=False)

    amp = torch.stack([s.amplitude for s in sols])
    wave_eq = WaveEquation(wf, space_order=2, damping_scale=1.0)
    sources = (
        torch.stack(
            [geom.source_field(i).to(dtype=wf.cdtype) for i in range(geom.n_shots)]
        )
        * g.t0**2
    )
    obs_traces = geom.extract_observations(amp)
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
