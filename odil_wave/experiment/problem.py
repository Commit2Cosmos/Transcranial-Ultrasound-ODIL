"""Construct the physical FWI problem once from a resolved config.

``build_problem(cfg)`` builds the grid, source, acquisition layout, truth /
initial velocity models and the (frequency-independent) broadband truth field,
so every solver operates on an identical problem. Per-band pieces
(``FrequencySelection``, geometry, observed wavefields, Helmholtz warm start)
are produced on demand by :meth:`Problem.make_band` / :meth:`Problem.warm_start`.

Numerical behaviour mirrors the notebook helpers ``build_setup`` /
``leapfrog_time_data`` / ``leapfrog_freq_wfs`` / ``helmholtz_solve`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from odil_wave.geometry import AcquisitionGeometry
from odil_wave.grid import FrequencySelection, Grid
from odil_wave.models import VelocityModel
from odil_wave.operator import HelmholtzSolver, LeapfrogSolver
from odil_wave.wavefield import Wavefield
from odil_wave.source import SourceSignal

from . import envinfo
from .config import ModelCfg, RunConfig


@dataclass
class BandContext:
    """Per-band frequency selection, geometry and observed wavefields."""

    frequencies_hz: List[float]
    freq: FrequencySelection
    geom: AcquisitionGeometry
    observed_wfs: List[Wavefield]


@dataclass
class Problem:
    """Frequency-independent problem pieces shared by every solver."""

    cfg: RunConfig
    device: torch.device
    dtype: torch.dtype
    grid: Grid
    source: SourceSignal
    center: Tuple[float, float]
    truth_velocity: VelocityModel
    init_velocity: VelocityModel
    head_mask: Optional[torch.Tensor]
    space_order: int
    pml_weight: float
    _truth_amp_time: Optional[torch.Tensor] = None

    # -- per-band construction -------------------------------------------- #
    def make_band(self, frequencies_hz: Sequence[float]) -> BandContext:
        freqs = list(frequencies_hz)
        freq = FrequencySelection.from_frequencies(self.grid, freqs)
        geom = self._geometry(freq)
        observed = self._observed(freq, geom)
        return BandContext(
            frequencies_hz=freqs, freq=freq, geom=geom, observed_wfs=observed
        )

    def _geometry(self, freq: FrequencySelection) -> AcquisitionGeometry:
        acq = self.cfg.acquisition
        return AcquisitionGeometry(
            self.grid,
            self.source,
            freq,
            n_receivers=acq.n_receivers,
            n_sources=acq.n_sources,
            sigma_s=acq.sigma_s,
            a_frac=acq.a_frac,
            b_frac=acq.b_frac,
            ring_center=self.center,
            source_spatial=acq.source_spatial,
        )

    def _observed(
        self, freq: FrequencySelection, geom: AcquisitionGeometry
    ) -> List[Wavefield]:
        """Observed complex wavefields (one per shot) on this band's bins."""
        method = self.cfg.observation.method
        verbose = self.cfg.observation.verbose
        if method == "helmholtz":
            wf = Wavefield(self.grid, freq, velocity_model=self.truth_velocity)
            solver = HelmholtzSolver(
                wf, geom, space_order=self.space_order, pml_weight=self.pml_weight
            )
            return solver.solve(verbose=verbose)
        if method == "leapfrog_fft":
            amp_time = self._ensure_truth_time(geom, verbose=verbose)
            amp_freq = freq.fft_time_series(amp_time, dim=-3)  # (n_shots, nf, nx, ny)
            outputs: List[Wavefield] = []
            for sh in range(amp_freq.shape[0]):
                wf = Wavefield(self.grid, freq, velocity_model=self.truth_velocity)
                wf.amplitude = amp_freq[sh]
                outputs.append(wf)
            return outputs
        raise ValueError(
            f"unknown observation.method {method!r}; "
            "expected 'leapfrog_fft' or 'helmholtz'."
        )

    def _ensure_truth_time(
        self, geom: AcquisitionGeometry, verbose: bool = False
    ) -> torch.Tensor:
        """Broadband leapfrog time field of the truth model (computed once).

        The source/receiver layout is frequency-independent, so a single
        time-domain solve is FFT'd onto every band's bins (avoids the inverse
        crime, matching the notebook).
        """
        if self._truth_amp_time is None:
            wf = Wavefield(
                self.grid, geom.frequency_selection, velocity_model=self.truth_velocity
            )
            solver = LeapfrogSolver(
                wf, geom, space_order=self.space_order, pml_weight=self.pml_weight
            )
            self._truth_amp_time = solver.solve(verbose=verbose)
        return self._truth_amp_time

    # -- warm start -------------------------------------------------------- #
    def warm_start(
        self, velocity_model: VelocityModel, band: BandContext
    ) -> Optional[List[Wavefield]]:
        """Helmholtz warm-start wavefields for ``u`` on the current model.

        Returns ``None`` when ``continuation.warm_start == 'none'``.
        """
        if self.cfg.continuation.warm_start == "none":
            return None
        wf = Wavefield(self.grid, band.freq, velocity_model=velocity_model)
        solver = HelmholtzSolver(
            wf, band.geom, space_order=self.space_order, pml_weight=self.pml_weight
        )
        return solver.solve(verbose=self.cfg.observation.verbose)


def _build_velocity(
    grid: Grid, spec: ModelCfg, center: Tuple[float, float]
) -> VelocityModel:
    kwargs = dict(spec.extra)
    kwargs["scale"] = spec.scale
    if spec.profile == "overdensity" and "center" not in kwargs:
        kwargs["center"] = center
    pml_c = spec.pml_c
    return VelocityModel(
        grid,
        profile=spec.profile,
        base=spec.base,
        contrast=spec.contrast,
        **({"pml_c": pml_c} if pml_c is not None else {}),
        **kwargs,
    )


def _resolve_center(cfg: RunConfig, grid: Grid) -> Tuple[float, float]:
    rc = cfg.acquisition.ring_center
    if isinstance(rc, str):
        if rc != "grid_center":
            raise ValueError(
                f"acquisition.ring_center string must be 'grid_center', got {rc!r}"
            )
        (x0, x1), (y0, y1) = grid.extent
        return ((x1 + x0) / 2.0, (y1 + y0) / 2.0)
    if isinstance(rc, (list, tuple)) and len(rc) == 2:
        return (float(rc[0]), float(rc[1]))
    raise ValueError(
        f"acquisition.ring_center must be 'grid_center' or [x, y], got {rc!r}"
    )


def build_problem(cfg: RunConfig) -> Problem:
    """Build the shared physical problem from a fully-resolved config."""
    device = envinfo.resolve_device(cfg.runtime.device)
    dtype = envinfo.resolve_dtype(cfg.runtime.dtype)
    if cfg.runtime.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.runtime.torch_num_threads))

    g = cfg.grid
    grid = Grid(
        interior_shape=tuple(g.interior_shape),
        c_min=g.c_min,
        c_max=g.c_max,
        interior_extent=tuple(tuple(e) for e in g.interior_extent),
        t_max=g.t_max,
        init_nt=g.init_nt,
        pml_width=g.pml_width,
        pml_power=g.pml_power,
        pml_R0=g.pml_R0,
        cfl_safety=g.cfl_safety,
        L0=g.L0,
        c0=g.c0,
        device=device,
        dtype=dtype,
    )

    center = _resolve_center(cfg, grid)

    s = cfg.source

    source = SourceSignal(
        grid,
        kind=s.kind,
        f0=s.f0,
        t0=s.t0,
        amplitude=s.amplitude,
        n_cycles=s.n_cycles,
        envelope=s.envelope,
        offset=s.offset,
        dimensionless=s.dimensionless,
    )

    truth_velocity = _build_velocity(grid, cfg.truth, center)
    init_velocity = _build_velocity(grid, cfg.init, center)
    head_mask = truth_velocity.head_mask  # interior-shaped bool tensor or None

    return Problem(
        cfg=cfg,
        device=device,
        dtype=dtype,
        grid=grid,
        source=source,
        center=center,
        truth_velocity=truth_velocity,
        init_velocity=init_velocity,
        head_mask=head_mask,
        space_order=cfg.physics.space_order,
        pml_weight=cfg.physics.pml_weight,
    )


def grid_summary(problem: Problem) -> dict:
    """Derived grid quantities recorded in metadata / band summaries."""
    grid = problem.grid
    return {
        "interior_shape": [int(grid.interior_nx), int(grid.interior_ny)],
        "full_shape": [int(grid.nx), int(grid.ny)],
        "nt": int(grid.nt),
        "dx": float(grid.dx),
        "dy": float(grid.dy),
        "dt": float(grid.dt),
        "pml_width": int(grid.pml_width),
        "extent": [list(map(float, ax)) for ax in grid.extent],
        "interior_extent": [list(map(float, ax)) for ax in grid.interior_extent],
        "c_min": (None if grid.c_min is None else float(grid.c_min)),
        "c_max": float(grid.c_max),
        "head_mask_cells": (
            None if problem.head_mask is None else int(problem.head_mask.sum())
        ),
    }
