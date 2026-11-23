"""Construct the physical FWI problem once from a resolved config.

``build_problem(cfg)`` builds the grid, source, acquisition layout, truth /
initial velocity models and the run's
:class:`~odil_wave.experiment.observation.ObservationPath`, so every solver
operates on an identical problem. :meth:`Problem.observe` generates a band's
observed data (the ``observation`` block picks the inverse crime or an
independent solve); :meth:`Problem.make_band` builds each band
(:class:`~odil_wave.band.Band`) from them and its initial model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from odil_wave.band import Band
from odil_wave.geometry import AcquisitionGeometry
from odil_wave.grid import FrequencySelection, Grid
from odil_wave.models import VelocityModel
from odil_wave.source import SourceSignal

from .config import ModelCfg, RunConfig
from .observation import (
    IndependentSolve,
    ObservationPath,
    Observations,
    prolonged_truth,
    refined_grid,
)


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
    damping_scale: float
    observation: ObservationPath

    # -- per-band construction -------------------------------------------- #
    def observe(self, frequencies_hz: Sequence[float]) -> Observations:
        """Observed data at ``frequencies_hz``, on the inversion grid."""
        freq = FrequencySelection.from_frequencies(self.grid, list(frequencies_hz))
        geom = self.acquisition_geometry(self.grid, self.source, freq)
        return self.observation.observe(geom)

    def make_band(
        self,
        frequencies_hz: Sequence[float],
        initial_model: VelocityModel,
        observations: Optional[Observations] = None,
    ) -> Band:
        """The band inverting ``frequencies_hz`` jointly from ``initial_model``.

        ``observations`` are :meth:`observe`'s at ``frequencies_hz``, ``None``
        to generate them. The warm start keeps its factors only when the
        wavefield preconditioner reuses them.
        """
        obs = self.observe(frequencies_hz) if observations is None else observations
        return Band(
            obs.geometry,
            initial_model,
            obs.traces,
            space_order=self.space_order,
            damping_scale=self.damping_scale,
            keep_factors=self.cfg.optimiser.wavefield_preconditioner != "none",
            true_wavefields=obs.true_wavefields,
            verbose=self.cfg.runtime.verbose,
        )

    def acquisition_geometry(
        self,
        grid: Grid,
        source: SourceSignal,
        freq: FrequencySelection,
    ) -> AcquisitionGeometry:
        """Acquisition layout for a band, *without* solving for observed data."""
        acq = self.cfg.acquisition
        return AcquisitionGeometry(
            grid,
            source,
            freq,
            n_transducers=acq.n_transducers,
            sigma_s=acq.sigma_s,
            a_frac=acq.a_frac,
            b_frac=acq.b_frac,
            ring_center=self.center,
            source_spatial=acq.source_spatial,
        )


def _build_velocity(
    grid: Grid, spec: ModelCfg, center: Tuple[float, float]
) -> VelocityModel:
    """Build a :class:`VelocityModel` from a :class:`ModelCfg`."""
    kwargs = dict(spec.extra)
    if spec.profile == "overdensity" and "center" not in kwargs:
        kwargs["center"] = center
    return VelocityModel(
        grid,
        profile=spec.profile,
        base=spec.base,
        contrast=spec.contrast,
        scale=spec.scale,
        skull_alpha=spec.skull_alpha,
        skull_sigma=spec.skull_sigma,
        **kwargs,
    )


def _resolve_center(cfg: RunConfig, grid: Grid) -> Tuple[float, float]:
    """The acquisition ring centre ``(x, y)`` in metres; a null
    ``ring_center`` means ``grid``'s interior centre."""
    rc = cfg.acquisition.ring_center
    if rc is None:
        return grid.interior_center
    return (float(rc[0]), float(rc[1]))


def _make_grid(cfg: RunConfig) -> Grid:
    """The config's inversion :class:`Grid`."""
    g, pml, tm = cfg.grid, cfg.pml, cfg.time
    return Grid(
        interior_shape=tuple(g.interior_shape),
        interior_extent=tuple(tuple(e) for e in g.interior_extent),
        c_min=g.c_min,
        c_max=g.c_max,
        pml_width=int(pml.width),
        pml_power=pml.power,
        pml_R0=pml.R0,
        t_max=tm.t_max,
        init_nt=tm.init_nt,
        cfl_safety=tm.cfl_safety,
        device=cfg.runtime.device,
        dtype=cfg.runtime.dtype,
    )


def _make_source(cfg: RunConfig, grid: Grid) -> SourceSignal:
    """The config's :class:`SourceSignal` on ``grid``."""
    s = cfg.source
    return SourceSignal(
        grid,
        kind=s.kind,
        f0=s.f0,
        amplitude=s.amplitude,
        n_cycles=s.n_cycles,
        envelope=s.envelope,
        dimensionless=s.dimensionless,
    )


def build_problem(cfg: RunConfig) -> Problem:
    """Build the shared physical problem from a fully-resolved config."""
    if cfg.runtime.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.runtime.torch_num_threads))

    grid = _make_grid(cfg)
    center = _resolve_center(cfg, grid)
    source = _make_source(cfg, grid)

    truth_velocity = _build_velocity(grid, cfg.truth, center)
    init_velocity = _build_velocity(grid, cfg.init, center)
    head_mask = truth_velocity.head_mask  # interior-shaped bool tensor or None

    return Problem(
        cfg=cfg,
        device=grid.device,
        dtype=grid.dtype,
        grid=grid,
        source=source,
        center=center,
        truth_velocity=truth_velocity,
        init_velocity=init_velocity,
        head_mask=head_mask,
        space_order=cfg.grid.space_order,
        damping_scale=cfg.pml.damping_scale,
        observation=_observation_path(cfg, grid, truth_velocity, center),
    )


def _observation_path(
    cfg: RunConfig, grid: Grid, truth: VelocityModel, center: Tuple[float, float]
) -> ObservationPath:
    """The ``observation`` block's :class:`ObservationPath` for the inversion
    ``grid`` and its ``truth``."""
    obs = cfg.observation
    independent = None
    if obs.mode == "independent":
        r = int(obs.refine)
        pml = round(float(obs.pml_thickness_factor) * int(cfg.pml.width) * r)
        fine = refined_grid(grid, r, pml)
        if obs.truth == "native":
            spec = dataclasses.replace(cfg.truth, skull_sigma=cfg.truth.skull_sigma * r)
            fine_truth = _build_velocity(fine, spec, center)
        else:
            fine_truth = prolonged_truth(truth, fine, r)
        independent = IndependentSolve(
            grid=fine,
            source=_make_source(cfg, fine),
            truth=fine_truth,
            refine=r,
            space_order=int(obs.space_order),
            time_symbols=obs.time_symbols,
        )
    return ObservationPath(
        truth,
        space_order=cfg.grid.space_order,
        damping_scale=cfg.pml.damping_scale,
        independent=independent,
        snr_db=obs.noise.snr_db,
        seed=int(obs.noise.seed),
        verbose=cfg.runtime.verbose,
    )


def grid_summary(problem: Problem) -> dict:
    """Derived grid quantities recorded in metadata / band summaries."""
    grid = problem.grid
    out = {
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
    return out


def observation_summary(problem: Problem) -> dict:
    """The observation settings and ``inverse_crime`` flag recorded in run
    metadata."""
    path = problem.observation
    out = {
        **dataclasses.asdict(problem.cfg.observation),
        "inverse_crime": path.inverse_crime,
    }
    if path.independent is not None:
        g = path.independent.grid
        out["grid_full_shape"] = [int(g.nx), int(g.ny)]
        out["grid_pml_width"] = int(g.pml_width)
    return out
