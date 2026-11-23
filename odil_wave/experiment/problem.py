"""Construct the physical FWI problem once from a resolved config.

``build_problem(cfg)`` builds the grid, source, acquisition layout, truth /
initial velocity models and the (frequency-independent) broadband truth field,
so every solver operates on an identical problem. :meth:`Problem.make_band`
builds each band (:class:`~odil_wave.band.Band`) from its initial model.
Observed data are always a Helmholtz solve on the truth model.

By default (``pml.observation_width`` null) observations are generated on the
inversion grid with the inversion operator: the inverse crime. When
``pml.observation_width`` differs from ``pml.width``, observations are instead
synthesised on a separate forward grid with that PML width and reduced to
receiver traces for the inverse solve (a deliberate modelling error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from odil_wave.band import Band
from odil_wave.geometry import AcquisitionGeometry
from odil_wave.grid import FrequencySelection, Grid
from odil_wave.models import VelocityModel
from odil_wave.operator import HelmholtzSolver
from odil_wave.wavefield import Wavefield
from odil_wave.source import SourceSignal

from .config import ModelCfg, RunConfig


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
    # Optional thicker-PML forward grid used only to synthesise observations.
    forward_grid: Optional[Grid] = None
    forward_truth: Optional[VelocityModel] = None
    forward_source: Optional[SourceSignal] = None

    @property
    def observation_grid(self) -> Grid:
        """Grid used to synthesise observed data (forward if PML-split)."""
        return self.forward_grid if self.forward_grid is not None else self.grid

    @property
    def observation_truth(self) -> VelocityModel:
        """Truth model used to synthesise data (forward truth if PML-split)."""
        return (
            self.forward_truth
            if self.forward_truth is not None
            else self.truth_velocity
        )

    @property
    def observation_source(self) -> SourceSignal:
        """Source used to synthesise data (forward source if PML-split)."""
        return self.forward_source if self.forward_source is not None else self.source

    @property
    def pml_split(self) -> bool:
        """True when observations use a different PML width than the inverse."""
        return self.forward_grid is not None

    # -- per-band construction -------------------------------------------- #
    def make_band(
        self, frequencies_hz: Sequence[float], initial_model: VelocityModel
    ) -> Band:
        """The band inverting ``frequencies_hz`` jointly from ``initial_model``.

        Observed data are reduced to traces on the observation grid. The
        warm start keeps its factors only when the wavefield preconditioner
        reuses them.
        """
        freqs = list(frequencies_hz)
        freq = FrequencySelection.from_frequencies(self.grid, freqs)
        geom = self.acquisition_geometry(self.grid, self.source, freq)
        if self.pml_split:
            # Same physical frequencies; bins must match (shared nt/dt).
            obs_freq = FrequencySelection.from_frequencies(self.observation_grid, freqs)
            obs_geom = self.acquisition_geometry(
                self.observation_grid, self.observation_source, obs_freq
            )
        else:
            obs_freq, obs_geom = freq, geom
        true_wavefields = self._observed(obs_freq, obs_geom)
        traces = obs_geom.extract_observations(
            torch.stack([wf.amplitude for wf in true_wavefields])
        )
        return Band(
            geom,
            initial_model,
            traces,
            space_order=self.space_order,
            damping_scale=self.damping_scale,
            keep_factors=self.cfg.optimiser.wavefield_preconditioner != "none",
            true_wavefields=true_wavefields,
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

    def _observed(
        self, freq: FrequencySelection, geom: AcquisitionGeometry
    ) -> List[Wavefield]:
        """Observed complex wavefields (one per shot) on this band's bins."""
        wf = Wavefield(geom.grid, freq, velocity_model=self.observation_truth)
        solver = HelmholtzSolver(
            wf, geom, space_order=self.space_order, damping_scale=self.damping_scale
        )
        return solver.solve(verbose=self.cfg.runtime.verbose)


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


def _make_grid(cfg: RunConfig, pml_width: int) -> Grid:
    """The config's :class:`Grid` with ``pml_width`` cells of PML (inverse or
    forward grid)."""
    g, pml, tm = cfg.grid, cfg.pml, cfg.time
    return Grid(
        interior_shape=tuple(g.interior_shape),
        interior_extent=tuple(tuple(e) for e in g.interior_extent),
        c_min=g.c_min,
        c_max=g.c_max,
        pml_width=int(pml_width),
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

    grid = _make_grid(cfg, cfg.pml.width)
    center = _resolve_center(cfg, grid)
    source = _make_source(cfg, grid)

    truth_velocity = _build_velocity(grid, cfg.truth, center)
    init_velocity = _build_velocity(grid, cfg.init, center)
    head_mask = truth_velocity.head_mask  # interior-shaped bool tensor or None

    forward_grid = None
    forward_truth = None
    forward_source = None
    obs_pml = cfg.pml.observation_width
    if obs_pml is not None and int(obs_pml) != int(cfg.pml.width):
        forward_grid = _make_grid(cfg, int(obs_pml))
        # Shared time base required so FFT bins line up across the split.
        if (
            int(forward_grid.nt) != int(grid.nt)
            or abs(float(forward_grid.dt) - float(grid.dt)) > 1e-18
        ):
            raise ValueError(
                "observation/inverse grids must share nt and dt for the PML "
                f"split; got fwd nt={forward_grid.nt} dt={float(forward_grid.dt)} "
                f"vs inv nt={grid.nt} dt={float(grid.dt)}. "
                "Set time.init_nt so both use the same time stepping."
            )
        forward_truth = _build_velocity(forward_grid, cfg.truth, center)
        forward_source = _make_source(cfg, forward_grid)

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
        forward_grid=forward_grid,
        forward_truth=forward_truth,
        forward_source=forward_source,
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
        "pml_width_inverse": int(grid.pml_width),
        "pml_width_forward": int(problem.observation_grid.pml_width),
        "extent": [list(map(float, ax)) for ax in grid.extent],
        "interior_extent": [list(map(float, ax)) for ax in grid.interior_extent],
        "c_min": (None if grid.c_min is None else float(grid.c_min)),
        "c_max": float(grid.c_max),
        "head_mask_cells": (
            None if problem.head_mask is None else int(problem.head_mask.sum())
        ),
    }
    return out
