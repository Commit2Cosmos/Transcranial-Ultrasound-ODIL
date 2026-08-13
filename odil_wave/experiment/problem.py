"""Construct the physical FWI problem once from a resolved config.

``build_problem(cfg)`` builds the grid, source, acquisition layout, truth /
initial velocity models and the (frequency-independent) broadband truth field,
so every solver operates on an identical problem. Per-band pieces
(``FrequencySelection``, geometry, observed wavefields, Helmholtz warm start)
are produced on demand by :meth:`Problem.make_band` / :meth:`Problem.warm_start`.

When ``observation.pml_width`` differs from ``grid.pml_width``, observations
are synthesised on a separate forward grid (thicker PML) and reduced to
receiver traces for the inverse solve — matching the notebook
``PML_FWD`` / ``PML_INV`` split.

Numerical behaviour mirrors the notebook helpers ``build_setup`` /
``leapfrog_time_data`` / ``leapfrog_freq_wfs`` / ``helmholtz_solve`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
    """Per-band frequency selection, geometry and observed data.

    ``observed_wfs`` are full complex fields on the grid used to synthesise
    data (inverse grid, or the thicker forward grid when PML is split) —
    useful for visualisation. When ``observed_traces`` is set, the inverse
    loss uses those receiver samples instead of resampling the wavefields
    (required for the PML_FWD / PML_INV split).
    """

    frequencies_hz: List[float]
    freq: FrequencySelection
    geom: AcquisitionGeometry
    observed_wfs: List[Wavefield]
    observed_traces: Optional[torch.Tensor] = None  # (n_shots, nf, n_recv)


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
    # Optional thicker-PML forward grid used only to synthesise observations.
    forward_grid: Optional[Grid] = None
    forward_truth: Optional[VelocityModel] = None
    forward_source: Optional[SourceSignal] = None
    # Broadband truth time field cache, keyed by the active source-ring layout
    # (tuple of ring indices). Distinct source_offsets get their own solve; the
    # common case (every band shares one layout) is computed once.
    # Values live on the observation grid (forward when split, else inverse).
    _truth_amp_time: Dict[Tuple[int, ...], torch.Tensor] = field(default_factory=dict)

    @property
    def observation_grid(self) -> Grid:
        """Grid used to synthesise observed data (forward if PML-split)."""
        return self.forward_grid if self.forward_grid is not None else self.grid

    @property
    def observation_truth(self) -> VelocityModel:
        return (
            self.forward_truth
            if self.forward_truth is not None
            else self.truth_velocity
        )

    @property
    def observation_source(self) -> SourceSignal:
        return self.forward_source if self.forward_source is not None else self.source

    @property
    def pml_split(self) -> bool:
        """True when observations use a different PML width than the inverse."""
        return self.forward_grid is not None

    # -- per-band construction -------------------------------------------- #
    def make_band(
        self,
        frequencies_hz: Sequence[float],
        source_offsets: Sequence[int] = (0,),
    ) -> BandContext:
        freqs = list(frequencies_hz)
        freq = FrequencySelection.from_frequencies(self.grid, freqs)
        geom = self._geometry(self.grid, self.source, freq, source_offsets)

        if self.pml_split:
            obs_grid = self.observation_grid
            obs_source = self.observation_source
            # Same physical frequencies; bins must match (shared nt/dt).
            obs_freq = FrequencySelection.from_frequencies(obs_grid, freqs)
            obs_geom = self._geometry(obs_grid, obs_source, obs_freq, source_offsets)
            observed_wfs = self._observed(obs_freq, obs_geom)
            traces = torch.stack(
                [obs_geom.extract_observations(wf.amplitude) for wf in observed_wfs],
                dim=0,
            )
            return BandContext(
                frequencies_hz=freqs,
                freq=freq,
                geom=geom,
                observed_wfs=observed_wfs,
                observed_traces=traces,
            )

        observed = self._observed(freq, geom)
        return BandContext(
            frequencies_hz=freqs, freq=freq, geom=geom, observed_wfs=observed
        )

    def _geometry(
        self,
        grid: Grid,
        source: SourceSignal,
        freq: FrequencySelection,
        source_offsets: Sequence[int] = (0,),
    ) -> AcquisitionGeometry:
        """Acquisition layout for a band, *without* solving for observed data."""
        acq = self.cfg.acquisition
        return AcquisitionGeometry(
            grid,
            source,
            freq,
            n_receivers=acq.n_receivers,
            n_sources=acq.n_sources,
            sigma_s=acq.sigma_s,
            a_frac=acq.a_frac,
            b_frac=acq.b_frac,
            ring_center=self.center,
            source_spatial=acq.source_spatial,
            source_offsets=tuple(source_offsets),
        )

    def _observed(
        self, freq: FrequencySelection, geom: AcquisitionGeometry
    ) -> List[Wavefield]:
        """Observed complex wavefields (one per shot) on this band's bins."""
        method = self.cfg.observation.method
        verbose = self.cfg.observation.verbose
        truth = self.observation_truth
        if method == "helmholtz":
            wf = Wavefield(geom.grid, freq, velocity_model=truth)
            solver = HelmholtzSolver(
                wf, geom, space_order=self.space_order, pml_weight=self.pml_weight
            )
            return solver.solve(verbose=verbose)
        if method == "leapfrog_fft":
            amp_time = self._ensure_truth_time(geom, verbose=verbose)
            amp_freq = freq.fft_time_series(amp_time, dim=-3)  # (n_shots, nf, nx, ny)
            outputs: List[Wavefield] = []
            for sh in range(amp_freq.shape[0]):
                wf = Wavefield(geom.grid, freq, velocity_model=truth)
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
        """Broadband leapfrog time field of the truth model.

        The source/receiver layout is frequency-independent, so a single
        time-domain solve is FFT'd onto every band's bins (avoids the inverse
        crime, matching the notebook). Computed once *per source layout* and
        cached by the active source-ring indices, so bands sharing a layout
        reuse it while a rotated source octet triggers its own solve.
        """
        key = tuple(int(i) for i in geom.source_ring_indices)
        cached = self._truth_amp_time.get(key)
        if cached is None:
            truth = self.observation_truth
            wf = Wavefield(geom.grid, geom.frequency_selection, velocity_model=truth)
            solver = LeapfrogSolver(
                wf, geom, space_order=self.space_order, pml_weight=self.pml_weight
            )
            cached = solver.solve(verbose=verbose)
            self._truth_amp_time[key] = cached
        return cached

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
    """Build a :class:`VelocityModel` from a :class:`ModelCfg`.

    First-class fields (``scale``, ``skull_alpha``, ``skull_sigma``) are
    forwarded as profile kwargs and override the same keys in ``spec.extra``.
    ``extra.skull_smooth`` is normalised to ``skull_sigma`` when the first-class
    ``skull_sigma`` is still at its default ``0`` (alias for
    :class:`~odil_wave.models.VelocityModel`).
    """
    kwargs = dict(spec.extra)
    kwargs["scale"] = spec.scale
    kwargs["skull_alpha"] = spec.skull_alpha
    if spec.skull_sigma != 0.0:
        kwargs["skull_sigma"] = spec.skull_sigma
        kwargs.pop("skull_smooth", None)
    elif "skull_sigma" in kwargs:
        kwargs["skull_sigma"] = float(kwargs["skull_sigma"])
        kwargs.pop("skull_smooth", None)
    elif "skull_smooth" in kwargs:
        kwargs["skull_sigma"] = float(kwargs.pop("skull_smooth"))
    else:
        kwargs["skull_sigma"] = spec.skull_sigma
    if spec.profile == "overdensity" and "center" not in kwargs:
        kwargs["center"] = center
    pml_c = spec.pml_c
    return VelocityModel(
        grid,
        profile=spec.profile,
        base=spec.base,
        contrast=spec.contrast,
        pml_fill=spec.pml_fill,
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


def _make_grid(
    cfg: RunConfig, device: torch.device, dtype: torch.dtype, pml_width: int
) -> Grid:
    g = cfg.grid
    return Grid(
        interior_shape=tuple(g.interior_shape),
        c_min=g.c_min,
        c_max=g.c_max,
        interior_extent=tuple(tuple(e) for e in g.interior_extent),
        t_max=g.t_max,
        init_nt=g.init_nt,
        pml_width=int(pml_width),
        pml_power=g.pml_power,
        pml_R0=g.pml_R0,
        cfl_safety=g.cfl_safety,
        L0=g.L0,
        c0=g.c0,
        device=device,
        dtype=dtype,
    )


def _make_source(cfg: RunConfig, grid: Grid) -> SourceSignal:
    s = cfg.source
    return SourceSignal(
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


def build_problem(cfg: RunConfig) -> Problem:
    """Build the shared physical problem from a fully-resolved config."""
    device = envinfo.resolve_device(cfg.runtime.device)
    dtype = envinfo.resolve_dtype(cfg.runtime.dtype)
    if cfg.runtime.torch_num_threads is not None:
        torch.set_num_threads(int(cfg.runtime.torch_num_threads))

    grid = _make_grid(cfg, device, dtype, cfg.grid.pml_width)
    center = _resolve_center(cfg, grid)
    source = _make_source(cfg, grid)

    truth_velocity = _build_velocity(grid, cfg.truth, center)
    init_velocity = _build_velocity(grid, cfg.init, center)
    head_mask = truth_velocity.head_mask  # interior-shaped bool tensor or None

    forward_grid = None
    forward_truth = None
    forward_source = None
    obs_pml = cfg.observation.pml_width
    if obs_pml is not None and int(obs_pml) != int(cfg.grid.pml_width):
        forward_grid = _make_grid(cfg, device, dtype, int(obs_pml))
        # Shared time base required so FFT bins line up across the split.
        if (
            int(forward_grid.nt) != int(grid.nt)
            or abs(float(forward_grid.dt) - float(grid.dt)) > 1e-18
        ):
            raise ValueError(
                "observation/inverse grids must share nt and dt for the PML "
                f"split; got fwd nt={forward_grid.nt} dt={float(forward_grid.dt)} "
                f"vs inv nt={grid.nt} dt={float(grid.dt)}. "
                "Set grid.init_nt so both use the same time stepping."
            )
        forward_truth = _build_velocity(forward_grid, cfg.truth, center)
        forward_source = _make_source(cfg, forward_grid)

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
