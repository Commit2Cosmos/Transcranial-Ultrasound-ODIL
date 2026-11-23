"""Observed data for a band: the inverse crime or an independent solve.

``build_problem`` builds one :class:`ObservationPath` per run from the
``observation`` config block; :meth:`ObservationPath.observe` solves the truth
at a band's frequencies and returns :class:`Observations`, always on the
inversion grid: traces at its receivers (noise added by :func:`add_noise`) and
the true wavefield restricted to its nodes. The band that consumes them is
built by ``Problem.make_band``; the operator is ``odil_wave.operator``.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from odil_wave.geometry import AcquisitionGeometry
from odil_wave.grid import FrequencySelection, Grid
from odil_wave.models import VelocityModel
from odil_wave.operator import HelmholtzSolver
from odil_wave.source import SourceSignal
from odil_wave.wavefield import Wavefield


@dataclass
class Observations:
    """A band's observed data and the true wavefield behind them, on the
    inversion grid."""

    geometry: AcquisitionGeometry  # the inversion grid's, for this band
    traces: torch.Tensor  # (n_shots, nf, n_transducers - 1) complex, noise included
    true_wavefields: List[Wavefield]  # one per shot, on geometry.grid
    # ‖d(c*) − d_obs‖ / ‖d_obs‖ with d(c*) the inversion's own solve at the truth:
    # the modelling-error-plus-noise floor of rel_data_residual.
    rel_data_residual_truth: float
    noise_floor: Optional[float]  # 10**(-snr_db / 20); None when noiseless
    solve_s: float  # wall seconds of the observation solve


@dataclass
class IndependentSolve:
    """The observation solve of ``observation.mode: independent``: the truth on
    a grid refined ``refine`` times, with its own stencil and time symbols."""

    grid: Grid
    source: SourceSignal
    truth: VelocityModel
    refine: int
    space_order: int
    time_symbols: str

    def node_offset(self, coarse: Grid) -> int:
        """Full-grid index of ``coarse``'s node 0 on :attr:`grid`; node ``i``
        is at ``node_offset + refine * i``."""
        return self.grid.pml_width - self.refine * coarse.pml_width


class ObservationPath:
    """Generates each band's :class:`Observations` from the true model.

    ``independent`` is ``None`` for the inverse crime: the inversion grid,
    stencil and truth make the data.
    """

    def __init__(
        self,
        truth: VelocityModel,
        *,
        space_order: int,
        damping_scale: float,
        independent: Optional[IndependentSolve],
        snr_db: Optional[float],
        seed: int,
        verbose: bool,
    ) -> None:
        """``truth``, ``space_order`` and ``damping_scale`` are the inversion's;
        ``snr_db`` and ``seed`` as in ``observation.noise``."""
        self.truth = truth
        self.space_order = space_order
        self.damping_scale = damping_scale
        self.independent = independent
        self.snr_db = snr_db
        self.seed = seed
        self.verbose = verbose

    @property
    def inverse_crime(self) -> bool:
        """True when the inversion's own discretisation makes the data."""
        return self.independent is None

    def observe(self, geometry: AcquisitionGeometry) -> Observations:
        """Observations at ``geometry``'s receivers and frequency bins
        (``geometry`` on the inversion grid)."""
        t0 = time.perf_counter()
        inversion_solve = self._solve(
            geometry, self.truth, self.space_order, self.damping_scale
        )
        if self.independent is None:
            true_wavefields = inversion_solve
            solve_s = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            true_wavefields = self._solve_independent(geometry)
            solve_s = time.perf_counter() - t0
            if self.verbose:
                print(f"  observation solve: {solve_s:.2f}s")
        clean = geometry.extract_observations(_stack(true_wavefields))
        traces = add_noise(
            clean, geometry.frequency_selection.fft_bins, self.snr_db, self.seed
        )
        predicted = geometry.extract_observations(_stack(inversion_solve))
        rel = float(
            torch.linalg.vector_norm(predicted - traces)
            / torch.linalg.vector_norm(traces)
        )
        return Observations(
            geometry=geometry,
            traces=traces,
            true_wavefields=true_wavefields,
            rel_data_residual_truth=rel,
            noise_floor=None if self.snr_db is None else 10 ** (-self.snr_db / 20),
            solve_s=solve_s,
        )

    def _solve(
        self,
        geometry: AcquisitionGeometry,
        truth: VelocityModel,
        space_order: int,
        damping_scale: float,
    ) -> List[Wavefield]:
        wf = Wavefield(
            geometry.grid, geometry.frequency_selection, velocity_model=truth
        )
        solver = HelmholtzSolver(
            wf, geometry, space_order=space_order, damping_scale=damping_scale
        )
        return solver.solve(verbose=self.verbose)

    def observation_geometry(
        self, geometry: AcquisitionGeometry
    ) -> AcquisitionGeometry:
        """The independent solve's counterpart of the inversion grid's
        ``geometry``: the same transducer nodes and frequency bins, on its
        grid with its time symbols. ``ValueError`` for the inverse crime."""
        ind = self.independent
        if ind is None:
            raise ValueError("the inverse crime observes on the inversion grid")
        off = ind.node_offset(geometry.grid)
        freq = FrequencySelection.from_bins(
            ind.grid,
            geometry.frequency_selection.fft_bins,
            requested_frequencies=geometry.frequency_selection.requested_frequencies,
            time_symbols=ind.time_symbols,
        )
        return AcquisitionGeometry(
            ind.grid,
            ind.source,
            freq,
            n_transducers=geometry.n_transducers,
            sigma_s=geometry.sigma_s,
            a_frac=geometry.a_frac,
            b_frac=geometry.b_frac,
            ring_center=geometry.ring_center,
            source_spatial=geometry.source_spatial,
            transducer_ij=off + ind.refine * geometry.transducer_ij,
        )

    def _solve_independent(self, geometry: AcquisitionGeometry) -> List[Wavefield]:
        """The independent solve, restricted to ``geometry.grid``'s nodes."""
        ind = self.independent
        coarse, r = geometry.grid, ind.refine
        off = ind.node_offset(coarse)
        fine = self._solve(
            self.observation_geometry(geometry),
            ind.truth,
            ind.space_order,
            self.damping_scale,
        )
        sx = slice(off, off + r * (coarse.nx - 1) + 1, r)
        sy = slice(off, off + r * (coarse.ny - 1) + 1, r)
        out = []
        for wf in fine:
            restricted = Wavefield(
                coarse, geometry.frequency_selection, velocity_model=self.truth
            )
            restricted.amplitude = wf.amplitude[:, sx, sy].contiguous()
            out.append(restricted)
        return out


def refined_grid(grid: Grid, refine: int, pml_width: int) -> Grid:
    """``grid`` with ``refine`` times finer spacing over the same interior, on
    the same time axis, with ``pml_width`` cells of PML. Every node of ``grid``
    is a node of it as long as ``pml_width >= refine * grid.pml_width``."""
    shape = tuple((n - 1) * refine + 1 for n in grid.interior_shape)
    return dataclasses.replace(
        grid, interior_shape=shape, pml_width=int(pml_width), init_nt=int(grid.nt)
    )


def prolonged_truth(truth: VelocityModel, grid: Grid, refine: int) -> VelocityModel:
    """``truth``'s interior on the ``refine`` times finer ``grid``, each node
    the speed of the nearest node of ``truth``'s grid."""
    sl = truth.grid.interior_slice
    c = truth.c[sl]

    def nearest(n: int) -> torch.Tensor:
        fine = torch.arange(n, device=c.device)
        return torch.div(fine + refine // 2, refine, rounding_mode="floor")

    return VelocityModel.from_interior(
        grid, c[nearest(grid.interior_nx)][:, nearest(grid.interior_ny)]
    )


def add_noise(
    traces: torch.Tensor,
    fft_bins: torch.Tensor,
    snr_db: Optional[float],
    seed: int,
) -> torch.Tensor:
    """``traces`` ``(n_shots, nf, n_receivers)`` plus circular complex Gaussian
    noise of ``sigma_f = RMS_f(|traces|) * 10**(-snr_db / 20)`` per frequency,
    seeded by ``(seed, fft_bin)``; ``traces`` itself when ``snr_db`` is None."""
    if snr_db is None:
        return traces
    n_shots, nf, n_rcv = traces.shape
    rms = traces.abs().square().mean(dim=(0, 2)).sqrt()
    sigma = rms * 10 ** (-float(snr_db) / 20)
    noise = torch.empty_like(traces)
    for k, b in enumerate(fft_bins.tolist()):
        z = np.random.default_rng([int(seed), int(b)]).standard_normal(
            (2, n_shots, n_rcv)
        )
        noise[:, k] = torch.from_numpy((z[0] + 1j * z[1]) / np.sqrt(2)).to(traces)
    return traces + sigma.view(1, nf, 1).to(traces.dtype) * noise


def _stack(wavefields: List[Wavefield]) -> torch.Tensor:
    return torch.stack([wf.amplitude for wf in wavefields])
