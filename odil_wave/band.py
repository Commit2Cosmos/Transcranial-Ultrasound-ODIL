"""One band of the inversion, as the joint solve receives it.

:class:`Band` holds a band's operator, sources, observed data, initial model
and warm start, built together so they cannot disagree. The experiment layer
builds one per band with ``Problem.make_band``; the joint solve
(:class:`~odil_wave.optimisation.JointFreqODIL`) reads everything from it.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch

from odil_wave.geometry import AcquisitionGeometry
from odil_wave.models import VelocityModel
from odil_wave.operator import HelmholtzSolver, WaveEquation
from odil_wave.wavefield import Wavefield


class Band:
    """A band's operator, sources, observed data, initial model and warm start.

    Constructing it runs the warm start on ``initial_model``. ``factors`` are
    the warm start's per-frequency ``splu`` factors when ``keep_factors``,
    else ``None``. ``space_order`` and ``damping_scale`` are configured
    (ADR 0001).
    """

    def __init__(
        self,
        geometry: AcquisitionGeometry,
        initial_model: VelocityModel,
        observed_traces: torch.Tensor,
        *,
        space_order: int,
        damping_scale: float,
        keep_factors: bool,
        true_wavefields: Optional[List[Wavefield]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        observed_traces : torch.Tensor
            Complex observed data ``(n_shots, nf, n_transducers - 1)``.
        true_wavefields : list of Wavefield, optional
            The true wavefield per shot, on the grid that generated the
            observed data (not necessarily ``geometry.grid``).
        """
        if initial_model.grid is not geometry.grid:
            raise ValueError("initial_model and geometry must share one grid")
        self.geometry = geometry
        self.frequency_selection = geometry.frequency_selection
        self.initial_model = initial_model
        self.wavefield = Wavefield(
            geometry.grid, self.frequency_selection, velocity_model=initial_model
        )
        self.wave_eq = WaveEquation(
            self.wavefield, space_order=space_order, damping_scale=damping_scale
        )
        # (n_shots, nf, nx, ny) complex
        self.sources = self.wave_eq.sources(geometry)

        shape = (
            geometry.n_shots,
            self.wavefield.n_frequencies,
            geometry.n_transducers - 1,
        )
        if tuple(observed_traces.shape) != shape:
            raise ValueError(
                f"observed_traces shape {tuple(observed_traces.shape)} != {shape}"
            )
        self.observed_traces = observed_traces.to(
            dtype=self.wavefield.cdtype, device=geometry.device
        )
        self.true_wavefields = true_wavefields

        solver = HelmholtzSolver(
            self.wavefield,
            geometry,
            space_order=space_order,
            damping_scale=damping_scale,
        )
        t0 = time.perf_counter()
        solved = solver.solve(verbose=verbose, return_factors=keep_factors)
        self.warm_start_s = time.perf_counter() - t0
        u_init, self.factors = solved if keep_factors else (solved, None)
        # (n_shots, nf, nx, ny) complex
        self.u_init = torch.stack([wf.amplitude for wf in u_init]).detach()

    def true_wavefield_interior(self) -> torch.Tensor:
        """The true wavefield over its grid's interior, ``(n_shots, nf, nx, ny)``.

        Raises ``ValueError`` when the band was built without true wavefields.
        """
        if self.true_wavefields is None:
            raise ValueError("band was built without true_wavefields")
        sl = self.true_wavefields[0].grid.interior_slice
        u_star = torch.stack([wf.amplitude for wf in self.true_wavefields])
        return u_star[:, :, sl[0], sl[1]].detach()
