"""Sequential frequency-band continuation for frequency-domain ODIL.

Each band jointly fits several FFT bins. Recovered ``c`` is carried to the next
band; ``u`` is re-warmed with Helmholtz on that ``c`` at the new bins, and a
fresh L-BFGS optimiser is started per band.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from odil_wave.geometry import AcquisitionGeometry, source_ring_indices
from odil_wave.grid import FrequencySelection, Grid
from odil_wave.loss import InverseLoss, LossConfig, LossTape
from odil_wave.models import VelocityModel
from odil_wave.operator import HelmholtzSolver, WaveEquation
from odil_wave.optimisation.base import LBFGSB
from odil_wave.source import SourceSignal
from odil_wave.wavefield import Wavefield


@dataclass
class FrequencyBand:
    """One continuation stage: several frequencies optimized jointly.

    Accepted forms::

        FrequencyBand([20e3, 30e3], n_iter=15)
        FrequencyBand(20e3, 30e3, n_iter=15)
        FrequencyBand([20e3], n_iter=15, source_offsets=(0,))
        FrequencyBand([20e3], n_iter=15, source_offsets=(0, 2, 4, 6))
        FrequencyBand(
            [20e3],
            n_iter=8,
            source_offsets=(0, 2, 4, 6),
            source_schedule="sequential",
        )
        FrequencyBand(
            [20e3],
            n_iter=8,
            source_offsets=(0, 2, 4, 6),
            source_schedule="cyclic",
        )

    ``source_offsets`` selects one or more rotated source octets on the
    receiver ring (see :func:`~odil_wave.geometry.source_ring_indices`).
    Default ``(0,)`` matches the historical single-octet layout.

    ``source_schedule``:
      - ``"joint"`` (default): all listed offsets are active together in one
        stage (e.g. 32 shots if four octets).
      - ``"sequential"``: run one octet after another at the same frequencies,
        carrying ``c`` between offsets (always 8 shots at a time when
        ``n_sources=8``). Each offset uses this band's ``n_iter``.
      - ``"cyclic"``: Stride-like mini-batch — for ``k = 0 .. n_iter-1`` use
        offset ``source_offsets[k % len(source_offsets)]`` with ``n_iter=1``
        each (e.g. ``0,2,4,6,0,2,4,6``). Always one octet per outer step.
    """

    frequencies_hz: Sequence[float]
    n_iter: Optional[int] = None
    source_offsets: Sequence[int] = (0,)
    source_schedule: str = "joint"

    def __init__(
        self,
        *freqs: Union[float, Sequence[float]],
        n_iter: Optional[int] = None,
        frequencies_hz: Optional[Sequence[float]] = None,
        source_offsets: Sequence[int] = (0,),
        source_schedule: str = "joint",
    ) -> None:
        if frequencies_hz is not None:
            resolved = list(frequencies_hz)
        elif len(freqs) == 1 and isinstance(freqs[0], (list, tuple)):
            resolved = list(freqs[0])
        else:
            resolved = [float(f) for f in freqs]  # type: ignore[arg-type]
        if len(resolved) == 0:
            raise ValueError("FrequencyBand must contain at least one frequency.")
        if not source_offsets:
            raise ValueError("source_offsets must contain at least one offset.")
        schedule = str(source_schedule).lower()
        if schedule not in ("joint", "sequential", "cyclic"):
            raise ValueError(
                "source_schedule must be 'joint', 'sequential', or 'cyclic'; "
                f"got {source_schedule!r}."
            )
        self.frequencies_hz = resolved
        self.n_iter = n_iter
        self.source_offsets = tuple(int(o) for o in source_offsets)
        self.source_schedule = schedule


def _expand_source_schedule(
    bands: Sequence[FrequencyBand],
    *,
    default_n_iter: int,
) -> List[FrequencyBand]:
    """Expand sequential/cyclic schedules into concrete per-offset stages."""
    stages: List[FrequencyBand] = []
    for band in bands:
        n_iter = int(default_n_iter if band.n_iter is None else band.n_iter)
        offs = band.source_offsets
        if band.source_schedule == "sequential" and len(offs) > 1:
            for off in offs:
                stages.append(
                    FrequencyBand(
                        frequencies_hz=list(band.frequencies_hz),
                        n_iter=n_iter,
                        source_offsets=(off,),
                        source_schedule="joint",
                    )
                )
        elif band.source_schedule == "cyclic" and len(offs) > 1:
            for k in range(n_iter):
                off = offs[k % len(offs)]
                stages.append(
                    FrequencyBand(
                        frequencies_hz=list(band.frequencies_hz),
                        n_iter=1,
                        source_offsets=(off,),
                        source_schedule="joint",
                    )
                )
        else:
            stages.append(band)
    return stages


@dataclass
class BandTimingStats:
    """Wall-clock and optimiser cost for one continuation band."""

    band_index: int
    frequencies_hz: List[float]
    n_iter_requested: int
    n_iter_run: int
    stopped_early: bool
    helmholtz_s: float
    optimise_s: float
    wall_s: float
    n_u_closure: int
    n_c_closure: int
    n_closure: int
    final_loss: Optional[float] = None
    source_offsets: Optional[Tuple[int, ...]] = None

    # Back-compat aliases used by earlier callers / prints.
    @property
    def n_iter_budget(self) -> int:
        return self.n_iter_requested

    @property
    def n_outer_iter(self) -> int:
        return self.n_iter_run

    @property
    def total_s(self) -> float:
        return self.wall_s

    def summary_line(self) -> str:
        f_label = ", ".join(f"{f * 1e-3:.0f}" for f in self.frequencies_hz)
        loss_s = (
            f"{self.final_loss:.6e}" if self.final_loss is not None else "n/a"
        )
        early = "yes" if self.stopped_early else "no"
        off_s = (
            f"  offsets={self.source_offsets}"
            if self.source_offsets is not None
            else ""
        )
        return (
            f"band{self.band_index + 1} [{f_label}] kHz: "
            f"requested={self.n_iter_requested}  run={self.n_iter_run}  "
            f"wall={self.wall_s:.2f}s "
            f"(helmholtz={self.helmholtz_s:.2f}s, optimise={self.optimise_s:.2f}s)  "
            f"closures={self.n_closure} "
            f"(u={self.n_u_closure}, c={self.n_c_closure})  "
            f"final_loss={loss_s}  early_stop={early}{off_s}"
        )


@dataclass
class FrequencyContinuationResult:
    """Outputs from :func:`run_frequency_continuation`."""

    wavefields: List[Wavefield]
    velocity_model: VelocityModel
    tapes: List[LossTape]
    band_start_models: List[VelocityModel]
    band_models: List[VelocityModel]
    bands: List[FrequencyBand] = field(default_factory=list)
    band_stats: List[BandTimingStats] = field(default_factory=list)

    def report_band_stats(self) -> None:
        """Print per-band requested/run iters, wall time, closures, loss, early-stop."""
        print("Per-band summary:")
        for s in self.band_stats:
            print(f"  {s.summary_line()}")


def _as_band(band: Union[FrequencyBand, Sequence[float]]) -> FrequencyBand:
    if isinstance(band, FrequencyBand):
        return band
    return FrequencyBand(frequencies_hz=band)


def _clone_velocity(vm: VelocityModel) -> VelocityModel:
    return VelocityModel.from_field(
        vm.grid, vm.c.detach().clone(), pml_c=vm.pml_c
    )


def _fft_obs_traces(
    freq_sel: FrequencySelection,
    observed_time_traces: torch.Tensor,
) -> torch.Tensor:
    """``(n_shots, nt, n_receivers)`` → ``(n_shots, nf, n_receivers)``."""
    traces = torch.as_tensor(observed_time_traces)
    if traces.ndim != 3:
        raise ValueError(
            "observed_time_traces must be (n_shots, nt, n_receivers), "
            f"got shape {tuple(traces.shape)}"
        )
    n_shots, nt, _ = traces.shape
    if nt != freq_sel.n_time:
        raise ValueError(
            f"observed_time_traces nt={nt} != FrequencySelection.n_time="
            f"{freq_sel.n_time}"
        )
    return torch.stack(
        [freq_sel.fft_time_series(traces[s], dim=0) for s in range(n_shots)],
        dim=0,
    )


def _helmholtz_warmstart(
    *,
    grid: Grid,
    freq_sel: FrequencySelection,
    geom: AcquisitionGeometry,
    velocity_model: VelocityModel,
    space_order: int,
    pml_weight: float,
    verbose: bool,
) -> torch.Tensor:
    wf = Wavefield(
        grid=grid,
        frequency_selection=freq_sel,
        velocity_model=velocity_model,
    )
    solved = HelmholtzSolver(
        wf, geom, space_order=space_order, pml_weight=pml_weight
    ).solve(verbose=verbose)
    return torch.stack([w.amplitude for w in solved], dim=0)


def run_frequency_continuation(
    *,
    grid: Grid,
    source: SourceSignal,
    bands: Sequence[Union[FrequencyBand, Sequence[float]]],
    observed_time_traces: torch.Tensor,
    velocity_model: VelocityModel,
    geometry_kwargs: Optional[Mapping[str, Any]] = None,
    weights: Optional[Mapping[str, float]] = None,
    default_n_iter: int = 40,
    space_order: int = 2,
    time_order: int = 2,
    pml_weight: float = 1.0,
    clamp: bool = True,
    normalize_data: str = "per_receiver",
    regulariser=None,
    free_mask=None,
    verbose: bool = True,
    **lbfgs_opts,
) -> FrequencyContinuationResult:
    """Run sequential multi-frequency ODIL bands with Helmholtz re-warm.

    Parameters
    ----------
    bands :
        Ordered stages. Each entry is a :class:`FrequencyBand` or a sequence of
        Hz values (typically 2-3 bins). Frequencies within a band are fit
        jointly; stages run in order.
    observed_time_traces :
        Real time-domain receiver gathers
        ``(n_catalog_shots, nt, n_receivers)`` ordered by **sorted unique
        source ring indices** over the union of all bands'
        ``source_offsets`` (see
        :func:`~odil_wave.geometry.source_ring_indices`). Re-FFTed and
        sliced onto each band's active shots.
    velocity_model :
        Starting ``c`` for band 0. Later bands start from the previous
        band's recovered model.
    geometry_kwargs :
        Forwarded to :class:`AcquisitionGeometry` (ring layout, counts, etc.).
        ``frequency_selection`` and ``source_offsets`` are set per band and
        must not be included.
    lbfgs_opts :
        Forwarded to :class:`LBFGSB` (``u_precond``, ``z_optim``, ``z_lr``,
        ``z_steps``, ``c_steps``, ``c_lr``, ``c_max_iter``, …).
        For ``u_precond="z"``, prefer ``z_optim="gd"`` (default) with
        ``z_steps=1`` and ``z_lr=1.0``; ``c`` is still updated with L-BFGS.
        Per-band ``n_iter`` overrides ``default_n_iter``. Early stopping is
        **off by default** (``early_stop_rtol=0``); pass ``early_stop_rtol``,
        ``early_stop_min_iter``, and ``early_stop_patience`` to enable it.

    Returns
    -------
    FrequencyContinuationResult
        Final wavefields / ``c``, plus per-band start/end models, loss tapes,
        and :class:`BandTimingStats` (requested/run iters, wall time, closures,
        final loss, early-stop flag).
    """
    if not bands:
        raise ValueError("bands must contain at least one FrequencyBand.")

    geom_kw: Dict[str, Any] = dict(geometry_kwargs or {})
    if "frequency_selection" in geom_kw:
        raise ValueError(
            "geometry_kwargs must not include frequency_selection; "
            "it is built per band."
        )
    if "source_offsets" in geom_kw:
        raise ValueError(
            "geometry_kwargs must not include source_offsets; "
            "set FrequencyBand.source_offsets per band."
        )

    n_receivers = int(geom_kw.get("n_receivers", 16))
    n_sources_per_offset = geom_kw.get("n_sources")
    if n_sources_per_offset is None:
        n_sources_per_offset = n_receivers
    else:
        n_sources_per_offset = int(n_sources_per_offset)

    w = dict(weights or {"pde": 1.0, "data": 100.0, "reg": 0.0})
    band_list = _expand_source_schedule(
        [_as_band(b) for b in bands],
        default_n_iter=default_n_iter,
    )

    catalog_indices = sorted(
        {
            idx
            for band in band_list
            for idx in source_ring_indices(
                n_receivers, n_sources_per_offset, band.source_offsets
            )
        }
    )
    catalog_position = {
        ring_idx: pos for pos, ring_idx in enumerate(catalog_indices)
    }

    obs_catalog = torch.as_tensor(observed_time_traces)
    if obs_catalog.ndim != 3:
        raise ValueError(
            "observed_time_traces must be (n_catalog_shots, nt, n_receivers), "
            f"got shape {tuple(obs_catalog.shape)}"
        )
    if obs_catalog.shape[0] != len(catalog_indices):
        raise ValueError(
            "observed_time_traces shot count must match the sorted unique "
            f"source-ring catalogue ({len(catalog_indices)} shots for ring "
            f"indices {catalog_indices}); got {obs_catalog.shape[0]} shots."
        )

    c_current = _clone_velocity(velocity_model)
    tapes: List[LossTape] = []
    band_start_models: List[VelocityModel] = []
    band_models: List[VelocityModel] = []
    band_stats: List[BandTimingStats] = []
    final_wfs: List[Wavefield] = []

    for band_idx, band in enumerate(band_list):
        n_iter = int(default_n_iter if band.n_iter is None else band.n_iter)
        freq_sel = FrequencySelection.from_frequencies(grid, band.frequencies_hz)
        geom = AcquisitionGeometry(
            grid,
            source,
            frequency_selection=freq_sel,
            source_offsets=band.source_offsets,
            **geom_kw,
        )
        band_obs_positions = [
            catalog_position[idx] for idx in geom.source_ring_indices
        ]
        obs = _fft_obs_traces(freq_sel, obs_catalog[band_obs_positions])

        # Snapshot for tests / diagnostics; optimiser may mutate the live model.
        band_start_models.append(_clone_velocity(c_current))
        start_vm = _clone_velocity(c_current)

        if verbose:
            f_label = ", ".join(f"{f * 1e-3:.1f}" for f in band.frequencies_hz)
            print(
                f"\n{'=' * 56}\n"
                f"  Band {band_idx + 1}/{len(band_list)}: "
                f"[{f_label}] kHz  (nf={freq_sel.n_frequencies}, "
                f"n_iter={n_iter}, offsets={geom.source_offsets}, "
                f"n_shots={geom.n_sources})\n"
                f"{'=' * 56}"
            )

        t_band0 = time.perf_counter()
        u_warm = _helmholtz_warmstart(
            grid=grid,
            freq_sel=freq_sel,
            geom=geom,
            velocity_model=start_vm,
            space_order=space_order,
            pml_weight=pml_weight,
            verbose=verbose,
        )
        helmholtz_s = time.perf_counter() - t_band0

        wf = Wavefield(
            grid=grid,
            frequency_selection=freq_sel,
            velocity_model=start_vm,
            init_amplitude=u_warm[0],
        )
        f_lo = float(min(band.frequencies_hz)) * 1e-3
        f_hi = float(max(band.frequencies_hz)) * 1e-3
        tape = LossTape(
            name=f"band{band_idx + 1}_{f_lo:.0f}-{f_hi:.0f}kHz",
            log_every=1,
            store_c_history=True,
        )
        loss = InverseLoss(
            config=LossConfig(
                WaveEquation(
                    wf,
                    space_order=space_order,
                    time_order=time_order,
                    pml_weight=pml_weight,
                ),
                geom,
                weights=w,
                regulariser=regulariser,
            ),
            observed_traces=obs,
            callback=tape,
            normalize_data=normalize_data,
        )
        opt = LBFGSB(
            wf,
            loss,
            clamp=clamp,
            u_init=u_warm,
            free_mask=free_mask,
            n_iter=n_iter,
            **lbfgs_opts,
        )
        t_opt0 = time.perf_counter()
        band_wfs, tape = opt.minimise()
        optimise_s = time.perf_counter() - t_opt0
        total_s = time.perf_counter() - t_band0

        opt_info = dict(getattr(tape, "result", None) or {})
        final_loss = opt_info.get("loss", None)
        if final_loss is not None:
            final_loss = float(final_loss)
        stats = BandTimingStats(
            band_index=band_idx,
            frequencies_hz=list(band.frequencies_hz),
            n_iter_requested=n_iter,
            n_iter_run=int(opt_info.get("n_outer_iter", n_iter)),
            stopped_early=bool(opt_info.get("stopped_early", False)),
            helmholtz_s=helmholtz_s,
            optimise_s=optimise_s,
            wall_s=total_s,
            n_u_closure=int(opt_info.get("n_u_closure", 0)),
            n_c_closure=int(opt_info.get("n_c_closure", 0)),
            n_closure=int(opt_info.get("n_closure", 0)),
            final_loss=final_loss,
            source_offsets=geom.source_offsets,
        )
        band_stats.append(stats)

        if verbose:
            print(f"  {stats.summary_line()}")

        recovered = VelocityModel.from_field(
            grid,
            band_wfs[0].velocity_model.c.detach().clone(),
            pml_c=start_vm.pml_c,
        )
        c_current = recovered
        final_wfs = band_wfs
        tapes.append(tape)
        band_models.append(_clone_velocity(recovered))

    return FrequencyContinuationResult(
        wavefields=final_wfs,
        velocity_model=c_current,
        tapes=tapes,
        band_start_models=band_start_models,
        band_models=band_models,
        bands=band_list,
        band_stats=band_stats,
    )

