"""Run orchestration: build the problem once, dispatch to the selected solver
per frequency band, and write a self-contained artifact directory per run.

Public entry points
--------------------
``run_inverse(config, run_directory) -> RunResult``  run every continuation band
``run_frequency_band(...) -> BandResult``            one continuation stage
``build_band_loss(...) -> (Wavefield, InverseLoss)`` one band's objective
``build_optimiser(...) -> JointFreqODIL``            one band's optimiser
``build_problem(config) -> Problem``                 (re-exported from problem.py)
"""

from __future__ import annotations

import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from odil_wave.loss import InverseLoss, LossConfig, LossTape, Regulariser
from odil_wave.operator import WaveEquation
from odil_wave.wavefield import Wavefield

from . import envinfo
from .arrays import BandArrays, save_problem
from .config import BandCfg, ConfigError, RunConfig, validate_config
from .health import HealthCheck
from .postprocess import draw_reported, postprocess_run
from .problem import Problem, build_problem, grid_summary
from .rundir import RunDirectory, write_json
from .tape import MetricTape

# Name of the (only) solver, recorded in run metadata and loss-tape names.
SOLVER_NAME = "joint"


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class BandResult:
    """Outcome of a single continuation stage (one frequency band)."""

    band_index: int
    frequencies_hz: List[float]
    label: str
    n_iter_requested: int
    wall_s: float
    start_metrics: Dict[str, Any]  # the band's start row of the tape
    final_metrics: Dict[str, Any]  # the tape row at the band's final iterate
    band_dir: Path  # holds arrays.npz and band_summary.json
    velocity_model: Any = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome of a whole (single- or multi-band) inversion run."""

    run_id: str
    run_dir: Path
    config: RunConfig
    status: str
    band_results: List[BandResult] = field(default_factory=list)
    final_velocity: Any = None
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


@dataclass
class LoggedRow:
    """One logged iterate as ``run_inverse(on_log=...)`` receives it."""

    row: Dict[str, Any]  # the tape.csv row just written
    c: np.ndarray  # interior velocity at this iterate (m/s)
    c_star: np.ndarray  # interior truth velocity (m/s)
    band_final: bool  # the row is the band's final iterate


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def band_label(frequencies_hz: List[float]) -> str:
    """Human-readable kHz label for a band's frequencies.

    Parameters
    ----------
    frequencies_hz : list of float
        Band frequencies in Hz.

    Returns
    -------
    str
        ``"60khz"`` for a single frequency, else ``"40-70khz"`` (min-max).
    """
    khz = [f / 1e3 for f in frequencies_hz]
    if len(khz) == 1:
        return f"{int(round(khz[0]))}khz"
    return f"{int(round(min(khz)))}-{int(round(max(khz)))}khz"


# --------------------------------------------------------------------------- #
# Optimiser dispatch (matches the notebook helpers exactly)
# --------------------------------------------------------------------------- #
def build_band_loss(
    problem: Problem,
    velocity_model: Any,
    band_ctx,
    callback: Optional[LossTape] = None,
) -> Tuple[Wavefield, InverseLoss]:
    """Inverse-grid wavefield on ``velocity_model`` and the band's objective.

    Parameters
    ----------
    problem : Problem
        The shared physical problem.
    velocity_model :
        The band's starting model.
    band_ctx :
        Per-band context (frequencies / geometry / observed data).
    callback : LossTape, optional
        Diagnostics tape of the loss.

    Returns
    -------
    tuple
        ``(wf_inv, loss)``.
    """
    cfg = problem.cfg
    wf_inv = Wavefield(problem.grid, band_ctx.freq, velocity_model=velocity_model)
    wave_eq = WaveEquation(
        wf_inv,
        space_order=problem.space_order,
        damping_scale=problem.damping_scale,
    )
    loss_config = LossConfig(
        wave_eq=wave_eq,
        geometry=band_ctx.geom,
        weights=dict(cfg.loss.weights),
        regulariser=_build_regulariser(cfg),
    )
    # PML split: observations synthesised on a separate forward grid are
    # reduced to receiver traces; otherwise use full observed wavefields.
    if getattr(band_ctx, "observed_traces", None) is not None:
        loss = InverseLoss(
            observed_traces=band_ctx.observed_traces,
            config=loss_config,
            callback=callback,
        )
    else:
        loss = InverseLoss(
            observed_wavefield=band_ctx.observed_wfs,
            config=loss_config,
            callback=callback,
        )
    return wf_inv, loss


def build_optimiser(
    problem: Problem,
    band_ctx,
    wf_inv: Wavefield,
    loss: InverseLoss,
    n_iter: int,
    u_init,
):
    """Construct the joint optimiser for one band.

    Parameters
    ----------
    problem : Problem
        The shared physical problem.
    band_ctx :
        Per-band context (frequencies / geometry / observed data).
    wf_inv : Wavefield
        Inverse-grid wavefield the optimiser acts on.
    loss : InverseLoss
        Objective for this band.
    n_iter : int
        Outer-iteration budget for this band.
    u_init :
        Per-shot Helmholtz warm-start wavefields.

    Returns
    -------
    JointFreqODIL
        The configured joint optimiser.
    """
    cfg = problem.cfg
    opt = cfg.optimiser

    # Pure joint full-space ODIL: one L-BFGS over (u, z_m); no alternation,
    # closed-form, WRI or multigrid. The objective weights come only from
    # loss.weights and are fixed for the run (never adapted).
    from odil_wave.optimisation import JointFreqODIL

    w = cfg.loss.weights
    ls = None if str(opt.line_search_fn).lower() == "none" else opt.line_search_fn
    return JointFreqODIL(
        wf_inv,
        loss,
        u_init=u_init,
        n_iter=n_iter,
        inner_max_iter=opt.inner_max_iter,
        history_size=opt.history_size,
        line_search_fn=ls,
        tolerance_grad=opt.tolerance_grad,
        tolerance_change=opt.tolerance_change,
        lbfgs_lr=opt.lbfgs_lr,
        pde_weight=float(w["pde"]),
        data_weight=float(w["data"]),
        reg_weight=float(w["reg"]),
        log_every=opt.log_every,
        verbose=cfg.runtime.verbose,
    )


# --------------------------------------------------------------------------- #
# One band / stage
# --------------------------------------------------------------------------- #
def run_frequency_band(
    problem: Problem,
    velocity_model: Any,
    band_cfg: BandCfg,
    tape: MetricTape,
    band_index: int,
    bands_dir: Path,
    *,
    warnings: Optional[List[str]] = None,
    on_log: Optional[Callable[[LoggedRow], None]] = None,
) -> BandResult:
    """Run a single continuation stage and write its per-band artifacts.

    ``velocity_model`` is the current (warm) starting model; the recovered
    model is returned in the :class:`BandResult` and becomes the next band's
    start. The band's ``arrays.npz`` and ``band_summary.json`` are written
    only once it completes. Each optimiser-health warning is printed when it
    fires and appended to ``warnings`` (the run's list) as well as the band's.
    ``on_log`` gets a :class:`LoggedRow` after each tape row is written.
    """
    cfg = problem.cfg
    freqs = list(band_cfg.frequencies_hz)
    label = band_label(freqs)
    n_iter = cfg.band_n_iter(band_cfg)

    t0 = time.perf_counter()
    t_start_epoch = time.time()
    band_ctx = problem.make_band(freqs)

    t_warm = time.perf_counter()
    u_init = problem.warm_start(velocity_model, band_ctx)
    warm_s = time.perf_counter() - t_warm

    band_tape = tape.start_band(band_index, band_ctx)
    # The loss callback keeps the solver's scalar history and result; the
    # tape rows come from ``record_row``.
    loss_tape = LossTape(
        name=f"{SOLVER_NAME}_{label}",
        log_every=cfg.optimiser.log_every,
        store_c_history=False,
    )
    wf_inv, loss = build_band_loss(problem, velocity_model, band_ctx, loss_tape)

    opt = build_optimiser(problem, band_ctx, wf_inv, loss, n_iter, u_init)

    band_arrays = BandArrays(problem, band_ctx)
    health = HealthCheck(c_min=problem.grid.c_min, c_max=problem.grid.c_max)

    def record_row(it):
        band_arrays(it)
        row = band_tape.record(it)
        for w in health(row):
            print(f"  [health warning] {w}")
            if warnings is not None:
                warnings.append(w)
        if on_log is not None:
            sl = problem.grid.interior_slice
            on_log(
                LoggedRow(
                    row=row,
                    c=it.c_full[sl].detach().cpu().numpy().astype(np.float64),
                    c_star=tape.model_error.truth,
                    band_final=int(it.iter) == n_iter,
                )
            )

    t_opt = time.perf_counter()
    recovered_wfs, _ = opt.minimise(on_log=record_row)
    optimise_s = time.perf_counter() - t_opt

    recovered_velocity = recovered_wfs[0].velocity_model
    opt_result = dict(loss_tape.result or {})
    start_metrics = band_tape.start_row
    final_metrics = band_tape.final_row

    # ---- per-band artifacts --------------------------------------------- #
    # Written into a hidden directory renamed into place once complete, so a
    # failed or interrupted band leaves no band directory.
    band_dir = bands_dir / f"band_{band_index:02d}_{label}"
    tmp_dir = bands_dir / f".{band_dir.name}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        band_arrays.save(tmp_dir / "arrays.npz")
        wall_s = time.perf_counter() - t0
        write_json(
            tmp_dir / "band_summary.json",
            {
                "band_index": band_index,
                "frequencies_hz": freqs,
                "label": label,
                "n_frequencies": int(band_ctx.freq.n_frequencies),
                "fft_bins": band_ctx.freq.fft_bins.detach().cpu().tolist(),
                "n_iter_requested": n_iter,
                "n_transducers": int(band_ctx.geom.n_transducers),
                "t_start_epoch_s": t_start_epoch,
                "wall_s": wall_s,
                "warm_start_s": warm_s,
                "optimise_s": optimise_s,
                "start_metrics": start_metrics,
                "final_metrics": final_metrics,
                "optimiser_result": opt_result,
                "warnings": list(health.warnings),
                "worst_fit_shot": band_arrays.worst_fit_shot,
                "saved_shots": band_arrays.shots,
            },
        )
        tmp_dir.replace(band_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(
        f"  [band {band_index} {label}] iters {n_iter} "
        f"loss={final_metrics['loss_total']:.6e}  "
        f"rmse_brain={final_metrics['rmse_brain']:.2f} m/s  "
        f"ssim_brain={final_metrics['ssim_brain']:.4f}  "
        f"wall={wall_s:.1f}s"
    )

    return BandResult(
        band_index=band_index,
        frequencies_hz=freqs,
        label=label,
        n_iter_requested=n_iter,
        wall_s=wall_s,
        start_metrics=start_metrics,
        final_metrics=final_metrics,
        band_dir=band_dir,
        velocity_model=recovered_velocity,
        warnings=list(health.warnings),
    )


def _build_regulariser(cfg: RunConfig):
    """Build the configured :class:`~odil_wave.loss.Regulariser`, or ``None``.

    The regulariser is active iff ``loss.weights['reg'] > 0``; then
    ``loss.regulariser.name`` selects the penalty kind (``tv_iso`` /
    ``tikhonov`` / ``tv_aniso``) and ``eps`` the ``tv_iso`` smoothing floor.
    Returns ``None`` otherwise, so :class:`~odil_wave.loss.LossConfig` omits the
    reg term. The scalar weight itself is applied by
    :class:`~odil_wave.optimisation.JointFreqODIL` in its objective.
    """
    if float(cfg.loss.weights["reg"]) <= 0.0:
        return None
    reg = cfg.loss.regulariser
    return Regulariser(kind=reg.name, eps=reg.eps)


# --------------------------------------------------------------------------- #
# Whole run
# --------------------------------------------------------------------------- #
def run_inverse(
    config: RunConfig,
    run_directory: RunDirectory,
    *,
    on_band_end: Optional[Callable[["BandResult", "RunResult"], None]] = None,
    on_log: Optional[Callable[[LoggedRow], None]] = None,
) -> RunResult:
    """Run the full (single- or multi-band) inversion into ``run_directory``.

    ``run_summary.json`` is always written, with ``status``: ``completed``,
    ``failed`` (any exception, including a non-finite loss, gradient or
    velocity; the traceback is saved) or ``interrupted`` (a
    ``KeyboardInterrupt``, e.g. Ctrl-C). A failed or interrupted run keeps its
    tape rows and completed bands and is returned, not raised; only a
    :class:`ConfigError` found at startup is re-raised, after the summary.
    Every returned run ends with post-processing
    (:func:`~odil_wave.experiment.postprocess.postprocess_run`) of what it saved.

    Parameters
    ----------
    config : RunConfig
        The run's Resolved config.
    run_directory : RunDirectory
        Where the run is written, from
        :func:`~odil_wave.experiment.rundir.new_run_directory`.
    on_band_end : callable, optional
        ``on_band_end(band_result, run_result)``, called after every completed
        band with the recovered model already in ``run_result.band_results``.
        An exception from it is reported, never aborting the run; a
        ``KeyboardInterrupt`` ends the run as ``interrupted``.
    on_log : callable, optional
        ``on_log(logged_row)``, called with a :class:`LoggedRow` after every
        tape row (e.g. :class:`odil_wave.experiment.LiveVelocityView`). Its
        first exception is reported and it is not called again, never aborting
        the run; a ``KeyboardInterrupt`` ends the run as ``interrupted``.

    Returns
    -------
    RunResult
        The run's status, band results and metrics summary.
    """
    warnings = validate_config(config)
    for w in warnings:
        print(f"  [config warning] {w}")
    # The run summary's warnings: the config's, then each health warning fired.
    run_warnings = list(warnings)

    run_dir = run_directory.path
    run_id = run_directory.run_id
    bands_dir = run_dir / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)

    # Config first, so a crash still leaves the run reproducible. The fully
    # resolved config records every effective value, so it is the single source
    # of truth for how the run was produced.
    config.to_yaml(run_dir / "config_resolved.yaml")

    seed_record = envinfo.seed_everything(
        config.run.seed, deterministic=config.runtime.deterministic
    )
    device = envinfo.resolve_device(config.runtime.device)
    meta = {
        **run_directory.provenance(),
        **envinfo.collect_metadata(
            optimiser=SOLVER_NAME,
            device_requested=config.runtime.device,
            device_resolved=str(device),
            dtype=config.runtime.dtype,
            seed=config.run.seed,
        ),
        "seed_record": seed_record,
        "n_bands": len(config.continuation.bands),
        "config_warnings": warnings,
    }

    if on_log is not None:
        on_log = _guarded(on_log)

    t_run0 = time.perf_counter()
    result = RunResult(run_id=run_id, run_dir=run_dir, config=config, status="running")
    try:
        problem = build_problem(config)
        meta["grid"] = grid_summary(problem)
        write_json(run_dir / "metadata.json", meta)

        # Raises ConfigError, before any band, if the truth gives nothing to score.
        tape = MetricTape(
            run_dir / "tape.csv", run_id=run_id, problem=problem, t0=t_run0
        )
        scored = tape.model_error
        save_problem(run_dir / "problem.npz", problem, scored.brain, scored.skull)

        current = problem.init_velocity
        for bi, band_cfg in enumerate(config.continuation.bands):
            band_res = run_frequency_band(
                problem,
                current,
                band_cfg,
                tape,
                bi,
                bands_dir,
                warnings=run_warnings,
                on_log=on_log,
            )
            result.band_results.append(band_res)
            current = band_res.velocity_model

            if on_band_end is not None:
                try:
                    on_band_end(band_res, result)
                except Exception as cb_exc:  # never let a monitor kill a run
                    print(f"  [on_band_end warning] {type(cb_exc).__name__}: {cb_exc}")
        result.final_velocity = current
        result.status = "completed"
    except KeyboardInterrupt:
        result.status = "interrupted"
    except Exception as exc:  # noqa: BLE001 - every failure is recorded
        result.status = "failed"
        result.error = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, ConfigError):
            _write_run_summary(result, run_warnings, time.perf_counter() - t_run0)
            raise
    _write_run_summary(result, run_warnings, time.perf_counter() - t_run0)
    # A post-processing error never changes the run's status.
    draw_reported(postprocess_run, run_dir, interruptible=False)
    return result


def _guarded(on_log: Callable[[LoggedRow], None]) -> Callable[[LoggedRow], None]:
    """``on_log`` made harmless: its first exception is reported and it is
    not called again. A ``KeyboardInterrupt`` still propagates."""
    failed = False

    def call(logged: LoggedRow) -> None:
        nonlocal failed
        if failed:
            return
        try:
            on_log(logged)
        except Exception as exc:  # never let a monitor kill a run
            failed = True
            print(
                f"  [on_log warning] {type(exc).__name__}: {exc} "
                "(not called again this run)"
            )

    return call


def _write_run_summary(result: RunResult, warnings: List[str], wall_s: float) -> None:
    """Fill ``result.metrics_summary`` from the completed bands, write
    ``run_summary.json`` and print the outcome.

    The metrics are tape rows: the run's first start row, the final iterate of
    its last completed band (the run's final iterate when it completed) and
    each completed band's final row.
    """
    bands = result.band_results
    result.metrics_summary = {
        "n_bands": len(bands),
        "start_metrics": bands[0].start_metrics if bands else None,
        "final_metrics": bands[-1].final_metrics if bands else None,
        "per_band": [b.final_metrics for b in bands],
    }
    write_json(
        result.run_dir / "run_summary.json",
        {
            "run_id": result.run_id,
            "optimiser": SOLVER_NAME,
            "status": result.status,
            "wall_s": wall_s,
            "warnings": list(warnings),
            "n_bands": len(result.config.continuation.bands),
            "metrics_summary": result.metrics_summary,
            "error": result.error,
        },
    )
    line = f"[run {result.run_id}] {result.status}"
    final = result.metrics_summary["final_metrics"]
    if result.error is not None:
        line += f": {result.error['exception_type']}: {result.error['message']}"
    elif final is not None:
        line += (
            f": final rmse_brain={final['rmse_brain']:.2f} m/s, "
            f"ssim_brain={final['ssim_brain']:.4f}"
        )
    print(line)
