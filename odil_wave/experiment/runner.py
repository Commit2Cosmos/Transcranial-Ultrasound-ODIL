"""Run orchestration: build the problem once, dispatch to the selected solver
per frequency band, and write a self-contained artifact directory per run.

Public entry points
--------------------
``run_inverse(config) -> RunResult``            run every continuation band
``run_frequency_band(...) -> BandResult``        one continuation stage
``build_problem(config) -> Problem``             (re-exported from problem.py)
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from odil_wave.loss import InverseLoss, LossConfig, LossTape, Regulariser
from odil_wave.operator import WaveEquation
from odil_wave.wavefield import Wavefield

from . import envinfo
from .config import BandCfg, RunConfig, validate_config
from .problem import Problem, build_problem, grid_summary
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
    c_final_path: Path
    velocity_model: Any = None


@dataclass
class RunResult:
    """Outcome of a whole (single- or multi-band) inversion run."""

    run_id: str
    run_dir: Path
    config: RunConfig
    status: str
    band_results: List[BandResult] = field(default_factory=list)
    final_velocity: Any = None
    final_c_path: Optional[Path] = None
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #
def _atomic_json(path: Path, obj: Any) -> None:
    """Atomically write ``obj`` to ``path`` as indented JSON.

    Parameters
    ----------
    path : Path
        Destination file (written via a temp file then renamed).
    obj : Any
        JSON-serialisable object (non-standard types via :func:`_json_default`).

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    """JSON fallback serialiser for numpy scalars and ``Path`` objects.

    Parameters
    ----------
    o : Any
        Object json cannot natively serialise.

    Returns
    -------
    Any
        A JSON-friendly value (float / int / str).
    """
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _band_label(frequencies_hz: List[float]) -> str:
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
def _build_optimiser(
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
) -> BandResult:
    """Run a single continuation stage and write its per-band artifacts.

    ``velocity_model`` is the current (warm) starting model; the recovered
    model is returned in the :class:`BandResult` and becomes the next band's
    start. One ``c_final.npy`` is written for this band.
    """
    cfg = problem.cfg
    freqs = list(band_cfg.frequencies_hz)
    label = _band_label(freqs)
    n_iter = int(
        band_cfg.n_iter if band_cfg.n_iter is not None else cfg.optimiser.n_iter
    )

    t0 = time.perf_counter()
    band_ctx = problem.make_band(freqs)

    t_warm = time.perf_counter()
    u_init = problem.warm_start(velocity_model, band_ctx)
    warm_s = time.perf_counter() - t_warm

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
    band_tape = tape.start_band(band_index, band_ctx)
    # The loss callback keeps the solver's scalar history and result; the
    # tape rows come from ``on_log``.
    loss_tape = LossTape(
        name=f"{SOLVER_NAME}_{label}",
        log_every=cfg.optimiser.log_every,
        store_c_history=False,
    )
    # PML split: observations synthesised on a separate forward grid are
    # reduced to receiver traces; otherwise use full observed wavefields.
    if getattr(band_ctx, "observed_traces", None) is not None:
        loss = InverseLoss(
            observed_traces=band_ctx.observed_traces,
            config=loss_config,
            callback=loss_tape,
        )
    else:
        loss = InverseLoss(
            observed_wavefield=band_ctx.observed_wfs,
            config=loss_config,
            callback=loss_tape,
        )

    opt = _build_optimiser(problem, band_ctx, wf_inv, loss, n_iter, u_init)

    t_opt = time.perf_counter()
    recovered_wfs, _ = opt.minimise(on_log=band_tape.record)
    optimise_s = time.perf_counter() - t_opt

    recovered_velocity = recovered_wfs[0].velocity_model
    opt_result = dict(loss_tape.result or {})
    start_metrics = band_tape.start_row
    final_metrics = band_tape.final_row

    # ---- per-band artifacts --------------------------------------------- #
    band_dir = bands_dir / f"band_{band_index:02d}_{label}"
    band_dir.mkdir(parents=True, exist_ok=True)

    c_full = recovered_velocity.c.detach().cpu().numpy().astype(np.float32)
    c_path = band_dir / "c_final.npy"
    _atomic_npy(c_path, c_full)

    _atomic_json(
        band_dir / "c_final_metadata.json",
        {
            "band_index": band_index,
            "frequencies_hz": freqs,
            "label": label,
            "fft_bins": band_ctx.freq.fft_bins.detach().cpu().tolist(),
            "shape": list(c_full.shape),
            "dtype": str(c_full.dtype),
            "c_min": float(c_full.min()),
            "c_max": float(c_full.max()),
            "c_mean": float(c_full.mean()),
            "grid": grid_summary(problem),
            "path": "c_final.npy",
        },
    )

    _dump_band_config(
        band_dir / "band_config.yaml",
        cfg,
        band_index,
        freqs,
        n_iter,
        band_ctx,
    )

    wall_s = time.perf_counter() - t0
    band_summary = {
        "band_index": band_index,
        "frequencies_hz": freqs,
        "label": label,
        "n_frequencies": int(band_ctx.freq.n_frequencies),
        "fft_bins": band_ctx.freq.fft_bins.detach().cpu().tolist(),
        "n_iter_requested": n_iter,
        "n_transducers": int(band_ctx.geom.n_transducers),
        "t_start_epoch_s": t0,
        "wall_s": wall_s,
        "warm_start_s": warm_s,
        "optimise_s": optimise_s,
        "start_metrics": start_metrics,
        "final_metrics": final_metrics,
        "optimiser_result": opt_result,
        "c_final": "c_final.npy",
    }
    _atomic_json(band_dir / "band_summary.json", band_summary)

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
        c_final_path=c_path,
        velocity_model=recovered_velocity,
    )


def _atomic_npy(path: Path, arr: np.ndarray) -> None:
    """Atomically write a numpy array to ``path`` as ``.npy``.

    Parameters
    ----------
    path : Path
        Destination file (written via a temp file then renamed).
    arr : numpy.ndarray
        Array to save.

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


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


def _dump_band_config(
    path: Path,
    cfg: RunConfig,
    band_index,
    freqs,
    n_iter,
    band_ctx,
):
    """Write the effective per-band configuration to ``path`` as YAML.

    Parameters
    ----------
    path : Path
        Destination ``band_config.yaml``.
    cfg : RunConfig
        The resolved run config.
    band_index :
        Index of this band.
    freqs :
        Band frequencies in Hz.
    n_iter :
        Effective outer-iteration budget for the band.
    band_ctx :
        Per-band context (used for FFT bins / shot count).

    Returns
    -------
    None
    """
    from .config import _asdict, _dump_yaml  # torch-free helpers

    payload = {
        "band_index": band_index,
        "frequencies_hz": freqs,
        "n_iter_effective": n_iter,
        "fft_bins": band_ctx.freq.fft_bins.detach().cpu().tolist(),
        "n_transducers": int(band_ctx.geom.n_transducers),
        "space_order": cfg.grid.space_order,
        "pml": _asdict(cfg.pml),
        "loss_weights": dict(cfg.loss.weights),
        "optimiser": _asdict(cfg.optimiser),
    }
    _dump_yaml(payload, path)


# --------------------------------------------------------------------------- #
# Whole run
# --------------------------------------------------------------------------- #
def run_inverse(
    config: RunConfig,
    *,
    input_config: Optional[Dict[str, Any]] = None,
    on_band_end: Optional[Callable[["BandResult", "RunResult"], None]] = None,
) -> RunResult:
    """Run the full (single- or multi-band) inversion and write all artifacts.

    ``input_config`` (the raw user dict before resolution) is accepted for
    backwards-compatible call sites but is no longer written to disk; the fully
    resolved ``config_resolved.yaml`` is the sole recorded configuration.

    ``on_band_end(band_result, run_result)`` is called after every completed
    frequency band, with the freshly recovered model already recorded in
    ``run_result.band_results``. Use it to monitor progress live and stop a
    stalled run early (e.g. :class:`odil_wave.experiment.LiveVelocityView`). A
    callback error is caught and reported, never aborting the run.
    """
    warnings = validate_config(config)
    for w in warnings:
        print(f"  [config warning] {w}")

    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    bands_dir = run_dir / "bands"
    final_dir = run_dir / "final"
    bands_dir.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True)

    # Config first, so a crash still leaves the run reproducible. The fully
    # resolved config records every effective value, so it is the single source
    # of truth for how the run was produced.
    config.to_yaml(run_dir / "config_resolved.yaml")

    seed_record = envinfo.seed_everything(
        config.run.seed, deterministic=config.runtime.deterministic
    )
    device = envinfo.resolve_device(config.runtime.device)
    meta = envinfo.collect_metadata(
        run_id=config.run.run_id,
        optimiser=SOLVER_NAME,
        device_requested=config.runtime.device,
        device_resolved=str(device),
        dtype=config.runtime.dtype,
        seed=config.run.seed,
        repo_root=Path(__file__).resolve().parents[2],
    )
    band_stages = list(config.continuation.bands)
    meta["seed_record"] = seed_record
    meta["n_bands"] = len(band_stages)
    meta["n_config_bands"] = len(config.continuation.bands)
    meta["config_warnings"] = warnings

    t_run0 = time.perf_counter()
    result = RunResult(
        run_id=config.run.run_id,
        run_dir=run_dir,
        config=config,
        status="running",
    )
    last_band_index = -1
    try:
        problem = build_problem(config)
        meta["grid"] = grid_summary(problem)
        _atomic_json(run_dir / "metadata.json", meta)

        # Raises ConfigError, before any band, if the truth gives nothing to score.
        tape = MetricTape(
            run_dir / "tape.csv",
            run_id=config.run.run_id,
            problem=problem,
            t0=t_run0,
        )

        current = problem.init_velocity
        for bi, band_cfg in enumerate(band_stages):
            last_band_index = bi
            band_res = run_frequency_band(
                problem, current, band_cfg, tape, bi, bands_dir
            )
            result.band_results.append(band_res)
            current = band_res.velocity_model

            if on_band_end is not None:
                try:
                    on_band_end(band_res, result)
                except Exception as cb_exc:  # never let a monitor kill a run
                    print(f"  [on_band_end warning] {type(cb_exc).__name__}: {cb_exc}")

        # ---- final artifacts -------------------------------------------- #
        final_c = current.c.detach().cpu().numpy().astype(np.float32)
        final_path = final_dir / "c_final.npy"
        _atomic_npy(final_path, final_c)
        result.final_velocity = current
        result.final_c_path = final_path

        # Tape rows: the run's first start row, its final iterate, and each
        # band's final iterate.
        final_metrics = result.band_results[-1].final_metrics
        result.metrics_summary = {
            "n_bands": len(result.band_results),
            "start_metrics": result.band_results[0].start_metrics,
            "final_metrics": final_metrics,
            "per_band": [b.final_metrics for b in result.band_results],
        }
        result.status = "completed"
        run_summary = {
            "run_id": config.run.run_id,
            "optimiser": SOLVER_NAME,
            "status": result.status,
            "wall_s": time.perf_counter() - t_run0,
            "n_bands": len(result.band_results),
            "final_c": "c_final.npy",
            "metrics_summary": result.metrics_summary,
            "grid": grid_summary(problem),
        }
        _atomic_json(final_dir / "run_summary.json", run_summary)
        print(
            f"[run {config.run.run_id}] completed: "
            f"final rmse_brain={final_metrics['rmse_brain']:.2f} m/s, "
            f"ssim_brain={final_metrics['ssim_brain']:.4f}"
        )
        return result

    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        result.status = "failed"
        # iter_total of the last tape row written, if any.
        last_row = getattr(locals().get("tape"), "last_row", None)
        last_completed_iter = None if last_row is None else last_row["iter_total"]
        err = {
            "run_id": config.run.run_id,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "last_band_index": last_band_index,
            "last_completed_logical_iteration": last_completed_iter,
            "wall_s": time.perf_counter() - t_run0,
        }
        result.error = err
        _atomic_json(run_dir / "failure.json", err)
        print(f"[run {config.run.run_id}] FAILED: {type(exc).__name__}: {exc}")
        raise
