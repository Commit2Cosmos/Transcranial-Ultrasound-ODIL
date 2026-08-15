"""Run orchestration: build the problem once, dispatch to the selected solver
per frequency band, and write a self-contained artifact directory per run.

Public entry points
--------------------
``run_inverse(config) -> RunResult``            dispatch by ``optimiser.name``
``run_inverse_lbfgsb(config) -> RunResult``      force the lbfgsb solver, then run
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
import torch

from odil_wave.loss import InverseLoss, LossConfig, Regulariser
from odil_wave.metrics import ssim as _ssim
from odil_wave.operator import WaveEquation
from odil_wave.wavefield import Wavefield

from . import envinfo
from .config import (
    BandCfg,
    ConfigError,
    RunConfig,
    canonical_optimiser_name,
    canonical_regulariser_name,
    deep_merge,
    resolve_config,
    validate_config,
)
from .problem import Problem, build_problem, grid_summary
from .recorder import RunRecorder


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
    n_iter_run: int
    termination_reason: str
    wall_s: float
    final_metrics: Dict[str, Any]
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
# Metrics helpers
# --------------------------------------------------------------------------- #
def _final_recovery_metrics(problem: Problem, velocity_model: Any) -> Dict[str, Any]:
    """SSIM (head ROI) + relative L2 c-error of a recovered model vs truth."""
    out: Dict[str, Any] = {
        "ssim_head_roi": None,
        "ssim_reason": None,
        "rel_c_error": None,
    }
    truth = problem.truth_velocity
    grid = problem.grid
    if truth is None:
        out["ssim_reason"] = "no ground truth"
        return out
    sc = problem.cfg.metrics.ssim
    mask = problem.head_mask if sc.mask == "head_roi" else None
    if sc.mask == "head_roi" and mask is None:
        out["ssim_reason"] = "no head_mask for truth profile"
    else:
        try:
            kwargs: Dict[str, Any] = {
                "win_size": int(sc.win_size),
                "gaussian_weights": bool(sc.gaussian_weights),
            }
            if sc.gaussian_weights:
                kwargs["sigma"] = float(sc.sigma)
            if sc.data_range != "auto":
                kwargs["data_range"] = float(sc.data_range)
            val = _ssim(velocity_model.c, truth.c, grid, mask=mask, **kwargs)
            out["ssim_head_roi"] = float(val) if np.isfinite(val) else None
        except Exception as exc:
            out["ssim_reason"] = f"ssim error: {type(exc).__name__}: {exc}"
    with torch.no_grad():
        c_int = (
            velocity_model.c[grid.interior_slice]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        t_int = truth.c[grid.interior_slice].detach().cpu().numpy().astype(np.float64)
    denom = float(np.linalg.norm(t_int))
    if denom > 0:
        out["rel_c_error"] = float(np.linalg.norm(c_int - t_int) / denom)
    return out


# --------------------------------------------------------------------------- #
# Optimiser dispatch (matches the notebook helpers exactly)
# --------------------------------------------------------------------------- #
def _effective_c_grad_smooth_sigma(lb, band_cfg: Optional[BandCfg]) -> float:
    """Per-band ``c_grad_smooth_sigma`` override, falling back to the global one."""
    if band_cfg is not None and band_cfg.c_grad_smooth_sigma is not None:
        return float(band_cfg.c_grad_smooth_sigma)
    return float(lb.c_grad_smooth_sigma)


def _schedule_tuple(sch) -> tuple:
    """Pack a :class:`SchedulerCfg` as the ``(factor, every_n, direction)`` tuple
    the optimiser consumes (torch-free, so LBFGSB need not import the config)."""
    return (float(sch.factor), int(sch.every_n), str(sch.direction).lower())


def _build_optimiser(
    problem: Problem,
    band_ctx,
    wf_inv: Wavefield,
    loss: InverseLoss,
    n_iter: int,
    u_init,
    band_cfg: Optional[BandCfg] = None,
):
    """Construct the configured optimiser for one band.

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
        Warm-start wavefield (or ``None``).
    band_cfg : BandCfg, optional
        Per-band overrides (e.g. ``c_grad_smooth_sigma``).

    Returns
    -------
    optimiser
        An ``LBFGSB`` or ``JointFreqODIL`` instance per ``optimiser.name``.

    Notes
    -----
    Raises :class:`ValueError` for an unhandled optimiser name.
    """
    cfg = problem.cfg
    opt = cfg.optimiser
    name = canonical_optimiser_name(opt.name)
    shared = dict(
        clamp=opt.clamp,
        u_init=u_init,
        line_search_fn=opt.line_search_fn,
        n_iter=n_iter,
        u_steps=opt.u_steps,
        max_iter=opt.max_iter,
        history_size=opt.history_size,
        tolerance_grad=opt.tolerance_grad,
        tolerance_change=opt.tolerance_change,
    )

    if name == "lbfgsb":
        from odil_wave.optimisation import LBFGSB

        lb = opt.lbfgsb
        c_grad_smooth_sigma = _effective_c_grad_smooth_sigma(lb, band_cfg)
        return LBFGSB(
            wf_inv,
            loss,
            **shared,
            c_steps=opt.c_steps,
            z_steps=lb.z_steps,
            u_precond=lb.u_precond,
            u_solve=lb.u_solve,
            z_optim=lb.z_optim,
            z_lr=lb.z_lr,
            c_lr=lb.c_lr,
            c_max_iter=lb.c_max_iter,
            c_history_size=lb.c_history_size,
            reset_c_history=lb.reset_c_history,
            c_param=lb.c_param,
            c_grad_smooth_sigma=c_grad_smooth_sigma,
            pde_weight_schedule=_schedule_tuple(lb.pde_weight_schedule),
            c_grad_smooth_sigma_schedule=_schedule_tuple(
                lb.c_grad_smooth_sigma_schedule
            ),
        )

    if name == "joint":
        # Pure joint full-space ODIL: one L-BFGS over (u, z_m); no alternation,
        # closed-form, WRI or multigrid. data_weight is fixed (never adapted).
        from odil_wave.optimisation import JointFreqODIL

        jo = opt.joint
        ls = None if str(jo.line_search_fn).lower() == "none" else jo.line_search_fn
        joint_kw = dict(
            clamp=opt.clamp,
            u_init=u_init,
            n_iter=n_iter,
            inner_max_iter=jo.inner_max_iter,
            history_size=jo.history_size,
            line_search_fn=ls,
            tolerance_grad=jo.tolerance_grad,
            tolerance_change=jo.tolerance_change,
            lbfgs_lr=jo.lbfgs_lr,
            data_weight=jo.data_weight,
            reg_weight=jo.reg_weight,
            c_min=jo.c_min,
            c_max=jo.c_max,
            eps_u=jo.eps_u,
            eps_data=jo.eps_data,
            eps_pde=jo.eps_pde,
            u_scale_factor=jo.u_scale_factor,
            pde_scale_factor=jo.pde_scale_factor,
            data_scale_factor=jo.data_scale_factor,
            z_scale=jo.z_scale,
            logit_clip=jo.logit_clip,
            log_every=opt.log_every,
            verbose=jo.verbose,
        )
        return JointFreqODIL(wf_inv, loss, **joint_kw)

    raise ValueError(f"unhandled optimiser {name!r}")


# --------------------------------------------------------------------------- #
# One band / stage
# --------------------------------------------------------------------------- #
def run_frequency_band(
    problem: Problem,
    velocity_model: Any,
    band_cfg: BandCfg,
    recorder: RunRecorder,
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
        pml_weight=problem.pml_weight,
        time_order=cfg.physics.time_order,
    )
    loss_config = LossConfig(
        wave_eq=wave_eq,
        geometry=band_ctx.geom,
        weights=dict(cfg.loss.weights),
        regulariser=_build_regulariser(cfg),
    )
    tape = recorder.band_tape(
        band_index=band_index,
        frequencies_hz=freqs,
        name=f"{cfg.optimiser.name}_{label}",
        log_every=cfg.optimiser.log_every,
    )
    # PML split: observations synthesised on a separate forward grid are
    # reduced to receiver traces; otherwise use full observed wavefields.
    if getattr(band_ctx, "observed_traces", None) is not None:
        loss = InverseLoss(
            observed_traces=band_ctx.observed_traces,
            config=loss_config,
            callback=tape,
            normalize_data=cfg.observation.normalize_data,
        )
    else:
        loss = InverseLoss(
            observed_wavefield=band_ctx.observed_wfs,
            config=loss_config,
            callback=tape,
            normalize_data=cfg.observation.normalize_data,
        )
    tape.bind_loss(loss)

    opt = _build_optimiser(problem, band_ctx, wf_inv, loss, n_iter, u_init, band_cfg)

    minimise_kwargs: Dict[str, Any] = {"on_iteration": tape.on_iteration}
    diag_cfg = getattr(cfg, "diagnostics", None)
    if (
        diag_cfg is not None
        and diag_cfg.enabled
        and canonical_optimiser_name(cfg.optimiser.name) == "lbfgsb"
        and problem.truth_velocity is not None
    ):
        from .diagnostics import DiagnosticsCollector

        diag_dir = bands_dir.parent / "diagnostics" / f"band_{band_index:02d}_{label}"
        minimise_kwargs["diagnostics"] = DiagnosticsCollector(
            out_dir=diag_dir,
            truth_velocity=problem.truth_velocity,
            grid=problem.grid,
            per_outer_field_maps=diag_cfg.per_outer_field_maps,
            u_depths=diag_cfg.u_depths,
            verify_hessian=diag_cfg.verify_hessian,
        )

    t_opt = time.perf_counter()
    recovered_wfs, _ = opt.minimise(**minimise_kwargs)
    optimise_s = time.perf_counter() - t_opt

    recovered_velocity = recovered_wfs[0].velocity_model
    opt_result = dict(getattr(tape, "result", None) or {})
    n_iter_run = int(opt_result.get("n_outer_iter", n_iter))
    termination = "budget_exhausted"

    recovery = _final_recovery_metrics(problem, recovered_velocity)
    hist = tape.history
    final_metrics = {
        "loss": opt_result.get("loss"),
        "pde_rms": (hist["pde_rms"][-1] if hist.get("pde_rms") else None),
        "data_rms": (hist["data_rms"][-1] if hist.get("data_rms") else None),
        "pde_src_ratio": (
            hist["pde_src_ratio"][-1] if hist.get("pde_src_ratio") else None
        ),
        **recovery,
    }

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
        band_cfg,
    )

    wall_s = time.perf_counter() - t0
    band_summary = {
        "band_index": band_index,
        "frequencies_hz": freqs,
        "label": label,
        "n_frequencies": int(band_ctx.freq.n_frequencies),
        "fft_bins": band_ctx.freq.fft_bins.detach().cpu().tolist(),
        "n_iter_requested": n_iter,
        "n_iter_run": n_iter_run,
        "termination_reason": termination,
        "n_shots": int(band_ctx.geom.n_sources),
        "t_start_epoch_s": t0,
        "wall_s": wall_s,
        "warm_start_s": warm_s,
        "optimise_s": optimise_s,
        "warm_start": cfg.continuation.warm_start,
        "final_metrics": final_metrics,
        "optimiser_result": opt_result,
        "c_final": "c_final.npy",
    }
    _atomic_json(band_dir / "band_summary.json", band_summary)

    ssim_s = (
        f"{final_metrics['ssim_head_roi']:.4f}"
        if final_metrics["ssim_head_roi"] is not None
        else "n/a"
    )
    loss_s = (
        f"{final_metrics['loss']:.6e}"
        if isinstance(final_metrics["loss"], (int, float))
        else "n/a"
    )
    print(
        f"  [band {band_index} {label}] iters {n_iter_run}/{n_iter} "
        f"loss={loss_s}  ssim_head_roi={ssim_s}  "
        f"wall={wall_s:.1f}s ({termination})"
    )

    return BandResult(
        band_index=band_index,
        frequencies_hz=freqs,
        label=label,
        n_iter_requested=n_iter,
        n_iter_run=n_iter_run,
        termination_reason=termination,
        wall_s=wall_s,
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

    ``cfg.loss.regulariser.name`` selects the penalty kind (``tikhonov`` /
    ``tv_iso`` / ``tv_aniso``; aliases resolved by
    :func:`~odil_wave.experiment.config.canonical_regulariser_name`) and
    ``params`` forwards keyword arguments to the constructor (only ``eps``, the
    ``tv_iso`` smoothing floor). Returns ``None`` when no regulariser is
    configured, so :class:`~odil_wave.loss.LossConfig` omits the reg term.

    The scalar weight is *not* set here: it is ``loss.weights['reg']``, applied
    by :class:`~odil_wave.loss.InverseLoss` (and by the closed-form proximal
    step) at evaluation time.
    """
    reg = cfg.loss.regulariser
    if reg is None or reg.name is None:
        return None
    kind = canonical_regulariser_name(reg.name)
    params = dict(reg.params or {})
    unknown = set(params) - {"eps"}
    if unknown:
        raise ConfigError(
            f"loss.regulariser.params has unknown key(s) {sorted(unknown)}; "
            "the only accepted param is 'eps'."
        )
    return Regulariser(kind=kind, **params)


def _dump_band_config(
    path: Path,
    cfg: RunConfig,
    band_index,
    freqs,
    n_iter,
    band_ctx,
    band_cfg: Optional[BandCfg] = None,
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
    band_cfg : BandCfg, optional
        Per-band overrides recorded for the lbfgsb optimiser.

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
        "n_shots": int(band_ctx.geom.n_sources),
        "warm_start": cfg.continuation.warm_start,
        "observation_method": cfg.observation.method,
        "normalize_data": cfg.observation.normalize_data,
        "physics": _asdict(cfg.physics),
        "loss_weights": dict(cfg.loss.weights),
        "optimiser": _asdict(cfg.optimiser),
    }
    name = canonical_optimiser_name(cfg.optimiser.name)
    if name == "lbfgsb":
        payload["effective_c_grad_smooth_sigma"] = _effective_c_grad_smooth_sigma(
            cfg.optimiser.lbfgsb, band_cfg
        )
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
        optimiser=config.optimiser.name,
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
    meta["observation_method"] = config.observation.method
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

        recorder = RunRecorder(
            run_id=config.run.run_id,
            solver_name=config.optimiser.name,
            out_dir=run_dir,
            grid=problem.grid,
            truth_velocity=problem.truth_velocity,
            metrics_cfg=config.metrics,
        )

        current = problem.init_velocity
        for bi, band_cfg in enumerate(band_stages):
            last_band_index = bi
            band_res = run_frequency_band(
                problem, current, band_cfg, recorder, bi, bands_dir
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

        recovery = _final_recovery_metrics(problem, current)
        result.metrics_summary = {
            "n_bands": len(result.band_results),
            "final_ssim_head_roi": recovery["ssim_head_roi"],
            "final_rel_c_error": recovery["rel_c_error"],
            "per_band": [
                {
                    "band_index": b.band_index,
                    "label": b.label,
                    "ssim_head_roi": b.final_metrics.get("ssim_head_roi"),
                    "rel_c_error": b.final_metrics.get("rel_c_error"),
                    "loss": b.final_metrics.get("loss"),
                    "n_iter_run": b.n_iter_run,
                    "termination_reason": b.termination_reason,
                }
                for b in result.band_results
            ],
        }
        result.status = "completed"
        run_summary = {
            "run_id": config.run.run_id,
            "optimiser": config.optimiser.name,
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
            f"final ssim_head_roi={recovery['ssim_head_roi']}, "
            f"rel_c_error={recovery['rel_c_error']}"
        )
        return result

    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        result.status = "failed"
        last_completed_iter = None
        try:
            # global_iter counts logged records emitted so far.
            last_completed_iter = getattr(
                locals().get("recorder", None), "_global_iter", None
            )
        except Exception:
            pass
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


def _rerun_with_optimiser(config: RunConfig, name: str) -> RunResult:
    """Re-resolve ``config`` with a forced optimiser and fresh run id, then run.

    Parameters
    ----------
    config : RunConfig
        Base configuration.
    name : str
        Optimiser name to force (e.g. ``"lbfgsb"``).

    Returns
    -------
    RunResult
        Result of the re-run inversion.
    """
    merged = deep_merge(
        config.to_dict(), {"optimiser": {"name": name}, "run": {"run_id": ""}}
    )
    return run_inverse(resolve_config(merged))


def run_inverse_lbfgsb(config: RunConfig) -> RunResult:
    """Run the inversion forcing the lbfgsb optimiser.

    Parameters
    ----------
    config : RunConfig
        Base configuration (its optimiser name is overridden).

    Returns
    -------
    RunResult
        Result of the lbfgsb inversion.
    """
    return _rerun_with_optimiser(config, "lbfgsb")
