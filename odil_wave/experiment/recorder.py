"""Per-iteration metric tape.

``RecordingTape`` subclasses :class:`odil_wave.loss.LossTape` and doubles as the
optimiser's ``on_iteration`` hook:

* ``on_iteration(i, c_full)`` fires once per **outer iteration** of the
  joint optimiser, *before* the tape logs. It stamps the outer
  iteration index and the current velocity field.
* the inherited ``log(loss, residuals, pde_src_ratio)`` fires only at the
  optimiser's ``log_every`` cadence and carries the exact loss the solver
  reports. Overriding it lets us emit one structured record per logged
  iteration with the correct outer index, loss, residuals and velocity field —
  with **zero changes to the optimiser**.

Semantics of one record: it corresponds to a single *logical outer iteration*
``i`` of the solver, **not** an L-BFGS closure / function
evaluation. The run-wide cumulative closure count of the joint optimiser is
logged separately as ``n_evaluations`` for diagnostics.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from odil_wave.loss import LossTape
from odil_wave.metrics import ssim as _ssim

from .config import MetricsCfg


# Ordered CSV columns == JSONL keys, as written today. The README's tape
# table is the planned replacement (#62), not these columns.
FIELDNAMES: List[str] = [
    "run_id",
    "solver",
    "band_index",
    "band_frequencies_hz",
    "solver_iter",
    "global_iter",
    "wall_s",
    "loss_total",
    "pde_term",
    "data_term",
    "reg_term",
    "pde_loss",
    "data_loss",
    "pde_rms",
    "data_rms",
    "pde_src_ratio",
    "c_min",
    "c_max",
    "c_mean",
    "c_std",
    "rel_c_error",
    "ssim_head_roi",
    "ssim_reason",
    "n_evaluations",
]


class RecordingTape(LossTape):
    """A ``LossTape`` that also emits structured per-iteration records.

    Pass the *same* instance as ``InverseLoss(callback=...)`` and as
    ``opt.minimise(on_iteration=tape.on_iteration)``.
    """

    def __init__(
        self,
        *,
        recorder: "RunRecorder",
        band_index: int,
        band_frequencies_hz: List[float],
        name: str,
        log_every: int,
        store_c_history: bool = False,
    ) -> None:
        """Create a per-band tape wired to a parent recorder.

        Parameters
        ----------
        recorder : RunRecorder
            Owner that formats and appends the emitted rows.
        band_index : int
            Index of the frequency band this tape records.
        band_frequencies_hz : list of float
            Frequencies of the band, in Hz.
        name : str
            Tape name forwarded to :class:`LossTape`.
        log_every : int
            Logging cadence in outer iterations.
        store_c_history : bool, optional
            Whether to retain the per-iteration velocity history.
        """
        super().__init__(
            name=name, log_every=log_every, store_c_history=store_c_history
        )
        self._rec = recorder
        self.band_index = int(band_index)
        self.band_frequencies_hz = [float(f) for f in band_frequencies_hz]
        self._cur_iter: Optional[int] = None
        self._cur_c: Optional[torch.Tensor] = None

    # -- optimiser hook (every outer iteration) ---------------------------- #
    def on_iteration(self, i: int, c_full: Optional[torch.Tensor]) -> None:
        """Stamp the current outer index and velocity field (optimiser hook).

        Parameters
        ----------
        i : int
            Outer-iteration index.
        c_full : torch.Tensor or None
            Current full-grid velocity field.

        Returns
        -------
        None
        """
        self._cur_iter = int(i)
        self._cur_c = c_full

    # -- logging hook (every log_every) ------------------------------------ #
    def log(
        self, loss: float, residuals, pde_src_ratio: Optional[float] = None
    ) -> None:
        """Log to the base tape and emit one structured record (logging hook).

        Parameters
        ----------
        loss : float
            Loss value reported by the solver.
        residuals :
            Residual dict (or object) carried through to the emitted row.
        pde_src_ratio : float, optional
            PDE-residual to source ratio for this iteration.

        Returns
        -------
        None

        Notes
        -----
        Fires only at the ``log_every`` cadence of the base :class:`LossTape`.
        """
        super().log(loss, residuals, pde_src_ratio=pde_src_ratio)
        self._rec.emit_row(
            tape=self,
            loss=float(loss),
            residuals=residuals,
            pde_src_ratio=pde_src_ratio,
        )


class RunRecorder:
    """Owns the run-level metric files and mints per-band recording tapes."""

    def __init__(
        self,
        *,
        run_id: str,
        solver_name: str,
        out_dir: Path,
        grid: Any,
        truth_velocity: Any,
        metrics_cfg: MetricsCfg,
        t0: Optional[float] = None,
    ) -> None:
        """Create the run-level metric writer and cache the ground truth.

        Parameters
        ----------
        run_id : str
            Identifier stamped on every emitted row.
        solver_name : str
            Name of the solver producing the run.
        out_dir : Path
            Directory for ``metrics.csv`` / ``metrics.jsonl``.
        grid :
            Grid object providing ``interior_slice``.
        truth_velocity :
            Ground-truth model (or ``None``) for error / SSIM metrics.
        metrics_cfg : MetricsCfg
            SSIM and velocity-statistics settings.
        t0 : float, optional
            Wall-clock start reference; defaults to ``time.perf_counter()``.
        """
        self.run_id = run_id
        self.solver_name = solver_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "metrics.csv"
        self.jsonl_path = self.out_dir / "metrics.jsonl"
        self.grid = grid
        self.metrics_cfg = metrics_cfg
        self.t0 = time.perf_counter() if t0 is None else t0
        self._global_iter = 0
        # Closures of finished bands, so ``n_evaluations`` spans the whole run.
        self._n_eval_done = 0
        self._last_tape: Optional[RecordingTape] = None

        # Ground truth (full grid) + interior arrays for error / SSIM.
        if truth_velocity is not None:
            self._truth_full = truth_velocity.c.detach()
            self._truth_int_np = (
                self._truth_full[grid.interior_slice].cpu().numpy().astype(np.float64)
            )
            hm = getattr(truth_velocity, "head_mask", None)
            self._head_mask = (
                None if hm is None else hm.detach().cpu().numpy().astype(bool)
            )
        else:
            self._truth_full = None
            self._truth_int_np = None
            self._head_mask = None

    # -- tape factory ------------------------------------------------------ #
    def band_tape(
        self,
        *,
        band_index: int,
        frequencies_hz: List[float],
        name: str,
        log_every: int,
    ) -> RecordingTape:
        """Mint a :class:`RecordingTape` bound to this recorder for one band.

        Parameters
        ----------
        band_index : int
            Index of the frequency band.
        frequencies_hz : list of float
            Frequencies of the band, in Hz.
        name : str
            Tape name.
        log_every : int
            Logging cadence in outer iterations.

        Returns
        -------
        RecordingTape
            A tape wired to this recorder.
        """
        # Fold the finished band's closure total in before starting the next.
        if self._last_tape is not None:
            prev = self._last_tape.result or {}
            self._n_eval_done += int(prev.get("n_evaluations", 0))
        self._last_tape = RecordingTape(
            recorder=self,
            band_index=band_index,
            band_frequencies_hz=frequencies_hz,
            name=name,
            log_every=log_every,
        )
        return self._last_tape

    # -- SSIM on the configured head ROI ----------------------------------- #
    def _ssim_head_roi(self, c_full: torch.Tensor) -> tuple:
        """Return ``(ssim_value_or_None, reason_or_None)``.

        Computed only on the configured mask (head ROI by default), which the
        underlying ``ssim`` restricts to the non-PML interior — PML and
        outside-mask pixels never enter the score.
        """
        if self._truth_full is None:
            return None, "no ground truth"
        sc = self.metrics_cfg.ssim
        if sc.mask == "head_roi":
            if self._head_mask is None:
                return None, "no head_mask for truth profile"
            mask = self._head_mask
        elif sc.mask == "interior":
            mask = None
        elif sc.mask == "none":
            mask = None
        else:
            return None, f"unknown ssim mask {sc.mask!r}"

        kwargs: Dict[str, Any] = {
            "win_size": int(sc.win_size),
            "gaussian_weights": bool(sc.gaussian_weights),
        }
        if sc.gaussian_weights:
            kwargs["sigma"] = float(sc.sigma)
        if sc.data_range != "auto":
            kwargs["data_range"] = float(sc.data_range)
        try:
            val = _ssim(c_full, self._truth_full, self.grid, mask=mask, **kwargs)
            if not np.isfinite(val):
                return None, "ssim non-finite"
            return float(val), None
        except Exception as exc:  # window too large for grid, etc.
            return None, f"ssim error: {type(exc).__name__}: {exc}"

    # -- one record -------------------------------------------------------- #
    def emit_row(
        self,
        *,
        tape: RecordingTape,
        loss: float,
        residuals: Any,
        pde_src_ratio: Optional[float],
    ) -> Dict[str, Any]:
        """Assemble one metric record, append it to disk, and return it.

        Parameters
        ----------
        tape : RecordingTape
            Tape supplying the band, outer index and current velocity field.
        loss : float
            Total loss reported by the solver.
        residuals : Any
            Residual dict with the per-term losses / RMS values.
        pde_src_ratio : float or None
            PDE-residual to source ratio.

        Returns
        -------
        dict
            The row written to ``metrics.csv`` / ``metrics.jsonl``.
        """
        res = residuals if isinstance(residuals, dict) else {}
        pde_loss = _f(res.get("pde_loss"))
        data_loss = _f(res.get("data_loss"))
        pde_rms = _f(res.get("pde_rms"))
        data_rms = _f(res.get("data_rms"))
        ratio = _f(
            pde_src_ratio if pde_src_ratio is not None else res.get("pde_src_ratio")
        )

        band_n_eval = res.get("n_evaluations")
        n_eval = None if band_n_eval is None else self._n_eval_done + int(band_n_eval)
        # Weighted terms as reported by the optimiser (which applies the
        # weights), from the same objective evaluation as ``loss``, so
        # pde_term + data_term + reg_term == loss_total.
        pde_term = _f(res.get("pde_term"))
        data_term = _f(res.get("data_term"))
        reg_term = _f(res.get("reg_term"))

        # Velocity-field summaries on the non-PML interior.
        c_min = c_max = c_mean = c_std = rel_err = None
        ssim_val = None
        ssim_reason = "no c field this iteration"
        c_full = tape._cur_c
        if c_full is not None:
            with torch.no_grad():
                c_int = c_full[self.grid.interior_slice]
                c_min = float(c_int.min())
                c_max = float(c_int.max())
                c_mean = float(c_int.mean())
                c_std = float(c_int.std())
            if self._truth_int_np is not None:
                c_int_np = c_int.detach().cpu().numpy().astype(np.float64)
                denom = float(np.linalg.norm(self._truth_int_np))
                if denom > 0:
                    rel_err = float(
                        np.linalg.norm(c_int_np - self._truth_int_np) / denom
                    )
            ssim_val, ssim_reason = self._ssim_head_roi(c_full)

        self._global_iter += 1
        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "solver": self.solver_name,
            "band_index": tape.band_index,
            "band_frequencies_hz": tape.band_frequencies_hz,
            "solver_iter": (tape._cur_iter if tape._cur_iter is not None else None),
            "global_iter": self._global_iter,
            "wall_s": round(time.perf_counter() - self.t0, 6),
            "loss_total": float(loss),
            "pde_term": pde_term,
            "data_term": data_term,
            "reg_term": reg_term,
            "pde_loss": pde_loss,
            "data_loss": data_loss,
            "pde_rms": pde_rms,
            "data_rms": data_rms,
            "pde_src_ratio": ratio,
            "c_min": c_min,
            "c_max": c_max,
            "c_mean": c_mean,
            "c_std": c_std,
            "rel_c_error": rel_err,
            "ssim_head_roi": ssim_val,
            "ssim_reason": ssim_reason,
            "n_evaluations": n_eval,
        }
        self._append(row)
        return row

    # -- crash-safe append ------------------------------------------------- #
    def _append(self, row: Dict[str, Any]) -> None:
        """Append one record to the CSV and JSONL files, flushing each.

        Parameters
        ----------
        row : dict
            The metric record to append.

        Returns
        -------
        None

        Notes
        -----
        Writes the CSV header on the first (empty) file and flushes both
        streams so a crash leaves the tape crash-safe.
        """
        new_csv = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            if new_csv:
                writer.writeheader()
            writer.writerow({k: _csv_cell(row.get(k)) for k in FIELDNAMES})
            f.flush()
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()


def _f(x: Any) -> Optional[float]:
    """Coerce ``x`` to a finite float, or ``None``.

    Parameters
    ----------
    x : Any
        Value to coerce.

    Returns
    -------
    float or None
        ``float(x)`` if finite; ``None`` if ``x`` is None, non-numeric or
        non-finite.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _csv_cell(v: Any) -> Any:
    """Render lists as JSON so CSV cells stay single-valued; None -> empty."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return v
