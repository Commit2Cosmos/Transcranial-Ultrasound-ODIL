"""The metric tape: one ``tape.csv`` row per logged iterate of a run.

:data:`COLUMNS` is the column set, each with its type, finiteness and meaning;
:func:`load_tape` reads the file back by them. :class:`MetricTape` owns the
file for a run; each band gets a :class:`BandTape` whose
:meth:`BandTape.record` is the joint optimiser's ``on_log`` hook.

Rows per band: a start row (``iter = 0``, after the warm start and before any
update, so its data residual is the forward-model misfit), one every
``optimiser.log_every`` outer iterations, and one at the band's final iterate.
The objective columns come from one evaluation at the logged point, so they
sum to ``loss_total``. Band and run summaries report these same columns.

The optimiser supplies the objective terms, residual ratios, gradient norms and
``sat_frac`` (:class:`~odil_wave.optimisation.Iterate`); this module adds the
run-wide counters and everything measured against the truth (regions as in
``GLOSSARY.md``). A truth that gives nothing to score is a config error
(:class:`ModelError` raises it) before the first band, never an empty cell; a
non-finite value is written as ``nan`` / ``inf``.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

from odil_wave.optimisation import Iterate

from .config import ConfigError, ssim_window_size


@dataclass(frozen=True)
class Column:
    """One ``tape.csv`` column: its name, the type its cells read as, what it
    measures, and whether a non-finite value in it fails the run."""

    name: str
    type: type
    doc: str
    finite: bool = False


# ``tape.csv`` columns, in order. An empty cell is ``None`` whatever the type.
COLUMNS: Tuple[Column, ...] = (
    # identity and cost
    Column(
        "run_id",
        str,
        "Run identity, so rows of several runs can be concatenated.",
    ),
    Column(
        "band_index",
        int,
        "Continuation stage; join to ``band_summary.json`` for its frequencies.",
    ),
    Column(
        "iter",
        int,
        "Outer iteration within the band (``0`` = start row).",
    ),
    Column(
        "iter_total",
        int,
        "Cumulative outer iterations over all bands up to this row.",
    ),
    Column(
        "wall_s",
        float,
        "Wall-clock seconds since the run started.",
    ),
    Column(
        "n_evaluations",
        int,
        "Cumulative L-BFGS closure calls over the whole run, diagnostic evaluations excluded; the line search makes several per outer iteration. The cost axis for equal-cost comparisons.",
    ),
    # objective
    Column(
        "loss_total",
        float,
        "Total objective as the solver reports it.",
        finite=True,
    ),
    Column(
        "pde_term",
        float,
        "Weighted PDE term (ADR-0002).",
        finite=True,
    ),
    Column(
        "data_term",
        float,
        "Weighted data term: a mean over traces, so independent of ``n_transducers``.",
        finite=True,
    ),
    Column(
        "reg_term",
        float,
        "``w_reg · R(c)`` on the interior; ``0.0`` without a regulariser.",
        finite=True,
    ),
    # normalised residuals
    Column(
        "rel_data_residual",
        float,
        "``‖d_syn − d_obs‖ / ‖d_obs‖`` over the band's shots × frequencies × receivers.",
    ),
    Column(
        "pde_rel_residual",
        float,
        "``sqrt(2 · pde_term / w_pde)``: the PDE residual in the PDE term's scale, independent of grid spacing, PML width and source normalisation.",
    ),
    Column(
        "rel_wavefield_error",
        float,
        "``‖u − u*‖ / ‖u*‖`` over the interior, all shots × band frequencies.",
    ),
    Column(
        "rel_model_update",
        float,
        "``‖c − c_prev‖ / ‖c_prev‖`` over the interior against the band's previous row; empty on a start row. Truth-free plateau signal.",
    ),
    # model error (m/s, against c*)
    Column(
        "rmse_brain",
        float,
        "RMS of ``c − c*`` over the brain region, m/s.",
    ),
    Column(
        "bias_brain",
        float,
        "Mean of ``c − c*`` over the brain region, m/s; its sign shows too slow or too fast.",
    ),
    Column(
        "rmse_skull",
        float,
        "RMS of ``c − c*`` over the skull region, m/s.",
    ),
    Column(
        "bias_skull",
        float,
        "Mean of ``c − c*`` over the skull region, m/s; catches a skull left at the initial model.",
    ),
    Column(
        "ssim_brain",
        float,
        "SSIM of ``c`` against ``c*`` over the brain region, eroded by the window radius (:class:`ModelError`).",
    ),
    # optimiser health
    Column(
        "c_min",
        float,
        "Minimum of ``c`` over the interior, m/s.",
        finite=True,
    ),
    Column(
        "c_max",
        float,
        "Maximum of ``c`` over the interior, m/s.",
        finite=True,
    ),
    Column(
        "sat_frac",
        float,
        "Fraction of cells pinned at a speed bound by the bounded-``c`` sigmoid.",
    ),
    Column(
        "grad_u_norm",
        float,
        "Gradient norm with respect to the scaled wavefield ``u_tilde``.",
        finite=True,
    ),
    Column(
        "grad_z_norm",
        float,
        "Gradient norm with respect to the unconstrained model variable.",
        finite=True,
    ),
)

TAPE_COLUMNS: List[str] = [c.name for c in COLUMNS]

_TYPES: Dict[str, type] = {c.name: c.type for c in COLUMNS}


class NonFiniteError(RuntimeError):
    """A tape row with a non-finite loss, gradient or velocity: the run has
    blown up. The row is written before this is raised."""


class ModelError:
    """Error of an interior velocity against the truth, by region.

    Built once per run; raises :class:`ConfigError` when the truth gives
    nothing to score (no brain or skull region, an SSIM window that leaves no
    brain cell after erosion, or a truth constant over the brain region).
    """

    def __init__(
        self,
        truth: np.ndarray,
        brain: np.ndarray,
        skull: np.ndarray,
        *,
        sigma: float,
    ) -> None:
        """``truth`` is the interior ``c*`` (m/s), ``brain`` and ``skull``
        interior-shaped bool, ``sigma`` the SSIM window std in cells."""
        self.truth = np.asarray(truth, dtype=np.float64)
        self.brain = np.asarray(brain, dtype=bool)
        self.skull = np.asarray(skull, dtype=bool)
        if not self.brain.any() or not self.skull.any():
            raise ConfigError("the truth model has an empty brain or skull region")
        self.sigma = float(sigma)
        win = ssim_window_size(self.sigma)
        self.win_size = win
        # Only windows lying wholly inside the brain region are scored.
        self.ssim_region = ndi.binary_erosion(
            self.brain, structure=np.ones((win, win), dtype=bool)
        )
        if not self.ssim_region.any():
            raise ConfigError(
                f"the {win}-cell SSIM window (metrics.ssim.sigma={self.sigma}) "
                "leaves no brain-region cell after erosion; lower the sigma"
            )
        self.data_range = float(np.ptp(self.truth[self.brain]))
        if self.data_range <= 0:
            raise ConfigError(
                "the truth is constant over the brain region, so ssim_brain is "
                "undefined; use a truth with brain-region structure"
            )

    @classmethod
    def from_problem(cls, problem) -> "ModelError":
        """Build from a :class:`~odil_wave.experiment.problem.Problem`."""
        truth = problem.truth_velocity
        if truth is None:
            raise ConfigError("the tape needs a truth model")
        try:
            _, brain, skull = truth.skull_region_masks()
        except ValueError as exc:
            raise ConfigError(f"no brain or skull region: {exc}") from None
        sl = problem.grid.interior_slice
        return cls(
            _np(truth.c[sl]),
            _np(brain[sl]),
            _np(skull[sl]),
            sigma=problem.cfg.metrics.ssim.sigma,
        )

    def __call__(self, c: np.ndarray) -> Dict[str, float]:
        """Region RMSE / bias (m/s) and ``ssim_brain`` of interior velocity ``c``."""
        diff = np.asarray(c, dtype=np.float64) - self.truth
        out: Dict[str, float] = {}
        for name, region in (("brain", self.brain), ("skull", self.skull)):
            d = diff[region]
            out[f"rmse_{name}"] = float(np.sqrt(np.mean(d**2)))
            out[f"bias_{name}"] = float(np.mean(d))
        _, ssim_map = structural_similarity(
            np.asarray(c, dtype=np.float64),
            self.truth,
            win_size=self.win_size,
            gaussian_weights=True,
            sigma=self.sigma,
            use_sample_covariance=False,
            data_range=self.data_range,
            full=True,
        )
        out["ssim_brain"] = float(ssim_map[self.ssim_region].mean())
        return out


class MetricTape:
    """A run's ``tape.csv``: appends one row per logged iterate, flushed per row."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        problem,
        t0: Optional[float] = None,
    ) -> None:
        """``t0`` is ``time.perf_counter()`` at run start, the zero of ``wall_s``.
        Raises :class:`ConfigError` as :class:`ModelError` does."""
        self.path = Path(path)
        self.run_id = run_id
        self.problem = problem
        self.model_error = ModelError.from_problem(problem)
        self.t0 = time.perf_counter() if t0 is None else t0
        self.last_row: Optional[Dict[str, Any]] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(TAPE_COLUMNS)

    def start_band(self, band_index: int, band) -> "BandTape":
        """A tape for one band (built with its true wavefields), continuing the
        run's counters."""
        return BandTape(self, band_index, band)

    def append(self, row: Dict[str, Any]) -> None:
        """Append one row (``None`` -> empty cell) and flush it to disk;
        :class:`ValueError` unless its keys are exactly :data:`TAPE_COLUMNS`."""
        _check_keys(row)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                ["" if row[k] is None else row[k] for k in TAPE_COLUMNS]
            )
            f.flush()
        self.last_row = row


class BandTape:
    """Writes one band's rows; :meth:`record` is the optimiser's ``on_log``."""

    def __init__(self, tape: MetricTape, band_index: int, band) -> None:
        self.tape = tape
        self.band_index = int(band_index)
        prev = tape.last_row
        # Counters run on from the previous band's final row.
        self._iter_offset = 0 if prev is None else int(prev["iter_total"])
        self._eval_offset = 0 if prev is None else int(prev["n_evaluations"])
        problem = tape.problem
        self._interior = problem.grid.interior_slice
        self._u_star = band.true_wavefield_interior()
        self._u_star_norm = float(torch.linalg.vector_norm(self._u_star))
        self._c_prev: Optional[np.ndarray] = None
        self.start_row: Optional[Dict[str, Any]] = None
        self.final_row: Optional[Dict[str, Any]] = None

    def record(self, it: Iterate) -> Dict[str, Any]:
        """Write the row for iterate ``it`` and return it; then
        :class:`NonFiniteError` if a :class:`Column` with ``finite`` is not."""
        sl = self._interior
        c = _np(it.c_full[sl])
        u = it.u[:, :, sl[0], sl[1]]
        rel_wavefield_error = float(
            torch.linalg.vector_norm(u - self._u_star.to(u.device))
        ) / max(self._u_star_norm, 1e-30)
        rel_model_update = (
            None
            if self._c_prev is None
            else float(np.linalg.norm(c - self._c_prev) / np.linalg.norm(self._c_prev))
        )
        row: Dict[str, Any] = {
            "run_id": self.tape.run_id,
            "band_index": self.band_index,
            "iter": int(it.iter),
            "iter_total": self._iter_offset + int(it.iter),
            "wall_s": time.perf_counter() - self.tape.t0,
            "n_evaluations": self._eval_offset + int(it.n_evaluations),
            "loss_total": it.loss,
            "pde_term": it.pde_term,
            "data_term": it.data_term,
            "reg_term": it.reg_term,
            "rel_data_residual": it.rel_data_residual,
            "pde_rel_residual": it.pde_rel_residual,
            "rel_wavefield_error": rel_wavefield_error,
            "rel_model_update": rel_model_update,
            **self.tape.model_error(c),
            "c_min": float(c.min()),
            "c_max": float(c.max()),
            "sat_frac": it.sat_frac,
            "grad_u_norm": it.grad_u_norm,
            "grad_z_norm": it.grad_z_norm,
        }
        self._c_prev = c
        if self.start_row is None:
            self.start_row = row
        self.final_row = row
        self.tape.append(row)
        bad = [
            c.name
            for c in COLUMNS
            if c.finite and row[c.name] is not None and not np.isfinite(row[c.name])
        ]
        if bad:
            raise NonFiniteError(
                f"non-finite {', '.join(bad)} at band {self.band_index}, "
                f"iter {row['iter']}"
            )
        return row


def load_tape(run_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a run's ``tape.csv``: one dict per row, each cell read as its
    :class:`Column` type (an empty cell is ``None``); :class:`ValueError` if
    the header names a column not in :data:`COLUMNS`.
    """
    path = Path(run_dir) / "tape.csv"
    if not path.exists():
        raise FileNotFoundError(f"no tape.csv under {str(run_dir)!r} ({path})")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        unknown = [k for k in reader.fieldnames or [] if k not in _TYPES]
        if unknown:
            raise ValueError(f"{path}: unknown tape columns {unknown}")
        return [
            {k: None if v == "" else _TYPES[k](v) for k, v in row.items()}
            for row in reader
        ]


def rows_by_band(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Tape rows grouped by ``band_index``, in band order."""
    by_band: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_band.setdefault(row["band_index"], []).append(row)
    return [by_band[b] for b in sorted(by_band)]


def _check_keys(row: Dict[str, Any]) -> None:
    unknown = [k for k in row if k not in _TYPES]
    missing = [k for k in TAPE_COLUMNS if k not in row]
    if unknown or missing:
        raise ValueError(
            f"tape row has unknown columns {unknown} and lacks columns {missing}"
        )


def _np(x) -> np.ndarray:
    """A detached float64 (or bool) numpy copy of a tensor or array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=bool if x.dtype == bool else np.float64)
