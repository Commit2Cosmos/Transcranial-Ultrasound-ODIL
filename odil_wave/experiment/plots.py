from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from odil_wave.grid import Grid
from odil_wave.loss import LossTape
from odil_wave.models import VelocityModel

from .config import RunConfig, load_config_file
from .postprocess import load_tape as load_metrics
from .problem import Problem, build_problem


RunDirOrRecords = Union[str, Path, List[dict]]


# --------------------------------------------------------------------------- #
# Rebuilding the physical problem from a run's saved config
# --------------------------------------------------------------------------- #
def load_config(run_dir: Union[str, Path]) -> RunConfig:
    """Reconstruct the resolved :class:`RunConfig` saved alongside a run."""
    path = Path(run_dir) / "config_resolved.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no config_resolved.yaml under {run_dir!r} ({path}); "
            "cannot rebuild the problem from this run directory"
        )
    return RunConfig.from_dict(load_config_file(path))


def rebuild_problem(run_dir: Union[str, Path]) -> Problem:
    """Rebuild the physical :class:`Problem` from a run's saved config."""
    return build_problem(load_config(run_dir))


def _resolve_grid_pml(
    run_dir: Union[str, Path], grid: Optional[Grid], pml_c: Optional[float]
) -> Tuple[Grid, Optional[float]]:
    """Fill a missing ``grid`` from the run's rebuilt problem and a missing
    ``pml_c`` with the run's frozen PML speed (in ``problem.npz``).

    Rebuilds the problem only when the grid is missing.
    """
    if pml_c is None:
        with np.load(Path(run_dir) / "problem.npz") as saved:
            pml_c = float(saved["pml_c"])
    if grid is None:
        grid = rebuild_problem(run_dir).grid
    return grid, pml_c


# --------------------------------------------------------------------------- #
# Loading saved artifacts
# --------------------------------------------------------------------------- #
def _records(source: RunDirOrRecords) -> List[dict]:
    """Accept either a run directory or an already-loaded list of records."""
    if isinstance(source, (str, Path)):
        return load_metrics(source)
    return list(source)


def load_final_velocity(
    run_dir: Union[str, Path],
    grid: Optional[Grid] = None,
    pml_c: Optional[float] = None,
) -> VelocityModel:
    """Load the last completed band's final model (the run's final model) into
    a :class:`VelocityModel`.

    ``grid`` and ``pml_c`` default to the ones rebuilt from the run's saved
    config, so only the run directory is required; pass them to reuse a
    ``Problem`` you already hold.
    """
    bands = load_band_velocities(run_dir, grid, pml_c)
    if not bands:
        raise FileNotFoundError(f"no completed band under {run_dir!r}")
    return bands[-1][2]


def load_band_velocities(
    run_dir: Union[str, Path],
    grid: Optional[Grid] = None,
    pml_c: Optional[float] = None,
) -> List[Tuple[int, str, VelocityModel]]:
    """Load every completed band's final model, in band order.

    Each band's ``arrays.npz`` holds the interior ``c``; the full-grid model is
    it padded with ``pml_c``. Returns ``[(band_index, label, VelocityModel),
    ...]`` sorted by band index. ``grid`` defaults to the one rebuilt from the
    run's saved config and ``pml_c`` to the run's frozen PML speed (in
    ``problem.npz``), so only the run directory is required.
    """
    grid, pml_c = _resolve_grid_pml(run_dir, grid, pml_c)
    bands_dir = Path(run_dir) / "bands"
    if not bands_dir.exists():
        raise FileNotFoundError(f"no bands/ directory under {run_dir!r}")
    out: List[Tuple[int, str, VelocityModel]] = []
    for summary_path in sorted(bands_dir.glob("band_*/band_summary.json")):
        summary = json.loads(summary_path.read_text())
        with np.load(summary_path.with_name("arrays.npz")) as saved:
            c = torch.from_numpy(saved["c"]).to(grid.dtype)
        p = grid.pml_width
        c_full = torch.nn.functional.pad(c, (p, p, p, p), value=float(pml_c))
        out.append(
            (
                int(summary["band_index"]),
                str(summary["label"]),
                VelocityModel.from_field(grid, c_full, pml_c=pml_c),
            )
        )
    out.sort(key=lambda t: t[0])
    return out


def load_loss_tape(source: RunDirOrRecords, name: str = "loaded run") -> LossTape:
    """Reconstruct a :class:`LossTape` (scalar history) from a saved run.

    The returned tape supports ``tape.show()`` and
    ``tape.show_velocity_recovery(...)`` exactly as during a live run.
    """
    return LossTape.from_records(_records(source), name=name)
