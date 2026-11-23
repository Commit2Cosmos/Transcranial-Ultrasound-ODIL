"""Notebook loaders of a saved run: its resolved config, the rebuilt problem,
its velocities and its loss history as a :class:`~odil_wave.loss.LossTape`.
Figures live in :mod:`.postprocess` (the fixed set) and :mod:`.on_demand`."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from odil_wave.grid import Grid
from odil_wave.loss import LossTape
from odil_wave.models import VelocityModel

from .config import RunConfig
from .problem import Problem, build_problem
from .saved import load_bands, load_run_config
from .tape import load_tape


RunDirOrRecords = Union[str, Path, List[dict]]


# --------------------------------------------------------------------------- #
# Rebuilding the physical problem from a run's saved config
# --------------------------------------------------------------------------- #
def load_config(run_dir: Union[str, Path]) -> RunConfig:
    """Reconstruct the resolved :class:`RunConfig` saved alongside a run."""
    return RunConfig.from_dict(load_run_config(run_dir))


def rebuild_problem(run_dir: Union[str, Path]) -> Problem:
    """Rebuild the physical :class:`Problem` from a run's saved config."""
    return build_problem(load_config(run_dir))


# --------------------------------------------------------------------------- #
# Loading saved artifacts
# --------------------------------------------------------------------------- #
def _records(source: RunDirOrRecords) -> List[dict]:
    """Accept either a run directory or an already-loaded list of records."""
    if isinstance(source, (str, Path)):
        return load_tape(source)
    return list(source)


def load_final_velocity(
    run_dir: Union[str, Path],
    grid: Optional[Grid] = None,
) -> VelocityModel:
    """Load the last completed band's final model (the run's final model) into
    a :class:`VelocityModel`.

    ``grid`` defaults to the one rebuilt from the run's saved config, so only
    the run directory is required; pass it to reuse a ``Problem`` you already
    hold.
    """
    bands = load_band_velocities(run_dir, grid)
    if not bands:
        raise FileNotFoundError(f"no completed band under {run_dir!r}")
    return bands[-1][2]


def load_band_velocities(
    run_dir: Union[str, Path],
    grid: Optional[Grid] = None,
) -> List[Tuple[int, str, VelocityModel]]:
    """Load every completed band's final model, in band order.

    Each band's ``arrays.npz`` holds the interior ``c``; the full-grid model is
    it with its edge replicated into the PML. Returns ``[(band_index, label,
    VelocityModel), ...]`` sorted by band index. ``grid`` defaults to the one
    rebuilt from the run's saved config, so only the run directory is required.
    """
    if grid is None:
        grid = rebuild_problem(run_dir).grid
    return [
        (
            band.index,
            band.label,
            VelocityModel.from_interior(grid, torch.from_numpy(band.c)),
        )
        for band in load_bands(run_dir)
    ]


def load_loss_tape(source: RunDirOrRecords, name: str = "loaded run") -> LossTape:
    """Reconstruct a :class:`LossTape` (scalar history) from a saved run.

    The returned tape supports ``tape.show()`` and
    ``tape.show_velocity_recovery(...)`` exactly as during a live run.
    """
    return LossTape.from_records(_records(source), name=name)
