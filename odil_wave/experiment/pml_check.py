"""PML check: the absorbing layer's reflection against its width.

For each PML width, frequency and source position it solves the Helmholtz
equation (the inversion's operator, float64) in homogeneous water on the
config's grid, and compares the wavefield with a reference whose layer is very
thick (same scheme): relative L2 error over the interior and at the
transducers. Sources are Gaussians ``acquisition.sigma_s`` wide, 30 mm inside
the interior's left edge at mid-height (``edge``) or 30 mm from its lower-left
corner (``corner``). Deterministic, no inversion; it writes no files::

    python -m odil_wave.experiment.pml_check [--config FILE] [-o KEY=VALUE ...]
        [--widths 5 8 10 12 15 20 40] [--frequencies 20e3 120e3]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import splu

from odil_wave.grid import FrequencySelection, Grid
from odil_wave.models import VelocityModel
from odil_wave.models.shepp_logan import SOS_WATER
from odil_wave.operator import HelmholtzSolver
from odil_wave.wavefield import Wavefield

from .config import ConfigError, RunConfig, load_config_file, resolve_config
from .problem import build_problem

REFERENCE_WIDTH = 150
REFERENCE_R0 = 1e-12
SOURCE_INSET = 30e-3  # m from the interior's edge
CASES = ("edge", "corner")


@dataclass(frozen=True)
class PmlRow:
    """One width, frequency and source position against the reference."""

    width: int
    interior_fraction: float  # N_int / N
    frequency_hz: float
    case: str
    interior_error: float
    receiver_error: float


def _check_config(cfg: RunConfig, width: int, R0: float) -> RunConfig:
    data = cfg.to_dict()
    data["pml"]["width"] = int(width)
    data["pml"]["R0"] = float(R0)
    data["pml"]["observation_width"] = None
    data["runtime"]["dtype"] = "float64"
    return RunConfig.from_dict(data)


def _source_position(grid: Grid, case: str) -> Tuple[float, float]:
    (x0, _), (y0, _) = grid.interior_extent
    yc = grid.interior_center[1]
    return (x0 + SOURCE_INSET, yc if case == "edge" else y0 + SOURCE_INSET)


def _solve(
    cfg: RunConfig, width: int, R0: float, frequencies_hz: Sequence[float]
) -> Tuple[Grid, Dict[str, np.ndarray], np.ndarray]:
    """``(grid, {case: interior u (nf, nx_int, ny_int)}, transducer interior ij)``."""
    problem = build_problem(_check_config(cfg, width, R0))
    grid = problem.grid
    freq = FrequencySelection.from_frequencies(grid, list(frequencies_hz))
    geom = problem.acquisition_geometry(grid, problem.source, freq)
    model = VelocityModel(grid, profile="homogeneous", base=SOS_WATER, contrast=0.0)
    solver = HelmholtzSolver(
        Wavefield(grid=grid, frequency_selection=freq, velocity_model=model),
        geom,
        space_order=problem.space_order,
        damping_scale=problem.damping_scale,
    )
    sigma = float(cfg.acquisition.sigma_s)
    rhs = []
    for case in CASES:
        xs, ys = _source_position(grid, case)
        r2 = (grid.X - xs) ** 2 + (grid.Y - ys) ** 2
        rhs.append(torch.exp(-r2 / sigma**2).reshape(-1).numpy())
    F = np.stack(rhs, axis=1).astype(np.complex128)
    sx, sy = grid.interior_slice
    out = {case: [] for case in CASES}
    for k in range(freq.n_frequencies):
        U = splu(solver.assemble_H_sparse(model.c, k).tocsc()).solve(F)
        for n, case in enumerate(CASES):
            out[case].append(U[:, n].reshape(grid.shape)[sx, sy])
    p = grid.pml_width
    ij = geom.transducer_ij.numpy() - p
    return grid, {case: np.stack(u) for case, u in out.items()}, ij


def _rel(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def pml_check(
    cfg: RunConfig, widths: Sequence[int], frequencies_hz: Sequence[float]
) -> List[PmlRow]:
    """Measure the PML's reflection on ``cfg``'s grid at each of ``widths``."""
    if any(int(w) < 1 for w in widths):
        raise ConfigError(f"PML widths must be >= 1, got {list(widths)}")
    _, ref, ij = _solve(cfg, REFERENCE_WIDTH, REFERENCE_R0, frequencies_hz)
    rows = []
    for width in widths:
        grid, u, _ = _solve(cfg, int(width), float(cfg.pml.R0), frequencies_hz)
        n_int = grid.interior_nx * grid.interior_ny
        for case in CASES:
            for k, f in enumerate(frequencies_hz):
                a, b = u[case][k], ref[case][k]
                rows.append(
                    PmlRow(
                        width=int(width),
                        interior_fraction=n_int / (grid.nx * grid.ny),
                        frequency_hz=float(f),
                        case=case,
                        interior_error=_rel(a, b),
                        receiver_error=_rel(
                            a[ij[:, 0], ij[:, 1]], b[ij[:, 0], ij[:, 1]]
                        ),
                    )
                )
    return rows


def format_table(rows: Sequence[PmlRow]) -> str:
    """The rows as a Markdown table."""
    lines = [
        "| width | N_int/N | f (kHz) | source | interior error | receiver error |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.width} | {r.interior_fraction:.3f} | {r.frequency_hz / 1e3:g} "
            f"| {r.case} | {r.interior_error:.2e} | {r.receiver_error:.2e} |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the PML check from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", "-c", default=None, help="YAML/JSON run config (differences only)."
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key override (repeatable).",
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=[5, 8, 10, 12, 15, 20, 40],
        help="PML widths in cells (default: 5 8 10 12 15 20 40).",
    )
    parser.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=[20e3, 120e3],
        help="Frequencies in Hz (default: 20e3 120e3).",
    )
    args = parser.parse_args(argv)
    try:
        user = load_config_file(args.config) if args.config else {}
        cfg = resolve_config(user or {}, overrides=args.override)
        rows = pml_check(cfg, args.widths, args.frequencies)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
