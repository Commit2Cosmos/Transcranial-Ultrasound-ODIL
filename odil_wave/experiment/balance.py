"""Balance check: the joint objective's PDE/data balance across grids.

For each :class:`GridCase` the baseline's interior shape is multiplied by
``scale`` and its PML width by ``pml_scale`` (default ``scale``: the Same
physical problem, with domain, PML thickness in metres and source width
fixed). On the first band it evaluates the solver's own objective twice:

* the PDE term at ``(u = H⁻¹(c_true), c_init)``, where the data term vanishes;
* the (unweighted) data term at the warm start ``(u = H⁻¹(c_init), c_init)``,
  where the PDE term vanishes.

When the balance is grid independent (ADR-0002) their ratio agrees across
cases up to discretisation error. Deterministic, no inversion; it writes no
files::

    python -m odil_wave.experiment.balance [--config FILE] [-o KEY=VALUE ...]
        [--cases 1 2 3 1:2]

A case ``k`` scales the interior and PML by ``k``; ``k:p`` scales the
interior by ``k`` and the PML by ``p``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from .config import ConfigError, RunConfig, load_config_file, resolve_config
from .problem import build_problem
from .runner import build_band_loss, build_optimiser


@dataclass(frozen=True)
class GridCase:
    """Multipliers of the baseline's interior shape and PML width."""

    scale: int
    pml_scale: Optional[int] = None

    @property
    def pml_factor(self) -> int:
        return self.scale if self.pml_scale is None else self.pml_scale

    @classmethod
    def parse(cls, text: str) -> "GridCase":
        """``"k"`` or ``"k:p"``."""
        k, _, p = text.partition(":")
        try:
            case = cls(int(k), int(p) if p else None)
        except ValueError:
            raise ConfigError(f"grid case {text!r} must be 'k' or 'k:p' integers")
        if case.scale < 1 or case.pml_factor < 1:
            raise ConfigError(f"grid case {text!r} must use factors >= 1")
        return case

    def __str__(self) -> str:
        return (
            str(self.scale)
            if self.pml_scale is None
            else f"{self.scale}:{self.pml_scale}"
        )


@dataclass(frozen=True)
class BalanceRow:
    """The balance measured on one grid."""

    case: GridCase
    interior_shape: tuple
    pml_width: int
    h: float
    interior_fraction: float  # N_int / N
    sigma_over_h: Optional[float]  # None for a point source
    pde_term: float
    data_term: float

    @property
    def ratio(self) -> float:
        return self.pde_term / self.data_term


def scaled_config(cfg: RunConfig, case: GridCase) -> RunConfig:
    """``cfg`` with its interior shape and PML width scaled by ``case``."""
    data = cfg.to_dict()
    data["grid"]["interior_shape"] = [
        case.scale * int(n) for n in cfg.grid.interior_shape
    ]
    data["pml"]["width"] = case.pml_factor * int(cfg.pml.width)
    return RunConfig.from_dict(data)


def balance_check(cfg: RunConfig, cases: Sequence[GridCase]) -> List[BalanceRow]:
    """Measure the PDE/data balance of ``cfg`` on each of ``cases``.

    Raises :class:`ConfigError` when ``pml.observation_width`` differs from
    ``pml.width``: that modelling error would mix into the data term.
    """
    obs = cfg.pml.observation_width
    if obs is not None and int(obs) != int(cfg.pml.width):
        raise ConfigError(
            "balance check needs observations on the inversion grid; got "
            f"pml.observation_width={obs} != pml.width={cfg.pml.width}"
        )
    return [_measure(scaled_config(cfg, case), case) for case in cases]


def _measure(cfg: RunConfig, case: GridCase) -> BalanceRow:
    problem = build_problem(cfg)
    band_ctx = problem.make_band(cfg.continuation.bands[0].frequencies_hz)
    init = problem.init_velocity
    u_init = problem.warm_start(init, band_ctx)
    wf_inv, loss = build_band_loss(problem, init, band_ctx)
    opt = build_optimiser(problem, band_ctx, wf_inv, loss, 1, u_init)

    u_re, u_im, z_m = opt.init_params()
    with torch.no_grad():
        opt.objective(u_re, u_im, z_m)
        data_term = opt.last_record["data_loss"]
        # Inverse crime (no observation PML): the observed fields are
        # H⁻¹(c_true) on the inversion grid.
        u_true = torch.stack([w.amplitude for w in band_ctx.observed_wfs])
        u_true = u_true.to(dtype=opt.cdtype, device=opt.device) / opt.u_ref
        opt.objective(u_true.real.contiguous(), u_true.imag.contiguous(), z_m)
        pde_term = opt.last_record["pde_term"]

    grid = problem.grid
    h = float(grid.dx)
    geom = band_ctx.geom
    gaussian = cfg.acquisition.source_spatial == "gaussian"
    return BalanceRow(
        case=case,
        interior_shape=tuple(int(n) for n in cfg.grid.interior_shape),
        pml_width=int(cfg.pml.width),
        h=h,
        interior_fraction=(grid.interior_nx * grid.interior_ny) / (grid.nx * grid.ny),
        sigma_over_h=float(geom.sigma_s) / h if gaussian else None,
        pde_term=float(pde_term),
        data_term=float(data_term),
    )


def format_table(rows: Sequence[BalanceRow]) -> str:
    """Markdown table of ``rows``; the last column is relative to the first row."""
    head = (
        "| case | interior | PML | h (mm) | N_int/N | σ/h | PDE term | data term "
        "| ratio | ratio / first |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    first = rows[0].ratio if rows else 1.0
    for r in rows:
        sigma = "point" if r.sigma_over_h is None else f"{r.sigma_over_h:.2f}"
        lines.append(
            f"| {r.case} | {r.interior_shape[0]}×{r.interior_shape[1]} "
            f"| {r.pml_width} | {r.h * 1e3:.3f} | {r.interior_fraction:.3f} "
            f"| {sigma} | {r.pde_term:.4e} | {r.data_term:.4e} "
            f"| {r.ratio:.4e} | {r.ratio / first:.3f} |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the balance check from the command line; return the exit code."""
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
        "--cases",
        nargs="+",
        default=["1", "2", "3"],
        help="Grid cases: 'k' scales interior and PML by k; 'k:p' scales the "
        "interior by k and the PML by p (default: 1 2 3).",
    )
    args = parser.parse_args(argv)
    try:
        user = load_config_file(args.config) if args.config else {}
        cfg = resolve_config(user or {}, overrides=args.override)
        rows = balance_check(cfg, [GridCase.parse(c) for c in args.cases])
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
