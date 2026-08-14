#!/usr/bin/env python
"""CLI entry point for reproducible frequency-domain FWI runs.

Examples
--------
    python main.py --config configs/default_inverse.yaml
    python main.py --config configs/default_inverse.yaml \
        --override optimiser.name=lbfgsb --override optimiser.n_iter=40
    python main.py --config configs/default_inverse.yaml --dry-run

Run with no ``--config`` to use the built-in defaults.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional


def _summarise(cfg) -> str:
    lines: List[str] = []
    a = lines.append
    a(f"run.run_id      : {cfg.run.run_id}")
    a(f"output dir      : {cfg.run_dir()}")
    a(
        f"optimiser       : {cfg.optimiser.name}  "
        f"(n_iter={cfg.optimiser.n_iter}, u_steps={cfg.optimiser.u_steps}, "
        f"c_steps={cfg.optimiser.c_steps}, log_every={cfg.optimiser.log_every})"
    )
    a(
        f"runtime         : device={cfg.runtime.device}, dtype={cfg.runtime.dtype}, "
        f"seed={cfg.run.seed}"
    )
    a(
        f"grid            : interior={cfg.grid.interior_shape}, pml={cfg.grid.pml_width}, "
        f"c=[{cfg.grid.c_min}, {cfg.grid.c_max}], init_nt={cfg.grid.init_nt}"
    )
    a(f"source          : {cfg.source.kind}  f0={cfg.source.f0} Hz")
    a(
        f"acquisition     : n_recv={cfg.acquisition.n_receivers}, "
        f"n_src={cfg.acquisition.n_sources}, ring={cfg.acquisition.ring_center}"
    )
    a(f"truth / init    : {cfg.truth.profile} / {cfg.init.profile}")
    a(
        f"observation     : method={cfg.observation.method}, "
        f"normalize={cfg.observation.normalize_data}"
    )
    a(f"loss weights    : {cfg.loss.weights}")
    a(f"warm_start      : {cfg.continuation.warm_start}")
    a(
        f"ssim metric     : mask={cfg.metrics.ssim.mask}, win={cfg.metrics.ssim.win_size}, "
        f"data_range={cfg.metrics.ssim.data_range}"
    )
    if cfg.optimiser.name == "lbfgsb":
        lb = cfg.optimiser.lbfgsb
        a(
            f"lbfgsb          : u_precond={lb.u_precond}, u_solve={lb.u_solve}, "
            f"z_steps={lb.z_steps}, z_optim={lb.z_optim}, z_lr={lb.z_lr}, "
            f"c_lr={lb.c_lr}, c_grad_smooth_sigma={lb.c_grad_smooth_sigma}"
        )
    if cfg.diagnostics.enabled:
        d = cfg.diagnostics
        a(
            f"diagnostics     : enabled (per_outer_field_maps="
            f"{d.per_outer_field_maps}, u_depths={d.u_depths}, "
            f"verify_hessian={d.verify_hessian})"
        )
    a(f"bands ({len(cfg.continuation.bands)}):")
    for i, b in enumerate(cfg.continuation.bands):
        khz = [f / 1e3 for f in b.frequencies_hz]
        a(f"  band {i:02d}: {khz} kHz  (n_iter={b.n_iter or cfg.optimiser.n_iter})")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible frequency-domain FWI runner."
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="YAML/JSON config file. Omit to use built-in defaults.",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key override, e.g. optimiser.n_iter=40 (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve+validate the config, print a summary, write nothing.",
    )
    args = parser.parse_args(argv)

    # config resolution is torch-free; import lazily so --help stays instant.
    from odil_wave.experiment.config import (
        ConfigError,
        load_config_file,
        resolve_config,
        validate_config,
    )

    user_dict: Optional[Dict[str, Any]] = None
    if args.config is not None:
        user_dict = load_config_file(args.config)

    try:
        cfg = resolve_config(user_dict or {}, overrides=args.override)
        warnings = validate_config(cfg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(_summarise(cfg))
    for w in warnings:
        print(f"[warning] {w}")

    if args.dry_run:
        print("\n[dry-run] configuration is valid; no artifacts written.")
        return 0

    from odil_wave.experiment.runner import run_inverse

    result = run_inverse(cfg, input_config=user_dict)
    print(f"\nartifacts: {result.run_dir}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
