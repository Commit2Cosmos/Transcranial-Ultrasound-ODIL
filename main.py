#!/usr/bin/env python
"""CLI entry point for reproducible frequency-domain FWI runs.

Examples
--------
    python main.py                                  # the default config
    python main.py --print-defaults                 # print the default config
    python main.py --config configs/<experiment>.yaml
    python main.py --config configs/<experiment>.yaml \
        --override optimiser.n_iter=40
    python main.py --config configs/<experiment>.yaml --dry-run

The default config is defined in code (the ``RunConfig`` dataclass defaults in
``odil_wave/experiment/config.py``); an experiment config names only the values
it changes.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import yaml


def _summarise(cfg) -> str:
    lines: List[str] = []
    lines.append(f"run.run_id      : {cfg.run.run_id}")
    lines.append(f"output dir      : {cfg.run_dir()}")
    lines.append(
        f"optimiser       : joint  "
        f"(n_iter={cfg.optimiser.n_iter}, log_every={cfg.optimiser.log_every})"
    )
    lines.append(
        f"runtime         : device={cfg.runtime.device}, dtype={cfg.runtime.dtype}, "
        f"seed={cfg.run.seed}"
    )
    lines.append(
        f"grid            : interior={cfg.grid.interior_shape}, pml={cfg.pml.width}, "
        f"c=[{cfg.grid.c_min}, {cfg.grid.c_max}], init_nt={cfg.time.init_nt}"
    )
    lines.append(f"source          : {cfg.source.kind}  f0={cfg.source.f0} Hz")
    lines.append(
        f"acquisition     : n_recv={cfg.acquisition.n_receivers}, "
        f"n_src={cfg.acquisition.n_sources}, ring={cfg.acquisition.ring_center}"
    )
    lines.append(f"truth / init    : {cfg.truth.profile} / {cfg.init.profile}")
    obs_pml = cfg.pml.observation_width
    lines.append(
        f"observation pml : {cfg.pml.width if obs_pml is None else obs_pml}"
        f"{' (inverse crime)' if obs_pml in (None, cfg.pml.width) else ''}"
    )
    lines.append(
        f"loss            : weights={cfg.loss.weights}, "
        f"normalize={cfg.loss.normalize_data}"
    )
    lines.append(f"warm_start      : {cfg.continuation.warm_start}")
    lines.append(
        f"ssim metric     : mask={cfg.metrics.ssim.mask}, win={cfg.metrics.ssim.win_size}, "
        f"data_range={cfg.metrics.ssim.data_range}"
    )
    lines.append(f"bands ({len(cfg.continuation.bands)}):")
    for i, b in enumerate(cfg.continuation.bands):
        khz = [f / 1e3 for f in b.frequencies_hz]
        lines.append(
            f"  band {i:02d}: {khz} kHz  (n_iter={b.n_iter or cfg.optimiser.n_iter})"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible frequency-domain FWI runner."
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="YAML/JSON experiment config (only the values that differ from the "
        "default config). Omit to run the default config.",
    )
    parser.add_argument(
        "--print-defaults",
        action="store_true",
        help="Print the full default config as YAML and exit.",
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
        RunConfig,
        load_config_file,
        resolve_config,
        validate_config,
    )

    if args.print_defaults:
        print(yaml.safe_dump(RunConfig().to_dict(), sort_keys=False), end="")
        return 0

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
