#!/usr/bin/env python
"""CLI entry point for reproducible frequency-domain FWI runs.

Examples
--------
    python main.py                                  # the default config
    python main.py --print-defaults                 # print the default config
    python main.py --config configs/<run-config>.yaml
    python main.py --config configs/<run-config>.yaml \
        --override optimiser.n_iter=40
    python main.py --config configs/<run-config>.yaml --dry-run
    python main.py --scratch --config configs/<run-config>.yaml  # dirty tree
    python main.py --live --config configs/<run-config>.yaml     # watch it
    python main.py --postprocess outputs/runs/<run>  # redraw its figures
    python main.py --experiment configs/experiments/NNNN-slug.yaml [--dry-run]

A run is written to ``outputs/runs/<YYMMDD-HHMMSS>_<sha7>[_scratch]/`` and
needs a clean git tree unless ``--scratch`` is given. An experiment spec runs
from a clean tree into ``outputs/experiments/NNNN-slug[_cuda|_mps]/``.

The default config is defined in code (the ``RunConfig`` dataclass defaults in
``odil_wave/experiment/config.py``); a run config names only the values
it changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _summarise(cfg) -> str:
    lines: List[str] = []
    lines.append(
        "optimiser       : joint  "
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
        f"acquisition     : n_transducers={cfg.acquisition.n_transducers}, "
        f"ring={cfg.acquisition.ring_center}"
    )
    lines.append(f"truth / init    : {cfg.truth.profile} / {cfg.init.profile}")
    obs_pml = cfg.pml.observation_width
    lines.append(
        f"observation pml : {cfg.pml.width if obs_pml is None else obs_pml}"
        f"{' (inverse crime)' if obs_pml in (None, cfg.pml.width) else ''}"
    )
    lines.append(f"loss            : weights={cfg.loss.weights}")
    lines.append(f"ssim_brain      : sigma={cfg.metrics.ssim.sigma}")
    lines.append(f"bands ({len(cfg.continuation.bands)}):")
    for i, b in enumerate(cfg.continuation.bands):
        khz = [f / 1e3 for f in b.frequencies_hz]
        lines.append(
            f"  band {i:02d}: {khz} kHz  (n_iter={b.n_iter or cfg.optimiser.n_iter})"
        )
    return "\n".join(lines)


def _postprocess(run_dir: Path) -> int:
    """``--postprocess``: draw ``run_dir``'s figures; exit 2 if it is no run."""
    from odil_wave.experiment.postprocess import postprocess_run

    try:
        written = postprocess_run(run_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"figures: {run_dir / 'figures'} ({len(written)} files)")
    return 0


def main(argv: Optional[List[str]] = None, *, repo_root: Optional[Path] = None) -> int:
    """Run the CLI; return the exit code.

    Exit codes: ``0`` the run completed (or a dry run / ``--print-defaults``),
    ``1`` it failed or was interrupted, ``2`` it could not start (a config
    error, a dirty tree without ``--scratch``, an existing ``--out``).
    ``repo_root`` is the code's git repository, checked for a clean tree;
    it defaults to the one holding this file.
    """
    parser = argparse.ArgumentParser(
        description="Reproducible frequency-domain FWI runner."
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="YAML/JSON run config (only the values that differ from the "
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
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Allow uncommitted changes: a scratch run, marked as such and "
        "saving its git diff. No experiment record may cite it.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Redraw a live view of the run in one window (needs a display).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        metavar="SPEC",
        help="Run an experiment spec (configs/experiments/NNNN-slug.yaml) into "
        "outputs/experiments/; re-running resumes it. Needs a clean tree.",
    )
    # The run (or experiment) directory itself, for --reproduce and tests;
    # normally the entry point picks outputs/runs/<YYMMDD-HHMMSS>_<sha7>[_scratch]/.
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--postprocess",
        metavar="RUN_DIR",
        default=None,
        help="Redraw a saved run's figures and CSVs into <RUN_DIR>/figures/ "
        "and exit. Reads only the run directory; never re-runs a solve.",
    )
    args = parser.parse_args(argv)

    if args.postprocess is not None:
        return _postprocess(Path(args.postprocess))
    if args.experiment is not None:
        return _experiment(args, repo_root)

    # config resolution is torch-free; import lazily so --help stays instant.
    import yaml

    from odil_wave.experiment.config import (
        ConfigError,
        default_config_dict,
        load_config_file,
        resolve_config,
        validate_config,
    )

    if args.print_defaults:
        print(yaml.safe_dump(default_config_dict(), sort_keys=False), end="")
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

    from odil_wave.experiment.rundir import RunStartError, new_run_directory
    from odil_wave.experiment.runner import run_inverse

    try:
        run_directory = new_run_directory(
            scratch=args.scratch, out=args.out, repo_root=repo_root
        )
    except RunStartError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    on_log = None
    if args.live:
        from odil_wave.experiment.live import LiveVelocityView

        on_log = LiveVelocityView()

    try:
        result = run_inverse(cfg, run_directory, on_log=on_log)
    except ConfigError as exc:  # found at startup, once the problem is built
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(f"\nartifacts: {result.run_dir}")
    return 0 if result.status == "completed" else 1


def _experiment(args: argparse.Namespace, repo_root: Optional[Path]) -> int:
    """``--experiment``: the committed spec alone defines the runs."""
    given = [
        flag
        for flag, used in (
            ("--config", args.config is not None),
            ("--override", bool(args.override)),
            ("--scratch", args.scratch),
        )
        if used
    ]
    if given:
        print(
            f"error: {', '.join(given)} can't be used with --experiment: "
            "the committed spec alone defines the runs",
            file=sys.stderr,
        )
        return 2
    if args.live:
        print(
            "error: --live watches a single run; it can't be used with --experiment",
            file=sys.stderr,
        )
        return 2

    from odil_wave.experiment.launch import launch

    return launch(
        args.experiment, dry_run=args.dry_run, out=args.out, repo_root=repo_root
    )


if __name__ == "__main__":
    raise SystemExit(main())
