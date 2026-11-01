"""Probe runs: estimate how long an experiment will take before launching it.

``main.py --experiment <spec> --probe`` calls :func:`probe`. A **Probe run** is
:data:`PROBE_ITERATIONS` iterations of the baseline's first band, run into
``outputs/probes/<YYMMDD-HHMMSS>/``; its seconds per iteration times each run's
total iterations is that run's estimate. It is allowed on a dirty tree and
reserves no experiment id. The run-layer README's ``--experiment`` section is
the contract.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Union

from .config import ConfigError, RunConfig, resolve_config
from .rundir import REPO_ROOT, RunStartError
from .spec import load_spec

PROBE_ITERATIONS = 3


def probe(
    spec_path: Union[str, Path],
    *,
    repo_root: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> int:
    """Run a Probe run of the spec at ``spec_path`` and print the estimates.

    ``repo_root`` is the code's repository, holding ``outputs/probes/``;
    ``device``, if given, replaces the spec's ``runtime.device``.
    Returns the exit code: ``0`` the probe completed, ``1`` it failed or was
    interrupted, ``2`` it could not start (a spec or config error).
    """
    from .rundir import new_run_directory
    from .runner import run_inverse

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        # No repo_root: a probe reserves no id, so the id number isn't checked.
        spec = load_spec(spec_path, device=device)
        config = probe_config(spec.baseline)
    except ConfigError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2
    out = root / "outputs" / "probes" / time.strftime("%y%m%d-%H%M%S")
    try:
        # A scratch run directory: the tree may be dirty, and its git.diff
        # stays with the probe.
        run_directory = new_run_directory(scratch=True, out=out, repo_root=root)
        result = run_inverse(config, run_directory)
    except (ConfigError, RunStartError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result.status != "completed":
        print(f"\nprobe {result.status}: {out}; no estimate", file=sys.stderr)
        return 1

    band = result.band_results[0]
    start, final = band.start_metrics, band.final_metrics
    s_per_iter = (final["wall_s"] - start["wall_s"]) / (
        final["iter_total"] - start["iter_total"]
    )
    print(f"\nprobe: {out}")
    print(
        f"{s_per_iter:.4g} s/iter over {PROBE_ITERATIONS} iterations of the "
        f"baseline's first band ({band.label}); excludes setup"
    )
    lines = [(run.name, total_iterations(run.config)) for run in spec.runs]
    lines.append(("experiment", sum(n for _, n in lines)))
    width = max(len(name) for name, _ in lines)
    for name, n in lines:
        print(f"  {name:<{width}} : {_estimate(n, s_per_iter)}")
    return 0


def probe_config(baseline: RunConfig) -> RunConfig:
    """The baseline cut to :data:`PROBE_ITERATIONS` iterations of its first band."""
    data = baseline.to_dict()
    first = dict(data["continuation"]["bands"][0], n_iter=PROBE_ITERATIONS)
    data["continuation"]["bands"] = [first]
    return resolve_config(data)


def total_iterations(config: RunConfig) -> int:
    """A run's iterations over all its bands (a band's ``n_iter`` defaults to
    ``optimiser.n_iter``)."""
    return sum(config.band_n_iter(b) for b in config.continuation.bands)


def _estimate(n_iter: int, s_per_iter: float) -> str:
    """``"<n> iterations ~ <seconds> s (h:mm:ss)"``."""
    seconds = n_iter * s_per_iter
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{n_iter} iterations ~ {seconds:.1f} s ({hours}:{minutes:02d}:{secs:02d})"
