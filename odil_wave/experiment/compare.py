"""Run comparison: does a code change alter a run's results?

Runs one config at a git ref and on this tree (uncommitted changes included),
then compares the two runs' results exactly::

    python -m odil_wave.experiment.compare <ref> [--config FILE]
        [-o KEY=VALUE ...] [--keep DIR]

The config defaults to ``configs/checks/tiny.yaml``. The ref runs in a
detached worktree with its own locked environment
(``uv run --frozen python main.py``, as ``--reproduce`` does), so a
dependency change counts as a change. Both are scratch runs. Results are every
``tape.csv`` column, every array of every ``*.npz`` and every value of each
``band_summary.json``, except the run's identity (``run_id``) and timings
(``wall_s`` and other keys ending ``_s``). Figures, ``metadata.json``, the
resolved config and ``run_summary.json`` (which repeats the band summaries)
are not results.

Exit codes: ``0`` identical results, ``1`` they differ (each difference is
printed), ``2`` a run could not start or failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from .rundir import REPO_ROOT, run_git

DEFAULT_CONFIG = REPO_ROOT / "configs" / "checks" / "tiny.yaml"


def _ignored(key: str) -> bool:
    """A run's identity or a timing: differs between any two runs."""
    return key == "run_id" or key.endswith("_s")


def compare_run_dirs(a: Path, b: Path) -> List[str]:
    """Every difference between the results of run directories ``a`` and ``b``.

    Returns
    -------
    list of str
        One line per difference, ``"<path relative to the run>: <what>"``;
        empty when the results are identical.
    """
    a, b = Path(a), Path(b)
    diffs: List[str] = []
    for rel in sorted(_result_files(a) | _result_files(b)):
        pa, pb = a / rel, b / rel
        if not pb.exists():
            diffs.append(f"{rel}: only in the first run")
        elif not pa.exists():
            diffs.append(f"{rel}: only in the second run")
        elif rel.name == "tape.csv":
            diffs += [f"{rel}: {d}" for d in _compare_tape(pa, pb)]
        elif rel.suffix == ".npz":
            diffs += [f"{rel}: {d}" for d in _compare_npz(pa, pb)]
        else:
            data_a = json.loads(pa.read_text())
            data_b = json.loads(pb.read_text())
            diffs += [f"{rel}: {d}" for d in _compare_json(data_a, data_b, "")]
    return diffs


def _result_files(run_dir: Path) -> set:
    """Result files of ``run_dir``, relative to it."""
    paths = [run_dir / "tape.csv", *run_dir.rglob("*.npz")]
    paths += run_dir.rglob("band_summary.json")
    return {p.relative_to(run_dir) for p in paths if p.is_file()}


def _delta(x: str, y: str) -> str:
    """`` (|Δ| …)`` when both are numbers, else nothing."""
    try:
        d = abs(float(x) - float(y))
    except ValueError:
        return ""
    return f" (|Δ| {d:.3g})" if math.isfinite(d) else ""


def _compare_tape(a: Path, b: Path) -> List[str]:
    """Columns whose values differ, with the first differing row."""
    with open(a, newline="") as f:
        rows_a = list(csv.DictReader(f))
    with open(b, newline="") as f:
        rows_b = list(csv.DictReader(f))
    if len(rows_a) != len(rows_b):
        return [f"{len(rows_a)} rows vs {len(rows_b)}"]
    columns = list(rows_a[0]) if rows_a else []
    if rows_b and list(rows_b[0]) != columns:
        return [f"columns differ: {columns} vs {list(rows_b[0])}"]
    diffs = []
    for col in columns:
        if _ignored(col):
            continue
        bad = [i for i, (x, y) in enumerate(zip(rows_a, rows_b)) if x[col] != y[col]]
        if bad:
            x, y = rows_a[bad[0]][col], rows_b[bad[0]][col]
            diffs.append(
                f"{col} differs in {len(bad)}/{len(rows_a)} rows; first at row "
                f"{bad[0]}: {x} vs {y}{_delta(x, y)}"
            )
    return diffs


def _compare_npz(a: Path, b: Path) -> List[str]:
    """Arrays that are missing from one file or differ in shape, dtype or value."""
    with np.load(a) as za, np.load(b) as zb:
        diffs = [f"{k}: only in the first run" for k in sorted(set(za) - set(zb))]
        diffs += [f"{k}: only in the second run" for k in sorted(set(zb) - set(za))]
        for k in sorted(set(za) & set(zb)):
            x, y = za[k], zb[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                diffs.append(
                    f"{k}: {x.dtype}{list(x.shape)} vs {y.dtype}{list(y.shape)}"
                )
            elif not np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
                if x.dtype.kind in "fc":
                    d = float(np.nanmax(np.abs(x.astype(complex) - y)))
                    diffs.append(f"{k} differs (max |Δ| {d:.3g})")
                else:
                    diffs.append(f"{k} differs")
    return diffs


def _compare_json(x: Any, y: Any, path: str) -> List[str]:
    """Dotted paths whose values differ, skipping identity and timings."""
    if isinstance(x, dict) and isinstance(y, dict):
        diffs = []
        for k in sorted(set(x) | set(y), key=str):
            if _ignored(str(k)):
                continue
            sub = f"{path}.{k}" if path else str(k)
            if k not in y:
                diffs.append(f"{sub}: only in the first run")
            elif k not in x:
                diffs.append(f"{sub}: only in the second run")
            else:
                diffs += _compare_json(x[k], y[k], sub)
        return diffs
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return [
            d
            for i, (u, v) in enumerate(zip(x, y))
            for d in _compare_json(u, v, f"{path}[{i}]")
        ]
    if x != y and not (x != x and y != y):  # NaN equals NaN here
        return [f"{path}: {x!r} vs {y!r}{_delta(str(x), str(y))}"]
    return []


def _run(cwd: Path, config: Path, overrides: Sequence[str], out: Path) -> int:
    """Run ``main.py`` in ``cwd``'s locked environment as a scratch run into
    ``out``; return its exit code (``2`` when ``uv`` can't be run)."""
    command = ["uv", "run", "--frozen", "python", "main.py", "--config", str(config)]
    command += [arg for o in overrides for arg in ("-o", o)]
    command += ["--scratch", "--out", str(out)]
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    except OSError as exc:  # no uv on PATH
        print(f"error: can't run {command[0]!r}: {exc}", file=sys.stderr)
        return 2
    if done.returncode != 0:
        print(done.stdout[-2000:] + done.stderr[-2000:], file=sys.stderr)
    return done.returncode


def compare(
    ref: str,
    *,
    config: Path = DEFAULT_CONFIG,
    overrides: Sequence[str] = (),
    keep: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
) -> int:
    """Run ``config`` at ``ref`` and on this tree, print the differences in
    their results and return the exit code (see the module docstring).

    ``keep`` is a new directory to leave both runs in (``ref/``, ``tree/``);
    by default they are deleted.
    """
    config = Path(config).resolve()
    if not config.is_file():
        print(f"error: no config {config}", file=sys.stderr)
        return 2
    if keep is not None and Path(keep).exists():
        print(f"error: {keep} already exists", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="compare-"))
    runs = Path(keep).resolve() if keep is not None else tmp / "runs"
    worktree = tmp / "worktree"
    try:
        try:
            added = run_git(
                repo_root, "worktree", "add", "--detach", str(worktree), ref
            )
            if added is None:
                print(f"error: can't check out {ref!r}", file=sys.stderr)
                return 2
            for name, cwd in (("ref", worktree), ("tree", repo_root)):
                label = ref if name == "ref" else "this tree"
                print(f"running {config.name} on {label}")
                code = _run(cwd, config, overrides, runs / name)
                if code != 0:
                    print(f"error: the run on {label} exited {code}", file=sys.stderr)
                    return 2
        finally:
            run_git(repo_root, "worktree", "remove", "--force", str(worktree))
        diffs = compare_run_dirs(runs / "ref", runs / "tree")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not diffs:
        print(f"identical results: {ref} vs this tree")
        return 0
    print(f"results differ: {ref} vs this tree")
    for d in diffs:
        print(f"  {d}")
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns the exit code."""
    parser = argparse.ArgumentParser(
        description="Run a config at a git ref and on this tree; compare the "
        "results exactly."
    )
    parser.add_argument("ref", help="Git ref to compare against, e.g. main.")
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG),
        help="Run config (default: configs/checks/tiny.yaml).",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key override passed to both runs (repeatable).",
    )
    parser.add_argument(
        "--keep",
        default=None,
        metavar="DIR",
        help="Leave both runs in DIR/ref and DIR/tree instead of deleting them.",
    )
    args = parser.parse_args(argv)
    keep = Path(args.keep) if args.keep else None
    return compare(
        args.ref, config=Path(args.config), overrides=args.override, keep=keep
    )


if __name__ == "__main__":
    sys.exit(main())
