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

Anything only the tree run has (a result file, ``tape.csv`` column, ``.npz``
array or JSON key) is printed as new and is not a difference; anything only
the ref run has, and any changed value, is.

Exit codes: ``0`` identical results, new results aside (each new one is
printed), ``1`` they differ (each difference is printed), ``2`` a run could
not start or failed.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .rundir import REPO_ROOT, run_git

DEFAULT_CONFIG = REPO_ROOT / "configs" / "checks" / "tiny.yaml"


def _ignored(key: str) -> bool:
    """A run's identity or a timing: differs between any two runs."""
    return key == "run_id" or key.endswith("_s")


@dataclass
class Comparison:
    """The results of a ref run against a tree run: ``differences`` (changed
    values and anything only the ref run has) and ``new`` (anything only the
    tree run has), one ``"<path relative to the run>: <what>"`` line each."""

    differences: List[str]
    new: List[str]


def compare_run_dirs(ref: Path, tree: Path) -> Comparison:
    """Compare the results of run directories ``ref`` and ``tree``; both lists
    are empty when the results are identical."""
    ref, tree = Path(ref), Path(tree)
    diffs: List[str] = []
    new: List[str] = []
    for rel in sorted(_result_files(ref) | _result_files(tree)):
        pa, pb = ref / rel, tree / rel
        if not pb.exists():
            diffs.append(f"{rel}: only in the ref run")
        elif not pa.exists():
            new.append(str(rel))
        else:
            if rel.name == "tape.csv":
                d, n = _compare_tape(pa, pb)
            elif rel.suffix == ".npz":
                d, n = _compare_npz(pa, pb)
            else:
                d, n = _compare_json(
                    json.loads(pa.read_text()), json.loads(pb.read_text()), ""
                )
            diffs += [f"{rel}: {x}" for x in d]
            new += [f"{rel}: {x}" for x in n]
    return Comparison(diffs, new)


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


def _compare_tape(a: Path, b: Path) -> Tuple[List[str], List[str]]:
    """Columns whose values differ (with the first differing row) or are only
    in ``a``, and columns only in ``b``."""
    with open(a, newline="") as f:
        reader = csv.DictReader(f)
        rows_a, columns_a = list(reader), list(reader.fieldnames or [])
    with open(b, newline="") as f:
        reader = csv.DictReader(f)
        rows_b, columns_b = list(reader), list(reader.fieldnames or [])
    columns_a = [c for c in columns_a if not _ignored(c)]
    columns_b = [c for c in columns_b if not _ignored(c)]
    diffs = [f"{c}: only in the ref run" for c in columns_a if c not in columns_b]
    new = [c for c in columns_b if c not in columns_a]
    if len(rows_a) != len(rows_b):
        return diffs + [f"{len(rows_a)} rows vs {len(rows_b)}"], new
    for col in columns_a:
        if col not in columns_b:
            continue
        bad = [i for i, (x, y) in enumerate(zip(rows_a, rows_b)) if x[col] != y[col]]
        if bad:
            x, y = rows_a[bad[0]][col], rows_b[bad[0]][col]
            diffs.append(
                f"{col} differs in {len(bad)}/{len(rows_a)} rows; first at row "
                f"{bad[0]}: {x} vs {y}{_delta(x, y)}"
            )
    return diffs, new


def _compare_npz(a: Path, b: Path) -> Tuple[List[str], List[str]]:
    """Arrays that are only in ``a`` or differ in shape, dtype or value, and
    arrays only in ``b``."""
    with np.load(a) as za, np.load(b) as zb:
        diffs = [f"{k}: only in the ref run" for k in sorted(set(za) - set(zb))]
        new = sorted(set(zb) - set(za))
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
    return diffs, new


def _compare_json(x: Any, y: Any, path: str) -> Tuple[List[str], List[str]]:
    """Dotted paths whose values differ or are only in ``x``, and dotted
    paths only in ``y``, skipping identity and timings."""
    diffs: List[str] = []
    new: List[str] = []
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y), key=str):
            if _ignored(str(k)):
                continue
            sub = f"{path}.{k}" if path else str(k)
            if k not in y:
                diffs.append(f"{sub}: only in the ref run")
            elif k not in x:
                new.append(sub)
            else:
                d, n = _compare_json(x[k], y[k], sub)
                diffs += d
                new += n
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            d, n = _compare_json(u, v, f"{path}[{i}]")
            diffs += d
            new += n
    elif x != y and not (x != x and y != y):  # NaN equals NaN here
        diffs.append(f"{path}: {x!r} vs {y!r}{_delta(str(x), str(y))}")
    return diffs, new


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
        comparison = compare_run_dirs(runs / "ref", runs / "tree")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return report(comparison, ref)


def report(comparison: Comparison, ref: str) -> int:
    """Print ``comparison`` of ``ref`` against this tree and return the exit
    code (see the module docstring)."""
    if comparison.differences:
        print(f"results differ: {ref} vs this tree")
    elif comparison.new:
        print(f"results unchanged, new results added: {ref} vs this tree")
    else:
        print(f"identical results: {ref} vs this tree")
    for d in comparison.differences:
        print(f"  {d}")
    for n in comparison.new:
        print(f"  new: {n}")
    return 1 if comparison.differences else 0


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
