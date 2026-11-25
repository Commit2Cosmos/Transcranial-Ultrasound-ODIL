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
from dataclasses import dataclass, field
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

    differences: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)

    def __add__(self, other: Comparison) -> Comparison:
        return Comparison(self.differences + other.differences, self.new + other.new)


def compare_run_dirs(ref: Path, tree: Path) -> Comparison:
    """Compare the results of run directories ``ref`` and ``tree``; both lists
    are empty when the results are identical."""
    ref, tree = Path(ref), Path(tree)
    total = Comparison()
    for rel in sorted(_result_files(ref) | _result_files(tree)):
        in_ref, in_tree = ref / rel, tree / rel
        if not in_tree.exists():
            total += Comparison(differences=[f"{rel}: only in the ref run"])
        elif not in_ref.exists():
            total += Comparison(new=[str(rel)])
        else:
            if rel.name == "tape.csv":
                found = _compare_tape(in_ref, in_tree)
            elif rel.suffix == ".npz":
                found = _compare_npz(in_ref, in_tree)
            else:
                found = _compare_json(
                    json.loads(in_ref.read_text()), json.loads(in_tree.read_text()), ""
                )
            total += Comparison(
                [f"{rel}: {d}" for d in found.differences],
                [f"{rel}: {n}" for n in found.new],
            )
    return total


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


def _read_tape(path: Path) -> Tuple[List[dict], List[str]]:
    """Rows and result columns (identity and timings left out) of a tape."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, [c for c in reader.fieldnames or [] if not _ignored(c)]


def _compare_tape(ref: Path, tree: Path) -> Comparison:
    """Tape columns, with the first differing row of each changed one."""
    rows_ref, columns_ref = _read_tape(ref)
    rows_tree, columns_tree = _read_tape(tree)
    diffs = [f"{c}: only in the ref run" for c in columns_ref if c not in columns_tree]
    new = [c for c in columns_tree if c not in columns_ref]
    if len(rows_ref) != len(rows_tree):
        diffs.append(f"{len(rows_ref)} rows vs {len(rows_tree)}")
        return Comparison(diffs, new)
    for col in columns_ref:
        if col not in columns_tree:
            continue
        bad = [
            i for i, (x, y) in enumerate(zip(rows_ref, rows_tree)) if x[col] != y[col]
        ]
        if bad:
            x, y = rows_ref[bad[0]][col], rows_tree[bad[0]][col]
            diffs.append(
                f"{col} differs in {len(bad)}/{len(rows_ref)} rows; first at row "
                f"{bad[0]}: {x} vs {y}{_delta(x, y)}"
            )
    return Comparison(diffs, new)


def _compare_npz(ref: Path, tree: Path) -> Comparison:
    """Arrays, by name; a shape or dtype change is a difference."""
    with np.load(ref) as zr, np.load(tree) as zt:
        diffs = [f"{k}: only in the ref run" for k in sorted(set(zr) - set(zt))]
        new = sorted(set(zt) - set(zr))
        for k in sorted(set(zr) & set(zt)):
            x, y = zr[k], zt[k]
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
    return Comparison(diffs, new)


def _compare_json(ref: Any, tree: Any, path: str) -> Comparison:
    """JSON values by dotted path, skipping identity and timings."""
    if isinstance(ref, dict) and isinstance(tree, dict):
        total = Comparison()
        for k in sorted(set(ref) | set(tree), key=str):
            if _ignored(str(k)):
                continue
            sub = f"{path}.{k}" if path else str(k)
            if k not in tree:
                total += Comparison(differences=[f"{sub}: only in the ref run"])
            elif k not in ref:
                total += Comparison(new=[sub])
            else:
                total += _compare_json(ref[k], tree[k], sub)
        return total
    if isinstance(ref, list) and isinstance(tree, list) and len(ref) == len(tree):
        total = Comparison()
        for i, (u, v) in enumerate(zip(ref, tree)):
            total += _compare_json(u, v, f"{path}[{i}]")
        return total
    if ref != tree and not (ref != ref and tree != tree):  # NaN equals NaN here
        return Comparison([f"{path}: {ref!r} vs {tree!r}{_delta(str(ref), str(tree))}"])
    return Comparison()


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
