"""Reproduce an experiment: regenerate its runs at its recorded commit.

``main.py --reproduce <experiment-id>`` calls :func:`reproduce`. It reads
``commit`` and ``spec`` from the record's frontmatter
(``docs/experiments/<id>.md``), adds a detached worktree at that commit, runs
the **recorded** entry point there with
``uv run --frozen python main.py --experiment <spec> --out <target>`` and
removes the worktree. It compares nothing and never edits the record. The
run-layer README's "``--reproduce``" section is the contract.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml

from .bank import read_record
from .config import ConfigError
from .rundir import REPO_ROOT, experiment_path, run_git
from .spec import DEVICE_KEY, device_override


class ReproduceError(RuntimeError):
    """The reproduction can't start."""


def reproduce(
    exp_id: str,
    *,
    overrides: Sequence[str] = (),
    repo_root: Optional[Union[str, Path]] = None,
) -> int:
    """Regenerate experiment ``exp_id`` into ``outputs/experiments/``.

    ``overrides`` may only be ``runtime.device=<device>``, to run on a device
    other than the spec's. Returns the exit code: ``0`` every run completed,
    ``1`` a run failed, ``2`` it could not start.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        try:
            device = device_override(overrides)
        except ConfigError as exc:
            raise ReproduceError(f"with --reproduce, {exc}") from None
        commit, spec = _recorded(root, exp_id)
        if device is None:
            device = _spec_device(root, commit, spec)
        target = _target(root, exp_id, device)
    except ReproduceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    command = ["uv", "run", "--frozen", "python", "main.py"]
    command += ["--experiment", spec, "--out", str(target)]
    if overrides:
        command += ["-o", f"{DEVICE_KEY}={device}"]
    print(f"reproducing {exp_id} at {commit} into {target}")
    tmp = Path(tempfile.mkdtemp(prefix=f"reproduce-{exp_id}-"))
    worktree = tmp / exp_id
    try:
        if run_git(root, "worktree", "add", "--detach", str(worktree), commit) is None:
            print(
                f"error: can't check out the recorded commit {commit} "
                f"(fetch it, e.g. its launch/{exp_id} tag)",
                file=sys.stderr,
            )
            return 2
        try:
            code = subprocess.run(command, cwd=worktree).returncode
        except OSError as exc:  # no uv on PATH
            print(f"error: can't run {command[0]!r}: {exc}", file=sys.stderr)
            return 2
    finally:
        run_git(root, "worktree", "remove", "--force", str(worktree))
        shutil.rmtree(tmp, ignore_errors=True)
    # The recorded entry point exits 0 / 1 / 2 like this one; anything else
    # (a crash, a signal) means a run did not complete.
    return code if code in (0, 1, 2) else 1


def _recorded(root: Path, exp_id: str) -> tuple[str, str]:
    """``commit`` and ``spec`` from the frontmatter of ``exp_id``'s record."""
    path = root / "docs" / "experiments" / f"{exp_id}.md"
    if not path.is_file():
        raise ReproduceError(f"no experiment record {path}")
    record = read_record(path)
    if record is None:
        raise ReproduceError(f"{path} has no YAML frontmatter")
    fields = [record.front.get("commit"), record.front.get("spec")]
    for name, value in zip(("commit", "spec"), fields):
        if not isinstance(value, str) or not value.strip():
            raise ReproduceError(f"{path}: frontmatter `{name}` is empty")
    return fields[0].strip(), fields[1].strip()


def _spec_device(root: Path, commit: str, spec: str) -> str:
    """The baseline's ``runtime.device`` in the spec as recorded at ``commit``.

    Read from the YAML alone, not resolved by this commit's config code, which
    may no longer accept an old spec that the recorded entry point does.
    """
    text = run_git(root, "show", f"{commit}:{spec}")
    if text is None:
        raise ReproduceError(f"the spec {spec} is not in the recorded commit {commit}")
    data: Any = yaml.safe_load(text)
    runtime = (data or {}).get("baseline", {}).get("runtime", {}) or {}
    return str(runtime.get("device", "cpu"))


def _target(root: Path, exp_id: str, device: str) -> Path:
    """``outputs/experiments/<id>[_<device>]``, suffixed for cuda / mps by the
    device the runs resolve to on this machine; it must not exist."""
    from .envinfo import resolve_device

    target = experiment_path(root, exp_id, resolve_device(device).type).resolve()
    if target.exists():
        raise ReproduceError(
            f"{target} already exists; --reproduce never overwrites a result "
            "(move or delete it first)"
        )
    return target
