"""Experiment specs: the committed file defining exactly an experiment's runs.

An **experiment spec** (``configs/experiments/NNNN-slug.yaml``) holds a
``baseline`` run config and a ``sweep``: one or more axes, each a dotted key
and its values. Every combination of one value per axis is a run, each value
applied at its key like a run config (dicts deep-merge, lists replace).
:func:`load_spec` parses and validates a spec, resolving every run's config; it
is torch-free, so ``--dry-run`` stays fast. The run-layer README's
"Experiment spec" section is the contract.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import (
    ConfigError,
    RunConfig,
    deep_merge,
    load_config_file,
    resolve_config,
    validate_config,
)

# Version of the spec format; readers keep a path for every version from 1.
# Version 1's ``sweep`` is a single axis, version 2's a list of axes.
SPEC_SCHEMA_VERSION = 2

_KEYS = ("schema_version", "id", "issue", "baseline", "sweep")
_SWEEP_KEYS = ("key", "values", "labels")
_ID = re.compile(r"^(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*$")
# A label names a run's directory, so it stays filesystem-safe.
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
# The one override an experiment accepts (--reproduce passes it to regenerate
# on another device), and the devices it may name.
DEVICE_KEY = "runtime.device"
_DEVICE = re.compile(r"^(cpu|mps|cuda(:\d+)?)$")


@dataclass(frozen=True)
class SweepRun:
    """One run of an experiment: one value per swept key applied to the baseline."""

    index: int
    label: str
    values: Dict[str, Any]  # swept key -> its value in this run, in axis order
    config: RunConfig
    is_baseline: bool

    @property
    def name(self) -> str:
        """The run's directory name, ``<idx>_<label>`` with a two-digit index."""
        return f"{self.index:02d}_{self.label}"


@dataclass(frozen=True)
class ExperimentSpec:
    """A parsed, validated experiment spec and its runs: the cross product of
    the axes, the last axis varying fastest."""

    path: Path
    id: str
    issue: int  # the protocol issue
    keys: List[str]  # the swept keys, in axis order
    baseline: RunConfig
    runs: List[SweepRun]
    sha256: str  # of the spec file's bytes


def load_spec(
    path: Union[str, Path],
    *,
    repo_root: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> ExperimentSpec:
    """Parse and validate the experiment spec at ``path``, resolving every run.

    Parameters
    ----------
    path : str or Path
        The spec, ``configs/experiments/NNNN-slug.yaml``.
    repo_root : str or Path, optional
        The repository whose ``configs/experiments/`` specs and
        ``docs/experiments/`` records must not use the spec's id number.
        Without it the id number isn't checked.
    device : str, optional
        ``runtime.device`` for the baseline and every run, in place of the
        spec's (see :func:`device_override`).

    Raises
    ------
    ConfigError
        The spec, or the config of one of its runs, is invalid.
    """
    path = Path(path)
    data = load_config_file(path)
    unknown = sorted(set(data) - set(_KEYS))
    if unknown:
        raise ConfigError(f"unknown spec key(s) {unknown}; valid keys: {list(_KEYS)}")
    version = data.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not 1 <= version <= SPEC_SCHEMA_VERSION
    ):
        raise ConfigError(
            f"schema_version must be 1 to {SPEC_SCHEMA_VERSION}, got {version!r}"
        )
    exp_id = _check_id(data.get("id"), path, repo_root)
    issue = data.get("issue")
    if not isinstance(issue, int) or isinstance(issue, bool) or issue < 1:
        raise ConfigError(f"issue must be the protocol's issue number, got {issue!r}")

    baseline_user = data.get("baseline")
    if not isinstance(baseline_user, dict):
        raise ConfigError("baseline must be a run config (a mapping)")
    on_device = {} if device is None else _nest(DEVICE_KEY, device)
    try:
        baseline = resolve_config(deep_merge(baseline_user, on_device))
        validate_config(baseline)
    except ConfigError as exc:
        raise ConfigError(f"baseline: {exc}") from None

    axes = _check_sweep(data.get("sweep"), version)
    keys = [key for key, _, _ in axes]
    combinations = itertools.product(
        *(list(zip(labels, values)) for _, values, labels in axes)
    )
    runs = []
    for index, combination in enumerate(combinations):
        label = "_".join(label for label, _ in combination)
        values = {key: value for key, (_, value) in zip(keys, combination)}
        name = f"{index:02d}_{label}"
        try:
            run_user = baseline_user
            for key, value in values.items():
                run_user = deep_merge(run_user, _nest(key, value))
            cfg = resolve_config(deep_merge(run_user, on_device))
            validate_config(cfg)
        except ConfigError as exc:
            raise ConfigError(f"run {name}: {exc}") from None
        runs.append(
            SweepRun(
                index=index,
                label=label,
                values=values,
                config=cfg,
                is_baseline=cfg.to_dict() == baseline.to_dict(),
            )
        )
    n_baseline = sum(run.is_baseline for run in runs)
    if n_baseline != 1:
        raise ConfigError(
            f"exactly one run must resolve to the baseline; {n_baseline} do"
        )
    return ExperimentSpec(
        path=path,
        id=exp_id,
        issue=issue,
        keys=keys,
        baseline=baseline,
        runs=runs,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def device_override(overrides: Sequence[str]) -> Optional[str]:
    """The device named by ``-o runtime.device=<device>`` among ``overrides``,
    or ``None`` when there are none.

    Raises
    ------
    ConfigError
        An override of another key, or a device other than cpu, cuda[:N], mps.
    """
    device = None
    for override in overrides:
        key, _, value = override.partition("=")
        if key.strip() != DEVICE_KEY:
            raise ConfigError(
                f"only -o {DEVICE_KEY}=<device> may be given, not {override!r}: "
                "the committed spec alone defines the runs"
            )
        device = value.strip()
        if not _DEVICE.match(device):
            raise ConfigError(f"{DEVICE_KEY} must be cpu, cuda or mps, not {value!r}")
    return device


def _check_id(exp_id: Any, path: Path, repo_root: Optional[Union[str, Path]]) -> str:
    """The spec's id: ``NNNN-slug``, its filename, its number used by no other
    spec or record in ``repo_root``."""
    if not isinstance(exp_id, str) or not _ID.match(exp_id):
        raise ConfigError(f"id must be NNNN-slug (lowercase), got {exp_id!r}")
    if exp_id != path.stem:
        raise ConfigError(f"id {exp_id!r} doesn't match the filename {path.name!r}")
    if repo_root is not None:
        root = Path(repo_root)
        taken = sorted(
            p.stem
            for p in [
                *(root / "configs" / "experiments").glob(f"{exp_id[:4]}-*.yaml"),
                *(root / "docs" / "experiments").glob(f"{exp_id[:4]}-*.md"),
            ]
            if p.stem != exp_id
        )
        if taken:
            raise ConfigError(
                f"id number {exp_id[:4]} is already used by {', '.join(taken)}"
            )
    return exp_id


def _check_sweep(sweep: Any, version: int) -> List[tuple]:
    """The sweep's axes, each ``(key, values, labels)``: one axis in a version 1
    spec, a non-empty list of them in version 2, no key equal to or nested in
    another."""
    if version == 1:
        return [_check_axis(sweep)]
    if not isinstance(sweep, list) or not sweep:
        raise ConfigError(
            "sweep must be a non-empty list of axes, each with key, values and labels"
        )
    axes = [_check_axis(axis) for axis in sweep]
    keys = [key for key, _, _ in axes]
    for a, b in itertools.permutations(keys, 2):
        if a == b or b.startswith(a + "."):
            raise ConfigError(f"sweep key {b!r} repeats or is nested in {a!r}")
    labels = ["_".join(c) for c in itertools.product(*(ls for _, _, ls in axes))]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"run labels must be unique, got {labels}")
    return axes


def _check_axis(sweep: Any) -> tuple:
    """A sweep axis's ``(key, values, labels)``, labels derived for scalars."""
    if not isinstance(sweep, dict):
        raise ConfigError("a sweep axis must be a mapping with key, values and labels")
    unknown = sorted(set(sweep) - set(_SWEEP_KEYS))
    if unknown:
        raise ConfigError(
            f"unknown sweep key(s) {unknown}; valid keys: {list(_SWEEP_KEYS)}"
        )
    key = sweep.get("key")
    if not isinstance(key, str) or not all(key.split(".")):
        raise ConfigError(f"sweep.key must be one dotted key, got {key!r}")
    values = sweep.get("values")
    if not isinstance(values, list) or not values:
        raise ConfigError("sweep.values must be a non-empty list")

    labels = sweep.get("labels")
    if labels is None:
        if any(isinstance(v, (list, dict)) for v in values):
            raise ConfigError(
                "sweep.labels are required when a value is a list or dict"
            )
        labels = [_scalar_label(key, v) for v in values]
    elif not isinstance(labels, list) or len(labels) != len(values):
        raise ConfigError(
            f"sweep.labels must be a list of one label per value ({len(values)})"
        )
    for label in labels:
        if not isinstance(label, str) or not _LABEL.match(label):
            raise ConfigError(
                f"label {label!r} must be letters, digits and . _ + - "
                "(give sweep.labels to name the runs)"
            )
    if len(set(labels)) != len(labels):
        raise ConfigError(f"run labels must be unique, got {labels}")
    return key, values, labels


def _nest(key: str, value: Any) -> Dict[str, Any]:
    """``a.b.c`` and ``value`` as the run config ``{a: {b: {c: value}}}``."""
    *parents, leaf = key.split(".")
    out: Dict[str, Any] = {leaf: value}
    for part in reversed(parents):
        out = {part: out}
    return out


def format_value(value: Any) -> str:
    """A scalar as written in a run label: ``1e-6`` not ``1e-06``, ``0`` not
    ``0.0``, ``true`` / ``null`` for booleans and ``None``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = repr(value)
        if text.endswith(".0"):
            text = text[:-2]
        mantissa, e, exponent = text.partition("e")
        if e:
            sign = "-" if exponent.startswith("-") else ""
            text = f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"
        return text
    return str(value)


def _scalar_label(key: str, value: Any) -> str:
    """``<last key segment, kebab-case>-<value>``, e.g. ``reg-1e-6``."""
    return f"{key.split('.')[-1].replace('_', '-')}-{format_value(value)}"
