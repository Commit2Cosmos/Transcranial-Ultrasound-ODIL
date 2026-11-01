"""Launch an experiment: run every run of an experiment spec into one directory.

``main.py --experiment <spec>`` calls :func:`launch`. The runs execute in
``values`` order, each into ``runs/<idx>_<label>/`` of the experiment directory
``outputs/experiments/NNNN-slug[_cuda|_mps]/``, which also holds a verbatim
``spec.yaml``, ``experiment.json`` (provenance and per-run status) and
``summary.csv`` (one row per run). The run-layer README's "Experiments" section
is the contract.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .config import ConfigError
from .rundir import REPO_ROOT, SCHEMA_VERSION, GitState, RunStartError, _git, write_json
from .spec import ExperimentSpec, SweepRun, format_value, load_spec
from .tape import TAPE_COLUMNS

# An extra per-run hook, called with each run directory once its run ends.
# ``run_inverse`` already draws each run's figures, so a launch needs none;
# ``--postprocess <experiment_dir>`` passes the redraw through
# :func:`postprocess_experiment`.
PostprocessRun = Callable[[Path], None]


def launch(
    spec_path: Union[str, Path],
    *,
    dry_run: bool = False,
    out: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    postprocess_run: Optional[PostprocessRun] = None,
) -> int:
    """Validate the spec at ``spec_path`` and, unless ``dry_run``, run it.

    Returns the exit code: ``0`` every run completed (or a valid dry run),
    ``1`` a run failed or was interrupted, ``2`` the experiment could not start.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        spec = load_spec(spec_path, repo_root=root)
    except ConfigError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2
    exp_dir = Path(out) if out is not None else experiment_directory(spec, root)
    _print_plan(spec, exp_dir)
    if dry_run:
        print("\n[dry-run] experiment spec is valid; no artifacts written.")
        return 0
    try:
        status = run_experiment(
            spec, exp_dir, repo_root=root, postprocess_run=postprocess_run
        )
    except RunStartError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nexperiment {status}: {exp_dir}")
    return 0 if status == "completed" else 1


def experiment_directory(spec: ExperimentSpec, repo_root: Path) -> Path:
    """``<repo_root>/outputs/experiments/<id>[_<device>]``, suffixed only for
    cuda / mps, by the device the baseline's runs use."""
    device = _device(spec)
    suffix = "" if device == "cpu" else f"_{device}"
    return repo_root / "outputs" / "experiments" / f"{spec.id}{suffix}"


def run_experiment(
    spec: ExperimentSpec,
    exp_dir: Path,
    *,
    repo_root: Path,
    postprocess_run: Optional[PostprocessRun] = None,
) -> str:
    """Run every run of ``spec`` into ``exp_dir``; return the experiment status.

    Runs execute in ``values`` order; a failed run doesn't stop the others. On
    an existing experiment directory this resumes: ``completed`` runs are
    skipped, any other is deleted and run again from scratch. The status is
    ``completed`` when every run completed, ``interrupted`` when a run was
    (the remaining runs are not started), else ``failed``.

    Raises
    ------
    RunStartError
        A dirty tree, a spec not committed in ``repo_root``, or an existing
        ``exp_dir`` that isn't this experiment at this commit and spec.
    """
    state = _clean_state(repo_root)
    spec_rel = _committed_spec(spec, repo_root)
    record = _resumable_record(spec, state, exp_dir)
    if record is None:
        record = _new_record(spec, state, spec_rel)
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "spec.yaml").write_bytes(spec.path.read_bytes())
    else:
        print(f"resuming {exp_dir}")
    record["status"] = "running"
    write_json(exp_dir / "experiment.json", record)

    try:
        for run, entry in zip(spec.runs, record["runs"]):
            if entry["status"] == "completed":
                print(f"\n=== run {run.name}: completed, skipped ===")
                continue
            run_path = exp_dir / "runs" / run.name
            if run_path.exists():
                shutil.rmtree(run_path)
            print(f"\n=== run {run.name} ===")
            entry["status"] = "running"
            write_json(exp_dir / "experiment.json", record)
            entry["status"] = _run_one(run, run_path, repo_root, postprocess_run)
            write_json(exp_dir / "experiment.json", record)
            write_summary(exp_dir)
            if entry["status"] == "interrupted":
                break
    except KeyboardInterrupt:  # between runs, or in post-processing
        for entry in record["runs"]:
            if entry["status"] == "running":
                entry["status"] = "interrupted"

    statuses = [r["status"] for r in record["runs"]]
    if "interrupted" in statuses:
        record["status"] = "interrupted"
    elif all(s == "completed" for s in statuses):
        record["status"] = "completed"
    else:
        record["status"] = "failed"
    record["finished_utc"] = _utc_now()
    write_json(exp_dir / "experiment.json", record)
    postprocess_experiment_level(exp_dir)
    return record["status"]


def postprocess_experiment(
    exp_dir: Union[str, Path], *, postprocess_run: Optional[PostprocessRun] = None
) -> None:
    """``--postprocess <experiment_dir>``: post-process every run directory
    again with ``postprocess_run``, then the experiment itself, reading only
    what the experiment directory holds."""
    exp_dir = Path(exp_dir)
    record = json.loads((exp_dir / "experiment.json").read_text())
    if postprocess_run is not None:
        for entry in record["runs"]:
            run_dir = exp_dir / entry["dir"]
            if run_dir.is_dir():
                postprocess_run(run_dir)
    postprocess_experiment_level(exp_dir)


def postprocess_experiment_level(exp_dir: Path) -> None:
    """The experiment's own post-processing, at the end of a launch and on
    ``--postprocess``, from the experiment directory alone: ``summary.csv``,
    and the place for experiment figures (written to ``figures/``)."""
    write_summary(exp_dir)


def _run_one(
    run: SweepRun,
    run_path: Path,
    repo_root: Path,
    postprocess_run: Optional[PostprocessRun],
) -> str:
    """Run ``run`` into ``run_path``, post-process it and return its status."""
    from .rundir import new_run_directory
    from .runner import run_inverse

    try:
        status = run_inverse(
            run.config, new_run_directory(out=run_path, repo_root=repo_root)
        ).status
    except (ConfigError, RunStartError) as exc:  # this run could not start
        print(f"  [run {run.name}] could not start: {exc}")
        return "failed"
    if postprocess_run is not None:
        try:
            postprocess_run(run_path)
        except Exception as exc:  # noqa: BLE001 - figures never fail a run
            print(f"  [postprocess warning] {type(exc).__name__}: {exc}")
    return status


def _clean_state(repo_root: Path) -> GitState:
    """The repository's state, which must be a clean tree at a commit."""
    state = GitState.of(repo_root)
    if state.sha is None:
        raise RunStartError(
            f"{repo_root} is not a git repository, so the experiment can't name "
            "its commit"
        )
    if state.dirty:
        raise RunStartError(
            "the git tree has uncommitted changes or untracked files; an "
            "experiment needs a clean tree (commit them first)"
        )
    return state


def write_summary(exp_dir: Path) -> None:
    """Write ``summary.csv``: one row per run with its swept value, whether it
    is the baseline, its status and its final-iterate tape columns (empty
    unless the run completed)."""
    record = json.loads((exp_dir / "experiment.json").read_text())
    header = ["index", "label", "value", "is_baseline", "status"] + TAPE_COLUMNS
    rows = []
    for entry in record["runs"]:
        final = _final_metrics(exp_dir / entry["dir"]) or {}
        if entry["status"] != "completed":
            final = {}
        row = {
            "index": entry["index"],
            "label": entry["label"],
            "value": _value_text(entry["value"]),
            "is_baseline": "true" if entry["is_baseline"] else "false",
            "status": entry["status"],
        }
        row.update({k: "" if final.get(k) is None else final[k] for k in TAPE_COLUMNS})
        rows.append(row)
    tmp = exp_dir / "summary.csv.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(exp_dir / "summary.csv")


def _final_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """The run's final tape row from its ``run_summary.json``, if any."""
    path = run_dir / "run_summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())["metrics_summary"]["final_metrics"]


def _value_text(value: Any) -> str:
    """A swept value as a ``summary.csv`` cell: JSON for a list or dict."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return format_value(value)


def _committed_spec(spec: ExperimentSpec, repo_root: Path) -> str:
    """The spec's path relative to ``repo_root``, where it must be committed."""
    try:
        rel = spec.path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel = None
    if rel is None or _git(repo_root, "ls-files", "--error-unmatch", rel) is None:
        raise RunStartError(
            f"the spec {spec.path} must be committed in {repo_root} before launch"
        )
    return rel


def _resumable_record(
    spec: ExperimentSpec, state: GitState, exp_dir: Path
) -> Optional[dict]:
    """The ``experiment.json`` of an earlier launch to resume, or ``None`` if
    ``exp_dir`` is new or empty.

    Raises
    ------
    RunStartError
        ``exp_dir`` holds files but no record, or the record's commit or spec
        hash differs from HEAD's or this spec's.
    """
    if not exp_dir.exists() or not any(exp_dir.iterdir()):
        return None
    path = exp_dir / "experiment.json"
    if not path.is_file():
        raise RunStartError(
            f"{exp_dir} already exists and is not an experiment directory"
        )
    record = json.loads(path.read_text())
    if record.get("commit") != state.sha:
        raise RunStartError(
            f"{exp_dir} was launched at commit {record.get('commit')}, not HEAD "
            f"{state.sha}; an experiment resumes only at its commit"
        )
    if record.get("spec_sha256") != spec.sha256:
        raise RunStartError(
            f"{exp_dir} was launched from a different spec (hash mismatch); an "
            "experiment resumes only with its spec unchanged"
        )
    return record


def _new_record(spec: ExperimentSpec, state: GitState, spec_rel: str) -> dict:
    """``experiment.json`` at launch: every run ``pending``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": spec.id,
        "issue": spec.issue,
        "spec": spec_rel,
        "spec_sha256": spec.sha256,
        "commit": state.sha,
        "device": _device(spec),
        "sweep_key": spec.key,
        "status": "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "runs": [
            {
                "index": run.index,
                "label": run.label,
                "dir": f"runs/{run.name}",
                "value": run.value,
                "is_baseline": run.is_baseline,
                "status": "pending",
            }
            for run in spec.runs
        ],
    }


def _device(spec: ExperimentSpec) -> str:
    """The device type the baseline's runs resolve to: cpu, cuda or mps."""
    from .envinfo import resolve_device

    return resolve_device(spec.baseline.runtime.device).type


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _print_plan(spec: ExperimentSpec, exp_dir: Path) -> None:
    """Print the experiment's id, sweep and runs."""
    print(f"experiment      : {spec.id}  (sweep {spec.key})")
    print(f"directory       : {exp_dir}")
    for run in spec.runs:
        mark = "  (baseline)" if run.is_baseline else ""
        print(f"  {run.name}: {_value_text(run.value)}{mark}")
