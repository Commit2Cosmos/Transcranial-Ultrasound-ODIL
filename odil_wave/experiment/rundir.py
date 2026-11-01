"""Where a run is written and what ties it to its code.

The entry point, not the Run config, picks a run's directory, so a regenerated
run's ``config_resolved.yaml`` is identical to the original's. A standalone run
lands in ``outputs/runs/<YYMMDD-HHMMSS>_<sha7>[_scratch]/`` under the code's
repository (``out`` overrides the location) and needs a clean git tree unless
it is a **scratch run**, which saves its ``git.diff``. :func:`new_run_directory`
applies that policy; :meth:`RunDirectory.provenance` is what ``metadata.json``
records about it. The run-layer README's "Where runs and experiments are
written" and "Standalone and scratch runs" are the contract.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .atomic import write_text_atomic

# Version of the run and experiment directories' formats (``metadata.json``,
# ``experiment.json``); a format change bumps it and readers keep a path for
# every version from 1 (:func:`load_versioned_json`).
SCHEMA_VERSION = 1

# The repository holding this code: the default ``repo_root``.
REPO_ROOT = Path(__file__).resolve().parents[2]


class RunStartError(RuntimeError):
    """A run that may not start: a dirty tree without ``scratch``, no git
    repository, or a run directory that already holds files."""


class SchemaVersionError(ValueError):
    """A ``metadata.json`` or ``experiment.json`` whose ``schema_version``
    this code can't read: newer than :data:`SCHEMA_VERSION`, or none."""


@dataclass(frozen=True)
class GitState:
    """The code's commit and whether the tree differs from it.

    ``sha`` and ``branch`` are ``None`` outside a git repository. ``dirty`` counts
    untracked, non-ignored files as well as tracked changes.
    """

    sha: Optional[str]
    branch: Optional[str]
    dirty: bool

    @classmethod
    def of(cls, repo_root: Path) -> "GitState":
        """The state of the git repository at ``repo_root``."""
        sha = run_git(repo_root, "rev-parse", "HEAD")
        if sha is None:
            return cls(sha=None, branch=None, dirty=True)
        status = run_git(repo_root, "status", "--porcelain", "--untracked-files=normal")
        branch = run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        return cls(
            sha=sha.strip(),
            branch=None if branch is None else branch.strip(),
            dirty=bool(status and status.strip()),
        )


@dataclass(frozen=True)
class RunDirectory:
    """A run's directory, created and checked by :func:`new_run_directory`."""

    path: Path
    scratch: bool
    git: GitState
    repo_root: Path

    @property
    def run_id(self) -> str:
        """The directory name, stamped on every tape row."""
        return self.path.name

    def provenance(self) -> Dict[str, Any]:
        """The ``metadata.json`` fields tying the run to its code and lockfile."""
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "scratch": self.scratch,
            "git": {
                "sha": self.git.sha,
                "branch": self.git.branch,
                "dirty": self.git.dirty,
            },
            "uv_lock_sha256": _sha256(self.repo_root / "uv.lock"),
        }


def new_run_directory(
    *,
    scratch: bool = False,
    out: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> RunDirectory:
    """Create a standalone run's directory after checking the code's git tree.

    Parameters
    ----------
    scratch : bool
        Allow a dirty tree (or no repository): ``_scratch`` is appended to the
        directory name and its ``git.diff`` (:func:`scratch_diff`: tracked
        changes and untracked, non-ignored files) is saved in it.
    out : str or Path, optional
        The run directory itself, instead of
        ``<repo_root>/outputs/runs/<YYMMDD-HHMMSS>_<sha7>[_scratch]``.
    repo_root : str or Path, optional
        The code's repository; defaults to the one holding this package.

    Returns
    -------
    RunDirectory
        The created, empty directory (``git.diff`` aside) and its provenance.

    Raises
    ------
    RunStartError
        A dirty tree or no repository without ``scratch``, or a directory that
        already holds files.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    if scratch:
        state = GitState.of(root)
    else:
        state = clean_git_state(
            root, needs="the run", remedy="; pass --scratch for a scratch run"
        )
    if out is not None:
        path = Path(out)
    else:
        sha7 = state.sha[:7] if state.sha else "nogit"
        name = f"{time.strftime('%y%m%d-%H%M%S')}_{sha7}"
        path = root / "outputs" / "runs" / (name + ("_scratch" if scratch else ""))
    if path.exists() and any(path.iterdir()):
        raise RunStartError(f"run directory {path} already exists and is not empty")
    path.mkdir(parents=True, exist_ok=True)
    if scratch:
        write_text_atomic(path / "git.diff", scratch_diff(root))
    return RunDirectory(path=path, scratch=scratch, git=state, repo_root=root)


def clean_git_state(repo_root: Path, *, needs: str, remedy: str = "") -> GitState:
    """The state of ``repo_root``, which must be a clean tree at a commit.

    ``needs`` names what needs it (``"the run"``) and ``remedy`` ends both
    error messages.

    Raises
    ------
    RunStartError
        ``repo_root`` is no git repository, or its tree is dirty.
    """
    state = GitState.of(repo_root)
    if state.sha is None:
        raise RunStartError(
            f"{repo_root} is not a git repository, so {needs} can't name its "
            f"commit{remedy}"
        )
    if state.dirty:
        raise RunStartError(
            "the git tree has uncommitted changes or untracked files; "
            f"{needs} needs a clean tree (commit them first){remedy}"
        )
    return state


def experiment_path(repo_root: Path, exp_id: str, device_type: str) -> Path:
    """``<repo_root>/outputs/experiments/<exp_id>[_<device_type>]``: an
    experiment's directory, suffixed only for cuda / mps, by the device type
    its runs use."""
    suffix = "" if device_type == "cpu" else f"_{device_type}"
    return repo_root / "outputs" / "experiments" / f"{exp_id}{suffix}"


def load_versioned_json(path: Path) -> Dict[str, Any]:
    """Read ``metadata.json`` or ``experiment.json`` at ``path``, whose
    ``schema_version`` must be one this code reads: 1 to :data:`SCHEMA_VERSION`.

    Raises
    ------
    SchemaVersionError
        The file's ``schema_version`` is missing or out of that range.
    """
    data = json.loads(Path(path).read_text())
    version = data.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not 1 <= version <= SCHEMA_VERSION
    ):
        raise SchemaVersionError(
            f"{path} has schema_version {version!r}; this code reads 1 to "
            f"{SCHEMA_VERSION} (check out a newer commit to read it)"
        )
    return data


def write_json(path: Path, obj: Any) -> None:
    """Atomically write ``obj`` to ``path`` as indented JSON (``str`` fallback)."""
    write_text_atomic(path, json.dumps(obj, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    """JSON fallback: numpy scalars to Python numbers, anything else to ``str``."""
    item = getattr(o, "item", None)
    if callable(item) and getattr(o, "shape", None) == ():
        return item()
    return str(o)


def _sha256(path: Path) -> Optional[str]:
    """Hex SHA-256 of the file at ``path``; ``None`` if it doesn't exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def scratch_diff(repo_root: Path) -> str:
    """A scratch run's ``git.diff``: every change from HEAD, the untracked,
    non-ignored files included as new files, so ``git apply`` on a clean
    checkout of the commit rebuilds the tree (binary files too)."""
    parts = [run_git(repo_root, "diff", "--binary", "HEAD") or ""]
    untracked = run_git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    for rel in filter(None, (untracked or "").split("\0")):
        # --no-index exits 1 when the files differ, as they always do here.
        new_file = ("diff", "--no-index", "--binary", "--", "/dev/null", rel)
        parts.append(run_git(repo_root, *new_file, ok=(0, 1)) or "")
    return "".join(parts)


def run_git(repo_root: Path, *args: str, ok: Tuple[int, ...] = (0,)) -> Optional[str]:
    """``git -C repo_root <args>``'s raw stdout, or ``None`` if it fails (exits
    with a code not in ``ok``)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode in ok else None
