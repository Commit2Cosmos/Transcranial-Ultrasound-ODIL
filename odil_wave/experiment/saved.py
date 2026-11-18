"""A run directory's format: the one module that writes what a run saves and
reads it back.

Writers: :class:`ProblemArrays` (``problem.npz``, once per run),
:class:`BandArraysCollector` and :func:`save_band` (``bands/band_<NN>_<label>/``
per completed band) and :func:`write_run_summary` (``run_summary.json``).
Readers: :class:`SavedRun` (with a :class:`SavedBand` per completed band)
and :func:`load_final_metrics`. ``tape.csv`` belongs to :mod:`.tape` and
``metadata.json``'s provenance to :mod:`.rundir`; :class:`SavedRun` reads both.

Every array is saved as float32 / complex64 whatever ``runtime.dtype`` (masks
bool, indices int64), in an uncompressed ``.npz`` written atomically. Velocities
cover the interior only; the full-grid model is the interior padded with
``pml_c`` (:meth:`~odil_wave.models.VelocityModel.from_interior`).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from odil_wave.band import Band
from odil_wave.optimisation import Iterate

from .atomic import atomic_open
from .config import load_config_file
from .rundir import load_versioned_json, write_json
from .tape import load_tape

PathLike = Union[str, Path]

CONFIG_FILE = "config_resolved.yaml"
METADATA_FILE = "metadata.json"
PROBLEM_FILE = "problem.npz"
RUN_SUMMARY_FILE = "run_summary.json"
BANDS_DIR = "bands"
BAND_ARRAYS_FILE = "arrays.npz"
BAND_SUMMARY_FILE = "band_summary.json"

# The fixed saved shots: the transducers nearest these ring angles.
SAVED_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)


def write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Atomically write ``arrays`` as an uncompressed ``.npz`` (tmp + rename)."""
    with atomic_open(path, "wb") as f:
        np.savez(f, **arrays)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _single(x) -> np.ndarray:
    """A numpy copy of a tensor or array in float32 / complex64 (bool kept)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype == bool:
        return x
    return x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)


def band_label(frequencies_hz: List[float]) -> str:
    """A band's kHz label in its directory name: ``"60khz"`` for a single
    frequency, else ``"40-70khz"`` (min-max)."""
    khz = [f / 1e3 for f in frequencies_hz]
    if len(khz) == 1:
        return f"{int(round(khz[0]))}khz"
    return f"{int(round(min(khz)))}-{int(round(max(khz)))}khz"


# Matches every :func:`band_dir_name`.
BAND_DIR_GLOB = "band_*"


def band_dir_name(band_index: int, label: str) -> str:
    """A completed band's directory under ``bands/``."""
    return f"band_{band_index:02d}_{label}"


def transducer_angles_deg(n_transducers: int) -> np.ndarray:
    """Ring angle of each transducer in degrees (transducer ``k`` at ``360 k / n``)."""
    return (360.0 * np.arange(n_transducers) / n_transducers).astype(np.float32)


def fixed_shots(n_transducers: int) -> List[int]:
    """The shots of the transducers nearest 0, 90, 180 and 270 degrees, in that
    order and without repeats (``n_transducers`` need not divide by 4)."""
    shots: List[int] = []
    for angle in SAVED_ANGLES_DEG:
        k = int(round(angle * n_transducers / 360.0)) % n_transducers
        if k not in shots:
            shots.append(k)
    return shots


# --------------------------------------------------------------------------- #
# problem.npz
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProblemArrays:
    """A run's ``problem.npz``: the truth, starting model and layout that every
    band shares. Velocities (m/s) and masks are ``(interior_nx, interior_ny)``."""

    c_star: np.ndarray
    c0: np.ndarray
    pml_c: float  # the frozen PML speed, m/s
    brain_mask: np.ndarray
    skull_mask: np.ndarray
    transducer_angles_deg: np.ndarray  # (n_transducers,): shot k's ring angle
    interior_slice: Tuple[slice, slice]  # the interior on the full grid

    @classmethod
    def of(cls, problem, brain: np.ndarray, skull: np.ndarray) -> "ProblemArrays":
        """The arrays of ``problem`` (a :class:`~.problem.Problem`), with its
        interior-shaped boolean brain and skull regions."""
        sl = problem.grid.interior_slice
        return cls(
            c_star=_single(problem.truth_velocity.c[sl]),
            c0=_single(problem.init_velocity.c[sl]),
            pml_c=float(problem.init_velocity.pml_c),
            brain_mask=np.asarray(brain, dtype=bool),
            skull_mask=np.asarray(skull, dtype=bool),
            transducer_angles_deg=transducer_angles_deg(
                problem.cfg.acquisition.n_transducers
            ),
            interior_slice=tuple(sl),
        )

    @classmethod
    def load(cls, path: Path) -> "ProblemArrays":
        z = _load_npz(path)
        (x0, x1), (y0, y1) = z.pop("interior_slice")
        return cls(
            **{**z, "pml_c": float(z["pml_c"])},
            interior_slice=(slice(int(x0), int(x1)), slice(int(y0), int(y1))),
        )

    def save(self, path: Path) -> None:
        write_npz(
            path,
            {
                "c_star": _single(self.c_star),
                "c0": _single(self.c0),
                "pml_c": np.float32(self.pml_c),
                "brain_mask": np.asarray(self.brain_mask, dtype=bool),
                "skull_mask": np.asarray(self.skull_mask, dtype=bool),
                "transducer_angles_deg": _single(self.transducer_angles_deg),
                "interior_slice": np.array(
                    [[s.start, s.stop] for s in self.interior_slice], dtype=np.int64
                ),
            },
        )

    @property
    def head_mask(self) -> np.ndarray:
        """The head region, interior-shaped."""
        return self.brain_mask | self.skull_mask


# --------------------------------------------------------------------------- #
# A completed band's directory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BandArrays:
    """A band's ``arrays.npz`` at its final iterate.

    ``d_obs``, ``d_syn_start`` (the start row's warm start) and ``d_syn_final``
    cover every shot, ``(n_shots, n_freq, n_transducers - 1)`` complex with shot
    ``s``'s receivers in ascending transducer order without ``s``. ``u`` and
    ``u_star`` cover the saved ``shots`` only: ``(n_saved_shots, n_freq, nx,
    ny)``, ``u`` on the full inversion grid and ``u_star`` on the interior.
    """

    freqs: np.ndarray  # (n_freq,) Hz: the FFT bins solved at
    c: np.ndarray  # (interior_nx, interior_ny) m/s
    d_obs: np.ndarray
    d_syn_start: np.ndarray
    d_syn_final: np.ndarray
    pde_scale: np.ndarray  # (n_shots, n_freq): the band's fixed |lambda_tt| * u_ref
    shots: np.ndarray  # (n_saved_shots,) int
    u: np.ndarray
    u_star: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "BandArrays":
        return cls(**_load_npz(path))

    def save(self, path: Path) -> None:
        arrays = {f.name: _single(getattr(self, f.name)) for f in fields(self)}
        arrays["shots"] = np.asarray(self.shots, dtype=np.int64)
        write_npz(path, arrays)


class BandArraysCollector:
    """Gathers a band's :class:`BandArrays` from its logged iterates.

    Call it with every :class:`~odil_wave.optimisation.Iterate` the band logs
    (alongside the tape's ``on_log``): the first is the start row's warm start,
    the last the band's final iterate. :meth:`arrays` then holds the file.
    """

    def __init__(self, band: Band, pde_scale: torch.Tensor) -> None:
        """``band`` must be built with its true wavefields; ``pde_scale`` is its
        fixed ``|lambda_tt| * u_ref``, ``(n_shots, n_freq)``."""
        self.geom = band.geometry
        self.pde_scale = pde_scale.detach().reshape(pde_scale.shape[0], -1)
        self.freqs = band.frequency_selection.frequencies
        self._interior = band.geometry.grid.interior_slice
        self.d_obs = band.observed_traces.detach()
        self._u_star = band.true_wavefield_interior()
        self.d_syn_start: Optional[torch.Tensor] = None
        self.d_syn_final: Optional[torch.Tensor] = None
        self._final: Optional[Iterate] = None

    def __call__(self, it: Iterate) -> None:
        """Take a logged iterate; the latest one is the band's final iterate."""
        d_syn = self.geom.extract_observations(it.u).detach()
        if self.d_syn_start is None:
            self.d_syn_start = d_syn
        self.d_syn_final = d_syn
        self._final = it

    @property
    def worst_fit_shot(self) -> int:
        """The shot with the highest ``rel_data_residual`` at the final iterate."""
        if self.d_syn_final is None:
            raise RuntimeError("no iterate logged yet")
        n = self.d_obs.shape[0]
        diff = (self.d_syn_final - self.d_obs).reshape(n, -1)
        obs = self.d_obs.reshape(n, -1)
        rel = torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(
            obs, dim=1
        ).clamp_min(1e-30)
        return int(torch.argmax(rel))

    @property
    def shots(self) -> List[int]:
        """The saved shots: :func:`fixed_shots` plus the worst-fit shot unless it
        is already one of them."""
        shots = fixed_shots(self.geom.n_transducers)
        worst = self.worst_fit_shot
        return shots if worst in shots else shots + [worst]

    def arrays(self) -> BandArrays:
        """The band's arrays at the final iterate."""
        if self._final is None:
            raise RuntimeError("no iterate logged yet")
        shots = self.shots
        idx = torch.as_tensor(shots, device=self._final.u.device)
        return BandArrays(
            freqs=_single(self.freqs),
            c=_single(self._final.c_full[self._interior]),
            d_obs=_single(self.d_obs),
            d_syn_start=_single(self.d_syn_start),
            d_syn_final=_single(self.d_syn_final),
            pde_scale=_single(self.pde_scale),
            shots=np.asarray(shots, dtype=np.int64),
            u=_single(self._final.u.index_select(0, idx)),
            u_star=_single(self._u_star.index_select(0, idx.to(self._u_star.device))),
        )


def save_band(bands_dir: Path, arrays: BandArrays, summary: Dict[str, Any]) -> Path:
    """Write a completed band's directory under ``bands_dir`` and return it.

    ``summary`` is ``band_summary.json``, named by its ``band_index`` and
    ``label``. The directory is written under a hidden name and renamed into
    place once complete, so a failed or interrupted band leaves none.
    """
    band_dir = bands_dir / band_dir_name(summary["band_index"], summary["label"])
    tmp_dir = bands_dir / f".{band_dir.name}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        arrays.save(tmp_dir / BAND_ARRAYS_FILE)
        write_json(tmp_dir / BAND_SUMMARY_FILE, summary)
        tmp_dir.replace(band_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return band_dir


@dataclass
class SavedBand:
    """A completed band of a saved run: its ``band_summary.json`` and (lazily)
    its arrays."""

    index: int
    label: str
    path: Path
    summary: Dict[str, Any]

    @cached_property
    def arrays(self) -> BandArrays:
        return BandArrays.load(self.path / BAND_ARRAYS_FILE)

    @property
    def c(self) -> np.ndarray:
        """The interior velocity at the band's final iterate (reads only ``c``
        unless the arrays are already loaded)."""
        if "arrays" in self.__dict__:
            return self.arrays.c
        with np.load(self.path / BAND_ARRAYS_FILE) as z:
            return z["c"]


def load_bands(run_dir: PathLike) -> List[SavedBand]:
    """The run's completed bands (each with a ``band_summary.json``), in band
    order; their arrays are read lazily."""
    bands = []
    pattern = f"{BANDS_DIR}/{BAND_DIR_GLOB}/{BAND_SUMMARY_FILE}"
    for summary_path in Path(run_dir).glob(pattern):
        summary = json.loads(summary_path.read_text())
        bands.append(
            SavedBand(
                index=int(summary["band_index"]),
                label=str(summary["label"]),
                path=summary_path.parent,
                summary=summary,
            )
        )
    return sorted(bands, key=lambda b: b.index)


# --------------------------------------------------------------------------- #
# The whole run
# --------------------------------------------------------------------------- #
def write_run_summary(run_dir: Path, summary: Dict[str, Any]) -> None:
    """Write ``run_summary.json``, which every run ends with whatever its status
    (``completed | failed | interrupted``); the runner builds ``summary``."""
    write_json(run_dir / RUN_SUMMARY_FILE, summary)


def load_final_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """The run's final tape row from its ``run_summary.json``; ``None`` without
    the file or a completed band."""
    path = run_dir / RUN_SUMMARY_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text())["metrics_summary"]["final_metrics"]


def load_run_config(run_dir: PathLike) -> Dict[str, Any]:
    """Read ``config_resolved.yaml``, the run's Resolved config, as a dict;
    :class:`FileNotFoundError` if ``run_dir`` is no run directory."""
    path = Path(run_dir) / CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{str(run_dir)!r} is not a run directory (no {CONFIG_FILE})"
        )
    return load_config_file(path)


@dataclass
class SavedRun:
    """A read-only view of a run directory.

    ``problem`` is ``None`` if the run stopped before writing ``problem.npz``,
    and ``bands`` holds the completed bands in band order.
    """

    path: Path
    config: Dict[str, Any]
    tape: List[Dict[str, Any]]
    problem: Optional[ProblemArrays]
    bands: List[SavedBand]

    @classmethod
    def load(cls, run_dir: PathLike) -> "SavedRun":
        """Read the run directory ``run_dir``; :class:`FileNotFoundError` if it
        is none, :class:`SchemaVersionError` if its ``metadata.json`` is of a
        format this code can't read."""
        path = Path(run_dir)
        config = load_run_config(path)
        if (path / METADATA_FILE).exists():  # a run may stop before writing it
            load_versioned_json(path / METADATA_FILE)
        problem_path = path / PROBLEM_FILE
        return cls(
            path=path,
            config=config,
            tape=load_tape(path) if (path / "tape.csv").exists() else [],
            problem=ProblemArrays.load(problem_path) if problem_path.exists() else None,
            bands=load_bands(path),
        )

    @property
    def interior_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Interior node coordinates ``(x, y)`` in metres."""
        nx, ny = self.config["grid"]["interior_shape"]
        (x0, x1), (y0, y1) = self.config["grid"]["interior_extent"]
        return np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
