"""The arrays a run saves: ``problem.npz`` once per run and ``arrays.npz`` per
completed band, as in the run-layer README's "Where each artifact is written".

Every array is saved as float32 / complex64 whatever ``runtime.dtype`` (masks
bool, indices int64), in an uncompressed ``.npz`` written atomically. Velocities
cover the interior only; the full-grid model is the interior padded with
``pml_c``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from odil_wave.optimisation import Iterate

# The fixed saved shots: the transducers nearest these ring angles.
SAVED_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)


def write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Atomically write ``arrays`` as an uncompressed ``.npz`` (tmp + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


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


def _single(x) -> np.ndarray:
    """A numpy copy of a tensor or array in float32 / complex64 (bool kept)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype == bool:
        return x
    return x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)


class BandArrays:
    """A band's ``arrays.npz``, gathered from its logged iterates.

    Call it with every :class:`~odil_wave.optimisation.Iterate` the band logs
    (alongside the tape's ``on_log``): the first is the start row's warm start,
    the last the band's final iterate. :meth:`save` then writes the file.
    """

    def __init__(self, problem, band) -> None:
        """
        Parameters
        ----------
        problem : Problem
            The run's problem (inversion and observation grids).
        band : BandContext
            The band's frequencies, geometry and observed wavefields ``u*``.
        """
        self.geom = band.geom
        self.freqs = band.freq.frequencies
        self._interior = problem.grid.interior_slice
        u_star = torch.stack([wf.amplitude for wf in band.observed_wfs]).detach()
        if band.observed_traces is not None:
            self.d_obs = band.observed_traces.detach()
        else:
            self.d_obs = self.geom.extract_observations(u_star)
        obs_sl = problem.observation_grid.interior_slice
        self._u_star = u_star[:, :, obs_sl[0], obs_sl[1]]
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

    def save(self, path: Path) -> None:
        """Write ``arrays.npz`` at the final iterate."""
        if self._final is None:
            raise RuntimeError("no iterate logged yet")
        sl = self._interior
        shots = self.shots
        idx = torch.as_tensor(shots, device=self._final.u.device)
        write_npz(
            path,
            {
                "freqs": _single(self.freqs),
                "c": _single(self._final.c_full[sl]),
                "d_obs": _single(self.d_obs),
                "d_syn_start": _single(self.d_syn_start),
                "d_syn_final": _single(self.d_syn_final),
                "shots": np.asarray(shots, dtype=np.int64),
                "u": _single(self._final.u.index_select(0, idx)),
                "u_star": _single(
                    self._u_star.index_select(0, idx.to(self._u_star.device))
                ),
            },
        )


def save_problem(path: Path, problem, brain: np.ndarray, skull: np.ndarray) -> None:
    """Write the run's ``problem.npz``.

    Parameters
    ----------
    path : Path
        Destination ``problem.npz``.
    problem : Problem
        The run's problem (truth, starting model, grid, acquisition).
    brain, skull : numpy.ndarray
        Interior-shaped boolean brain and skull regions.
    """
    grid = problem.grid
    sl = grid.interior_slice
    write_npz(
        path,
        {
            "c_star": _single(problem.truth_velocity.c[sl]),
            "c0": _single(problem.init_velocity.c[sl]),
            "pml_c": np.float32(problem.init_velocity.pml_c),
            "brain_mask": np.asarray(brain, dtype=bool),
            "skull_mask": np.asarray(skull, dtype=bool),
            "transducer_angles_deg": transducer_angles_deg(
                problem.cfg.acquisition.n_transducers
            ),
            "interior_slice": np.array([[s.start, s.stop] for s in sl], dtype=np.int64),
        },
    )
