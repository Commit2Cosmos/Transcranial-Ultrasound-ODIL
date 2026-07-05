"""Stride forward reference for Shepp Logan (pure Stride API, no odil_wave).

Run from the stride conda env (repo root):

    conda activate stride
    bash stride/run_stride_forward_shepp.sh

On macOS, use the shell wrapper above — it sets Devito OpenMP paths for
Homebrew ``libomp`` *before* ``mrun`` spawns workers. Running the Python file
directly often fails with ``omp.h not found``.

Manual run (after configuring Devito env vars yourself):

    MPL_BACKEND=MacOSX mrun python scripts/stride_forward_shepp.py

Outputs observed traces to ``output/stride_forward/`` as ``shot_XXXX.npy``
(shape ``n_receivers x n_time``) plus ``grid_meta.json``.

Also exports full forward wavefields (with Stride absorbing layers) to
``output/stride_forward/wavefields/wavefield_XXXX.npy`` (shape ``n_time x nx x ny``)
for ODIL warm-start.
"""

from __future__ import annotations

import json
from pathlib import Path

import mosaic
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from stride import Problem, ScalarField, Space, Time, forward, IsoAcousticDevito
from stride.utils import wavelets

SOS_WATER = 1500.0
SOS_SOFT = 1540.0
SOS_SKULL = 2800.0

# Match ODIL (2nd-order FD). OT2 is auto-selected at this dt anyway unless OT4 is forced.
STRIDE_KERNEL = "OT4"
STRIDE_EXTRA = (30, 30)
STRIDE_ABSORBING = (20, 20)

OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "stride_forward"
EXPORT_WAVEFIELDS = True


async def export_wavefields(problem, vp_true, num: int) -> tuple[int, ...] | None:
    """Per-shot local forward with ``save_wavefield=True`` (absorbing BC active)."""
    wf_dir = OUT_DIR / "wavefields"
    wf_dir.mkdir(parents=True, exist_ok=True)
    pde = IsoAcousticDevito(grid=problem.grid)
    shape = None
    n_shots = len(problem.acquisitions.shots)
    print(f"Exporting Stride wavefields -> {wf_dir} ({n_shots} shots)")
    for i, shot in enumerate(problem.acquisitions.shots):
        sub = problem.sub_problem(shot.id)
        await pde(
            sub.shot.wavelets,
            vp_true,
            problem=sub,
            kernel=STRIDE_KERNEL,
            fw3d_mode=False,
            interpolation_type="hicks",
            save_wavefield=True,
            time_bounds=(0, num - 1),
            save_undersampling=1,
        )
        u = np.asarray(pde.wavefield.data, dtype=np.float32)
        np.save(wf_dir / f"wavefield_{shot.id:04d}.npy", u)
        shape = tuple(u.shape)
        pde.deallocate_wavefield(deallocate=True)
        pde.clear_operators()
        if i == 0 or (i + 1) % 8 == 0 or i + 1 == n_shots:
            print(f"  shot {shot.id:04d} ({i + 1}/{n_shots}) shape {u.shape}")
    return shape


def shepp_logan_sos(
    shape: tuple[int, int],
    *,
    phantom_scale: float = 0.90,
) -> np.ndarray:
    original = np.rot90(shepp_logan_phantom(), k=1).astype(np.float32)
    phantom_shape = (
        int(round(shape[0] * phantom_scale)),
        int(round(shape[1] * phantom_scale)),
    )
    smaller = resize(
        original, phantom_shape, anti_aliasing=True, mode="reflect", preserve_range=True
    ).astype(np.float32)
    smaller -= smaller.min()
    if smaller.max() > 0:
        smaller /= smaller.max()

    phantom = np.zeros(shape, dtype=np.float32)
    ox = (shape[0] - phantom_shape[0]) // 2
    oy = (shape[1] - phantom_shape[1]) // 2
    phantom[ox : ox + phantom_shape[0], oy : oy + phantom_shape[1]] = smaller

    model = np.full(shape, SOS_WATER, dtype=np.float32)
    outer_head = binary_fill_holes(phantom > 0.05)
    model[outer_head] = SOS_SOFT
    model[outer_head] = (1450.0 + 300.0 * phantom)[outer_head]

    erosion = max(2, int(round(0.02 * min(phantom_shape))))
    inner = binary_erosion(outer_head, iterations=erosion)
    model[outer_head & ~inner] = SOS_SKULL
    return model


async def main(runtime):
    interior_shape = (30, 30)
    domain_m = 0.25
    f_centre = 200e3
    n_cycles = 3
    n_receivers = 64

    spacing = (
        domain_m / (interior_shape[0] - 1),
        domain_m / (interior_shape[1] - 1),
    )
    dx = min(spacing)
    dt = 0.3 * dx / SOS_SKULL
    t_max = 2.0 * domain_m * np.sqrt(2.0) / SOS_WATER
    num = int(np.ceil(t_max / dt)) + 1

    space = Space(
        shape=interior_shape,
        extra=STRIDE_EXTRA,
        absorbing=STRIDE_ABSORBING,
        spacing=spacing,
    )
    time = Time(start=0.0, step=dt, num=num)
    problem = Problem(name="shepp_ref", space=space, time=time)

    vp_array = shepp_logan_sos(interior_shape)
    vp_true = ScalarField(name="vp", grid=problem.grid)
    vp_true.allocate()
    vp_true.fill(SOS_WATER)
    vp_true.data[:] = vp_array
    problem.medium.add(vp_true)

    problem.transducers.default()
    problem.geometry.default("elliptical", n_receivers)
    problem.acquisitions.default()

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(
            f_centre, n_cycles, time.num, time.step
        )

    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    await forward(
        problem,
        pde,
        vp_true,
        dump=True,
        kernel=STRIDE_KERNEL,
        fw3d_mode=False,
        interpolation_type="hicks",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "vp_true.npy", vp_array)

    transducer_xy = []
    for loc in problem.geometry.locations:
        xy = np.asarray(loc.coordinates[:2], dtype=np.float64)
        transducer_xy.append(xy.tolist())
    shot_source_ids = [
        int(shot.sources[0].id) for shot in problem.acquisitions.shots
    ]
    (OUT_DIR / "geometry.json").write_text(
        json.dumps(
            {
                "transducer_xy_m": transducer_xy,
                "shot_source_ids": shot_source_ids,
                "n_receivers": n_receivers,
                "n_shots": len(problem.acquisitions.shots),
            },
            indent=2,
        )
    )

    meta = {
        "interior_shape": list(interior_shape),
        "domain_m": domain_m,
        "spacing_m": list(spacing),
        "dt": dt,
        "num_time": num,
        "f_centre_hz": f_centre,
        "n_cycles": n_cycles,
        "n_receivers": n_receivers,
        "n_shots": len(problem.acquisitions.shots),
        "kernel": STRIDE_KERNEL,
        "stride_extra": list(STRIDE_EXTRA),
        "stride_absorbing": list(STRIDE_ABSORBING),
    }
    (OUT_DIR / "grid_meta.json").write_text(json.dumps(meta, indent=2))

    for shot in problem.acquisitions.shots:
        np.save(
            OUT_DIR / f"shot_{shot.id:04d}.npy",
            np.asarray(shot.observed.data, dtype=np.float32),
        )

    wf_shape = None
    if EXPORT_WAVEFIELDS:
        wf_shape = await export_wavefields(problem, vp_true, num)
        meta["wavefield"] = {
            "layout": "nt_nx_ny",
            "dir": "wavefields",
            "filename": "wavefield_{shot:04d}.npy",
            "shape": list(wf_shape) if wf_shape is not None else None,
            "stride_extent_m": [list(stride_wavefield_extent(meta)[0]),
                                list(stride_wavefield_extent(meta)[1])],
        }
        (OUT_DIR / "grid_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved {len(problem.acquisitions.shots)} shots to {OUT_DIR}")
    if wf_shape is not None:
        print(f"Saved wavefields shape {wf_shape} under {OUT_DIR / 'wavefields'}")


def stride_wavefield_extent(meta: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    domain_m = float(meta["domain_m"])
    dx, dy = float(meta["spacing_m"][0]), float(meta["spacing_m"][1])
    ex, ey = int(meta["stride_extra"][0]), int(meta["stride_extra"][1])
    return ((-ex * dx, domain_m + ex * dx), (-ey * dy, domain_m + ey * dy))


if __name__ == "__main__":
    mosaic.run(main)