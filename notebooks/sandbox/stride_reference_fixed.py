"""Helpers for Stride reference forward + FWI (sandbox only, not src/)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# SoS table [m/s]
SOS_WATER = 1500.0
SOS_SOFT = 1540.0
SOS_SKULL = 2800.0

# confirm git pull worked.
REF_VERSION = 5


def configure_devito() -> str:
    """Set Devito options before import stride.

    Stride defaults to OpenMP kernels. On macOS, Apple clang has no omp.h,
    and linking Homebrew libomp often clashes with conda's OpenMP (crash or
    CompileError). We therefore use serial C kernels here - correct, just
    slower.

    Returns a short label describing which path was chosen.
    """
    import platform

    if platform.system() == "Darwin":
        os.environ["DEVITO_LANGUAGE"] = "C"
        for key in ("CFLAGS", "CXXFLAGS", "LDFLAGS"):
            os.environ.pop(key, None)
        return "serial C (macOS — avoids omp.h / libomp clash)"

    for prefix in (Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")):
        if (prefix / "include" / "omp.h").is_file():
            inc = str(prefix / "include")
            lib = str(prefix / "lib")
            omp_flags = f"-Xpreprocessor -fopenmp -I{inc}"
            os.environ["CFLAGS"] = omp_flags
            os.environ["CXXFLAGS"] = omp_flags
            os.environ["LDFLAGS"] = f"-L{lib} -lomp"
            os.environ.setdefault("DEVITO_LANGUAGE", "openmp")
            return f"openmp ({prefix})"

    os.environ["DEVITO_LANGUAGE"] = "C"
    return "serial C (no libomp found)"


async def ensure_mosaic_runtime(num_workers: int = 1, log_level: str = "info"):
    """Shut down and restart Mosaic (fixes ``async_for cannot be nested``)."""
    import mosaic

    try:
        await mosaic.interactive("off")
    except Exception:
        pass
    mosaic.clear_runtime()
    await mosaic.interactive("on", num_workers=num_workers, log_level=log_level)
    runtime = mosaic.runtime()
    if runtime is None:
        raise RuntimeError("Mosaic runtime failed to start")
    return runtime


def set_medium_vp(problem, vp) -> None:
    """Register inversion velocity field on the problem medium."""
    problem.medium.add(vp)  # replaces existing field with the same name


def inversion_vp_from_array(problem, array_2d: np.ndarray, *, name: str = "vp"):
    """Create a fresh Stride inversion parameter from an interior NumPy array."""
    from stride import ScalarField

    vp = ScalarField.parameter(name=name, grid=problem.grid, needs_grad=True)
    fill_vp_from_numpy(vp, np.asarray(array_2d, dtype=np.float32))
    return vp


def perfect_skull_start(
    true_model: np.ndarray,
    *,
    c_water: float = SOS_WATER,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return water background + exact skull values, and the skull mask.

    The threshold defaults to the midpoint between soft tissue and skull speeds.
    This keeps the skull exact while deliberately removing all soft tissue truth
    from the starting model.
    """
    true_model = np.asarray(true_model, dtype=np.float32)
    if threshold is None:
        threshold = 0.5 * (SOS_SOFT + SOS_SKULL)

    skull_mask = true_model >= float(threshold)
    start = np.full(true_model.shape, c_water, dtype=np.float32)
    start[skull_mask] = true_model[skull_mask]
    return start, skull_mask


def inversion_vp_from_true(problem, vp_true):
    """Deprecated: a full-truth start is not a perfect-skull start."""
    raise RuntimeError(
        "Do not initialise FWI from the complete true model. "
        "Use perfect_skull_start(...) followed by inversion_vp_from_array(...)."
    )


def project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "src").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return path.parents[2]


def reference_dir(resolution: int) -> Path:
    out = project_root() / "inputs" / "reference" / str(resolution)
    out.mkdir(parents=True, exist_ok=True)
    return out


from scipy.ndimage import binary_erosion, binary_fill_holes

def shepp_logan_sos(
    shape: tuple[int, int],
    c_water: float = SOS_WATER,
    c_soft: float = SOS_SOFT,
    c_skull: float = SOS_SKULL,
) -> np.ndarray:
    """Shepp–Logan interior with skull only on the outer head boundary."""

    phantom = np.rot90(
        shepp_logan_phantom(),
        k=1,
    ).astype(np.float32)

    phantom = resize(
        phantom,
        shape,
        anti_aliasing=True,
        mode="reflect",
        preserve_range=True,
    ).astype(np.float32)

    phantom -= phantom.min()
    if phantom.max() > 0:
        phantom /= phantom.max()

    model = np.full(
        shape,
        c_water,
        dtype=np.float32,
    )

    # Outer head support only
    outer_head = phantom > 0.05
    outer_head = binary_fill_holes(outer_head)

    # Fill the whole head with soft tissue first
    model[outer_head] = c_soft

    # Add Shepp Logan internal sound-speed variations
    interior_values = 1450.0 + 300.0 * phantom
    model[outer_head] = interior_values[outer_head]

    # Build skull only from the outer head boundary
    erosion_pixels = max(
        2,
        int(round(0.02 * min(shape))),
    )

    inner_head = binary_erosion(
        outer_head,
        iterations=erosion_pixels,
    )

    skull_mask = outer_head & ~inner_head
    model[skull_mask] = c_skull

    return model


def water_model(shape: tuple[int, int], c_water: float = SOS_WATER) -> np.ndarray:
    return np.full(shape, c_water, dtype=np.float32)


def cfl_dt(spacing_m: float, c_max: float, safety: float = 0.3) -> float:
    """2nd-order acoustic CFL: dt < safety * dx / c_max."""
    return safety * spacing_m / c_max


def fill_vp_from_numpy(vp, array_2d: np.ndarray, exterior: float = SOS_WATER) -> None:
    """Write interior SoS into a Stride ScalarField (including PML padding)."""
    if array_2d.shape != vp.grid.space.shape:
        raise ValueError(
            f"expected interior {vp.grid.space.shape}, got array {array_2d.shape}"
        )
    vp.allocate()
    vp.fill(exterior)
    vp.data[:] = np.asarray(array_2d, dtype=np.float32)
    check_vp_finite(vp)


def check_vp_finite(vp, name: str = "vp") -> None:
    """Raise if velocity field contains NaN/Inf (common setup failure)."""
    data = np.asarray(vp.extended_data)
    if not np.isfinite(data).all():
        bad = np.count_nonzero(~np.isfinite(data))
        raise ValueError(f"{name}: {bad} non-finite values in extended_data")


def save_vp_npy(vp, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(vp.data, dtype=np.float32))


def save_vp_npy_from_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array, dtype=np.float32))


def save_grid_meta(problem, path: Path, *, f_centre: float, n_cycles: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": problem.name,
        "interior_shape": list(problem.space.shape),
        "spacing_m": list(problem.space.spacing),
        "time_step_s": problem.time.step,
        "time_num": problem.time.num,
        "f_centre_hz": f_centre,
        "n_cycles": n_cycles,
        "output_folder": str(problem.output_folder),
    }
    path.write_text(json.dumps(meta, indent=2))


def export_shot_observed_npy(problem, out_dir: Path) -> None:
    """Save observed traces per shot as (n_receivers, n_time) .npy files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for shot in problem.acquisitions.shots:
        if shot.observed is None or not shot.observed.allocated:
            continue
        data = np.asarray(shot.observed.data, dtype=np.float32)
        np.save(out_dir / f"shot_{shot.id:04d}_observed.npy", data)


def time_axis(problem) -> np.ndarray:
    """Physical time samples for a Stride problem."""
    return problem.time.start + np.arange(problem.time.num, dtype=np.float64) * problem.time.step


def transfer_observed_to_problem(
    source_problem,
    target_problem,
    *,
    check_geometry: bool = True,
) -> None:
    """Resample fine-grid observed traces onto another Stride problem's time grid.

    Both problems must describe the same acquisition ordering. Spatial receiver
    coordinates remain physical coordinates; only the trace time axis is
    interpolated. The target shot.observed objects are allocated in place.
    """
    source_shots = list(source_problem.acquisitions.shots)
    target_shots = list(target_problem.acquisitions.shots)

    if len(source_shots) != len(target_shots):
        raise ValueError(
            f"shot-count mismatch: source={len(source_shots)}, target={len(target_shots)}"
        )

    t_source = time_axis(source_problem)
    t_target = time_axis(target_problem)

    for source_shot, target_shot in zip(source_shots, target_shots):
        if source_shot.id != target_shot.id:
            raise ValueError(
                f"shot ordering mismatch: source id={source_shot.id}, "
                f"target id={target_shot.id}"
            )
        if source_shot.observed is None or not source_shot.observed.allocated:
            raise RuntimeError(f"source shot {source_shot.id} has no observed data")

        source_data = np.asarray(source_shot.observed.data, dtype=np.float32)
        if source_data.ndim != 2:
            raise ValueError(
                f"shot {source_shot.id}: expected 2-D traces, got {source_data.shape}"
            )

        if target_shot.observed is None:
            raise RuntimeError(
                f"target shot {target_shot.id} has no observed trace container"
            )
        if not target_shot.observed.allocated:
            target_shot.observed.allocate()

        target_data = np.asarray(target_shot.observed.data)
        if source_data.shape[0] != target_data.shape[0]:
            raise ValueError(
                f"shot {source_shot.id}: receiver-count mismatch "
                f"{source_data.shape[0]} != {target_data.shape[0]}"
            )

        if check_geometry:
            if list(source_shot.receiver_ids) != list(target_shot.receiver_ids):
                raise ValueError(f"shot {source_shot.id}: receiver ordering differs")
            if list(source_shot.source_ids) != list(target_shot.source_ids):
                raise ValueError(f"shot {source_shot.id}: source ordering differs")

        for receiver_id in range(source_data.shape[0]):
            target_data[receiver_id, :] = np.interp(
                t_target,
                t_source,
                source_data[receiver_id, :],
                left=0.0,
                right=0.0,
            ).astype(np.float32)

    check_forward_ready(target_problem)


def observed_trace_difference(source_problem, target_problem, shot_id: int = 0) -> dict:
    """Simple diagnostics after resampling observations."""
    source = np.asarray(source_problem.acquisitions.get(shot_id).observed.data)
    target = np.asarray(target_problem.acquisitions.get(shot_id).observed.data)
    return {
        "source_shape": tuple(source.shape),
        "target_shape": tuple(target.shape),
        "source_max_abs": float(np.max(np.abs(source))),
        "target_max_abs": float(np.max(np.abs(target))),
    }


def downsample_vp(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return resize(array, target_shape, anti_aliasing=True, mode="reflect").astype(
        np.float32
    )


def count_observed_shots(problem) -> int:
    """Number of shots with allocated observed traces (after ``acquisitions.load``)."""
    n = 0
    for shot in problem.acquisitions.shots:
        if shot.observed is not None and shot.observed.allocated:
            n += 1
    return n


def check_forward_ready(problem, *, min_shots: int | None = None) -> int:
    """Raise if forward observed data is missing, return number of ready shots."""
    n = count_observed_shots(problem)
    need = min_shots if min_shots is not None else len(problem.acquisitions.shots)
    if n < need:
        raise RuntimeError(
            f"Forward not ready: {n}/{len(problem.acquisitions.shots)} shots have "
            f"observed data (need {need}). Re-run forward after a kernel restart."
        )
    print(f"Forward OK: {n}/{len(problem.acquisitions.shots)} shots with observed")
    return n


def domain_extent_mm(domain_m: float) -> tuple[float, float, float, float]:
    """Imshow extent (x0, x1, y0, y1) in mm for a square domain."""
    mm = domain_m * 1000.0
    return (0.0, mm, 0.0, mm)


def plot_vp(
    vp,
    *,
    domain_m=0.25,
    title="vp",
    ax=None,
    show=True,
    vmin=None,
    vmax=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.asarray(vp.data if hasattr(vp, "data") else vp)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=(0, domain_m * 1000, 0, domain_m * 1000),
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, label="m/s")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_vp_overview(
    problem,
    vp,
    *,
    domain_m: float = 0.25,
    shot_id: int = 0,
):
    """Static overview: vp + transducers + shot wavelet (replaces problem.plot())."""
    import matplotlib.pyplot as plt

    data = np.asarray(vp.data if hasattr(vp, "data") else vp)
    shot = problem.acquisitions.get(shot_id)
    ext = domain_extent_mm(domain_m)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im = axes[0].imshow(data.T, origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="m/s", fraction=0.046)
    axes[0].set_title("vp")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")

    for loc_id in shot.receiver_ids:
        c = problem.geometry.get(loc_id).coordinates * 1000.0
        axes[0].scatter(
            c[0],
            c[1],
            c="white",
            edgecolors="black",
            s=12,
            linewidths=0.5,
            zorder=3,
        )
    for loc_id in shot.source_ids:
        c = problem.geometry.get(loc_id).coordinates * 1000.0
        axes[0].scatter(
            c[0],
            c[1],
            c="red",
            marker="*",
            s=100,
            zorder=4,
            label="source",
        )
    axes[0].legend(loc="upper right", fontsize=8)

    t = np.arange(problem.time.num) * problem.time.step
    w = np.asarray(shot.wavelets.data[0, :])
    axes[1].plot(t * 1e6, w, "k-", lw=1)
    axes[1].set_xlabel("time (µs)")
    axes[1].set_ylabel("amplitude")
    axes[1].set_title(f"shot {shot_id} wavelet")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def build_stride_problem(
    *,
    name: str,
    interior_shape: tuple[int, int],
    domain_m: float = 0.25,
    extra: tuple[int, int] = (20, 20),
    absorbing: tuple[int, int] = (20, 20),
    f_centre: float = 200e3,
    n_cycles: int = 3,
    n_receivers: int = 64,
    c_max: float = SOS_SKULL,
    use_shepp_logan: bool = True,
):
    """Create Stride Problem with Shepp-Logan vp_true (or water box)."""
    from stride import Problem, ScalarField, Space, Time
    from stride.utils import wavelets

    spacing = (
        domain_m / (interior_shape[0] - 1),
        domain_m / (interior_shape[1] - 1),
    )
    dx = min(spacing)
    dt = cfl_dt(dx, c_max)
    diag = domain_m * np.sqrt(2.0)
    t_max = 2.0 * diag / SOS_WATER
    num = int(np.ceil(t_max / dt)) + 1

    space = Space(
        shape=interior_shape,
        extra=extra,
        absorbing=absorbing,
        spacing=spacing,
    )
    time = Time(start=0.0, step=dt, num=num)
    problem = Problem(name=name, space=space, time=time)

    if use_shepp_logan:
        vp_array = shepp_logan_sos(interior_shape)
    else:
        vp_array = water_model(interior_shape)

    vp_true = ScalarField(name="vp", grid=problem.grid)
    fill_vp_from_numpy(vp_true, vp_array)
    check_vp_finite(vp_true, "vp_true")
    problem.medium.add(vp_true)

    problem.transducers.default()
    problem.geometry.default("elliptical", n_receivers)
    problem.acquisitions.default()

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(
            f_centre, n_cycles, time.num, time.step
        )

    return problem, vp_true, vp_array, dict(
        f_centre=f_centre,
        n_cycles=n_cycles,
        spacing=spacing,
        dt=dt,
    )
