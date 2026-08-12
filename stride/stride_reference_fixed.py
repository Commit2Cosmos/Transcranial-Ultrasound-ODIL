"""Helpers for Stride reference forward + FWI (sandbox only, not src/)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# SoS table [m/s]
SOS_WATER = 1500.0
SOS_SOFT = 1540.0
SOS_SKULL = 3000.0

# confirm git pull worked.
REF_VERSION = 10

# Defaults matched to odil_wave Shepp–Logan experiment configs
# (e.g. shepp_logan scale=0.85, a_frac=b_frac=0.9, t_max=500 µs).
SHEPP_PHANTOM_SCALE = 0.85
RING_FRAC = 0.9
T_MAX_S = 500e-6
C_CLAMP_MIN = 1300.0
C_CLAMP_MAX = 3100.0
# Absorbing layer width in cells (odil_wave PML split).
# Stride: ``absorbing`` ≈ PML thickness; ``extra`` is the padded halo and
# should be a bit larger than absorbing (forward 50/40, inverse 70/60).
PML_FWD = 40
PML_INV = 60
EXTRA_FWD = 50
EXTRA_INV = 70
STRIDE_EXTRA = (EXTRA_INV, EXTRA_INV)
STRIDE_ABSORBING = (PML_INV, PML_INV)
STRIDE_EXTRA_FWD = (EXTRA_FWD, EXTRA_FWD)
STRIDE_ABSORBING_FWD = (PML_FWD, PML_FWD)


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


def _shepp_logan_phantom_and_masks(
    shape: tuple[int, int],
    *,
    phantom_scale: float = SHEPP_PHANTOM_SCALE,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalised Shepp–Logan phantom + head / interior / rim masks.

    Mirrors ``odil_wave.models.VelocityModel._shepp_logan_interior_phantom``.
    Returns ``(phantom, head, inner, rim)``.
    """
    original = np.rot90(shepp_logan_phantom(), k=1).astype(np.float32)
    phantom_shape = (
        max(2, int(round(shape[0] * phantom_scale))),
        max(2, int(round(shape[1] * phantom_scale))),
    )
    smaller = resize(
        original,
        phantom_shape,
        anti_aliasing=True,
        mode="reflect",
        preserve_range=True,
    ).astype(np.float32)
    smaller -= smaller.min()
    if smaller.max() > 0:
        smaller /= smaller.max()

    phantom = np.zeros(shape, dtype=np.float32)
    ox = (shape[0] - phantom_shape[0]) // 2
    oy = (shape[1] - phantom_shape[1]) // 2
    phantom[ox : ox + phantom_shape[0], oy : oy + phantom_shape[1]] = smaller

    head = binary_fill_holes(phantom > threshold)
    erosion_pixels = max(2, int(round(0.02 * min(phantom_shape))))
    inner = binary_erosion(head, iterations=erosion_pixels)
    rim = head & ~inner
    return phantom, head, inner, rim


def shepp_logan_sos(
    shape: tuple[int, int],
    c_water: float = SOS_WATER,
    c_soft: float = SOS_SOFT,
    c_skull: float = SOS_SKULL,
    phantom_scale: float = SHEPP_PHANTOM_SCALE,
) -> np.ndarray:
    """Full Shepp–Logan SoS (soft tissue + hard skull rim), odil ``shepp_logan``."""
    del c_soft  # kept for API compatibility; soft tissue uses 1450 + 300·phantom
    phantom, head, _inner, rim = _shepp_logan_phantom_and_masks(
        shape, phantom_scale=phantom_scale
    )
    model = np.full(shape, c_water, dtype=np.float32)
    model[head] = (1450.0 + 300.0 * phantom)[head]
    model[rim] = c_skull
    return model


def shepp_logan_skull_sos(
    shape: tuple[int, int],
    *,
    c_water: float = SOS_WATER,
    c_skull: float = SOS_SKULL,
    phantom_scale: float = SHEPP_PHANTOM_SCALE,
    threshold: float = 0.05,
    interior_value: float | None = None,
    skull_alpha: float = 1.0,
    skull_sigma: float = 0.0,
) -> np.ndarray:
    """Water + skull rim only (odil ``shepp_logan_skull``).

    Perfect-skull geometry with optional contrast scaling / Gaussian smoothing:
    ``c = c_water + alpha * G_σ(c_perfect - c_water)``.

    * ``skull_alpha`` in ``[0, 1]`` — ``1`` full rim, ``0`` all water.
    * ``skull_sigma`` ≥ 0 — Gaussian σ in **grid cells** (try ``1``–``3``);
      ``0`` keeps a sharp rim.
    """
    if skull_alpha < 0.0 or skull_alpha > 1.0:
        raise ValueError(f"skull_alpha must be in [0, 1]; got {skull_alpha}")
    if skull_sigma < 0.0:
        raise ValueError(f"skull_sigma must be >= 0; got {skull_sigma}")

    phantom, _head, inner, rim = _shepp_logan_phantom_and_masks(
        shape, phantom_scale=phantom_scale, threshold=threshold
    )
    del phantom  # geometry only; no soft-tissue features

    fill_interior = c_water if interior_value is None else float(interior_value)
    model = np.full(shape, c_water, dtype=np.float32)
    if fill_interior != c_water:
        model[inner] = fill_interior
    model[rim] = c_skull

    cw = np.float32(c_water)
    contrast = model - cw
    if skull_sigma > 0.0:
        contrast = gaussian_filter(contrast, sigma=skull_sigma)
    model = cw + np.float32(skull_alpha) * contrast
    return model.astype(np.float32)

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


def head_mask(
    shape: tuple[int, int],
    *,
    phantom_scale: float = SHEPP_PHANTOM_SCALE,
    threshold: float = 0.05,
) -> np.ndarray:
    """Boolean head ROI matching odil ``ssim`` mask ``head_roi``."""
    _phantom, head, _inner, _rim = _shepp_logan_phantom_and_masks(
        shape, phantom_scale=phantom_scale, threshold=threshold
    )
    return head


def evaluate_vp_vs_true(
    pred,
    true: np.ndarray,
    *,
    head_roi: np.ndarray | None = None,
    phantom_scale: float = SHEPP_PHANTOM_SCALE,
) -> dict:
    """Per-model metrics vs truth (interior arrays), ODIL-style.

    Returns ``rms``, ``rel_c_error`` (= ‖Δc‖₂ / ‖c_true‖₂), and
    ``ssim_head_roi`` (skimage SSIM averaged over the head mask).
    """
    from skimage.metrics import structural_similarity as sk_ssim

    p = np.asarray(
        pred.data if hasattr(pred, "data") else pred, dtype=np.float64
    )
    t = np.asarray(true, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs true {t.shape}")

    rms = float(np.sqrt(np.mean((p - t) ** 2)))
    denom = float(np.linalg.norm(t))
    rel_c_error = float(np.linalg.norm(p - t) / denom) if denom > 0 else None

    if head_roi is None:
        head_roi = head_mask(t.shape, phantom_scale=phantom_scale)
    m = np.asarray(head_roi, dtype=bool)
    ssim_val = None
    ssim_reason = None
    if m.shape != t.shape or not np.any(m):
        ssim_reason = "empty or mismatched head mask"
    else:
        try:
            data_range = float(t[m].max() - t[m].min())
            if data_range <= 0:
                ssim_reason = "zero data_range inside head mask"
            else:
                _, S = sk_ssim(p, t, full=True, data_range=data_range)
                ssim_val = float(S[m].mean())
                if not np.isfinite(ssim_val):
                    ssim_val = None
                    ssim_reason = "ssim non-finite"
        except Exception as exc:  # noqa: BLE001 — surface reason in metrics
            ssim_reason = f"ssim error: {type(exc).__name__}: {exc}"

    return {
        "rms": rms,
        "rel_c_error": rel_c_error,
        "ssim_head_roi": ssim_val,
        "ssim_reason": ssim_reason,
        "c_min": float(p.min()),
        "c_max": float(p.max()),
        "c_mean": float(p.mean()),
        "c_std": float(p.std()),
    }


class BandRunRecorder:
    """Save per-band vp snapshots + metrics (ODIL ``bands/`` + ``metrics.jsonl``)."""

    def __init__(
        self,
        out_dir: Path | str,
        true_model: np.ndarray,
        *,
        phantom_scale: float = SHEPP_PHANTOM_SCALE,
        run_id: str = "stride_fwi",
    ):
        self.out_dir = Path(out_dir)
        self.bands_dir = self.out_dir / "bands"
        self.bands_dir.mkdir(parents=True, exist_ok=True)
        self.true_model = np.asarray(true_model, dtype=np.float32)
        self.phantom_scale = float(phantom_scale)
        self.head_roi = head_mask(
            self.true_model.shape, phantom_scale=self.phantom_scale
        )
        self.run_id = run_id
        self.jsonl_path = self.out_dir / "metrics.jsonl"
        self.csv_path = self.out_dir / "metrics.csv"
        self.rows: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.frame_labels: list[str] = []
        # truncate previous metrics for a fresh run
        self.jsonl_path.write_text("")
        if self.csv_path.is_file():
            self.csv_path.unlink()

    def record(
        self,
        *,
        band_index: int,
        frequency_hz: float | None,
        vp,
        label: str | None = None,
    ) -> dict:
        arr = np.asarray(
            vp.data if hasattr(vp, "data") else vp, dtype=np.float32
        )
        metrics = evaluate_vp_vs_true(
            arr,
            self.true_model,
            head_roi=self.head_roi,
            phantom_scale=self.phantom_scale,
        )
        if label is None:
            if frequency_hz is None:
                label = "initial"
            else:
                label = f"{frequency_hz / 1e3:.0f}kHz"
        band_name = f"band_{band_index:02d}_{label}"
        band_dir = self.bands_dir / band_name
        band_dir.mkdir(parents=True, exist_ok=True)
        np.save(band_dir / "c_final.npy", arr)
        meta = {
            "band_index": int(band_index),
            "label": label,
            "frequency_hz": None if frequency_hz is None else float(frequency_hz),
            **metrics,
        }
        (band_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        row = {
            "run_id": self.run_id,
            "band_index": int(band_index),
            "band_frequencies_hz": (
                [] if frequency_hz is None else [float(frequency_hz)]
            ),
            "label": label,
            **metrics,
        }
        self.rows.append(row)
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self._rewrite_csv()

        self.frames.append(arr.copy())
        self.frame_labels.append(f"band {band_index}: {label}")
        return row

    def _rewrite_csv(self) -> None:
        if not self.rows:
            return
        keys = list(self.rows[0].keys())
        lines = [",".join(keys)]
        for row in self.rows:
            vals = []
            for k in keys:
                v = row.get(k)
                if isinstance(v, list):
                    vals.append(" ".join(str(x) for x in v))
                elif v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            lines.append(",".join(vals))
        self.csv_path.write_text("\n".join(lines) + "\n")

    def save_final(self, vp) -> Path:
        final_dir = self.out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        path = final_dir / "c_final.npy"
        if hasattr(vp, "data"):
            save_vp_npy(vp, path)
        else:
            save_vp_npy_from_array(vp, path)
        return path

    def write_gif(
        self,
        *,
        domain_m: float = 0.25,
        filename: str = "c_bands.gif",
        fps: int = 2,
        title: str = "Recovered vp across bands",
    ) -> Path:
        return animate_band_gif(
            self.frames,
            self.frame_labels,
            self.out_dir / filename,
            domain_m=domain_m,
            fps=fps,
            title=title,
        )


def load_band_history(run_dir: Path | str) -> dict:
    """Load per-band metrics written by :class:`BandRunRecorder`.

    Reads ``metrics.jsonl`` under ``run_dir`` and returns plot-ready series:
    ``rms``, ``rel_c_error``, ``ssim_head_roi``, ``labels``, and ``gif`` path.
    """
    run_dir = Path(run_dir)
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labels: list[str] = []
    for r in rows:
        lab = r.get("label")
        if not lab:
            freqs = r.get("band_frequencies_hz") or []
            lab = "Initial" if not freqs else f"{float(freqs[0]) / 1e3:.0f} kHz"
        elif lab == "initial":
            lab = "Initial"
        elif isinstance(lab, str) and lab.endswith("kHz") and " " not in lab:
            lab = lab.replace("kHz", " kHz")
        labels.append(str(lab))
    return {
        "rms": [r["rms"] for r in rows],
        "rel_c_error": [r.get("rel_c_error") for r in rows],
        "ssim_head_roi": [r.get("ssim_head_roi") for r in rows],
        "labels": labels,
        "gif": run_dir / "c_bands.gif",
        "rows": rows,
    }


def _load_final_vp_array(run_dir: Path) -> np.ndarray:
    """Best available recovered velocity under a BandRunRecorder run dir."""
    for cand in (
        run_dir / "vp_recovered.npy",
        run_dir / "final" / "c_final.npy",
    ):
        if cand.is_file():
            return np.load(cand)
    bands = sorted((run_dir / "bands").glob("band_*/c_final.npy"))
    if bands:
        return np.load(bands[-1])
    raise FileNotFoundError(f"no recovered vp under {run_dir}")


def plot_recovery_progress(
    run_dir: Path | str,
    *,
    domain_m: float = 0.25,
    title: str | None = None,
    vmin: float = 1400.0,
    vcenter: float = 1600.0,
    vmax: float = 3000.0,
    skip_initial: bool = True,
    figsize: tuple[float, float] = (12, 4.5),
    show: bool = True,
):
    """ODIL ``LiveVelocityView``-style figure from a finished Stride FWI run.

    Left: recovered ``c`` (final / last band). Right: per-band head-ROI SSIM
    and relative c-error read from ``metrics.jsonl``. Safe to call on another
    machine as long as ``run_dir`` points at the saved artifacts.
    """
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    hist = load_band_history(run_dir)
    rows = list(hist["rows"])
    if skip_initial:
        rows = [
            r
            for r in rows
            if str(r.get("label", "")).lower() not in ("initial", "init")
        ]
    if not rows:
        raise RuntimeError(f"no band metrics in {run_dir / 'metrics.jsonl'}")

    c = _load_final_vp_array(run_dir)
    # Prefer the last band snapshot if it exists (matches live-view “current band”).
    last = rows[-1]
    band_idx = int(last.get("band_index", len(rows) - 1))
    lab = str(last.get("label", ""))
    band_glob = list((run_dir / "bands").glob(f"band_{band_idx:02d}_*/c_final.npy"))
    if band_glob:
        c = np.load(band_glob[0])

    tick_labels = [f"{int(r.get('band_index', i))}:{r.get('label', i)}" for i, r in enumerate(rows)]
    ssim = [r.get("ssim_head_roi") for r in rows]
    rel_err = [r.get("rel_c_error") for r in rows]
    last_ssim = ssim[-1]

    norm = sos_norm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    extent = domain_extent_mm(domain_m)

    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=figsize)
    im = ax_c.imshow(np.asarray(c).T, origin="lower", extent=extent, cmap="viridis", norm=norm)
    ax_c.set_xlabel("x [mm]")
    ax_c.set_ylabel("y [mm]")
    ax_c.set_aspect("equal")
    ssim_s = f"  SSIM={last_ssim:.3f}" if isinstance(last_ssim, (int, float)) else ""
    ax_c.set_title(f"recovered c — band {band_idx} ({lab}){ssim_s}")
    plt.colorbar(im, ax=ax_c, shrink=0.85, label="c [m/s]")

    x = list(range(len(rows)))
    if any(s is not None for s in ssim):
        ax_m.plot(
            x,
            [np.nan if s is None else s for s in ssim],
            "o-",
            color="tab:green",
            label="SSIM (head ROI)",
        )
        ax_m.set_ylabel("SSIM (head ROI)", color="tab:green")
        ax_m.tick_params(axis="y", labelcolor="tab:green")
    if any(e is not None for e in rel_err):
        ax_r = ax_m.twinx()
        ax_r.plot(
            x,
            [np.nan if e is None else e for e in rel_err],
            "s--",
            color="tab:red",
            label="rel c-error",
        )
        ax_r.set_ylabel("rel c-error", color="tab:red")
        ax_r.tick_params(axis="y", labelcolor="tab:red")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_m.set_xlabel("completed band")
    ax_m.set_title("progress per band")
    ax_m.grid(alpha=0.3)

    sup = title or f"Inversion progress — {run_dir.name}"
    fig.suptitle(sup)
    fig.tight_layout()
    if show:
        plt.show()
    return fig






def animate_band_gif(
    frames: list[np.ndarray],
    labels: list[str],
    out_path: Path | str,
    *,
    domain_m: float = 0.25,
    fps: int = 2,
    title: str = "Recovered vp across bands",
    vmin: float = 1400.0,
    vcenter: float = 1600.0,
    vmax: float = 3000.0,
) -> Path:
    """Write an ODIL-style ``c_bands.gif`` from per-band velocity arrays."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not frames:
        raise RuntimeError("no frames to animate")
    if len(labels) != len(frames):
        raise ValueError("labels length must match frames")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = sos_norm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    extent = domain_extent_mm(domain_m)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(
        frames[0].T, origin="lower", extent=extent, cmap="viridis", norm=norm
    )
    title_txt = ax.set_title(f"{title}\n{labels[0]}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="m/s")
    fig.tight_layout()

    def _update(i):
        im.set_data(frames[i].T)
        title_txt.set_text(f"{title}\n{labels[i]}")
        return (im, title_txt)

    anim = FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 / max(fps, 1), blit=False
    )
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


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


def load_grid_meta(path: Path | str) -> dict:
    """Load a ``grid_meta_*.json`` written by :func:`save_grid_meta`."""
    return json.loads(Path(path).read_text())


def time_axis_from_meta(meta: dict, *, start: float = 0.0) -> np.ndarray:
    """Build a physical time axis [s] from saved grid metadata."""
    dt = float(meta["time_step_s"])
    num = int(meta["time_num"])
    return float(start) + np.arange(num, dtype=np.float64) * dt


def load_observed_shot(out_dir: Path | str, shot_id: int) -> np.ndarray:
    """Load one ``shot_XXXX_observed.npy`` exported by :func:`export_shot_observed_npy`."""
    path = Path(out_dir) / f"shot_{int(shot_id):04d}_observed.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def load_observed_shots(out_dir: Path | str) -> dict[int, np.ndarray]:
    """Load all ``shot_*_observed.npy`` files under ``out_dir`` keyed by shot id."""
    out_dir = Path(out_dir)
    shots: dict[int, np.ndarray] = {}
    for path in sorted(out_dir.glob("shot_*_observed.npy")):
        # shot_0012_observed.npy
        try:
            shot_id = int(path.name.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"unexpected observed filename: {path.name}") from exc
        shots[shot_id] = np.load(path)
    if not shots:
        raise FileNotFoundError(f"no shot_*_observed.npy under {out_dir}")
    return shots


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


def sos_norm(vmin: float = 1400.0, vcenter: float = 1600.0, vmax: float = 3000.0):
    """Two-slope (nonlinear) normalisation for SoS colorbars.

    Maps half the colormap to [vmin, vcenter] (soft-tissue range, ~200 m/s)
    and half to [vcenter, vmax] (skull range, ~1400 m/s), so soft-tissue
    contrast is not visually crushed by the much higher skull speed.
    """
    from matplotlib.colors import TwoSlopeNorm

    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


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
    vmin=1400.0,
    vmax=3000.0,
    norm=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.asarray(vp.data if hasattr(vp, "data") else vp)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    if norm is None:
        norm = sos_norm(vmin=vmin, vcenter=1600.0, vmax=vmax)

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=(0, domain_m * 1000, 0, domain_m * 1000),
        norm=norm,
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
    norm=None,
):
    """Static overview: vp + transducers + shot wavelet (replaces problem.plot())."""
    import matplotlib.pyplot as plt

    data = np.asarray(vp.data if hasattr(vp, "data") else vp)
    shot = problem.acquisitions.get(shot_id)
    ext = domain_extent_mm(domain_m)

    if norm is None:
        norm = sos_norm()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im = axes[0].imshow(data.T, origin="lower", extent=ext, cmap="viridis", norm=norm)
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


def elliptical_ring_radius(
    domain_m: float,
    ring_frac: float = RING_FRAC,
) -> tuple[float, float]:
    """Semi-axes matching odil_wave ``a_frac`` / ``b_frac`` (fraction of half-extent)."""
    half = 0.5 * float(domain_m)
    r = float(ring_frac) * half
    return (r, r)


def build_stride_problem(
    *,
    name: str,
    interior_shape: tuple[int, int],
    domain_m: float = 0.25,
    extra: tuple[int, int] = STRIDE_EXTRA,
    absorbing: tuple[int, int] = STRIDE_ABSORBING,
    f_centre: float = 80e3,
    n_cycles: int = 3,
    n_receivers: int = 100,
    c_max: float = SOS_SKULL,
    use_shepp_logan: bool = True,
    profile: str | None = None,
    phantom_scale: float = SHEPP_PHANTOM_SCALE,
    ring_frac: float = RING_FRAC,
    t_max: float | None = T_MAX_S,
    skull_alpha: float = 1.0,
    skull_sigma: float = 0.0,
    interior_value: float | None = None,
):
    """Create Stride Problem with a velocity model + acquisition.

    Defaults follow the odil_wave Shepp–Logan experiment setup: 80 kHz tone
    burst, 100 elliptical transducers at ``ring_frac`` of the half-extent,
    phantom scale 0.85, ``t_max=500`` µs.

    ``profile`` (preferred) selects the velocity model:
    ``\"shepp_logan\"``, ``\"shepp_logan_skull\"`` (optional smooth rim via
    ``skull_alpha`` / ``skull_sigma``), or ``\"water\"`` / ``\"homogeneous\"``.
    If ``profile`` is omitted, ``use_shepp_logan`` keeps the old boolean API.
    """
    from stride import Problem, ScalarField, Space, Time
    from stride.utils import wavelets

    if profile is None:
        profile = "shepp_logan" if use_shepp_logan else "water"
    profile = str(profile).lower()

    spacing = (
        domain_m / (interior_shape[0] - 1),
        domain_m / (interior_shape[1] - 1),
    )
    dx = min(spacing)
    dt = cfl_dt(dx, c_max)
    if t_max is None:
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

    if profile == "shepp_logan":
        vp_array = shepp_logan_sos(
            interior_shape,
            phantom_scale=phantom_scale,
        )
    elif profile == "shepp_logan_skull":
        vp_array = shepp_logan_skull_sos(
            interior_shape,
            phantom_scale=phantom_scale,
            skull_alpha=skull_alpha,
            skull_sigma=skull_sigma,
            interior_value=interior_value,
        )
    elif profile in ("water", "homogeneous"):
        vp_array = water_model(interior_shape)
    else:
        raise ValueError(
            "Unknown profile "
            f"{profile!r}; expected shepp_logan, shepp_logan_skull, or water"
        )

    vp_true = ScalarField(name="vp", grid=problem.grid)
    fill_vp_from_numpy(vp_true, vp_array)
    check_vp_finite(vp_true, "vp_true")
    problem.medium.add(vp_true)

    half = 0.5 * float(domain_m)
    radius = elliptical_ring_radius(domain_m, ring_frac=ring_frac)
    centre = (half, half)

    problem.transducers.default()
    problem.geometry.default(
        "elliptical",
        n_receivers,
        radius=radius,
        centre=centre,
    )
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
        t_max=float(t_max),
        time_num=num,
        profile=profile,
        phantom_scale=float(phantom_scale),
        ring_frac=float(ring_frac),
        skull_alpha=float(skull_alpha),
        skull_sigma=float(skull_sigma),
        radius=list(radius),
        centre=list(centre),
    )

