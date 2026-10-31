"""Environment capture, seeding, and device/dtype resolution.

Everything here is best-effort and side-effect-light: capturing metadata must
never fail a run. ``torch`` is imported lazily so ``--dry-run`` can summarise a
config without importing the numerical stack.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any, Dict, Optional


def seed_everything(seed: int, deterministic: bool = False) -> Dict[str, Any]:
    """Seed Python, NumPy and torch RNGs. Returns a record of what was seeded."""
    import random

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    record: Dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch": True,
        "deterministic": bool(deterministic),
    }
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - platform dependent
            record["deterministic_error"] = repr(exc)
    return record


def resolve_dtype(name: str):
    """Map a dtype name (``float32``/``float64``) to a ``torch.dtype``."""
    import torch

    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "f32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "f64": torch.float64,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; expected float32 or float64")
    return mapping[key]


def resolve_device(name: str):
    """Resolve a device string to a ``torch.device`` with CPU fallback.

    ``Grid`` also guards against unavailable accelerators; this mirrors that so
    metadata records the *resolved* device.
    """
    import warnings

    import torch

    req = str(name).strip().lower()
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(req)
        warnings.warn("CUDA requested but not available; using CPU.", RuntimeWarning)
        return torch.device("cpu")
    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested but not available; using CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device("cpu")


def platform_info() -> Dict[str, Any]:
    """Host / OS / Python info that does not require importing torch."""
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def _package_versions() -> Dict[str, Any]:
    """Best-effort torch / numpy / odil_wave versions and device flags.

    Returns
    -------
    dict
        Available package versions plus torch device availability; import
        failures are recorded as ``*_error`` keys rather than raised.
    """
    versions: Dict[str, Any] = {}
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda_available"] = bool(torch.cuda.is_available())
        mps = getattr(torch.backends, "mps", None)
        versions["mps_available"] = bool(mps is not None and mps.is_available())
        versions["torch_num_threads"] = int(torch.get_num_threads())
    except Exception as exc:  # pragma: no cover
        versions["torch_error"] = repr(exc)
    try:
        import numpy as np

        versions["numpy"] = np.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import odil_wave

        versions["odil_wave"] = getattr(odil_wave, "__version__", "0.1.0")
    except Exception:  # pragma: no cover
        pass
    return versions


def collect_metadata(
    *,
    optimiser: str,
    device_requested: str,
    device_resolved: Optional[str] = None,
    dtype: str,
    seed: int,
    include_torch: bool = True,
) -> Dict[str, Any]:
    """Assemble the environment part of the metadata record.

    The code's commit and lockfile come from
    :meth:`~odil_wave.experiment.rundir.RunDirectory.provenance`.

    ``include_torch=False`` skips importing torch.
    """
    meta: Dict[str, Any] = {
        "optimiser": optimiser,
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch_s": time.time(),
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "dtype": dtype,
        "seed": seed,
        "argv": list(sys.argv),
        "platform": platform_info(),
    }
    if include_torch:
        meta["versions"] = _package_versions()
    return meta
