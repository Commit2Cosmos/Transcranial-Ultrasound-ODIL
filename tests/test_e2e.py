"""End-to-end tests: the ``main.py`` CLI runs a tiny inversion to completion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from odil_wave.experiment import build_problem, deep_merge, resolve_config
from odil_wave.experiment.runner import _final_recovery_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import main  # noqa: E402

MPS_AVAILABLE = getattr(torch.backends, "mps", None) is not None and (
    torch.backends.mps.is_available()
)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=[
                pytest.mark.mps,
                pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available"),
            ],
        ),
    ],
)
def test_cli_tiny_two_band_run_float32(tmp_path, tiny_experiment, device):
    """A float32 2-band run through the CLI completes and improves the model."""
    user_cfg = deep_merge(
        tiny_experiment,
        {
            "run": {"run_id": f"e2e_{device}"},
            "continuation": {
                "bands": [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]
            },
        },
    )
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(user_cfg))
    overrides = [f"runtime.device={device}", "runtime.dtype=float32"]

    argv = ["--config", str(config_path)]
    for o in overrides:
        argv += ["--override", o]
    assert main.main(argv) == 0

    cfg = resolve_config(user_cfg, overrides=overrides)
    run_dir = cfg.run_dir()

    # The run used the requested device and precision (no silent CPU fallback).
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["device_resolved"] == device
    assert meta["dtype"] == "float32"

    summary = json.loads((run_dir / "final" / "run_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["n_bands"] == 2
    final_err = summary["metrics_summary"]["final_rel_c_error"]
    assert final_err is not None and np.isfinite(final_err)

    problem = build_problem(cfg)
    init_err = _final_recovery_metrics(problem, problem.init_velocity)["rel_c_error"]
    assert final_err < init_err

    c_final = np.load(run_dir / "final" / "c_final.npy")
    assert c_final.shape == tuple(problem.grid.shape)
