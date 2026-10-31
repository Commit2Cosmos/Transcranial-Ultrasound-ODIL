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
def test_cli_tiny_two_band_run_float32(tmp_path, git_repo, tiny_run_config, device):
    """A float32 2-band run through the CLI completes and improves the model."""
    user_cfg = deep_merge(
        tiny_run_config,
        {
            "continuation": {
                "bands": [{"frequencies_hz": [30e3]}, {"frequencies_hz": [40e3]}]
            },
        },
    )
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(user_cfg))
    overrides = [f"runtime.device={device}", "runtime.dtype=float32"]

    run_dir = tmp_path / "run"
    argv = ["--config", str(config_path), "--out", str(run_dir)]
    for o in overrides:
        argv += ["--override", o]
    assert main.main(argv, repo_root=git_repo) == 0

    cfg = resolve_config(user_cfg, overrides=overrides)

    # The run used the requested device and precision (no silent CPU fallback).
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["device_resolved"] == device
    assert meta["dtype"] == "float32"

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["n_bands"] == 2
    metrics = summary["metrics_summary"]
    final_err = metrics["final_metrics"]["rmse_brain"]
    assert np.isfinite(final_err)
    # the run's start row scores the starting model
    assert final_err < metrics["start_metrics"]["rmse_brain"]

    problem = build_problem(cfg)

    c_final = np.load(sorted((run_dir / "bands").iterdir())[-1] / "c_final.npy")
    assert c_final.shape == tuple(problem.grid.shape)
