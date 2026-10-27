"""Tests for the ``main.py`` CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from odil_wave.experiment import load_config_file, resolve_config

from conftest import TINY_EXPERIMENT

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import main  # noqa: E402


def _dotted_overrides(cfg: dict, prefix: str = "") -> list:
    """Flatten a nested config into ``a.b=<json>`` CLI overrides, one per leaf."""
    out = []
    for key, val in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            out += _dotted_overrides(val, dotted + ".")
        else:
            out.append(f"{dotted}={json.dumps(val)}")
    return out


@pytest.fixture
def experiment_file(tmp_path) -> Path:
    """The tiny experiment config written to a YAML file."""
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_EXPERIMENT))
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out
    assert "--print-defaults" in out


def test_print_defaults_prints_the_default_config(tmp_path, capsys):
    """--print-defaults prints YAML that resolves to the default config."""
    assert main.main(["--print-defaults"]) == 0
    printed = tmp_path / "defaults.yaml"
    printed.write_text(capsys.readouterr().out)

    resolved = resolve_config(load_config_file(printed)).to_dict()
    defaults = resolve_config({}).to_dict()
    resolved["run"].pop("run_id")
    defaults["run"].pop("run_id")
    assert resolved == defaults


def test_dry_run_default_config(capsys):
    code = main.main(["--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "n_transducers=64" in out
    assert "bands (1):" in out
    assert "[dry-run] configuration is valid; no artifacts written." in out


def test_dry_run_with_config_file(experiment_file, capsys):
    code = main.main(["--config", str(experiment_file), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "interior=[20, 20]" in out
    assert "bands (1):" in out


def test_override_applies(experiment_file, capsys):
    code = main.main(
        [
            "--config",
            str(experiment_file),
            "--override",
            "optimiser.n_iter=40",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "n_iter=40" in out


def test_no_config_runs_the_default_config(tmp_path):
    """Without --config a run starts from the default config (overrides keep it
    tiny)."""
    run = {"output_root": str(tmp_path), "run_id": "no_config"}
    overrides = _dotted_overrides({**TINY_EXPERIMENT, "run": run})
    overrides.append("optimiser.n_iter=2")
    argv = []
    for o in overrides:
        argv += ["--override", o]
    assert main.main(argv) == 0

    resolved = load_config_file(tmp_path / "no_config" / "config_resolved.yaml")
    assert resolved == resolve_config({}, overrides=overrides).to_dict()


def test_malformed_override_returns_config_error(experiment_file, capsys):
    code = main.main(["--config", str(experiment_file), "--override", "no-equals-sign"])
    err = capsys.readouterr().err
    assert code == 2
    assert "config error:" in err


def test_unknown_override_key_returns_config_error(experiment_file, capsys):
    code = main.main(
        [
            "--config",
            str(experiment_file),
            "--override",
            "not_a_real_section.x=1",
        ]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "config error:" in err


def test_missing_config_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        main.main(["--config", str(missing), "--dry-run"])
