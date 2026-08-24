"""Tests for the ``main.py`` CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import main  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_inverse.yaml"


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main.main(["--help"])
    assert exc_info.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_dry_run_default_config(capsys):
    code = main.main(["--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[dry-run] configuration is valid; no artifacts written." in out


def test_dry_run_with_config_file(capsys):
    code = main.main(["--config", str(DEFAULT_CONFIG), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "optimiser       : lbfgsb" in out
    assert "bands (1):" in out


def test_override_applies(capsys):
    code = main.main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--override",
            "optimiser.n_iter=40",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "n_iter=40" in out


def test_malformed_override_returns_config_error(capsys):
    code = main.main(["--config", str(DEFAULT_CONFIG), "--override", "no-equals-sign"])
    err = capsys.readouterr().err
    assert code == 2
    assert "config error:" in err


def test_unknown_override_key_returns_config_error(capsys):
    code = main.main(
        [
            "--config",
            str(DEFAULT_CONFIG),
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
