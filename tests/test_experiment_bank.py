"""Tests for the experiment bank index (``python -m odil_wave.experiment.bank``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from odil_wave.experiment import bank

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_record(bank_dir: Path, rec_id: str, **fields):
    """Write a minimal experiment record; ``fields`` override or (None) drop keys."""
    front = {
        "id": rec_id,
        "title": f"Does {rec_id} matter?",
        "date": "2026-10-31",
        "status": "current",
        "outcome": "answered",
        "issue": 100,
        "commit": "abc1234",
        "spec": f"configs/experiments/{rec_id}.yaml",
        "experiment_dir": f"outputs/experiments/{rec_id}",
        "device": "cpu",
        "variable": "loss.weights.reg",
        "baseline": "configs/example_multiband.yaml",
        "answer": "Yes, a little.",
    }
    for key, val in fields.items():
        if val is None:
            front.pop(key)
        else:
            front[key] = val
    text = (
        "---\n"
        + yaml.safe_dump(front, sort_keys=False)
        + "---\n\n# Title\n\n## Question\n\nQ?\n\n## Validity\n\nCurrent.\n"
    )
    (bank_dir / f"{rec_id}.md").write_text(text)


@pytest.fixture
def bank_dir(tmp_path) -> Path:
    path = tmp_path / "experiments"
    path.mkdir()
    return path


def _check(bank_dir: Path) -> int:
    return bank.main(["--check", "--bank", str(bank_dir)])


def test_check_passes_on_sequential_complete_records(bank_dir):
    _write_record(bank_dir, "0001-tv-weight")
    _write_record(bank_dir, "0002-band-ablation")
    assert _check(bank_dir) == 0


@pytest.mark.parametrize("field", ["answer", "commit"])
def test_check_fails_on_a_missing_frontmatter_field(bank_dir, field, capsys):
    _write_record(bank_dir, "0001-tv-weight")
    _write_record(bank_dir, "0002-band-ablation", **{field: None})
    assert _check(bank_dir) == 1
    err = capsys.readouterr().err
    assert "0002-band-ablation" in err and field in err


@pytest.mark.parametrize(
    "ids",
    [
        ["0001-tv-weight", "0003-band-ablation"],  # 0002 on another exp/* branch
        ["0002-tv-weight"],  # 0001 voided, or not merged yet
    ],
)
def test_check_passes_on_ids_with_gaps(bank_dir, ids):
    """A branch holds only its own records: the other ids are on other exp/*
    branches or were voided, so gaps are expected."""
    for rec_id in ids:
        _write_record(bank_dir, rec_id)
    assert _check(bank_dir) == 0


def test_check_fails_on_a_number_used_twice(bank_dir, capsys):
    _write_record(bank_dir, "0001-tv-weight")
    _write_record(bank_dir, "0001-band-ablation")
    assert _check(bank_dir) == 1
    assert "0001" in capsys.readouterr().err


def test_check_fails_when_the_frontmatter_id_is_not_the_filename(bank_dir, capsys):
    _write_record(bank_dir, "0001-tv-weight", id="0001-tv")
    assert _check(bank_dir) == 1
    assert "0001-tv-weight" in capsys.readouterr().err


README = """\
# Experiment bank

Intro.

## Index

<!-- index:start -->
stale table
<!-- index:end -->

Footer.
"""


def test_regenerate_writes_the_index_table_from_frontmatter(bank_dir):
    (bank_dir / "README.md").write_text(README)
    _write_record(bank_dir, "0001-tv-weight", status="invalidated")
    _write_record(
        bank_dir,
        "0002-tv-weight-fine",
        title="Does the TV weight matter on a finer grid?",
        answer="Yes | above 1e-6",
    )

    assert bank.main(["--bank", str(bank_dir)]) == 0

    assert (bank_dir / "README.md").read_text() == (
        "# Experiment bank\n\nIntro.\n\n## Index\n\n"
        "<!-- index:start -->\n"
        "| id | question | variable | status | answer |\n"
        "|---|---|---|---|---|\n"
        "| [0001-tv-weight](0001-tv-weight.md) | Does 0001-tv-weight matter? "
        "| loss.weights.reg | invalidated | Yes, a little. |\n"
        "| [0002-tv-weight-fine](0002-tv-weight-fine.md) "
        "| Does the TV weight matter on a finer grid? "
        "| loss.weights.reg | current | Yes \\| above 1e-6 |\n"
        "<!-- index:end -->\n\nFooter.\n"
    )


def test_regenerate_leaves_the_readme_alone_when_the_check_fails(bank_dir):
    (bank_dir / "README.md").write_text(README)
    _write_record(bank_dir, "0001-tv-weight")
    _write_record(bank_dir, "0001-band-ablation")
    assert bank.main(["--bank", str(bank_dir)]) == 1
    assert (bank_dir / "README.md").read_text() == README


def test_check_does_not_write_the_readme(bank_dir):
    (bank_dir / "README.md").write_text(README)
    _write_record(bank_dir, "0001-tv-weight")
    assert _check(bank_dir) == 0
    assert (bank_dir / "README.md").read_text() == README


REAL_BANK = REPO_ROOT / "docs" / "experiments"


def test_a_record_copied_from_the_template_passes_the_check(bank_dir):
    template = (REAL_BANK / "TEMPLATE.md").read_text()
    assert "id: NNNN-slug" in template
    copy = template.replace("id: NNNN-slug", "id: 0001-from-template")
    (bank_dir / "0001-from-template.md").write_text(copy)
    assert _check(bank_dir) == 0


def test_the_real_bank_passes_the_check():
    """CI gates the committed bank's format through the module entry point."""
    result = subprocess.run(
        [sys.executable, "-m", "odil_wave.experiment.bank", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
