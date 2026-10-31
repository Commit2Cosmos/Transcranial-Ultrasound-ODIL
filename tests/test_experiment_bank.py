"""Tests for the experiment bank index (``python -m odil_wave.experiment.bank``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from odil_wave.experiment import bank

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_record(bank_dir: Path, rec_id: str, *, validity="Current.", **fields):
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
        "supersedes": [],
    }
    for key, val in fields.items():
        if val is None:
            front.pop(key)
        else:
            front[key] = val
    text = (
        "---\n"
        + yaml.safe_dump(front, sort_keys=False)
        + "---\n\n# Title\n\n## Question\n\nQ?\n\n## Validity\n\n"
        + validity
        + "\n"
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


@pytest.mark.parametrize("field", ["answer", "supersedes", "commit"])
def test_check_fails_on_a_missing_frontmatter_field(bank_dir, field, capsys):
    _write_record(bank_dir, "0001-tv-weight")
    _write_record(bank_dir, "0002-band-ablation", **{field: None})
    assert _check(bank_dir) == 1
    err = capsys.readouterr().err
    assert "0002-band-ablation" in err and field in err


@pytest.mark.parametrize(
    "ids",
    [
        ["0001-tv-weight", "0003-band-ablation"],  # gap
        ["0002-tv-weight"],  # doesn't start at 0001
        ["0001-tv-weight", "0001-band-ablation"],  # number used twice
    ],
)
def test_check_fails_on_non_sequential_ids(bank_dir, ids, capsys):
    for rec_id in ids:
        _write_record(bank_dir, rec_id)
    assert _check(bank_dir) == 1
    assert "sequential" in capsys.readouterr().err


def test_check_fails_when_the_frontmatter_id_is_not_the_filename(bank_dir, capsys):
    _write_record(bank_dir, "0001-tv-weight", id="0001-tv")
    assert _check(bank_dir) == 1
    assert "0001-tv-weight" in capsys.readouterr().err


def _superseded_by(rec_id: str) -> str:
    return f"- 2026-10-31 · superseded by [{rec_id}]({rec_id}.md) · reran at finer grid"


def test_check_passes_when_supersedes_targets_are_flipped_and_noted(bank_dir):
    _write_record(
        bank_dir,
        "0001-tv-weight",
        status="superseded",
        validity=_superseded_by("0003-tv-weight-fine"),
    )
    _write_record(bank_dir, "0002-band-ablation", status="invalidated")
    _write_record(
        bank_dir,
        "0003-tv-weight-fine",
        supersedes=["0001-tv-weight", "0002-band-ablation"],
    )
    assert _check(bank_dir) == 0


def test_check_passes_on_a_transitive_supersession_chain(bank_dir):
    """0003 lists every record it replaces; 0001 keeps 0002's entry."""
    _write_record(
        bank_dir,
        "0001-tv-weight",
        status="superseded",
        validity=_superseded_by("0002-tv-weight-fine"),
    )
    _write_record(
        bank_dir,
        "0002-tv-weight-fine",
        status="superseded",
        supersedes=["0001-tv-weight"],
        validity=_superseded_by("0003-tv-weight-finer"),
    )
    _write_record(
        bank_dir,
        "0003-tv-weight-finer",
        supersedes=["0001-tv-weight", "0002-tv-weight-fine"],
    )
    assert _check(bank_dir) == 0


@pytest.mark.parametrize(
    "target_fields",
    [
        {"status": "current", "validity": "Current."},  # not flipped
        {"status": "superseded", "validity": "Current."},  # no superseded-by entry
    ],
)
def test_check_fails_on_a_supersedes_target_left_unmaintained(
    bank_dir, target_fields, capsys
):
    _write_record(bank_dir, "0001-tv-weight", **target_fields)
    _write_record(bank_dir, "0002-tv-weight-fine", supersedes=["0001-tv-weight"])
    assert _check(bank_dir) == 1
    assert "0001-tv-weight" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["0009-no-such-record", 1])
def test_check_fails_on_a_supersedes_target_that_is_not_a_record(
    bank_dir, target, capsys
):
    _write_record(bank_dir, "0001-tv-weight", supersedes=[target])
    assert _check(bank_dir) == 1
    assert repr(target) in capsys.readouterr().err


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
    _write_record(
        bank_dir,
        "0001-tv-weight",
        status="superseded",
        validity=_superseded_by("0002-tv-weight-fine"),
    )
    _write_record(
        bank_dir,
        "0002-tv-weight-fine",
        title="Does the TV weight matter on a finer grid?",
        answer="Yes | above 1e-6",
        supersedes=["0001-tv-weight"],
    )

    assert bank.main(["--bank", str(bank_dir)]) == 0

    assert (bank_dir / "README.md").read_text() == (
        "# Experiment bank\n\nIntro.\n\n## Index\n\n"
        "<!-- index:start -->\n"
        "| id | question | variable | status | answer |\n"
        "|---|---|---|---|---|\n"
        "| [0001-tv-weight](0001-tv-weight.md) | Does 0001-tv-weight matter? "
        "| loss.weights.reg | superseded | Yes, a little. |\n"
        "| [0002-tv-weight-fine](0002-tv-weight-fine.md) "
        "| Does the TV weight matter on a finer grid? "
        "| loss.weights.reg | current | Yes \\| above 1e-6 |\n"
        "<!-- index:end -->\n\nFooter.\n"
    )


def test_regenerate_leaves_the_readme_alone_when_the_check_fails(bank_dir):
    (bank_dir / "README.md").write_text(README)
    _write_record(bank_dir, "0002-tv-weight")
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
