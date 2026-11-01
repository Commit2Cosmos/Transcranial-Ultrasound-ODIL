"""Experiment bank index: regenerate the README's index table, or check the bank.

    uv run python -m odil_wave.experiment.bank            # regenerate the index
    uv run python -m odil_wave.experiment.bank --check    # validate only

Conventions are in ``docs/experiments/README.md``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_BANK = Path(__file__).resolve().parents[2] / "docs" / "experiments"

REQUIRED_FIELDS = (
    "id",
    "title",
    "date",
    "status",
    "outcome",
    "issue",
    "commit",
    "spec",
    "experiment_dir",
    "device",
    "variable",
    "baseline",
    "answer",
    "supersedes",
)

_RECORD_NAME = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_SUPERSEDED_BY = re.compile(r"superseded by \[(\d{4}-[a-z0-9-]+)\]")
INDEX_START = "<!-- index:start -->"
INDEX_END = "<!-- index:end -->"


@dataclass
class Record:
    """One experiment record: its file stem, frontmatter and body."""

    name: str
    front: Dict[str, Any]
    body: str


def load_records(bank_dir: Path) -> tuple[List[Record], List[str]]:
    """Read every ``NNNN-slug.md`` record in ``bank_dir``, sorted by filename.

    Returns the records whose frontmatter parsed, and a problem for each that
    did not.
    """
    records, problems = [], []
    for path in sorted(bank_dir.glob("*.md")):
        if not _RECORD_NAME.match(path.name):
            continue
        record = read_record(path)
        if record is None:
            problems.append(f"{path.stem}: no YAML frontmatter between '---' lines")
            continue
        records.append(record)
    return records, problems


def read_record(path: Path) -> Optional[Record]:
    """The record at ``path``, or ``None`` if it has no YAML frontmatter."""
    match = _FRONTMATTER.match(path.read_text())
    front = yaml.safe_load(match.group(1)) if match else None
    if not isinstance(front, dict):
        return None
    return Record(path.stem, front, match.group(2))


def check(bank_dir: Path) -> List[str]:
    """Return every problem found in the bank's records (empty when valid)."""
    records, problems = load_records(bank_dir)
    for rec in records:
        missing = [f for f in REQUIRED_FIELDS if f not in rec.front]
        if missing:
            problems.append(
                f"{rec.name}: missing frontmatter fields: {', '.join(missing)}"
            )
        if "id" in rec.front and str(rec.front["id"]) != rec.name:
            problems.append(
                f"{rec.name}: frontmatter id {rec.front['id']!r} is not the filename"
            )

    numbers = [int(rec.name[:4]) for rec in records]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(
            "ids are not sequential from 0001 without gaps or repeats: "
            + ", ".join(rec.name for rec in records)
        )

    problems += _check_supersession(records)
    return problems


def _check_supersession(records: List[Record]) -> List[str]:
    """Every ``supersedes`` target is a record, no longer ``current``, and, if
    ``superseded``, has a superseded-by Validity entry naming a record that
    lists it in ``supersedes``."""
    by_name = {rec.name: rec for rec in records}
    superseders: Dict[str, set] = {}
    problems = []
    for rec in records:
        targets = rec.front.get("supersedes") or []
        if not isinstance(targets, list):
            problems.append(f"{rec.name}: supersedes must be a list of record ids")
            continue
        for target in targets:
            if not isinstance(target, str) or target not in by_name:
                problems.append(
                    f"{rec.name}: supersedes target {target!r} is not a record id "
                    "(write the full NNNN-slug)"
                )
                continue
            superseders.setdefault(target, set()).add(rec.name)

    for target, newer in sorted(superseders.items()):
        status = by_name[target].front.get("status")
        if status == "current":
            problems.append(
                f"{target}: still current, but superseded by {', '.join(sorted(newer))}"
            )
        elif status == "superseded":
            named = set(_SUPERSEDED_BY.findall(_validity(by_name[target].body)))
            if not named & newer:
                problems.append(
                    f"{target}: no Validity entry 'superseded by [<id>]' naming "
                    f"{' or '.join(sorted(newer))}"
                )
    return problems


def _validity(body: str) -> str:
    """The text of the record's Validity section."""
    match = re.search(r"^## Validity[^\n]*\n(.*?)(?=^## |\Z)", body, re.M | re.S)
    return match.group(1) if match else ""


def index_table(records: List[Record]) -> str:
    """The bank's index: id | question (the title) | variable | status | answer."""

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| id | question | variable | status | answer |",
        "|---|---|---|---|---|",
    ]
    for rec in records:
        f = rec.front
        cells = [f"[{rec.name}]({rec.name}.md)"] + [
            cell(f[key]) for key in ("title", "variable", "status", "answer")
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def regenerate(bank_dir: Path, records: List[Record]) -> Optional[str]:
    """Rewrite the table between the README's index markers.

    Returns a problem if the README or its markers are missing, else ``None``.
    """
    readme = bank_dir / "README.md"
    text = readme.read_text() if readme.exists() else ""
    start, end = text.find(INDEX_START), text.find(INDEX_END)
    if start < 0 or end < start:
        return f"{readme}: no '{INDEX_START}' ... '{INDEX_END}' markers"
    head = text[: start + len(INDEX_START)]
    readme.write_text(head + "\n" + index_table(records) + text[end:])
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m odil_wave.experiment.bank", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--check", action="store_true", help="validate only")
    parser.add_argument(
        "--bank", type=Path, default=DEFAULT_BANK, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    problems = check(args.bank)
    if not problems and not args.check:
        problem = regenerate(args.bank, load_records(args.bank)[0])
        problems = [problem] if problem else []
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
