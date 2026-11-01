"""No module imports another module's private (``_``-prefixed) names, so renaming
one can't silently break another module. Ruff's PLC2701 exempts imports within
the same package, which is every import here, so the rule is checked here;
ruff's SLF001 covers private attribute access."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted((REPO_ROOT / "odil_wave").rglob("*.py")) + [REPO_ROOT / "main.py"]


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def test_no_module_imports_a_private_name():
    offences = [
        f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports {alias.name}"
        for path in SOURCES
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if _is_private(alias.name)
    ]
    assert not offences, "\n".join(offences)
