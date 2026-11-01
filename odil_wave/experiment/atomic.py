"""Atomic file writes: a temp file beside the target, renamed over it once
complete, so a reader never sees a half-written file."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator, Union


@contextmanager
def atomic_open(path: Union[str, Path], mode: str = "w", **kwargs: Any) -> Iterator[IO]:
    """Open ``<path>.tmp`` for writing (``open``'s ``mode`` and ``kwargs``);
    on a clean exit it is flushed to disk and renamed to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, mode, **kwargs) as f:
        yield f
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_text_atomic(path: Union[str, Path], text: str) -> None:
    """Atomically write ``text`` to ``path``."""
    with atomic_open(path) as f:
        f.write(text)
