"""SI-prefix helpers for plot axes and titles."""

from typing import Tuple

import numpy as np


_TIME_PREFIXES = (
    (1.0, "s"),
    (1e-3, "ms"),
    (1e-6, "µs"),
    (1e-9, "ns"),
)

_LENGTH_PREFIXES = (
    (1e3, "km"),
    (1.0, "m"),
    (1e-3, "mm"),
    (1e-6, "µm"),
)

_FREQ_PREFIXES = (
    (1e9, "GHz"),
    (1e6, "MHz"),
    (1e3, "kHz"),
    (1.0, "Hz"),
)


def _pick(magnitude: float, table) -> Tuple[float, str]:
    """Choose `(multiplier, label)` such that `magnitude * multiplier` is ~O(1-1000)."""
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return 1.0, table[0][1]
    m = abs(float(magnitude))
    for base, label in table:
        if m >= base:
            return 1.0 / base, label
    base, label = table[-1]
    return 1.0 / base, label


def time_scale(t_max: float) -> Tuple[float, str]:
    """Multiplier and unit label for a time axis whose largest value is ``t_max`` seconds."""
    return _pick(t_max, _TIME_PREFIXES)


def length_scale(l_max: float) -> Tuple[float, str]:
    """Multiplier and unit label for a spatial axis whose extent is ``l_max`` metres."""
    return _pick(l_max, _LENGTH_PREFIXES)


def frequency_scale(f: float) -> Tuple[float, str]:
    """Multiplier and unit label for a frequency value ``f`` in hertz."""
    return _pick(f, _FREQ_PREFIXES)


def format_time(t: float, unit: str | None = None, fmt: str = ".2f") -> str:
    """Format a time value with an auto-picked SI prefix (e.g. ``"123.45 µs"``)."""
    if unit is None:
        mult, unit = time_scale(t)
    else:
        # Look up the multiplier for the requested unit.
        mult = next((1.0 / b for b, lbl in _TIME_PREFIXES if lbl == unit), 1.0)
    return f"{t * mult:{fmt}} {unit}"
