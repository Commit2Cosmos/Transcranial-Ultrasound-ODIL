"""Optimiser-health warnings, from a band's tape rows as they are written.

The rules and thresholds are fixed in code ("Standard post-processing and plot
set", #53): a warning fires when

- ``c`` comes within 1% of a speed bound (``c_min`` ≤ 1.01 · bound, or
  ``c_max`` ≥ 0.99 · bound);
- ``sat_frac`` exceeds 1%;
- ``grad_u_norm`` or ``grad_z_norm`` rises at 3 consecutive logged rows (4
  strictly increasing values).

Each rule fires at most once per band. A fired warning is printed and listed
under ``warnings`` in ``band_summary.json`` and ``run_summary.json``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

BOUND_MARGIN = 0.01  # fraction of a speed bound
SAT_FRAC_LIMIT = 0.01
RISING_ROWS = 3  # consecutive logged rows at which a gradient norm rises
GRAD_NORMS = ("grad_u_norm", "grad_z_norm")


class HealthCheck:
    """Checks one band's tape rows against the health rules.

    Call it with each row (a ``tape.csv`` row as a dict); it returns the
    warnings that fired at that row and keeps every fired one in
    :attr:`warnings`.
    """

    def __init__(self, *, c_min: float, c_max: float) -> None:
        """
        Parameters
        ----------
        c_min, c_max : float
            The grid's speed bounds (``grid.c_min`` / ``grid.c_max``), m/s.
        """
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.warnings: List[str] = []
        self._fired: set = set()
        self._prev: Optional[Dict[str, Any]] = None
        self._rises = {k: 0 for k in GRAD_NORMS}

    def __call__(self, row: Dict[str, Any]) -> List[str]:
        """The warnings that fire at ``row`` (those not fired before in the band)."""
        found: Dict[str, str] = {}
        if row["c_min"] <= self.c_min * (1 + BOUND_MARGIN):
            found["c_min"] = (
                f"c_min {row['c_min']:.1f} m/s within 1% of the speed bound "
                f"{self.c_min:g} m/s"
            )
        if row["c_max"] >= self.c_max * (1 - BOUND_MARGIN):
            found["c_max"] = (
                f"c_max {row['c_max']:.1f} m/s within 1% of the speed bound "
                f"{self.c_max:g} m/s"
            )
        if row["sat_frac"] > SAT_FRAC_LIMIT:
            found["sat_frac"] = (
                f"sat_frac {row['sat_frac']:.3g} above {SAT_FRAC_LIMIT:g}: cells "
                "pinned at a speed bound"
            )
        for key in GRAD_NORMS:
            rose = self._prev is not None and row[key] > self._prev[key]
            self._rises[key] = self._rises[key] + 1 if rose else 0
            if self._rises[key] >= RISING_ROWS:
                found[key] = f"{key} rose over {RISING_ROWS} consecutive logged rows"
        self._prev = row

        new = []
        for rule, text in found.items():
            if rule in self._fired:
                continue
            self._fired.add(rule)
            new.append(f"band {row['band_index']}, iter {row['iter']}: {text}")
        self.warnings += new
        return new
