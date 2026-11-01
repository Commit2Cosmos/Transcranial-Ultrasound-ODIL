"""The live view: one matplotlib window redrawn during a run.

:class:`LiveVelocityView` is the ``run_inverse(on_log=...)`` callback behind
``python main.py --live`` and the notebook, as decided under "Live view" in
"Standard post-processing and plot set" (#53). Its panels are c − c*, the
truth-free stopping pair (``rel_data_residual``, ``pde_rel_residual``) and
``rmse_brain`` / ``ssim_brain``, all from the run's tape rows.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Dict, List

import matplotlib
import numpy as np

# Seconds between redraws; each band's final row is drawn regardless.
REDRAW_INTERVAL_S = 5.0

_TREND_COLUMNS = ("rel_data_residual", "pde_rel_residual", "rmse_brain", "ssim_brain")


def has_display() -> bool:
    """Whether matplotlib can show a figure: an interactive backend (a window)
    or Jupyter's inline backend."""
    from matplotlib.backends import BackendFilter, backend_registry

    backend = matplotlib.get_backend().lower()
    if "inline" in backend:
        return True
    return backend not in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)


class LiveVelocityView:
    """Redraws one window during a run so a stalled run can be stopped early.

    Pass an instance as ``run_inverse(..., on_log=view)``. Every logged row
    is kept; the window is redrawn at a logged row at most once every
    :data:`REDRAW_INTERVAL_S` seconds, and always at each band's final row.
    It prints nothing. Without a display it warns once (on construction) and
    does nothing. Interrupt the run (Ctrl-C) to stop it early: the run ends
    as ``interrupted``.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """
        Parameters
        ----------
        clock : callable, optional
            Seconds now, read once per logged row; for the redraw interval.
        """
        self.clock = clock
        self.figure = None
        self.enabled = has_display()
        if not self.enabled:
            print(
                "warning: no display (matplotlib backend "
                f"{matplotlib.get_backend()!r}); running without the live view",
                file=sys.stderr,
            )
        self._rows: List[Dict[str, float]] = []
        self._last_draw: float = -np.inf

    def __call__(self, logged) -> None:
        """Take one logged row and redraw if it is due.

        Parameters
        ----------
        logged : LoggedRow
            The row just written to the tape, with the interior ``c`` and
            ``c_star`` (:class:`~odil_wave.experiment.runner.LoggedRow`).
        """
        if not self.enabled:
            return
        now = self.clock()
        self._rows.append(logged.row)
        if logged.band_final or now - self._last_draw >= REDRAW_INTERVAL_S:
            self._last_draw = now
            self._draw(logged)

    # ------------------------------------------------------------------ #
    def _draw(self, logged) -> None:
        """Redraw every panel from the rows so far and show the window."""
        from matplotlib.figure import Figure

        first = self.figure is None
        if first:
            if _inline():
                self.figure = Figure(figsize=(14, 4.2))
            else:
                import matplotlib.pyplot as plt

                self.figure = plt.figure(figsize=(14, 4.2))
        fig = self.figure
        fig.clear()
        ax_c, ax_fit, ax_err = fig.subplots(1, 3)

        diff = logged.c - logged.c_star
        lim = float(np.max(np.abs(diff))) or 1.0
        im = ax_c.imshow(diff.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax_c, shrink=0.85, label="c − c* [m/s]")
        row = logged.row
        ax_c.set_title(f"c − c*  (band {row['band_index']}, iter {row['iter']})")
        ax_c.set_xticks([])
        ax_c.set_yticks([])

        x = [r["iter_total"] for r in self._rows]
        col = {k: [r[k] for r in self._rows] for k in _TREND_COLUMNS}
        band_starts = [
            r["iter_total"]
            for prev, r in zip(self._rows, self._rows[1:])
            if r["band_index"] != prev["band_index"]
        ]

        ax_fit.semilogy(x, col["rel_data_residual"], "o-", label="rel_data_residual")
        ax_fit.semilogy(x, col["pde_rel_residual"], "s-", label="pde_rel_residual")
        ax_fit.set_title("stopping pair (truth-free)")
        ax_fit.legend(loc="best", fontsize="small")

        ax_err.plot(x, col["rmse_brain"], "o-", color="tab:red", label="rmse_brain")
        ax_err.set_ylabel("rmse_brain [m/s]", color="tab:red")
        ax_ssim = ax_err.twinx()
        ax_ssim.plot(x, col["ssim_brain"], "s-", color="tab:green", label="ssim_brain")
        ax_ssim.set_ylabel("ssim_brain", color="tab:green")
        ax_err.set_title("model error (brain region)")

        for ax in (ax_fit, ax_err):
            ax.set_xlabel("iter_total")
            ax.grid(alpha=0.3)
            for b in band_starts:
                ax.axvline(b, color="0.6", lw=0.8, ls=":")
        fig.suptitle("Live run — Ctrl-C / interrupt the kernel to stop early")
        fig.tight_layout()
        _show(fig, first=first)


def _inline() -> bool:
    return "inline" in matplotlib.get_backend().lower()


def _show(fig, *, first: bool) -> None:
    """Show ``fig``: replace the cell output in Jupyter, else update the
    window (opened on the ``first`` draw)."""
    if _inline():
        from IPython.display import clear_output, display

        clear_output(wait=True)
        display(fig)
        return
    if first:
        fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
