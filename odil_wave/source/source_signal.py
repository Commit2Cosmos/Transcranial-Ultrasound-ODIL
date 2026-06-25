import math
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from odil_wave.grid import Grid


class SourceSignal:
    """Temporal source waveform s(t) for an ODIL shot.

    Split out from `AcquisitionGeometry` so the same transducer ring can
    drive multiple frequencies (e.g. a low-frequency pass to resolve the
    skull, then a higher-frequency pass to sharpen the interior).
    """

    def __init__(
        self,
        grid: Grid,
        kind: str = "ricker",
        f0: float = 4.0,
        t0: Optional[float] = None,
        amplitude: float = 1.0,
    ):
        self.grid = grid
        self.kind = kind
        self.f0 = f0
        self.t0 = 1.0 / f0 if t0 is None else t0  # causal delay default
        self.amplitude = amplitude

    def _ricker(self, t: torch.Tensor) -> torch.Tensor:
        arg = (math.pi * self.f0 * (t - self.t0)) ** 2
        return self.amplitude * (1.0 - 2.0 * arg) * torch.exp(-arg)

    def waveform(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the analytical waveform at times `t`."""
        if self.kind == "ricker":
            return self._ricker(t)
        raise ValueError(f"{self.kind} source isn't implemented")

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.waveform(t)

    def plot_pulse(self, ax=None):
        """Plot the temporal waveform s(t) over the grid's time axis."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 3.5))
        s = self.waveform(self.grid.t).cpu().numpy()
        peak_t = int(np.argmax(np.abs(s)))
        ax.plot(self.grid.t.cpu().numpy(), s)
        ax.axvline(self.grid.t[peak_t].item(), color="gray", ls=":", alpha=0.7)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("s(t) [a.u.]")
        ax.set_title(
            rf"{self.kind} pulse ($f_0$={self.f0} Hz, $t_0$={self.t0:.3f} s, "
            rf"A={self.amplitude:g})"
        )
        ax.grid(alpha=0.3)
        return ax
