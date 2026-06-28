import math
from typing import Optional

import numpy as np
import scipy.signal
import torch
import matplotlib.pyplot as plt

from odil_wave.grid import Grid
from odil_wave.plot_utils import frequency_scale, time_scale


class SourceSignal:
    """Temporal source waveform s(t) for an ODIL shot.

    Split out from `AcquisitionGeometry` so the same transducer ring can
    drive multiple frequencies (e.g. a low-frequency pass to resolve the
    skull, then a higher-frequency pass to sharpen the interior).

    Supported kinds
    ---------------
    ``"ricker"``
        Parameters: ``f0``, ``t0``, ``amplitude``.

    ``"tone_burst"``
        Gaussian (or rectangular) enveloped sinusoidal burst, matching
        ``stride.utils.wavelets.tone_burst``.
        Parameters: ``f0``, ``n_cycles``, ``envelope``, ``offset``,
        ``amplitude``.
    """

    def __init__(
        self,
        grid: Grid,
        kind: str = "ricker",
        f0: float = 4.0,
        t0: Optional[float] = None,
        amplitude: float = 1.0,
        # tone_burst-only parameters
        n_cycles: float = 3.0,
        envelope: str = "gaussian",
        offset: int = 0,
    ):
        self.grid = grid
        self.kind = kind
        self.f0 = f0
        self.t0 = 1.0 / f0 if t0 is None else t0  # causal delay default for ricker
        self.amplitude = amplitude
        self.n_cycles = n_cycles
        self.envelope = envelope
        self.offset = offset

        if kind == "tone_burst":
            self._tone_burst_signal = self._tone_burst()

    # Ricker
    def _ricker(self, t: torch.Tensor) -> torch.Tensor:
        arg = (math.pi * self.f0 * (t - self.t0)) ** 2
        return self.amplitude * (1.0 - 2.0 * arg) * torch.exp(-arg)

    # Tone burst (Stride)
    def _tone_burst(self) -> torch.Tensor:
        """Sample the tone-burst onto the grid's time axis."""
        dt = float(self.grid.dt)
        n_samples = self.grid.nt
        tone_length = self.n_cycles / self.f0
        n_tone = int(tone_length // dt + 1)

        time_array = np.linspace(0, tone_length, n_tone, endpoint=False)
        signal = np.sin(2 * np.pi * self.f0 * time_array)

        if self.envelope == "gaussian":
            window_x = np.linspace(-3.0, 3.0, n_tone)
            window = np.exp(-(window_x**2) / 2)
        elif self.envelope == "rectangular":
            window = np.ones(n_tone)
        else:
            raise ValueError(
                f"envelope '{self.envelope}' not supported, "
                "use 'gaussian' or 'rectangular'"
            )

        signal = signal * window
        signal = signal * scipy.signal.get_window(("tukey", 0.05), n_tone, False)

        pad_after = n_samples - self.offset - n_tone
        signal = np.pad(
            signal,
            ((self.offset, pad_after),),
            mode="constant",
            constant_values=0.0,
        )
        return torch.tensor(
            signal * self.amplitude,
            dtype=self.grid.dtype,
            device=self.grid.device,
        )

    def waveform(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the source waveform at times ``t``."""
        if self.kind == "ricker":
            return self._ricker(t)
        if self.kind == "tone_burst":
            return self._tone_burst_signal
        raise ValueError(
            f"Unknown source kind '{self.kind}'. Use 'ricker' or 'tone_burst'."
        )

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.waveform(t)

    def plot_pulse(self, ax=None):
        """Plot the temporal waveform s(t) over the grid's time axis."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 3.5))
        s = self.waveform(self.grid.t).cpu().numpy()
        t = self.grid.t.cpu().numpy()
        peak_t = int(np.argmax(np.abs(s)))

        t_mult, t_unit = time_scale(float(t[-1]) if t.size else 1.0)
        f_mult, f_unit = frequency_scale(self.f0)

        ax.plot(t * t_mult, s)
        ax.axvline(self.grid.t[peak_t].item() * t_mult, color="gray", ls=":", alpha=0.7)
        ax.set_xlabel(f"t [{t_unit}]")
        ax.set_ylabel("s(t) [a.u.]")
        if self.kind == "tone_burst":
            ax.set_title(
                rf"tone_burst ($f_0$={self.f0 * f_mult:g} {f_unit}, "
                rf"{self.n_cycles} cycles, {self.envelope} env, A={self.amplitude:g})"
            )
        else:
            ax.set_title(
                rf"{self.kind} pulse ($f_0$={self.f0 * f_mult:g} {f_unit}, "
                rf"$t_0$={self.t0 * t_mult:.2f} {t_unit}, A={self.amplitude:g})"
            )
        ax.grid(alpha=0.3)
        return ax
