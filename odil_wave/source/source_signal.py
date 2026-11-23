import math
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from odil_wave.grid import Grid
from odil_wave.plot_utils import frequency_scale, time_scale


class SourceSignal:
    """Temporal source waveform s(t) for an ODIL shot."""

    def __init__(
        self,
        grid: Grid,
        *,
        kind: str,
        f0: float,
        amplitude: float = 1.0,
        # tone_burst-only parameters
        n_cycles: float = 3.0,
        envelope: str = "gaussian",
        dimensionless: bool = True,
        t_centre: Optional[float] = None,
        t_start: float = 0.0,
    ):
        """Build a source pulse on `grid`; defaults follow ADR 0001.

        ``f0`` is in Hz, ``amplitude`` the peak of `s(t)` before the
        ``dimensionless`` scaling by ``grid.natural_source_amplitude(f0)``.
        ``n_cycles``, ``envelope`` and ``t_start`` (s) are tone_burst only;
        the burst must start and end on the grid's time axis.
        ``t_centre`` is the Ricker centre in s, ``None`` -> ``1 / f0``.
        """
        self.grid = grid
        self.kind = kind
        self.f0 = f0
        self.t_centre = 1.0 / f0 if t_centre is None else t_centre
        self.dimensionless = dimensionless
        # User-facing amplitude (kept as-is so plot titles read naturally),
        # and the internal physical amplitude actually fed to the waveform.
        self.amplitude = amplitude
        self._amplitude_phys = (
            amplitude * grid.natural_source_amplitude(f0)
            if dimensionless
            else amplitude
        )
        self.n_cycles = n_cycles
        self.envelope = envelope
        self.t_start = t_start

        if kind == "ricker":
            self._signal = self._ricker(grid.t)
        elif kind == "tone_burst":
            self._signal = self._tone_burst()
        else:
            raise ValueError(
                f"Unknown source kind '{kind}'. Use 'ricker' or 'tone_burst'."
            )

    def _ricker(self, t: torch.Tensor) -> torch.Tensor:
        arg = (math.pi * self.f0 * (t - self.t_centre)) ** 2
        return self._amplitude_phys * (1.0 - 2.0 * arg) * torch.exp(-arg)

    # Tone burst (Stride's shape, defined in continuous time)
    def _tone_burst(self) -> torch.Tensor:
        """Sample the tone burst at ``grid.t``, so its spectrum does not depend
        on ``dt``."""
        tone_length = self.n_cycles / self.f0
        if self.t_start < 0 or self.t_start + tone_length > self.grid.t_max:
            raise ValueError(
                f"tone burst of {tone_length:g} s starting at t_start = "
                f"{self.t_start:g} s does not fit the grid's time axis "
                f"[0, {self.grid.t_max:g}] s"
            )
        if self.envelope not in ("gaussian", "rectangular"):
            raise ValueError(
                f"envelope '{self.envelope}' not supported, "
                "use 'gaussian' or 'rectangular'"
            )

        t = np.arange(self.grid.nt) * float(self.grid.dt) - self.t_start
        x = t / tone_length  # burst on [0, 1)
        inside = (x >= 0) & (x < 1)
        x = np.where(inside, x, 0.0)

        signal = np.sin(2 * np.pi * self.n_cycles * x)
        if self.envelope == "gaussian":
            signal = signal * np.exp(-((6 * x - 3) ** 2) / 2)
        signal = signal * _tukey(x, alpha=0.05)
        signal = np.where(inside, signal, 0.0)
        return torch.tensor(
            signal * self._amplitude_phys,
            dtype=self.grid.dtype,
            device=self.grid.device,
        )

    def waveform(self) -> torch.Tensor:
        """The source waveform sampled on ``grid.t``."""
        return self._signal

    def spectrum(self, frequency_selection) -> torch.Tensor:
        """Complex spectrum of ``s(t)`` on ``FrequencySelection`` bins, as
        :meth:`FrequencySelection.fft_time_series` of the waveform on ``grid.t``."""
        s_t = self.waveform()
        return frequency_selection.fft_time_series(s_t, dim=0)

    def plot_pulse(self, ax=None):
        """Plot the temporal waveform s(t) over the grid's time axis."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 3.5))
        s = self.waveform().cpu().numpy()
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
                rf"$t_c$={self.t_centre * t_mult:.2f} {t_unit}, A={self.amplitude:g})"
            )
        ax.grid(alpha=0.3)
        return ax

    def plot_spectrum(self, frequencies=None, ax=None, title="Source spectrum"):
        """Plot the normalised source magnitude spectrum ``|S(f)|``.

        ``frequencies`` optionally marks selected bins (e.g. the inversion
        frequencies) as vertical lines. It accepts a ``FrequencySelection``
        (its ``.frequencies`` are used), a tensor/array of hertz values, or a
        sequence of floats.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 3.2))

        sig = self.waveform().cpu().numpy()
        mag = np.abs(np.fft.rfft(sig))
        mag /= mag.max() + 1e-30
        fr = np.fft.rfftfreq(self.grid.nt, d=self.grid.dt)

        sel = _as_frequency_array(frequencies)
        f_ref = float(np.max(sel)) if sel is not None and sel.size else self.f0
        f_mult, f_unit = frequency_scale(f_ref if f_ref > 0 else self.f0)

        ax.plot(fr * f_mult, mag, label="|S(f)|")
        if sel is not None and sel.size:
            for k, f in enumerate(sel):
                ax.axvline(
                    f * f_mult,
                    color="crimson",
                    ls="--",
                    alpha=0.85,
                    label="selected bins" if k == 0 else None,
                )
            ax.set_xlim(0, float(np.max(sel)) * f_mult * 2.2)
        ax.set_xlabel(f"f [{f_unit}]")
        ax.set_ylabel("normalised |S(f)|")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        return ax


def _as_frequency_array(frequencies):
    """Coerce a FrequencySelection / tensor / array / sequence to 1D |Hz|."""
    if frequencies is None:
        return None
    freqs = getattr(frequencies, "frequencies", frequencies)
    if isinstance(freqs, torch.Tensor):
        freqs = freqs.detach().abs().cpu().numpy()
    else:
        freqs = np.abs(np.asarray(freqs, dtype=float))
    return np.atleast_1d(freqs)


def _tukey(x: np.ndarray, alpha: float) -> np.ndarray:
    """Continuous Tukey window on ``x`` in [0, 1]: cosine tapers over the first
    and last ``alpha / 2``."""
    edge = np.minimum(x, 1 - x)
    taper = 0.5 * (1 - np.cos(2 * np.pi * edge / alpha))
    return np.where(edge < alpha / 2, taper, 1.0)
