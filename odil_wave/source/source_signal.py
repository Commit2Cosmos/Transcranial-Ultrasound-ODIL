import math
from typing import Optional

import numpy as np
import scipy.signal
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
        amplitude: float,
        # tone_burst-only parameters
        n_cycles: float,
        envelope: str,
        dimensionless: bool,
        t_centre: Optional[float] = None,
        offset: int = 0,
    ):
        """Build a source pulse on `grid`.

        Every parameter set by the run config's ``source`` block is a required
        keyword (ADR 0001); ``t_centre`` and ``offset`` keep library defaults.

        Parameters
        ----------
        kind
            ``"tone_burst"`` | ``"ricker"``.
        f0
            Centre frequency in Hz.
        amplitude
            Peak of the waveform `s(t)` (before the ``dimensionless`` scaling).
        n_cycles, envelope
            tone_burst only: cycle count and ``"gaussian"`` | ``"rectangular"``
            envelope.
        dimensionless
            If true the physical amplitude is ``amplitude * 2 * f0**2``.
        t_centre
            Ricker time centre in s; ``None`` -> ``1 / f0``.
        offset
            tone_burst sample offset delaying the wavelet; the burst must start
            and end on the grid's time axis.
        """
        self.grid = grid
        self.kind = kind
        self.f0 = f0
        # causal delay default for ricker
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
        self.offset = offset

        if kind == "ricker":
            self._signal = self._ricker(grid.t)
        elif kind == "tone_burst":
            self._signal = self._tone_burst()
        else:
            raise ValueError(
                f"Unknown source kind '{kind}'. Use 'ricker' or 'tone_burst'."
            )

    # Ricker
    def _ricker(self, t: torch.Tensor) -> torch.Tensor:
        arg = (math.pi * self.f0 * (t - self.t_centre)) ** 2
        return self._amplitude_phys * (1.0 - 2.0 * arg) * torch.exp(-arg)

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
        if self.offset < 0 or pad_after < 0:
            raise ValueError(
                f"tone burst of {n_tone} samples at offset {self.offset} does not "
                f"fit the grid's {n_samples} time samples"
            )
        signal = np.pad(
            signal,
            ((self.offset, pad_after),),
            mode="constant",
            constant_values=0.0,
        )
        return torch.tensor(
            signal * self._amplitude_phys,
            dtype=self.grid.dtype,
            device=self.grid.device,
        )

    def waveform(self) -> torch.Tensor:
        """The source waveform sampled on ``grid.t``."""
        return self._signal

    def spectrum(self, frequency_selection) -> torch.Tensor:
        """Complex spectrum of ``s(t)`` on ``FrequencySelection`` bins.

        FFT of the waveform on ``grid.t`` with the same norm/bins as observations.
        """
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
