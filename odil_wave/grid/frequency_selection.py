from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import torch

from odil_wave.grid.grid import Grid


def complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Map a real dtype to its matching complex dtype."""
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    raise ValueError(f"Unsupported real dtype for complex promotion: {real_dtype}")


def discrete_lambda_t(omega_nd: torch.Tensor, dt_nd: float) -> torch.Tensor:
    """Centered 2nd-order D_t symbol: i * sin(omega_nd * dt_nd) / dt_nd."""
    return 1j * torch.sin(omega_nd * dt_nd) / dt_nd


def discrete_lambda_tt(omega_nd: torch.Tensor, dt_nd: float) -> torch.Tensor:
    """Centered 2nd-order D_tt symbol: -(4 / dt_nd^2) * sin(omega_nd * dt_nd / 2)^2."""
    return -(4.0 / dt_nd**2) * torch.sin(0.5 * omega_nd * dt_nd) ** 2


# The time-derivative symbols a FrequencySelection can carry: the leapfrog
# scheme's centred differences, or the continuous ``iω`` and ``-ω²``.
TIME_SYMBOLS = ("leapfrog", "exact")


def _map_frequency_to_bin(freqs_all: torch.Tensor, f_hz: float) -> tuple[int, float]:
    """Nearest FFT bin for a requested physical frequency (avoids the 0 Hz bin)."""
    f = float(f_hz)
    k = int(torch.argmin(torch.abs(freqs_all - f)).item())
    if freqs_all[k].item() == 0.0 and f > 0.0:
        k = int(torch.argmin(torch.abs(freqs_all[1:] - f)).item()) + 1
    return k, float(freqs_all[k].item())


@dataclass
class FrequencySelection:
    """FFT bins and matching physical / normalised frequencies.

    All frequency-domain sources and observations must use the same instance
    (same bins). ``time_symbols`` picks ``lambda_t`` / ``lambda_tt``:
    ``"leapfrog"`` (the inversion's) or ``"exact"`` (``iω``, ``-ω²``); defaults
    follow ADR 0001.
    """

    grid: Grid
    fft_bins: torch.Tensor
    frequencies: torch.Tensor
    omega: torch.Tensor
    omega_nd: torch.Tensor
    n_time: int
    dt: float
    dt_nd: float
    requested_frequencies: Optional[torch.Tensor] = None
    time_symbols: str = "leapfrog"
    lambda_t: torch.Tensor = field(init=False, repr=False)
    lambda_tt: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalise field shapes/dtypes and derive lambda_t/lambda_tt."""
        device = self.grid.device
        self.fft_bins = torch.as_tensor(
            self.fft_bins, dtype=torch.long, device=device
        ).reshape(-1)
        self.frequencies = torch.as_tensor(
            self.frequencies, dtype=self.grid.dtype, device=device
        ).reshape(-1)
        self.omega = torch.as_tensor(
            self.omega, dtype=self.grid.dtype, device=device
        ).reshape(-1)
        self.omega_nd = torch.as_tensor(
            self.omega_nd, dtype=self.grid.dtype, device=device
        ).reshape(-1)
        if self.requested_frequencies is not None:
            self.requested_frequencies = torch.as_tensor(
                self.requested_frequencies,
                dtype=self.grid.dtype,
                device=device,
            ).reshape(-1)
        if not (
            self.fft_bins.numel()
            == self.frequencies.numel()
            == self.omega.numel()
            == self.omega_nd.numel()
        ):
            raise ValueError("FrequencySelection fields must share the same length.")
        if (
            self.requested_frequencies is not None
            and self.requested_frequencies.numel() != self.fft_bins.numel()
        ):
            raise ValueError(
                "requested_frequencies length must match selected fft_bins."
            )
        if self.fft_bins.numel() == 0:
            raise ValueError("FrequencySelection must contain at least one bin.")
        if self.time_symbols == "leapfrog":
            self.lambda_t = discrete_lambda_t(self.omega_nd, self.dt_nd)
            self.lambda_tt = discrete_lambda_tt(self.omega_nd, self.dt_nd)
        elif self.time_symbols == "exact":
            self.lambda_t = 1j * self.omega_nd
            self.lambda_tt = -(self.omega_nd**2)
        else:
            raise ValueError(
                f"time_symbols must be one of {list(TIME_SYMBOLS)}, "
                f"got {self.time_symbols!r}"
            )

    @property
    def n_frequencies(self) -> int:
        """Number of selected frequency bins."""
        return int(self.fft_bins.numel())

    @property
    def cdtype(self) -> torch.dtype:
        """Complex dtype matching the grid's real dtype."""
        return complex_dtype(self.grid.dtype)

    def symbols_broadcast(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``lambda_t``, ``lambda_tt`` shaped ``(1, nf, 1, 1)`` for shot batches."""
        nf = self.n_frequencies
        lt = self.lambda_t.to(dtype=self.cdtype).view(1, nf, 1, 1)
        ltt = self.lambda_tt.to(dtype=self.cdtype).view(1, nf, 1, 1)
        return lt, ltt

    @classmethod
    def from_bins(
        cls,
        grid: Grid,
        fft_bins: Union[Sequence[int], torch.Tensor],
        *,
        requested_frequencies: Optional[Union[Sequence[float], torch.Tensor]] = None,
        time_symbols: str = "leapfrog",
    ) -> "FrequencySelection":
        """Build from integer rFFT / FFT bin indices on ``grid.t``."""
        bins = torch.as_tensor(fft_bins, dtype=torch.long, device=grid.device).reshape(
            -1
        )
        n_time = int(grid.nt)
        dt = float(grid.dt)
        if (bins < 0).any() or (bins >= n_time).any():
            raise ValueError(
                f"fft_bins must lie in [0, {n_time - 1}], got {bins.tolist()}"
            )
        # Physical frequencies for the unnormalized DFT on grid.t
        freqs_all = torch.fft.fftfreq(
            n_time, d=dt, device=grid.device, dtype=grid.dtype
        )
        frequencies = freqs_all[bins]
        omega = 2.0 * math.pi * frequencies
        omega_nd = omega * grid.t0
        return cls(
            grid=grid,
            fft_bins=bins,
            frequencies=frequencies,
            omega=omega,
            omega_nd=omega_nd,
            n_time=n_time,
            dt=dt,
            dt_nd=float(grid.dt_nd),
            requested_frequencies=requested_frequencies,
            time_symbols=time_symbols,
        )

    @classmethod
    def from_frequencies(
        cls,
        grid: Grid,
        frequencies_hz: Union[Sequence[float], torch.Tensor],
    ) -> "FrequencySelection":
        """Map physical frequencies (Hz) to nearest positive FFT bins on ``grid.t``."""
        f_req = torch.as_tensor(
            frequencies_hz, dtype=grid.dtype, device=grid.device
        ).reshape(-1)
        n_time = int(grid.nt)
        dt = float(grid.dt)
        freqs_all = torch.fft.fftfreq(
            n_time, d=dt, device=grid.device, dtype=grid.dtype
        )

        bins = []
        for f in f_req:
            k, _ = _map_frequency_to_bin(freqs_all, float(f.item()))
            bins.append(k)
        return cls.from_bins(
            grid,
            bins,
            requested_frequencies=f_req,
        )

    def fft_time_series(self, signal: torch.Tensor, dim: int = -3) -> torch.Tensor:
        """Continuous Fourier transform of ``signal`` along ``dim`` on the
        selected bins, approximated as DFT x ``dt``, so it does not depend on
        ``nt`` at a fixed bin frequency.

        ``signal`` is real-valued in time. Returns complex spectrum with the
        selected frequency axis replacing the time axis at ``dim``; units of
        ``signal`` x s.
        """
        spec = torch.fft.fft(signal, dim=dim) * self.dt
        return torch.index_select(spec, dim, self.fft_bins)
