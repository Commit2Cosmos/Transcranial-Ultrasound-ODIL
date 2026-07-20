"""Shared frequency-bin selection for sources, data, and the Helmholtz residual.

Uses the NumPy / PyTorch FFT convention::

    U[k] = sum_n u[n] exp(-2 π i k n / N)

Temporal derivatives in the frequency residual use *discrete* symbols of the
centered second-order FD stencils (not continuous ±iω / -ω²), so that the FFT
of a leapfrog wavefield can have a small PDE residual::

    λ_t  = i * sin(ω_nd Δt') / Δt'
    λ_tt = -(4 / Δt'²) * sin²(ω_nd Δt' / 2)

Continuous +i ω_nd / -ω_nd² are the low-frequency limit (sin x ≈ x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import torch

from odil_wave.grid.grid import Grid


def complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    raise ValueError(f"Unsupported real dtype for complex promotion: {real_dtype}")


def discrete_lambda_t(omega_nd: torch.Tensor, dt_nd: float) -> torch.Tensor:
    """Centered 2nd-order D_t symbol: i sin(ω_nd Δt') / Δt'."""
    return 1j * torch.sin(omega_nd * dt_nd) / dt_nd


def discrete_lambda_tt(omega_nd: torch.Tensor, dt_nd: float) -> torch.Tensor:
    """Centered 2nd-order D_tt symbol: -(4/Δt'²) sin²(ω_nd Δt'/2)."""
    return -(4.0 / dt_nd**2) * torch.sin(0.5 * omega_nd * dt_nd) ** 2


@dataclass
class FrequencySelection:
    """FFT bins and matching physical / normalized frequencies.

    All frequency-domain sources and observations must use the same instance
    (same bins and FFT normalization).
    """

    grid: Grid
    fft_bins: torch.Tensor
    frequencies: torch.Tensor
    omega: torch.Tensor
    omega_nd: torch.Tensor
    n_time: int
    dt: float
    dt_nd: float
    fft_norm: Optional[str] = None  # torch.fft norm flag; None = unnormalized
    lambda_t: torch.Tensor = field(init=False, repr=False)
    lambda_tt: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
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
        if not (
            self.fft_bins.numel()
            == self.frequencies.numel()
            == self.omega.numel()
            == self.omega_nd.numel()
        ):
            raise ValueError("FrequencySelection fields must share the same length.")
        if self.fft_bins.numel() == 0:
            raise ValueError("FrequencySelection must contain at least one bin.")
        self.lambda_t = discrete_lambda_t(self.omega_nd, self.dt_nd)
        self.lambda_tt = discrete_lambda_tt(self.omega_nd, self.dt_nd)

    @property
    def n_frequencies(self) -> int:
        return int(self.fft_bins.numel())

    @property
    def cdtype(self) -> torch.dtype:
        return complex_dtype(self.grid.dtype)

    def symbols_broadcast(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``λ_t``, ``λ_tt`` shaped ``(1, nf, 1, 1)`` for shot-batched fields."""
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
        fft_norm: Optional[str] = None,
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
        freqs_all = torch.fft.fftfreq(n_time, d=dt, device=grid.device, dtype=grid.dtype)
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
            fft_norm=fft_norm,
        )

    @classmethod
    def from_frequencies(
        cls,
        grid: Grid,
        frequencies_hz: Union[Sequence[float], torch.Tensor],
        *,
        fft_norm: Optional[str] = None,
    ) -> "FrequencySelection":
        """Map physical frequencies (Hz) to nearest positive FFT bins on ``grid.t``."""
        f_req = torch.as_tensor(
            frequencies_hz, dtype=grid.dtype, device=grid.device
        ).reshape(-1)
        n_time = int(grid.nt)
        dt = float(grid.dt)
        freqs_all = torch.fft.fftfreq(n_time, d=dt, device=grid.device, dtype=grid.dtype)
        # Prefer non-negative bins for real time series (rfft-compatible).
        bins = []
        for f in f_req:
            # Nearest bin by absolute frequency distance
            k = int(torch.argmin(torch.abs(freqs_all - f)).item())
            if freqs_all[k].item() == 0.0 and f.item() > 0:
                # Avoid DC if a positive frequency was requested
                k = int(torch.argmin(torch.abs(freqs_all[1:] - f)).item()) + 1
            bins.append(k)
        return cls.from_bins(grid, bins, fft_norm=fft_norm)

    def fft_time_series(self, signal: torch.Tensor, dim: int = -3) -> torch.Tensor:
        """FFT ``signal`` along ``dim`` and gather selected bins.

        ``signal`` is real-valued in time. Returns complex spectrum with the
        selected frequency axis replacing the time axis at ``dim``.
        """
        spec = torch.fft.fft(signal, dim=dim, norm=self.fft_norm)
        return torch.index_select(spec, dim, self.fft_bins)
