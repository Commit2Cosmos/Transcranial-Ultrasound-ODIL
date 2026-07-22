"""Shared frequency-bin selection for sources, data, and the Helmholtz residual.

Uses the NumPy / PyTorch FFT convention::

    U[k] = sum_n u[n] exp(-2 π i k n / N)

Temporal derivatives in the frequency residual use *discrete* symbols of the
centered second-order FD stencils (not continuous ±iω / -ω²), so that the FFT
of a leapfrog wavefield can have a small PDE residual::

    λ_t  = i * sin(ω_nd Δt') / Δt'
    λ_tt = -(4 / Δt'²) * sin²(ω_nd Δt' / 2)

Continuous +i ω_nd / -ω_nd² are the low-frequency limit (sin x ≈ x).

Optional practical limits (``validate_limits=True``) reject FFT bins that are
usable mathematically but poorly supported by recording length, PML thickness,
spatial PPW, Nyquist margin, or source spectrum. Limits are **off by default**
so existing experiments are unchanged until callers opt in.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

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


def default_min_ppw(space_order: int) -> float:
    """Points-per-wavelength defaults by spatial FD order.

    * ``space_order=2`` -> 10 PPW (2nd-order Laplacian is dispersive)
    * ``space_order=4`` -> 6 PPW
    * ``space_order=10`` -> 4 PPW
    """
    if space_order <= 2:
        return 10.0
    if space_order <= 4:
        return 6.0
    return 4.0


@dataclass
class FrequencyLimits:
    """Global usable frequency interval and the ingredients that define it."""

    nt: int
    dt: float
    record_duration: float
    delta_f: float
    f_nyquist: float
    nyquist_fraction: float
    f_temporal_max: float
    f_duration_min: float
    f_pml_min: float
    f_spatial_max: float
    f_source_min: Optional[float]
    f_source_max: Optional[float]
    f_lo: float
    f_hi: float
    c_min: float
    c_max: float
    min_ppw: float
    pml_thickness: float
    lowest_positive_fft_hz: float


def usable_frequency_limits(
    grid: Grid,
    *,
    source_spectrum: Optional[torch.Tensor] = None,
    min_record_cycles: float = 2.0,
    source_rel_threshold: float = 0.05,
    min_pml_wavelengths: float = 0.5,
    nyquist_fraction: float = 0.85,
    min_ppw: Optional[float] = None,
    space_order: int = 2,
    c_min: Optional[float] = None,
    c_max: Optional[float] = None,
) -> FrequencyLimits:
    """Compute practical lower/upper FFT frequency limits for ``grid``.

    Parameters
    ----------
    source_spectrum :
        Full complex FFT of the source waveform on ``grid.t``, length ``nt``,
        same convention as ``torch.fft.fft`` / ``fftfreq``. If ``None``, source
        bounds do not tighten the usable interval.
    c_min, c_max :
        Velocities for spatial PPW and PML wavelength checks. Default to
        ``grid.c_min`` / ``grid.c_max``. ``c_min`` is required (explicitly or
        on the grid) when limits are used for spatial checks.
    """
    nt = int(grid.nt)
    dt = float(grid.dt)
    if nt < 2 or dt <= 0.0:
        raise ValueError(f"Invalid time axis: nt={nt}, dt={dt}")

    record_duration = nt * dt # T = Nt * dt
    delta_f = 1.0 / record_duration # bin spacing = 1/T
    f_nyquist = 1.0 / (2.0 * dt) # in theory: 1/ 2*dt, highest freq
    f_temporal_max = float(nyquist_fraction) * f_nyquist
    f_duration_min = float(min_record_cycles) / record_duration

    c_max_used = float(grid.c_max if c_max is None else c_max)
    if c_min is not None:
        c_min_used = float(c_min)
    elif grid.c_min is not None:
        c_min_used = float(grid.c_min)
    else:
        raise ValueError(
            "c_min is required for spatial PPW limits; pass c_min=... or set "
            "grid.c_min."
        )
    if c_min_used <= 0.0 or c_max_used <= 0.0:
        raise ValueError(f"c_min/c_max must be positive, got {c_min_used}, {c_max_used}")

    dx = float(grid.dx)
    dy = float(grid.dy)
    pml_thickness = float(grid.pml_width) * dx
    if grid.pml_width <= 0 or pml_thickness <= 0.0:
        f_pml_min = 0.0
    else:
        # Longest wavelength at fixed f uses c_max → hardest PML requirement.
        f_pml_min = float(min_pml_wavelengths) * c_max_used / pml_thickness

    ppw = float(default_min_ppw(space_order) if min_ppw is None else min_ppw)
    if ppw <= 0.0:
        raise ValueError(f"min_ppw must be positive, got {ppw}")
    f_spatial_max = c_min_used / (ppw * max(dx, dy))

    freqs_all = torch.fft.fftfreq(nt, d=dt, dtype=torch.float64)
    pos = freqs_all > 0
    if not bool(pos.any()):
        raise ValueError("No positive FFT frequencies on this time axis.")
    lowest_positive = float(freqs_all[pos].min().item())

    f_source_min: Optional[float] = None
    f_source_max: Optional[float] = None
    if source_spectrum is not None:
        spec = torch.as_tensor(source_spectrum).reshape(-1)
        if spec.numel() != nt:
            raise ValueError(
                f"source_spectrum length {spec.numel()} != grid.nt={nt}. "
                "Pass the full FFT of s(t) on grid.t."
            )
        amp = spec.abs().to(dtype=torch.float64)
        peak = float(amp.max().item())
        if peak <= 0.0:
            raise ValueError("source_spectrum has zero peak amplitude.")
        rel = amp / peak
        supported = pos & (rel >= float(source_rel_threshold))
        if bool(supported.any()):
            f_sup = freqs_all[supported]
            f_source_min = float(f_sup.min().item())
            f_source_max = float(f_sup.max().item())
        else:
            f_source_min = math.inf
            f_source_max = 0.0

    lo_candidates = [f_duration_min, f_pml_min]
    hi_candidates = [f_temporal_max, f_spatial_max]
    if f_source_min is not None:
        lo_candidates.append(f_source_min)
    if f_source_max is not None:
        hi_candidates.append(f_source_max)

    f_lo = max(lo_candidates)
    f_hi = min(hi_candidates)

    return FrequencyLimits(
        nt=nt,
        dt=dt,
        record_duration=record_duration,
        delta_f=delta_f,
        f_nyquist=f_nyquist,
        nyquist_fraction=float(nyquist_fraction),
        f_temporal_max=f_temporal_max,
        f_duration_min=f_duration_min,
        f_pml_min=f_pml_min,
        f_spatial_max=f_spatial_max,
        f_source_min=f_source_min,
        f_source_max=f_source_max,
        f_lo=f_lo,
        f_hi=f_hi,
        c_min=c_min_used,
        c_max=c_max_used,
        min_ppw=ppw,
        pml_thickness=pml_thickness,
        lowest_positive_fft_hz=lowest_positive,
    )


def _map_frequency_to_bin(
    freqs_all: torch.Tensor, f_hz: float
) -> tuple[int, float]:
    """Nearest FFT bin for a requested physical frequency (prefer non-DC)."""
    f = float(f_hz)
    k = int(torch.argmin(torch.abs(freqs_all - f)).item())
    if freqs_all[k].item() == 0.0 and f > 0.0:
        k = int(torch.argmin(torch.abs(freqs_all[1:] - f)).item()) + 1
    return k, float(freqs_all[k].item())


def _source_rel_at_bin(
    source_spectrum: Optional[torch.Tensor], bin_idx: int
) -> Optional[float]:
    if source_spectrum is None:
        return None
    spec = torch.as_tensor(source_spectrum).reshape(-1)
    peak = float(spec.abs().max().item())
    if peak <= 0.0:
        return 0.0
    return float(spec[bin_idx].abs().item() / peak)


@dataclass
class FrequencySelection:
    """FFT bins and matching physical / normalised frequencies.

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
    requested_frequencies: Optional[torch.Tensor] = None
    lambda_t: torch.Tensor = field(init=False, repr=False)
    lambda_tt: torch.Tensor = field(init=False, repr=False)
    limits: Optional[FrequencyLimits] = field(default=None, repr=False)
    validation_rows: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

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
        requested_frequencies: Optional[Union[Sequence[float], torch.Tensor]] = None,
        limits: Optional[FrequencyLimits] = None,
        validation_rows: Optional[List[Dict[str, Any]]] = None,
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
            requested_frequencies=requested_frequencies,
            limits=limits,
            validation_rows=validation_rows,
        )

    @classmethod
    def from_frequencies(
        cls,
        grid: Grid,
        frequencies_hz: Union[Sequence[float], torch.Tensor],
        *,
        fft_norm: Optional[str] = None,
        source_spectrum: Optional[torch.Tensor] = None,
        validate_limits: bool = False,
        strict: bool = True,
        min_record_cycles: float = 2.0,
        source_rel_threshold: float = 0.05,
        min_pml_wavelengths: float = 0.5,
        nyquist_fraction: float = 0.85,
        min_ppw: Optional[float] = None,
        space_order: int = 2,
        c_min: Optional[float] = None,
        c_max: Optional[float] = None,
        verbose: bool = True,
    ) -> "FrequencySelection":
        """Map physical frequencies (Hz) to nearest positive FFT bins on ``grid.t``.

        When ``validate_limits=False`` (default), behaviour matches the historical
        nearest-bin mapping with no filtering.

        When ``validate_limits=True``, requested frequencies are checked against
        recording duration, PML thickness, spatial PPW, Nyquist margin, and
        optional source-spectrum support. Invalid entries raise (``strict=True``)
        or are dropped with warnings (``strict=False``). Frequencies are never
        silently clamped to a different usable value.
        """
        f_req = torch.as_tensor(
            frequencies_hz, dtype=grid.dtype, device=grid.device
        ).reshape(-1)
        n_time = int(grid.nt)
        dt = float(grid.dt)
        freqs_all = torch.fft.fftfreq(n_time, d=dt, device=grid.device, dtype=grid.dtype)

        if not validate_limits:
            bins = []
            for f in f_req:
                k, _ = _map_frequency_to_bin(freqs_all, float(f.item()))
                bins.append(k)
            return cls.from_bins(
                grid,
                bins,
                fft_norm=fft_norm,
                requested_frequencies=f_req,
            )

        limits = usable_frequency_limits(
            grid,
            source_spectrum=source_spectrum,
            min_record_cycles=min_record_cycles,
            source_rel_threshold=source_rel_threshold,
            min_pml_wavelengths=min_pml_wavelengths,
            nyquist_fraction=nyquist_fraction,
            min_ppw=min_ppw,
            space_order=space_order,
            c_min=c_min,
            c_max=c_max,
        )

        rows: List[Dict[str, Any]] = []
        selected_bins: List[int] = []
        selected_req: List[float] = []
        seen_bins: Dict[int, float] = {}
        removed: List[str] = []

        for f_tensor in f_req:
            f_requested = float(f_tensor.item())
            bin_idx, f_mapped = _map_frequency_to_bin(freqs_all, f_requested)
            src_rel = _source_rel_at_bin(source_spectrum, bin_idx)

            pass_duration = f_mapped >= limits.f_duration_min - 1e-15
            pass_pml = f_mapped >= limits.f_pml_min - 1e-15
            pass_spatial = f_mapped <= limits.f_spatial_max + 1e-15
            pass_temporal = f_mapped <= limits.f_temporal_max + 1e-15
            pass_source = True
            if src_rel is not None:
                pass_source = src_rel >= float(source_rel_threshold) - 1e-15

            reasons: List[str] = []
            if not pass_duration:
                reasons.append(
                    f"below duration limit ({f_mapped:.6g} < {limits.f_duration_min:.6g} Hz)"
                )
            if not pass_pml:
                reasons.append(
                    f"below PML limit ({f_mapped:.6g} < {limits.f_pml_min:.6g} Hz)"
                )
            if not pass_spatial:
                reasons.append(
                    f"above spatial PPW limit ({f_mapped:.6g} > {limits.f_spatial_max:.6g} Hz)"
                )
            if not pass_temporal:
                reasons.append(
                    f"above Nyquist margin ({f_mapped:.6g} > {limits.f_temporal_max:.6g} Hz)"
                )
            if not pass_source:
                reasons.append(
                    f"insufficient source amplitude "
                    f"(rel={src_rel:.3g} < {source_rel_threshold:g})"
                )
            if bin_idx in seen_bins:
                reasons.append(
                    f"duplicate FFT bin {bin_idx} "
                    f"(also requested {seen_bins[bin_idx]:.6g} Hz)"
                )

            ok = len(reasons) == 0
            decision = "keep" if ok else "reject"
            reason = "; ".join(reasons) if reasons else ""
            row = {
                "requested_hz": f_requested,
                "mapped_hz": f_mapped,
                "fft_bin": bin_idx,
                "source_rel": src_rel,
                "pass_duration": pass_duration,
                "pass_pml": pass_pml,
                "pass_spatial": pass_spatial,
                "pass_temporal": pass_temporal,
                "pass_source": pass_source,
                "decision": decision,
                "reason": reason,
            }
            rows.append(row)

            if ok:
                seen_bins[bin_idx] = f_requested
                selected_bins.append(bin_idx)
                selected_req.append(f_requested)
            else:
                msg = (
                    f"reject {f_requested:.6g} Hz → bin {bin_idx} "
                    f"({f_mapped:.6g} Hz): {reason}"
                )
                removed.append(msg)
                if strict:
                    if verbose:
                        _print_frequency_diagnostics(limits, rows, selected_req)
                    raise ValueError(
                        "Requested frequency failed validation "
                        f"(strict=True):\n  {msg}\n"
                        f"Usable interval: [{limits.f_lo:.6g}, {limits.f_hi:.6g}] Hz"
                    )

        if verbose:
            _print_frequency_diagnostics(limits, rows, selected_req)

        if not selected_bins:
            raise ValueError(
                "No valid FFT frequencies remain after filtering.\n"
                + ("\n".join(f"  {m}" for m in removed) if removed else "")
            )

        if removed and not strict:
            warnings.warn(
                "Dropped invalid requested frequencies (strict=False):\n"
                + "\n".join(f"  - {m}" for m in removed),
                UserWarning,
                stacklevel=2,
            )

        return cls.from_bins(
            grid,
            selected_bins,
            fft_norm=fft_norm,
            requested_frequencies=selected_req,
            limits=limits,
            validation_rows=rows,
        )

    def fft_time_series(self, signal: torch.Tensor, dim: int = -3) -> torch.Tensor:
        """FFT ``signal`` along ``dim`` and gather selected bins.

        ``signal`` is real-valued in time. Returns complex spectrum with the
        selected frequency axis replacing the time axis at ``dim``.
        """
        spec = torch.fft.fft(signal, dim=dim, norm=self.fft_norm)
        return torch.index_select(spec, dim, self.fft_bins)


def _fmt_hz(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "inf"
    return f"{x:.6g}"


def _print_frequency_diagnostics(
    limits: FrequencyLimits,
    rows: Sequence[Dict[str, Any]],
    selected_req: Sequence[float],
) -> None:
    print("FrequencySelection limits")
    print(
        f"  nt={limits.nt}  dt={limits.dt:.6g} s  "
        f"duration={limits.record_duration:.6g} s  "
        f"df={limits.delta_f:.6g} Hz"
    )
    print(
        f"  lowest +FFT bin={_fmt_hz(limits.lowest_positive_fft_hz)} Hz  "
        f"Nyquist={_fmt_hz(limits.f_nyquist)} Hz  "
        f"Nyquist x{limits.nyquist_fraction:g}={_fmt_hz(limits.f_temporal_max)} Hz"
    )
    print(
        f"  duration_min={_fmt_hz(limits.f_duration_min)} Hz  "
        f"pml_min={_fmt_hz(limits.f_pml_min)} Hz "
        f"(thickness={limits.pml_thickness:.6g} m, c_max={limits.c_max:.6g})  "
        f"spatial_max={_fmt_hz(limits.f_spatial_max)} Hz "
        f"(c_min={limits.c_min:.6g}, min_ppw={limits.min_ppw:g})"
    )
    print(
        f"  source-supported=[{_fmt_hz(limits.f_source_min)}, "
        f"{_fmt_hz(limits.f_source_max)}] Hz"
    )
    print(
        f"  usable=[{_fmt_hz(limits.f_lo)}, {_fmt_hz(limits.f_hi)}] Hz"
    )
    print(
        "  req_Hz     map_Hz   bin   src_rel  dur  pml  spat  nyq  decision  reason"
    )
    for r in rows:
        src = "n/a" if r["source_rel"] is None else f"{r['source_rel']:.3f}"
        print(
            f"  {r['requested_hz']:9.4g} "
            f"{r['mapped_hz']:9.4g} "
            f"{r['fft_bin']:4d} "
            f"{src:>7}  "
            f"{'Y' if r['pass_duration'] else 'N':>3}  "
            f"{'Y' if r['pass_pml'] else 'N':>3}  "
            f"{'Y' if r['pass_spatial'] else 'N':>3}  "
            f"{'Y' if r['pass_temporal'] else 'N':>3}  "
            f"{r['decision']:<7}  {r['reason']}"
        )
    sel = ", ".join(f"{f:.6g}" for f in selected_req) if selected_req else "(none)"
    print(f"  final selected (requested): {sel}")
    mapped = [r["mapped_hz"] for r in rows if r["decision"] == "keep"]
    print(
        "  final selected (mapped Hz): "
        + (", ".join(f"{f:.6g}" for f in mapped) if mapped else "(none)")
    )
