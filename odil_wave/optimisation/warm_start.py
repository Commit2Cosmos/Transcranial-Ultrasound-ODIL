"""Forward warm-start helpers for inverse problems."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from odil_wave.geometry import AcquisitionGeometry
from odil_wave.grid import Grid
from odil_wave.operator.conditions import NeumannMirrorBC2nd

from odil_wave.wavefield import Wavefield


def smooth_amplitude(
    amplitude: Union[torch.Tensor, np.ndarray],
    sigma: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Gaussian-smooth ``u(x, y)`` at each time frame.

    ``sigma`` is in grid cells; ``sigma <= 0`` returns the input unchanged.
    """
    if sigma <= 0:
        out = torch.as_tensor(amplitude)
        if dtype is not None:
            out = out.to(dtype=dtype)
        if device is not None:
            out = out.to(device=device)
        return out

    arr = np.asarray(amplitude, dtype=np.float64)
    smoothed = np.empty_like(arr)
    for t in range(arr.shape[0]):
        smoothed[t] = gaussian_filter(arr[t], sigma=sigma, mode="nearest")

    out = torch.from_numpy(smoothed.astype(np.float32))
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    return out


def scale_amplitude_to_traces(
    amplitude: Union[torch.Tensor, np.ndarray],
    geometry: AcquisitionGeometry,
    target_traces: Union[torch.Tensor, np.ndarray],
    *,
    method: str = "max",
    eps: float = 1e-30,
) -> torch.Tensor:
    """Rescale ``u`` so receiver traces match the magnitude of ``target_traces``.

    Explicit time marching on a Stride-matched grid yields ``|u| ~ dt^2`` (tiny)
    while Stride observations are O(1). Scaling aligns the warm-start amplitude
    with the data term before L-BFGS.
    """
    amp = torch.as_tensor(amplitude)
    target = torch.as_tensor(target_traces, dtype=amp.dtype, device=amp.device)
    pred = geometry.extract_observations(amp)
    if method == "max":
        num = target.abs().max()
        den = pred.abs().max()
    elif method == "rms":
        num = torch.sqrt(torch.mean(target**2))
        den = torch.sqrt(torch.mean(pred**2))
    else:
        raise ValueError(f"Unknown trace scale method {method!r}; use 'max' or 'rms'.")

    if float(den) < eps:
        raise RuntimeError(
            "Cannot scale warm-start: time-march produced zero receiver traces."
        )
    return amp * (num / den)


def per_shot_amplitudes(wavefields: Sequence[Wavefield]) -> List[torch.Tensor]:
    """Full ``(NT, NX, NY)`` amplitude tensors, one per shot."""
    return [wf.amplitude.detach().clone() for wf in wavefields]


def _spatial_laplacian(
    u: torch.Tensor,
    bc: NeumannMirrorBC2nd,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """5-point Laplacian on a single ``(NX, NY)`` slice."""

    # ``patch_spatial_neighbors`` expects ``(NT, NX, NY)``.
    u3 = u.unsqueeze(0)

    uxm = torch.roll(u3, 1, dims=1)
    uxp = torch.roll(u3, -1, dims=1)
    uym = torch.roll(u3, 1, dims=2)
    uyp = torch.roll(u3, -1, dims=2)

    uxm, uxp, uym, uyp = bc.patch_spatial_neighbors(
        uxm, uxp, uym, uyp, u3
    )

    lap = (
        (uxm - 2.0 * u3 + uxp) / dx**2
        + (uym - 2.0 * u3 + uyp) / dy**2
    )

    return lap.squeeze(0)


def time_march_amplitude(
    grid: Grid,
    wavespeed: torch.Tensor,
    source: torch.Tensor,
    *,
    init_ut: Optional[torch.Tensor] = None,
    pml_weight: float = 0.0,
) -> torch.Tensor:
    """Explicit 2nd-order FD time step matching ODIL's spatial/PML operators.

    Solves the damped wave equation consistent with ``WaveEquation.residual``:

        u_tt + pml_w*(sigma_x+sigma_y)*u_t + pml_w*sigma_x*sigma_y*u
            = c(x,y)^2 * lap(u) + f

    Returns ``(NT, NX, NY)`` with hard ``u(0)=0``.

    Notes
    -----
    ``pml_weight`` defaults to ``0`` because ODIL's PML ``sigma`` profiles are
    tuned for the global residual, not explicit time integration (``pml_weight=1``
    typically blows up within a few hundred steps).
    """
    device = grid.device
    dtype = grid.dtype
    nt = grid.nt
    c2 = torch.as_tensor(wavespeed, dtype=dtype, device=device).reshape(grid.nx, grid.ny) ** 2
    src = torch.as_tensor(source, dtype=dtype, device=device).reshape(nt, grid.nx, grid.ny)

    if init_ut is None:
        init_ut = torch.zeros(grid.nx, grid.ny, dtype=dtype, device=device)
    else:
        init_ut = torch.as_tensor(init_ut, dtype=dtype, device=device).reshape(grid.nx, grid.ny)

    c_max = float(torch.sqrt(c2.max()).item())
    cfl = grid.cfl(c_max)
    if cfl > 0.7:
        print(
            f"  warning: CFL={cfl:.3f} > 0.7 — time-march may be unstable "
            f"(dt={grid.dt:.3e}, c_max={c_max:.0f})",
            flush=True,
        )

    bc = NeumannMirrorBC2nd()
    sigma_sum = grid.sigma_x + grid.sigma_y
    sigma_prod = grid.sigma_x * grid.sigma_y
    dt = grid.dt
    dt2 = dt * dt

    u_prev = -dt * init_ut
    u_curr = torch.zeros(grid.nx, grid.ny, dtype=dtype, device=device)
    amps = torch.zeros(nt, grid.nx, grid.ny, dtype=dtype, device=device)
    amps[0] = u_curr

    for n in range(nt - 1):
        lap_u = _spatial_laplacian(u_curr, bc, grid.dx, grid.dy)
        u_t = (u_curr - u_prev) / dt
        rhs = (
            c2 * lap_u
            + src[n]
            - pml_weight * (sigma_sum * u_t + sigma_prod * u_curr)
        )
        u_next = 2.0 * u_curr - u_prev + dt2 * rhs
        if not torch.isfinite(u_next).all():
            raise RuntimeError(
                f"time-march became non-finite at step {n}; CFL={cfl:.3f}"
            )
        u_prev = u_curr
        u_curr = u_next
        amps[n + 1] = u_curr

    return amps


def time_march_shots(
    grid: Grid,
    wavespeed: torch.Tensor,
    geometry: AcquisitionGeometry,
    *,
    pml_weight: float = 0.0,
    smooth_sigma: float = 0.0,
    target_traces: Optional[Sequence[Union[torch.Tensor, np.ndarray]]] = None,
    trace_scale: str = "max",
) -> List[torch.Tensor]:
    """Time-march each shot; optional smooth and trace-amplitude scaling."""
    amps: List[torch.Tensor] = []
    for s in range(geometry.n_sources):
        u = time_march_amplitude(
            grid,
            wavespeed,
            geometry.source_field(s),
            pml_weight=pml_weight,
        )
        if smooth_sigma > 0:
            u = smooth_amplitude(u, smooth_sigma, device=grid.device, dtype=grid.dtype)
        if target_traces is not None:
            u = scale_amplitude_to_traces(
                u, geometry, target_traces[s], method=trace_scale
            )
        amps.append(u)
    return amps
