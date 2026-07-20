"""Explicit leapfrog time-stepping forward solver (FFT sanity reference)."""

import math
import warnings
from typing import Optional

import torch

from odil_wave.wavefield import Wavefield
from .spatial import Laplacian2ndOrder, Laplacian4thOrder, Laplacian10thOrder
from .temporal import (
    TimeOperator2ndOrder,
    _first_time_derivative,
    _make_endpoint_masks,
)


class LeapfrogSolver:
    """Exact explicit solve of the ``time_order=2`` discrete wave equation.

    Retained as a **time-domain reference** for FFT sanity tests. Public ODIL
    path uses :class:`HelmholtzSolver` / frequency residuals.
    """

    def __init__(
        self,
        wavefield: Wavefield,
        geometry,
        space_order: int = 4,
        pml_weight: float = 1.0,
    ) -> None:
        if space_order == 2:
            lap_cls = Laplacian2ndOrder
        elif space_order == 4:
            lap_cls = Laplacian4thOrder
        elif space_order == 10:
            lap_cls = Laplacian10thOrder
        else:
            raise ValueError(f"Invalid space order: {space_order}")

        self.wavefield = wavefield
        self.geometry = geometry
        self.space_order = space_order
        self.pml_weight = float(pml_weight)
        self._lap = lap_cls(wavefield)
        self.diagnostics: Optional[dict] = None

        grid = wavefield.grid
        c_max = float(wavefield.velocity_model.c.max())
        cfl = grid.cfl(c_max)
        if space_order == 2:
            cfl_limit = 1.0
        elif space_order == 4:
            cfl_limit = math.sqrt(3.0) / 2.0
        elif space_order == 10:
            cfl_limit = 2.0 / math.sqrt(6.826666666666667)
        else:
            raise ValueError(f"Invalid space order: {space_order}")
        if cfl >= cfl_limit:
            raise ValueError(
                f"CFL number {cfl:.3f} >= {cfl_limit:.3f} (space_order="
                f"{space_order}): leapfrog is unstable. Reduce dt "
                "(increase nt) or coarsen the spatial grid."
            )
        if cfl > 0.9 * cfl_limit:
            warnings.warn(
                f"CFL number {cfl:.3f} is close to the stability limit "
                f"{cfl_limit:.3f}; consider a smaller dt.",
                RuntimeWarning,
            )

    def solve_time(self, verbose: bool = True) -> torch.Tensor:
        """Time-step all shots; returns real ``(n_shots, nt, nx, ny)``."""
        wf = self.wavefield
        grid = wf.grid
        device, dtype = grid.device, grid.dtype
        NT = grid.nt
        Nx, Ny = grid.shape
        n_shots = self.geometry.n_sources

        sources = (
            torch.stack(
                [
                    self.geometry.source_field_time(s).to(dtype=dtype, device=device)
                    for s in range(n_shots)
                ]
            )
            * grid.t0**2
        )

        dt = grid.dt_nd
        k = (wf.velocity_model.c / grid.c0) ** 2
        sig_s_raw = grid.sigma_x_nd + grid.sigma_y_nd
        sig_p_raw = grid.sigma_x_nd * grid.sigma_y_nd
        sig_s = self.pml_weight * sig_s_raw
        sig_p = self.pml_weight * sig_p_raw
        denom = 1.0 / dt**2 + sig_s / (2.0 * dt)

        amp = torch.zeros(n_shots, NT, Nx, Ny, dtype=dtype, device=device)
        amp[:, 1] = grid.dt * wf.init_ut.to(dtype=dtype, device=device)

        with torch.no_grad():
            for t in range(1, NT - 1):
                u_t, u_tm = amp[:, t], amp[:, t - 1]
                rhs = (
                    (2.0 / dt**2 - sig_p) * u_t
                    + (sig_s / (2.0 * dt) - 1.0 / dt**2) * u_tm
                    + k * self._lap.apply(u_t)
                    + sources[:, t]
                )
                amp[:, t + 1] = rhs / denom

        # Time-domain residual health check (legacy operators).
        time_op = TimeOperator2ndOrder(wf)
        mask_first, mask_last = _make_endpoint_masks(NT, device)
        with torch.no_grad():
            utt = time_op.apply(amp)
            lap = self._lap.apply(amp)
            r = utt - k * lap - sources
            u_t = _first_time_derivative(
                amp, dt, wf.init_ut_nd, mask_first, mask_last
            )
            r = r + self.pml_weight * (sig_s_raw * u_t + sig_p_raw * amp)
            r_rms = float(r.pow(2).mean().sqrt())
            src_rms = float(sources.pow(2).mean().sqrt())
        self.diagnostics = {
            "r_rms": r_rms,
            "src_rms": src_rms,
            "ratio": r_rms / max(src_rms, 1e-30),
            "u_absmax": float(amp.abs().max()),
            "loss_pde": float((r**2).mean(dim=(1, 2, 3)).sum()),
        }
        if verbose:
            print(
                f"leapfrog ({n_shots} shots, space_order={self.space_order}): "
                f"|r_pde|/|src| = {self.diagnostics['ratio']:.3e}, "
                f"|u|_max = {self.diagnostics['u_absmax']:.3e}"
            )
        return amp

    def solve(self, verbose: bool = True) -> torch.Tensor:
        """Alias for :meth:`solve_time` (returns time-domain amplitudes)."""
        return self.solve_time(verbose=verbose)
