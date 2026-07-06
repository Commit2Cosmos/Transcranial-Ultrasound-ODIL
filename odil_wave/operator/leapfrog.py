"""Explicit leapfrog time-stepping forward solver.
"""

import math
import warnings
from typing import List, Optional

import torch

from odil_wave.wavefield import Wavefield
from .spatial import Laplacian2ndOrder, Laplacian4thOrder, Laplacian10thOrder
from .utils import WaveEquation


class LeapfrogSolver:
    """Exact explicit solve of the ``time_order=2`` discrete wave equation."""

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
        # Leapfrog limit: dt^2 * lambda_max(-c^2 lap_h) <= 4. The 2nd-order
        # Laplacian has max eigenvalue 4/h^2 per axis -> cfl <= 1; the
        # 4th-order stencil reaches (16/3)/h^2 -> cfl <= sqrt(3)/2.
        if space_order == 2:
            cfl_limit = 1.0
        elif space_order == 4:
            cfl_limit = math.sqrt(3.0) / 2.0
        elif space_order == 10:
            # 10th-order 1D second-derivative stencil has max eigenvalue
            # approximately 6.8267 / h^2, so the 2D CFL limit is 2/sqrt(6.8267).
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

    def solve(self, verbose: bool = True) -> List[Wavefield]:
        """Time-step all shots; returns one `Wavefield` per shot."""
        wf = self.wavefield
        grid = wf.grid
        device, dtype = grid.device, grid.dtype
        NT = grid.nt
        Nx, Ny = grid.shape
        n_shots = self.geometry.n_sources

        # Same non-dimensional scaling as DiscreteLoss: f' = t0^2 * f.
        sources = (
            torch.stack(
                [
                    self.geometry.source_field(s).to(dtype=dtype, device=device)
                    for s in range(n_shots)
                ]
            )
            * grid.t0**2
        )

        dt = grid.dt_nd
        k = (wf.velocity_model.c / grid.c0) ** 2
        sig_s = self.pml_weight * (grid.sigma_x_nd + grid.sigma_y_nd)
        sig_p = self.pml_weight * (grid.sigma_x_nd * grid.sigma_y_nd)
        denom = 1.0 / dt**2 + sig_s / (2.0 * dt)

        amp = torch.zeros(n_shots, NT, Nx, Ny, dtype=dtype, device=device)
        # Hard ICs matching LBFGSB's fixed rows: u[0] = 0, u[1] = dt * u_t(0).
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

        # Health check with the matching discrete residual.
        wave_eq = WaveEquation(
            wf,
            time_order=2,
            space_order=self.space_order,
            pml_weight=self.pml_weight,
        )
        with torch.no_grad():
            r = wave_eq.residual(amp, wf.velocity_model.c, sources)
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

        outputs: List[Wavefield] = []
        for s in range(n_shots):
            out = Wavefield(grid=grid, velocity_model=wf.velocity_model)
            out.amplitude = amp[s]
            outputs.append(out)
        return outputs
