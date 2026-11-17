"""Frequency-domain wave equation residual (discrete temporal symbols)."""

from dataclasses import dataclass, field

import torch

from odil_wave.wavefield import Wavefield
from .spatial import Laplacian
from .conditions import Sponge


@dataclass
class WaveEquation:
    """Discrete frequency-domain acoustic residual matching the time ODIL form.

    Time residual (normalized)::

        r = D_tt[u] - c'^2 ∇'^2 u - f'
          + w_sponge [ σ_sum_nd D_t[u] + σ_prod_nd u ]

    with ``w_sponge = damping_scale`` (``1`` = designed PML absorption, ``0``
    disables it). ``space_order`` (2 | 4 | 6 | 8 | 10) is required; the library
    owns the ``damping_scale`` default (ADR 0001).

    Frequency form (2nd-order centered symbols from FrequencySelection)::

        r(ω) = λ_tt u - c'^2 ∇'^2 u - f̂'
             + w_sponge [ σ_sum_nd λ_t u + σ_prod_nd u ]

    with ``λ_t = i sin(ω_nd Δt') / Δt'`` and
    ``λ_tt = -(4/Δt'²) sin²(ω_nd Δt'/2)``.

    Amplitudes / sources are complex ``(n_shots, n_frequencies, nx, ny)``.
    """

    wavefield: Wavefield
    space_order: int = field(kw_only=True)
    damping_scale: float = field(default=1.0, kw_only=True)

    _lap: Laplacian = field(init=False)
    _sponge: Sponge = field(init=False)

    def __post_init__(self):
        """Build the Laplacian for ``space_order`` and the sponge."""
        self._lap = Laplacian(self.wavefield, space_order=self.space_order)
        self._sponge = Sponge(self.wavefield, self.damping_scale)

    def sources(self, geometry) -> torch.Tensor:
        """Every shot's source term, scaled by ``t0**2`` as :meth:`residual`
        takes it: complex ``(n_shots, n_frequencies, nx, ny)``."""
        grid = self.wavefield.grid
        cdtype = self.wavefield.cdtype
        return (
            torch.stack(
                [
                    geometry.source_field(i).to(dtype=cdtype, device=grid.device)
                    for i in range(geometry.n_shots)
                ]
            )
            * grid.t0**2
        )

    def residual(
        self, amp: torch.Tensor, wsp: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Complex dimensionless PDE residual on shot-batched frequency fields.

        Parameters
        ----------
        amp : torch.Tensor
            Complex amplitudes ``(n_shots, n_frequencies, nx, ny)``.
        wsp : torch.Tensor
            Physical wavespeed field (m/s).
        source : torch.Tensor
            Source term pre-scaled by ``t0**2``.

        Returns
        -------
        torch.Tensor
            Complex residual, same shape as ``amp``.
        """
        c0 = self.wavefield.grid.c0
        wsp_nd = wsp / c0
        freq = self.wavefield.frequency_selection
        lambda_t, lambda_tt = freq.symbols_broadcast()
        # Ensure symbol dtype matches amp
        lambda_t = lambda_t.to(dtype=amp.dtype, device=amp.device)
        lambda_tt = lambda_tt.to(dtype=amp.dtype, device=amp.device)

        lap = self._lap.apply(amp)
        r = lambda_tt * amp - wsp_nd**2 * lap - source
        return self._sponge.apply_residual(r, amp, lambda_t)
