"""Frequency-domain wave equation residual (discrete temporal symbols)."""

from dataclasses import dataclass, field

import torch

from odil_wave.wavefield import Wavefield
from .base import DenseOperator
from .spatial import (
    Laplacian2ndOrder,
    Laplacian4thOrder,
    Laplacian6thOrder,
    Laplacian8thOrder,
    Laplacian10thOrder,
)
from .conditions import Conditions, NeumannMirrorBC2nd, NeumannMirrorBC4th, Sponge


@dataclass
class WaveEquation:
    """Discrete frequency-domain acoustic residual matching the time ODIL form.

    Time residual (normalized)::

        r = D_tt[u] - c'^2 ∇'^2 u - f'
          + w_sponge [ σ_sum_nd D_t[u] + σ_prod_nd u ]

    Frequency form (2nd-order centered symbols from FrequencySelection)::

        r(ω) = λ_tt u - c'^2 ∇'^2 u - f̂'
             + w_sponge [ σ_sum_nd λ_t u + σ_prod_nd u ]

    with ``λ_t = i sin(ω_nd Δt') / Δt'`` and
    ``λ_tt = -(4/Δt'²) sin²(ω_nd Δt'/2)``.

    Amplitudes / sources are complex ``(n_shots, n_frequencies, nx, ny)``.
    """

    wavefield: Wavefield
    space_order: int = 2
    pml_weight: float = 1.0
    # Accepted for API compatibility with older call sites; ignored.
    time_order: int = 2

    _lap: DenseOperator = field(init=False)
    _bc: Conditions = field(init=False)
    _sponge: Sponge = field(init=False)

    def __post_init__(self):
        if self.space_order == 2:
            self._lap = Laplacian2ndOrder(self.wavefield)
            self._bc = NeumannMirrorBC2nd()
        elif self.space_order == 4:
            self._lap = Laplacian4thOrder(self.wavefield)
            self._bc = NeumannMirrorBC4th()
        elif self.space_order == 6:
            self._lap = Laplacian6thOrder(self.wavefield)
            self._bc = NeumannMirrorBC4th()
        elif self.space_order == 8:
            self._lap = Laplacian8thOrder(self.wavefield)
            self._bc = NeumannMirrorBC4th()
        elif self.space_order == 10:
            self._lap = Laplacian10thOrder(self.wavefield)
            self._bc = NeumannMirrorBC4th()
        else:
            raise ValueError(f"Invalid space order: {self.space_order}")
        self._sponge = Sponge(self.wavefield, self.pml_weight)

    def residual(
        self, amp: torch.Tensor, wsp: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Complex dimensionless residual on shot-batched frequency fields.

        ``wsp`` is physical wavespeed (m/s). ``source`` is pre-scaled by ``t0**2``.
        """
        c0 = self.wavefield.grid.c0
        wsp_nd = wsp / c0
        freq = self.wavefield.frequency_selection
        lambda_t, lambda_tt = freq.symbols_broadcast()
        # Ensure symbol dtype matches amp
        lambda_t = lambda_t.to(dtype=amp.dtype, device=amp.device)
        lambda_tt = lambda_tt.to(dtype=amp.dtype, device=amp.device)

        lap = self._lap.apply(amp, bc=self._bc)
        r = lambda_tt * amp - wsp_nd**2 * lap - source
        return self._sponge.apply(r, amp, lambda_t)
