"""Frequency-domain wave equation residual (discrete temporal symbols)."""

from dataclasses import dataclass, field

import torch

from odil_wave.wavefield import Wavefield
from .spatial import Laplacian
from .conditions import Pml


@dataclass
class WaveEquation:
    """Discrete frequency-domain acoustic residual with a stretched-coordinate
    PML (:class:`~odil_wave.operator.conditions.Pml`)::

        r(ω) = λ_tt u - c'^2 (Λ̃_x u / s_x + Λ̃_y u / s_y) - f̂' / (s_x s_y)

    (the conservative stretched form with its rows divided by ``s_x s_y``,
    ADR 0002) with ``s = 1 + damping_scale σ_nd / (iω_nd)`` (``1`` = designed
    PML absorption, ``0`` disables it) and the 2nd-order centred symbol
    ``λ_tt = -(4/Δt'²) sin²(ω_nd Δt'/2)`` from FrequencySelection. In the
    interior ``s = 1`` and ``r = λ_tt u - c'^2 ∇'^2 u - f̂'``. Defaults follow
    ADR 0001.

    Amplitudes / sources are complex ``(n_shots, n_frequencies, nx, ny)``.
    """

    wavefield: Wavefield
    space_order: int = field(kw_only=True)
    damping_scale: float = field(default=1.0, kw_only=True)

    pml: Pml = field(init=False)
    _lap: Laplacian = field(init=False)

    def __post_init__(self):
        """Build the second differences for ``space_order`` and the PML."""
        self._lap = Laplacian(self.wavefield, space_order=self.space_order)
        self.pml = Pml(self.wavefield, self.damping_scale)

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
        """Complex dimensionless PDE residual, shaped as ``amp`` ``(n_shots,
        n_frequencies, nx, ny)``; ``wsp`` in m/s, ``source`` scaled by
        ``t0**2``."""
        c0 = self.wavefield.grid.c0
        wsp_nd = wsp / c0
        _, lambda_tt = self.wavefield.frequency_selection.symbols_broadcast()
        lambda_tt = lambda_tt.to(dtype=amp.dtype, device=amp.device)
        u_xx, u_yy = self._lap.axes(amp)
        lap = self.pml.stretched_laplacian(amp, u_xx, u_yy)
        return lambda_tt * amp - wsp_nd**2 * lap - self.pml.row_weight * source
