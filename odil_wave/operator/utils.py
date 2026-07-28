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
    The residual is affine in ``k = (c/c0)²`` per cell, so ``c_closed_form``
    gives the exact per-cell wavespeed that minimises ``Σ|r|²`` for a fixed
    ``u`` (variable projection).
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

    def c_closed_form(
        self,
        amp: torch.Tensor,
        source: torch.Tensor,
        c_current: torch.Tensor | None = None,
        illum_rel_floor: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-cell least-squares wavespeed given the complex wavefield.

        Because the residual is affine in ``k = (c/c0)²`` at each cell::

            r = b - k a,   a = ∇'^2 u,   b = λ_tt u - f̂' + sponge(u),

        minimising ``Σ_{shots,freq} |r|²`` over the *real* per-cell ``k`` has the
        closed form (least squares of a complex vector onto a real scalar)::

            k = Σ Re(conj(a) b) / Σ |a|² ,   c* = c0 √k .

        This is the frequency-domain variable-projection update: it replaces the
        inner c-optimiser of :class:`~odil_wave.optimisation.LBFGSB`. The data
        term is independent of ``c``, so ``c*`` is also the argmin of the full
        (pde + data) loss over ``c`` for a fixed ``u``.

        Parameters
        ----------
        amp : (n_shots, nf, nx, ny) complex wavefield.
        source : (n_shots, nf, nx, ny) complex source, pre-scaled by ``t0²``.
        c_current : optional (nx, ny) fallback for poorly illuminated cells.
        illum_rel_floor : cells with illumination ``Σ|a|²`` below this fraction
            of the peak keep ``c_current`` (or a stabilised value).

        Returns
        -------
        (c_star, illumination) : both (nx, ny); illumination is ``Σ|a|²``.
        """
        with torch.no_grad():
            c0 = self.wavefield.grid.c0
            freq = self.wavefield.frequency_selection
            lambda_t, lambda_tt = freq.symbols_broadcast()
            lambda_t = lambda_t.to(dtype=amp.dtype, device=amp.device)
            lambda_tt = lambda_tt.to(dtype=amp.dtype, device=amp.device)

            a = self._lap.apply(amp, bc=self._bc)
            b = self._sponge.apply(lambda_tt * amp - source, amp, lambda_t)

            num = (a.conj() * b).real.sum(dim=(0, 1))
            den = a.abs().square().sum(dim=(0, 1))
            floor = illum_rel_floor * den.max()
            k = num / den.clamp(min=floor)
            c_star = c0 * k.clamp(min=1e-12).sqrt()
            if c_current is not None:
                c_star = torch.where(den > floor, c_star, c_current)
            return c_star, den
