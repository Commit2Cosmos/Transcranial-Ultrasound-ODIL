from dataclasses import dataclass, field
import torch
from odil_wave.wavefield import Wavefield
from .base import DenseOperator
from .temporal import TimeOperator2ndOrder, TimeOperator4thOrder
from .spatial import Laplacian2ndOrder, Laplacian4thOrder
from .conditions import (
    Conditions,
    NeumannMirrorBC2nd,
    NeumannMirrorBC4th,
    PML,
)


@dataclass
class WaveEquation:
    """Discrete acoustic wave equation u_tt - c^2*lap(u) = f.

    Internally non-dimensional: the residual returned is
    ``u_{t't'} - (c/c0)^2 * L0^2 * lap(u) - t0^2 * f`` evaluated with
    the grid's ``dx_nd, dy_nd, dt_nd`` and ``sigma_*_nd`` (see
    :class:`Grid` for the scaling derivation). Callers pass the physical
    wavespeed ``wsp`` (m/s) and a *pre-scaled* dimensionless ``source``
    (``DiscreteLoss`` precomputes ``t0**2 * f`` once).

    Initial conditions (u(0,x,y)=0 and u_t(0,x,y)=init_ut) are hard-
    constrained.
    """

    wavefield: Wavefield
    time_order: int = 2
    space_order: int = 2
    pml_weight: float = 1.0

    _time_op: DenseOperator = field(init=False)
    _lap: DenseOperator = field(init=False)
    _bc: Conditions = field(init=False)
    _pml: PML = field(init=False)

    def __post_init__(self):
        if self.time_order == 2:
            self._time_op = TimeOperator2ndOrder(self.wavefield)
        elif self.time_order == 4:
            self._time_op = TimeOperator4thOrder(self.wavefield)
        else:
            raise ValueError(f"Invalid time order: {self.time_order}")

        if self.space_order == 2:
            self._lap = Laplacian2ndOrder(self.wavefield)
        elif self.space_order == 4:
            self._lap = Laplacian4thOrder(self.wavefield)
        else:
            raise ValueError(f"Invalid space order: {self.space_order}")

        self._bc = (
            NeumannMirrorBC2nd() if self.space_order == 2 else NeumannMirrorBC4th()
        )
        self._pml = PML(self.wavefield, self.pml_weight)

    def residual(
        self, amp: torch.Tensor, wsp: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Dimensionless residual u_{t't'} - c'^2 * lap'(u) - f' with PML damping.

        ``wsp`` is the physical wavespeed (m/s); it is normalised by
        ``grid.c0`` here so the optimiser can keep operating in physical
        units. ``source`` is expected pre-scaled by ``t0**2``.
        """
        c0 = self.wavefield.grid.c0
        wsp_nd = wsp / c0
        utt = self._time_op.apply(amp)
        lap = self._lap.apply(amp, bc=self._bc)
        r = utt - wsp_nd**2 * lap - source
        return self._pml.apply(r, amp)
