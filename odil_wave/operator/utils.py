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
        """u_tt - c^2*(u_xx + u_yy) - f with PML damping."""
        utt = self._time_op.apply(amp)
        lap = self._lap.apply(amp, bc=self._bc)
        r = utt - wsp**2 * lap - source
        return self._pml.apply(r, amp)
