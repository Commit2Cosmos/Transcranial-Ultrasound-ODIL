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

    def c_closed_form(
        self,
        amp: torch.Tensor,
        source: torch.Tensor,
        c_current: torch.Tensor | None = None,
        illum_rel_floor: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-cell least-squares wavespeed given the wavefield (variable
        projection).

        Parameters
        ----------
        amp : (n_shots, NT, NX, NY) wavefield.
        source : (n_shots, NT, NX, NY) sources, pre-scaled by ``t0**2``.
        c_current : optional (NX, NY) fallback for unilluminated cells.

        Returns
        -------
        (c_star, illumination) : both (NX, NY).
        """
        with torch.no_grad():
            utt = self._time_op.apply(amp)
            a = self._lap.apply(amp, bc=self._bc)
            b = self._pml.apply(utt - source, amp)
            num = (a * b).sum(dim=(0, 1))
            den = (a * a).sum(dim=(0, 1))
            floor = illum_rel_floor * float(den.max())
            k = num / den.clamp(min=floor)
            c0 = self.wavefield.grid.c0
            c_star = c0 * k.clamp(min=1e-12).sqrt()
            if c_current is not None:
                c_star = torch.where(den > floor, c_star, c_current)
            return c_star, den
