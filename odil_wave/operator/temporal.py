"""Temporal finite-difference operators(u_tt)"""

from abc import abstractmethod
from dataclasses import dataclass

import torch

from .base import DenseOperator
from .spatial import _fourth_derivative_1d
from odil_wave.wavefield import Wavefield


def _first_time_derivative(
    u: torch.Tensor, dt: float, init_ut: torch.Tensor
) -> torch.Tensor:
    """Centered du/dt on the full (nt, nx, ny) field, IC-consistent at the ends."""
    utm1 = torch.roll(u, 1, dims=0)
    utp1 = torch.roll(u, -1, dims=0)
    utm1 = utm1.clone()
    utp1 = utp1.clone()
    # fake past consistent with the stencil's velocity IC
    utm1[0, :, :] = u[0, :, :] - dt * init_ut
    # one-sided at the final step (no future sample)
    utp1[-1, :, :] = u[-1, :, :]
    return (utp1 - utm1) / (2.0 * dt)


def _time_stencil_2point(
    u: torch.Tensor,
    utm: torch.Tensor,
    utmm: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
) -> torch.Tensor:
    """u_tt at every t via the backward 3-point stencil.

    At t=0 and t=1 the stencil would otherwise read wrapped values
    ``u[Nt-1]``, ``u[Nt-2]`` introduced by ``torch.roll``. We patch
    those slices using the soft IC ``u_t(0) = init_ut`` so the residual
    at t=0 does not couple the hard zero IC to the final time samples
    (a wraparound that otherwise drags late-time amplitudes to zero).
    """
    u_t_tm = u - utm  # u^t - u^{t-1}
    u_t_tmm = utm - utmm  # u^{t-1} - u^{t-2}
    u_t_tm = u_t_tm.clone()
    u_t_tmm = u_t_tmm.clone()
    # IC patch at t=0: u^0 - u^{-1} = dt * init_ut and u^{-1} - u^{-2} = dt * init_ut
    u_t_tm[0, :, :] = dt * init_ut
    u_t_tmm[0, :, :] = dt * init_ut
    u_t_tmm[1, :, :] = dt * init_ut  # IC patch at t=1
    return (u_t_tm - u_t_tmm) / dt**2  # u_tt


def _roll_time_4th(u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    utm2 = torch.roll(u, 2, dims=0)
    utm1 = torch.roll(u, 1, dims=0)
    utp1 = torch.roll(u, -1, dims=0)
    utp2 = torch.roll(u, -2, dims=0)
    return utm2, utm1, utp1, utp2


def _patch_time_neighbors_4th(
    u: torch.Tensor,
    utm2: torch.Tensor,
    utm1: torch.Tensor,
    utp1: torch.Tensor,
    utp2: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    utm1 = utm1.clone()
    utm2 = utm2.clone()
    utp1 = utp1.clone()
    utp2 = utp2.clone()

    # inventing fake past and future values
    utm1[0, :, :] = u[0, :, :] - dt * init_ut  # fake u_{-1} u0 - dt * ut0
    utm2[0, :, :] = u[0, :, :] - 2.0 * dt * init_ut  # fake u_{-2} u0 - 2 * dt * ut0
    utm2[1, :, :] = u[0, :, :] - dt * init_ut

    utp1[-1, :, :] = u[-2, :, :]  # pretend u6 is u4
    utp2[-1, :, :] = u[-3, :, :]  # pretend u7 is u3
    utp2[-2, :, :] = u[-1, :, :]  # pretend u6 is u4
    return utm2, utm1, utp1, utp2


def _time_stencil_4th(
    u: torch.Tensor,
    utm2: torch.Tensor,
    utm1: torch.Tensor,
    utp1: torch.Tensor,
    utp2: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
) -> torch.Tensor:
    # Compute 4th-order u_tt everywhere
    u_tt = _fourth_derivative_1d(utm2, utm1, u, utp1, utp2) / dt**2
    # Compute 2nd-order u_tt (IC-safe)
    # near the start not enough values, so use 2nd-order u_tt
    u_t_tm = u - utm1
    u_t_tmm = utm1 - utm2
    u_t_tmm = u_t_tmm.clone()
    u_t_tmm[1, :, :] = dt * init_ut
    u_tt_2pt = (u_t_tm - u_t_tmm) / dt**2
    u_tt = u_tt.clone()
    u_tt[1, :, :] = u_tt_2pt[1, :, :]
    u_tt[2, :, :] = u_tt_2pt[2, :, :]
    return u_tt


class TemporalOperator(DenseOperator):
    """Base class for time stencil operators."""

    @abstractmethod
    def apply(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return dt^2 * u_tt on the full (nt, nx, ny) field."""
        raise NotImplementedError


@dataclass
class TimeOperator2ndOrder(TemporalOperator):
    """2nd-order time stencil."""

    wavefield: Wavefield

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        # Non-dimensional time: returns t0^2 * u_tt = u_{t't'}, IC matches u_t'.
        init_ut = self.wavefield.init_ut_nd
        utm = torch.roll(u, 1, dims=0)
        utmm = torch.roll(u, 2, dims=0)
        return _time_stencil_2point(u, utm, utmm, self.wavefield.grid.dt_nd, init_ut)


@dataclass
class TimeOperator4thOrder(TemporalOperator):
    """4th-order time stencil."""

    wavefield: Wavefield

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        # Non-dimensional time: returns t0^2 * u_tt = u_{t't'}, IC matches u_t'.
        init_ut = self.wavefield.init_ut_nd
        dt_nd = self.wavefield.grid.dt_nd
        utm2, utm1, utp1, utp2 = _roll_time_4th(u)
        utm2, utm1, utp1, utp2 = _patch_time_neighbors_4th(
            u, utm2, utm1, utp1, utp2, dt_nd, init_ut
        )
        return _time_stencil_4th(u, utm2, utm1, utp1, utp2, dt_nd, init_ut)
