"""Temporal finite-difference operators (u_tt).
"""

from abc import abstractmethod
from dataclasses import dataclass

import torch

from .base import DenseOperator
from .spatial import _fourth_derivative_1d
from odil_wave.wavefield import Wavefield


def _ic_ghost_row(u: torch.Tensor, dt: float, init_ut: torch.Tensor, k: float):
    """Single time-row ghost ``u[..., 0:1, :, :] - k*dt*init_ut`` for IC consistency."""
    # init_ut is (NX, NY); broadcasts against u[..., 0:1, :, :] of shape (..., 1, NX, NY).
    return u[..., 0:1, :, :] - k * dt * init_ut


def _first_time_derivative(
    u: torch.Tensor, dt: float, init_ut: torch.Tensor
) -> torch.Tensor:
    """Centered du/dt on the full field, IC-consistent at the ends."""
    utm1 = torch.roll(u, 1, dims=-3)
    utp1 = torch.roll(u, -1, dims=-3)
    utm1 = torch.cat([_ic_ghost_row(u, dt, init_ut, 1.0), utm1[..., 1:, :, :]], dim=-3)
    # One-sided at the final step (no future sample) -> reuse u[-1].
    utp1 = torch.cat([utp1[..., :-1, :, :], u[..., -1:, :, :]], dim=-3)
    return (utp1 - utm1) / (2.0 * dt)


def _time_stencil_2point(
    u: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
) -> torch.Tensor:
    """Centred leapfrog u_tt in the interior; one-sided 4-point at endpoints."""
    utm = torch.roll(u, 1, dims=-3)
    utp = torch.roll(u, -1, dims=-3)
    u_tt_centered = (utp - 2.0 * u + utm) / dt**2

    u_tt_first = (
        2.0 * u[..., 0, :, :]
        - 5.0 * u[..., 1, :, :]
        + 4.0 * u[..., 2, :, :]
        - u[..., 3, :, :]
    ) / dt**2
    u_tt_last = (
        2.0 * u[..., -1, :, :]
        - 5.0 * u[..., -2, :, :]
        + 4.0 * u[..., -3, :, :]
        - u[..., -4, :, :]
    ) / dt**2

    return torch.cat(
        [
            u_tt_first.unsqueeze(-3),
            u_tt_centered[..., 1:-1, :, :],
            u_tt_last.unsqueeze(-3),
        ],
        dim=-3,
    )


def _build_time_neighbors_4th(
    u: torch.Tensor, dt: float, init_ut: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble (utm2, utm1, utp1, utp2) with IC + endpoint patches, out-of-place."""
    utm2 = torch.roll(u, 2, dims=-3)
    utm1 = torch.roll(u, 1, dims=-3)
    utp1 = torch.roll(u, -1, dims=-3)
    utp2 = torch.roll(u, -2, dims=-3)

    ghost_m1 = _ic_ghost_row(u, dt, init_ut, 1.0)  # u_{-1} = u[0] - dt*init_ut
    ghost_m2 = _ic_ghost_row(u, dt, init_ut, 2.0)  # u_{-2} = u[0] - 2*dt*init_ut

    # utm1[0] <- ghost_m1
    utm1 = torch.cat([ghost_m1, utm1[..., 1:, :, :]], dim=-3)
    # utm2[0] <- ghost_m2, utm2[1] <- ghost_m1
    utm2 = torch.cat([ghost_m2, ghost_m1, utm2[..., 2:, :, :]], dim=-3)

    # utp1[-1] <- u[-2]
    utp1 = torch.cat([utp1[..., :-1, :, :], u[..., -2:-1, :, :]], dim=-3)
    # utp2[-1] <- u[-3], utp2[-2] <- u[-1]
    utp2 = torch.cat(
        [utp2[..., :-2, :, :], u[..., -1:, :, :], u[..., -3:-2, :, :]], dim=-3
    )
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
    """4th-order centred u_tt with one-sided 5-point stencils at t=0 and t=1."""
    u_tt_centered = _fourth_derivative_1d(utm2, utm1, u, utp1, utp2) / dt**2

    u_tt_0 = (
        35.0 * u[..., 0, :, :]
        - 104.0 * u[..., 1, :, :]
        + 114.0 * u[..., 2, :, :]
        - 56.0 * u[..., 3, :, :]
        + 11.0 * u[..., 4, :, :]
    ) / (12.0 * dt**2)
    u_tt_1 = (
        11.0 * u[..., 0, :, :]
        - 20.0 * u[..., 1, :, :]
        + 6.0 * u[..., 2, :, :]
        + 4.0 * u[..., 3, :, :]
        - u[..., 4, :, :]
    ) / (12.0 * dt**2)

    return torch.cat(
        [
            u_tt_0.unsqueeze(-3),
            u_tt_1.unsqueeze(-3),
            u_tt_centered[..., 2:, :, :],
        ],
        dim=-3,
    )


class TemporalOperator(DenseOperator):
    """Base class for time stencil operators."""

    @abstractmethod
    def apply(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return dt^2 * u_tt on the full (..., NT, NX, NY) field."""
        raise NotImplementedError


@dataclass
class TimeOperator2ndOrder(TemporalOperator):
    """2nd-order time stencil."""

    wavefield: Wavefield

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        # Non-dimensional time: returns t0^2 * u_tt = u_{t't'}, IC matches u_t'.
        init_ut = self.wavefield.init_ut_nd
        return _time_stencil_2point(u, self.wavefield.grid.dt_nd, init_ut)


@dataclass
class TimeOperator4thOrder(TemporalOperator):
    """4th-order time stencil."""

    wavefield: Wavefield

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        # Non-dimensional time: returns t0^2 * u_tt = u_{t't'}, IC matches u_t'.
        init_ut = self.wavefield.init_ut_nd
        dt_nd = self.wavefield.grid.dt_nd
        utm2, utm1, utp1, utp2 = _build_time_neighbors_4th(u, dt_nd, init_ut)
        return _time_stencil_4th(u, utm2, utm1, utp1, utp2, dt_nd, init_ut)
