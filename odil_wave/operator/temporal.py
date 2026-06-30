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
    dt: float,
    init_ut: torch.Tensor,
) -> torch.Tensor:
    utm = torch.roll(u, 1, dims=0)
    utp = torch.roll(u, -1, dims=0)

    utm = utm.clone()
    utm[0, :, :] = u[0, :, :] - dt * init_ut

    u_tt = ((utp - 2.0 * u + utm) / dt**2).clone()

    u_tt[-1, :, :] = (
        2.0 * u[-1, :, :] - 5.0 * u[-2, :, :] + 4.0 * u[-3, :, :] - u[-4, :, :]
    ) / dt**2

    return u_tt


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
) -> torch.Tensor:
    u_tt = (_fourth_derivative_1d(utm2, utm1, u, utp1, utp2) / dt**2).clone()

    # left boundary
    u_tt[0, :, :] = (
        35.0 * u[0, :, :]
        - 104.0 * u[1, :, :]
        + 114.0 * u[2, :, :]
        - 56.0 * u[3, :, :]
        + 11.0 * u[4, :, :]
    ) / (12.0 * dt**2)

    u_tt[1, :, :] = (
        11.0 * u[0, :, :]
        - 20.0 * u[1, :, :]
        + 6.0 * u[2, :, :]
        + 4.0 * u[3, :, :]
        - 1.0 * u[4, :, :]
    ) / (12.0 * dt**2)

    # right boundary: mirror of the left boundary
    u_tt[-2, :, :] = (
        -1.0 * u[-5, :, :]
        + 4.0 * u[-4, :, :]
        + 6.0 * u[-3, :, :]
        - 20.0 * u[-2, :, :]
        + 11.0 * u[-1, :, :]
    ) / (12.0 * dt**2)

    u_tt[-1, :, :] = (
        11.0 * u[-5, :, :]
        - 56.0 * u[-4, :, :]
        + 114.0 * u[-3, :, :]
        - 104.0 * u[-2, :, :]
        + 35.0 * u[-1, :, :]
    ) / (12.0 * dt**2)

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
        return _time_stencil_2point(u, self.wavefield.grid.dt_nd, init_ut)


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
        return _time_stencil_4th(u, utm2, utm1, utp1, utp2, dt_nd)
