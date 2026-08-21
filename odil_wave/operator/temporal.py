"""Temporal finite-difference operators (u_tt)."""

from abc import abstractmethod
from dataclasses import dataclass, field

import torch

from .base import DenseOperator
from .spatial import _fourth_derivative_1d
from odil_wave.wavefield import Wavefield


def _make_endpoint_masks(
    NT: int,
    device: torch.device,
    *,
    second_layer: bool = False,
):
    """Boolean masks selecting endpoint time rows.

    Parameters
    ----------
    NT : int
        Number of time steps.
    device : torch.device
        Mask device.
    second_layer : bool, optional
        Also return masks for the second and second-to-last rows.

    Returns
    -------
    tuple of torch.Tensor
        ``(mask_first, mask_last)``, or with ``second_layer`` the 4-tuple
        ``(mask_first, mask_first2, mask_last, mask_last2)``; each ``(NT, 1, 1)``.
    """
    mask_first = torch.zeros(NT, 1, 1, dtype=torch.bool, device=device)
    mask_first[0] = True
    mask_last = torch.zeros(NT, 1, 1, dtype=torch.bool, device=device)
    mask_last[-1] = True
    if not second_layer:
        return mask_first, mask_last
    mask_first2 = torch.zeros(NT, 1, 1, dtype=torch.bool, device=device)
    mask_first2[1] = True
    mask_last2 = torch.zeros(NT, 1, 1, dtype=torch.bool, device=device)
    mask_last2[-2] = True
    return mask_first, mask_first2, mask_last, mask_last2


def _ic_ghost_row(u: torch.Tensor, dt: float, init_ut: torch.Tensor, k: float):
    """Initial-condition ghost row enforcing the ``u_t`` initial condition.

    Parameters
    ----------
    u : torch.Tensor
        Field with time on dim ``-3``.
    dt : float
        Non-dimensional time step.
    init_ut : torch.Tensor
        Initial time-derivative field ``(NX, NY)``.
    k : float
        Ghost offset multiplier (1 for ``u_{-1}``, 2 for ``u_{-2}``).

    Returns
    -------
    torch.Tensor
        Ghost row ``u[..., 0:1, :, :] - k*dt*init_ut``.
    """
    # init_ut is (NX, NY); broadcasts against u[..., 0:1, :, :] of shape (..., 1, NX, NY).
    return u[..., 0:1, :, :] - k * dt * init_ut


def _first_time_derivative(
    u: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
    mask_first: torch.Tensor,
    mask_last: torch.Tensor,
) -> torch.Tensor:
    """Centered ``du/dt`` on the full field, IC-consistent at the ends.

    Parameters
    ----------
    u : torch.Tensor
        Field with time on dim ``-3``.
    dt : float
        Non-dimensional time step.
    init_ut : torch.Tensor
        Initial time-derivative field used for the front ghost row.
    mask_first, mask_last : torch.Tensor
        Endpoint masks from :func:`_make_endpoint_masks`.

    Returns
    -------
    torch.Tensor
        Centered first time derivative, same shape as ``u``.
    """
    utm1 = torch.roll(u, 1, dims=-3)
    utp1 = torch.roll(u, -1, dims=-3)
    # IC: at t=0, the rolled-in value should be u[0] - dt*init_ut, not u[-1].
    ghost_m1 = _ic_ghost_row(u, dt, init_ut, 1.0)
    utm1 = torch.where(mask_first, ghost_m1, utm1)
    # No future sample at the last step: reuse u[-1] (one-sided).
    utp1 = torch.where(mask_last, u[..., -1:, :, :], utp1)
    return (utp1 - utm1) / (2.0 * dt)


def _time_stencil_2point(
    u: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
    mask_first: torch.Tensor,
    mask_last: torch.Tensor,
) -> torch.Tensor:
    """Second-order ``u_tt``: centred interior, one-sided 4-point endpoints.

    Parameters
    ----------
    u : torch.Tensor
        Field with time on dim ``-3``.
    dt : float
        Non-dimensional time step.
    init_ut : torch.Tensor
        Initial time-derivative field (unused; kept for signature parity).
    mask_first, mask_last : torch.Tensor
        Endpoint masks from :func:`_make_endpoint_masks`.

    Returns
    -------
    torch.Tensor
        ``u_tt`` on the full field, same shape as ``u``.
    """
    utm = torch.roll(u, 1, dims=-3)
    utp = torch.roll(u, -1, dims=-3)
    u_tt_centered = (utp - 2.0 * u + utm) / dt**2  # wrong at t=0 and t=-1; masked below

    u_tt_first = (
        2.0 * u[..., 0, :, :]
        - 5.0 * u[..., 1, :, :]
        + 4.0 * u[..., 2, :, :]
        - u[..., 3, :, :]
    ).unsqueeze(-3) / dt**2
    u_tt_last = (
        2.0 * u[..., -1, :, :]
        - 5.0 * u[..., -2, :, :]
        + 4.0 * u[..., -3, :, :]
        - u[..., -4, :, :]
    ).unsqueeze(-3) / dt**2

    u_tt = torch.where(mask_first, u_tt_first, u_tt_centered)
    u_tt = torch.where(mask_last, u_tt_last, u_tt)
    return u_tt


def _build_time_neighbors_4th(
    u: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
    mask_first: torch.Tensor,
    mask_first2: torch.Tensor,
    mask_last: torch.Tensor,
    mask_last2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the four time neighbours for the 4th-order stencil.

    Parameters
    ----------
    u : torch.Tensor
        Field with time on dim ``-3``.
    dt : float
        Non-dimensional time step.
    init_ut : torch.Tensor
        Initial time-derivative field for the front ghost rows.
    mask_first, mask_first2, mask_last, mask_last2 : torch.Tensor
        Endpoint masks from :func:`_make_endpoint_masks`.

    Returns
    -------
    tuple of torch.Tensor
        ``(utm2, utm1, utp1, utp2)`` with IC and endpoint patches applied.
    """
    utm2 = torch.roll(u, 2, dims=-3)
    utm1 = torch.roll(u, 1, dims=-3)
    utp1 = torch.roll(u, -1, dims=-3)
    utp2 = torch.roll(u, -2, dims=-3)

    ghost_m1 = _ic_ghost_row(u, dt, init_ut, 1.0)  # u_{-1} = u[0] - dt*init_ut
    ghost_m2 = _ic_ghost_row(u, dt, init_ut, 2.0)  # u_{-2} = u[0] - 2*dt*init_ut

    # Front: utm1[0] <- ghost_m1; utm2[0] <- ghost_m2; utm2[1] <- ghost_m1.
    utm1 = torch.where(mask_first, ghost_m1, utm1)
    utm2 = torch.where(mask_first, ghost_m2, utm2)
    utm2 = torch.where(mask_first2, ghost_m1, utm2)

    # Back: utp1[-1] <- u[-2]; utp2[-1] <- u[-3]; utp2[-2] <- u[-1].
    utp1 = torch.where(mask_last, u[..., -2:-1, :, :], utp1)
    utp2 = torch.where(mask_last, u[..., -3:-2, :, :], utp2)
    utp2 = torch.where(mask_last2, u[..., -1:, :, :], utp2)

    return utm2, utm1, utp1, utp2


def _time_stencil_4th(
    u: torch.Tensor,
    utm2: torch.Tensor,
    utm1: torch.Tensor,
    utp1: torch.Tensor,
    utp2: torch.Tensor,
    dt: float,
    init_ut: torch.Tensor,
    mask_first: torch.Tensor,
    mask_first2: torch.Tensor,
    mask_last: torch.Tensor,
    mask_last2: torch.Tensor,
) -> torch.Tensor:
    """Fourth-order ``u_tt``; rows 0, 1, NT-2 and NT-1 use one-sided 5-point stencils.

    Parameters
    ----------
    u : torch.Tensor
        Field with time on dim ``-3``.
    utm2, utm1, utp1, utp2 : torch.Tensor
        Time neighbours from :func:`_build_time_neighbors_4th`.
    dt : float
        Non-dimensional time step.
    init_ut : torch.Tensor
        Initial time-derivative field (unused; kept for signature parity).
    mask_first, mask_first2, mask_last, mask_last2 : torch.Tensor
        Endpoint masks from :func:`_make_endpoint_masks`.

    Returns
    -------
    torch.Tensor
        ``u_tt`` on the full field, same shape as ``u``.
    """
    u_tt_centered = _fourth_derivative_1d(utm2, utm1, u, utp1, utp2) / dt**2

    u_tt_0 = (
        35.0 * u[..., 0, :, :]
        - 104.0 * u[..., 1, :, :]
        + 114.0 * u[..., 2, :, :]
        - 56.0 * u[..., 3, :, :]
        + 11.0 * u[..., 4, :, :]
    ).unsqueeze(-3) / (12.0 * dt**2)
    u_tt_1 = (
        11.0 * u[..., 0, :, :]
        - 20.0 * u[..., 1, :, :]
        + 6.0 * u[..., 2, :, :]
        + 4.0 * u[..., 3, :, :]
        - u[..., 4, :, :]
    ).unsqueeze(-3) / (12.0 * dt**2)
    # Mirror images of u_tt_0 / u_tt_1: same coefficients, points counted
    # backward from the last row instead of forward from the first (the
    # 2nd-derivative one-sided stencil is symmetric under time reversal).
    u_tt_last = (
        35.0 * u[..., -1, :, :]
        - 104.0 * u[..., -2, :, :]
        + 114.0 * u[..., -3, :, :]
        - 56.0 * u[..., -4, :, :]
        + 11.0 * u[..., -5, :, :]
    ).unsqueeze(-3) / (12.0 * dt**2)
    u_tt_second_last = (
        11.0 * u[..., -1, :, :]
        - 20.0 * u[..., -2, :, :]
        + 6.0 * u[..., -3, :, :]
        + 4.0 * u[..., -4, :, :]
        - u[..., -5, :, :]
    ).unsqueeze(-3) / (12.0 * dt**2)

    u_tt = torch.where(mask_first, u_tt_0, u_tt_centered)
    u_tt = torch.where(mask_first2, u_tt_1, u_tt)
    u_tt = torch.where(mask_last2, u_tt_second_last, u_tt)
    u_tt = torch.where(mask_last, u_tt_last, u_tt)
    return u_tt


class TemporalOperator(DenseOperator):
    """Base class for time stencil operators."""

    @abstractmethod
    def apply(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the non-dimensional second time derivative of the field.

        Parameters
        ----------
        u : torch.Tensor
            Field of shape ``(..., NT, NX, NY)`` with time on dim ``-3``.

        Returns
        -------
        torch.Tensor
            Scaled ``u_tt`` on the full field.
        """
        raise NotImplementedError


@dataclass
class TimeOperator2ndOrder(TemporalOperator):
    """2nd-order time stencil."""

    wavefield: Wavefield
    _mask_first: torch.Tensor = field(init=False, repr=False)
    _mask_last: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        """Build the endpoint masks for the grid's time axis."""
        g = self.wavefield.grid
        self._mask_first, self._mask_last = _make_endpoint_masks(g.nt, g.device)

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        """Non-dimensional 2nd-order ``u_{t't'}`` on the full field.

        Parameters
        ----------
        u : torch.Tensor
            Field with time on dim ``-3``.

        Returns
        -------
        torch.Tensor
            Second time derivative with the IC matched to ``u_t'``.
        """
        init_ut = self.wavefield.init_ut_nd
        return _time_stencil_2point(
            u,
            self.wavefield.grid.dt_nd,
            init_ut,
            self._mask_first,
            self._mask_last,
        )


@dataclass
class TimeOperator4thOrder(TemporalOperator):
    """4th-order time stencil."""

    wavefield: Wavefield
    _mask_first: torch.Tensor = field(init=False, repr=False)
    _mask_first2: torch.Tensor = field(init=False, repr=False)
    _mask_last: torch.Tensor = field(init=False, repr=False)
    _mask_last2: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        """Build the two-layer endpoint masks for the grid's time axis."""
        g = self.wavefield.grid
        (
            self._mask_first,
            self._mask_first2,
            self._mask_last,
            self._mask_last2,
        ) = _make_endpoint_masks(g.nt, g.device, second_layer=True)

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        """Non-dimensional 4th-order ``u_{t't'}`` on the full field.

        Parameters
        ----------
        u : torch.Tensor
            Field with time on dim ``-3``.

        Returns
        -------
        torch.Tensor
            Second time derivative with the IC matched to ``u_t'``.
        """
        init_ut = self.wavefield.init_ut_nd
        dt_nd = self.wavefield.grid.dt_nd
        utm2, utm1, utp1, utp2 = _build_time_neighbors_4th(
            u,
            dt_nd,
            init_ut,
            self._mask_first,
            self._mask_first2,
            self._mask_last,
            self._mask_last2,
        )
        return _time_stencil_4th(
            u,
            utm2,
            utm1,
            utp1,
            utp2,
            dt_nd,
            init_ut,
            self._mask_first,
            self._mask_first2,
            self._mask_last,
            self._mask_last2,
        )
