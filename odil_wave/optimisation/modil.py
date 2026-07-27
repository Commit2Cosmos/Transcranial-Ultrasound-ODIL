"""Simultaneous multilevel (mODIL) velocity parameterization for FWI.

This module implements the multigrid-*decomposition* mODIL algorithm, which is
distinct from the coarse-to-fine continuation in
:mod:`odil_wave.optimisation.coarse_to_fine`. Instead of solving a sequence of
independent problems, mODIL represents the finest-grid velocity as a
*simultaneous* additive sum of trainable fields living on a hierarchy of
resolutions::

    c_fine = c_base + Σ_l  s_l · P_l( δc_l )

where ``δc_l`` is a trainable interior correction on level ``level`` (level 0 is the
finest, higher ``level`` are coarser), ``P_l`` prolongates level ``level`` to the finest
interior, ``s_l`` is an optional (numerically-neutral by default) per-level
scale, and ``c_base`` is a fixed (non-trainable) background model. **All** level
corrections remain independent optimisation variables and are optimised
together; the PDE/data loss is evaluated only on the reconstructed finest-grid
velocity. The representation is intentionally over-parameterized — it is not
made unique; instead corrections initialise to zero (so the initial model
equals ``c_base`` exactly) and optional regularization keeps it well-behaved.

Contrast with coarse-to-fine
----------------------------
* :class:`~odil_wave.optimisation.coarse_to_fine.CoarseToFineInversion` — a
  sequence of *independent* solves, each warm-started from the previous grid;
  coarse degrees of freedom are discarded once a finer solve begins.
* :class:`MODILVelocityParameterization` / :class:`MODILInversion` — a *single*
  joint optimisation with a multilevel additive representation, evaluating one
  finest-grid loss per closure and keeping every level's field trainable
  throughout.

Public API
----------
* :func:`build_grid_hierarchy` — validated factor-of-two grid hierarchy;
* :class:`MODILVelocityParameterization` — the ``nn.Module`` parameterization;
* :class:`MODILInversion` — an :class:`~odil_wave.optimisation.base.Optimiser`
  that jointly optimises the wavefield ``u`` and all level corrections against
  the repository's :class:`~odil_wave.loss.InverseLoss`.

Example
-------
>>> # finest grid + a validated factor-2 hierarchy over the same interior
>>> hierarchy = build_grid_hierarchy(grid, num_levels=3, coarsening_factor=2)
>>> # baseline (background) model as the finest interior velocity
>>> base_interior = init_vm.c[grid.interior_slice].clone()
>>> param = MODILVelocityParameterization(hierarchy, base_interior)
>>> # integrate with the actual wavefield / loss / optimiser API
>>> wf = Wavefield(grid, freq, velocity_model=init_vm)
>>> loss = InverseLoss(observed_traces=d_obs, config=cfg, ...)
>>> opt = MODILInversion(wf, loss, param, clamp=True, n_iter=60, u_steps=6)
>>> outputs, tape = opt.minimise()
>>> c_final = outputs[0].velocity_model.c        # full recovered velocity
>>> deltas = param.native_corrections()          # per-level own-resolution δc_l
>>> contribs = param.prolonged_contributions()   # each level's finest-grid share
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from odil_wave.grid import Grid
from odil_wave.loss import InverseLoss
from odil_wave.models import VelocityModel
from odil_wave.wavefield import Wavefield

from .base import Optimiser
from .grid_transfer import (
    _canonical_staggering,
    assert_compatible_interiors,
    resample_interior_field,
)


# --------------------------------------------------------------------------- #
# Grid hierarchy                                                               #
# --------------------------------------------------------------------------- #
def _coarsen_node_count(n_fine: int, factor: int) -> int:
    """Vertex-centred coarsening: (n-1) intervals must divide by ``factor``."""
    if (n_fine - 1) % factor != 0:
        raise ValueError(
            f"cannot coarsen a vertex-centred axis of {n_fine} nodes by factor "
            f"{factor}: {n_fine - 1} intervals is not divisible by {factor}. "
            f"Choose a finest node count n with (n-1) divisible by "
            f"{factor}**(num_levels-1)."
        )
    return (n_fine - 1) // factor + 1


def build_grid_hierarchy(
    finest_grid: Grid,
    num_levels: int,
    coarsening_factor: int = 2,
    *,
    min_interior_nodes: int = 3,
    nyquist_frequencies_hz: Optional[Sequence[float]] = None,
    min_ppw: float = 3.0,
    require_time_domain_cfl: bool = False,
) -> List[Grid]:
    """Build ``num_levels`` grids (finest first) over the same physical interior.

    Each coarser level halves (for ``coarsening_factor == 2``) the number of
    interior *intervals*, keeping the two endpoints coincident so vertex-centred
    prolongation is exact for affine fields. Every level shares ``finest_grid``'s
    physical interior extent, velocity bounds, PML parameters, time sampling and
    non-dimensionalisation; only the interior resolution (and hence ``dx``)
    changes.

    Validation (raises :class:`ValueError` with a specific reason on failure):

    * ``num_levels >= 1`` and ``coarsening_factor >= 2``;
    * each coarsening is exact (``(n-1)`` divisible by ``coarsening_factor``);
    * the coarsest interior has at least ``min_interior_nodes`` nodes per axis
      (``3`` is the practical floor for bilinear prolongation);
    * (optional, ``nyquist_frequencies_hz``) the coarsest grid still resolves
      ``max(nyquist_frequencies_hz)`` at ``>= min_ppw`` points per wavelength —
      the meaningful *spatial* adequacy criterion for a correction lattice;
    * (opt-in, ``require_time_domain_cfl``) the coarsest grid's CFL number at the
      shared ``dt`` stays ``< 1``. This is **off by default**: mODIL correction
      grids feed a frequency-domain Helmholtz residual and pure interpolation,
      neither of which time-steps, so CFL imposes no stability constraint here.
      Enable it only if a level will be handed to a time-domain solver.

    Returns
    -------
    list[Grid] of length ``num_levels``, finest to coarsest.
    """
    if num_levels < 1:
        raise ValueError(f"num_levels must be >= 1, got {num_levels}.")
    if coarsening_factor < 2:
        raise ValueError(f"coarsening_factor must be >= 2, got {coarsening_factor}.")
    if num_levels == 1:
        return [finest_grid]

    c_min = (
        float(finest_grid.c_min)
        if finest_grid.c_min is not None
        else float(finest_grid.c_max)
    )
    grids: List[Grid] = [finest_grid]
    nx, ny = int(finest_grid.interior_nx), int(finest_grid.interior_ny)

    for level in range(1, num_levels):
        nx = _coarsen_node_count(nx, coarsening_factor)
        ny = _coarsen_node_count(ny, coarsening_factor)
        if nx < min_interior_nodes or ny < min_interior_nodes:
            raise ValueError(
                f"coarsest level {level} would have interior {nx}x{ny} nodes, "
                f"below the minimum {min_interior_nodes}. Reduce num_levels "
                f"({num_levels}) or the coarsening factor ({coarsening_factor})."
            )
        grids.append(_coarse_grid_like(finest_grid, nx, ny))

    coarsest = grids[-1]
    if require_time_domain_cfl:
        cfl = coarsest.cfl(float(finest_grid.c_max))
        if not cfl < 1.0:
            raise ValueError(
                f"coarsest grid CFL={cfl:.3f} is not < 1 at the shared dt; the "
                f"finest grid itself is CFL-unstable for time stepping. Disable "
                f"require_time_domain_cfl for a frequency-domain-only hierarchy."
            )
    if nyquist_frequencies_hz is not None:
        f_max = float(max(nyquist_frequencies_hz))
        dx_max = max(coarsest.dx, coarsest.dy)
        ppw = c_min / (f_max * dx_max) if f_max > 0 else math.inf
        if ppw < min_ppw:
            raise ValueError(
                f"coarsest grid resolves {ppw:.2f} points/wavelength at "
                f"{f_max:g} Hz (c_min={c_min:g}), below min_ppw={min_ppw}. "
                f"Use fewer levels or a finer coarsest grid."
            )
    return grids


def _coarse_grid_like(finest: Grid, interior_nx: int, interior_ny: int) -> Grid:
    """A coarser :class:`Grid` sharing all of ``finest``'s physics but resolution."""
    return Grid(
        interior_shape=(interior_nx, interior_ny),
        interior_extent=finest.interior_extent,
        c_min=finest.c_min,
        c_max=finest.c_max,
        pml_width=finest.pml_width,
        pml_power=finest.pml_power,
        pml_R0=finest.pml_R0,
        t_max=finest.t_max,
        init_nt=finest.nt,
        cfl_safety=finest.cfl_safety,
        L0=finest.L0,
        c0=finest.c0,
        device=finest.device,
        dtype=finest.dtype,
    )


# --------------------------------------------------------------------------- #
# Parameterization                                                            #
# --------------------------------------------------------------------------- #
class MODILVelocityParameterization(nn.Module):
    """Multilevel additive parameterization of a scalar velocity field.

    The reconstructed finest-grid interior velocity is::

        c(x) = c_base + c_ref · Σ_l  level_scales[level] · P_l( δc_l )

    with one trainable interior correction ``δc_l`` per level (a
    :class:`torch.nn.Parameter`, zero-initialised so the initial model equals
    ``c_base`` exactly) and ``P_l`` a validated vertex-centred prolongation from
    level ``level`` directly to the finest interior.

    ``c_ref`` is a fixed velocity scale (default: the mean of the background
    model) that makes the trainable corrections **dimensionless and O(1)**, so
    L-BFGS sees a well-conditioned variable — mirroring how
    :class:`~odil_wave.optimisation.LBFGSB` normalises its single-grid ``c``.
    With physical-unit corrections (``c_ref = 1``) the multilevel c-subproblem is
    badly scaled and barely steps; the default normalisation fixes that. It is
    numerically neutral for the reconstruction (any ``c_ref`` spans the same
    velocity space), only rescaling the optimisation variable.

    Parameters
    ----------
    level_grids : sequence of Grid, finest first
        Every grid must describe the same physical interior (validated). Usually
        produced by :func:`build_grid_hierarchy`.
    base_interior_velocity : Tensor of finest interior shape, or float
        Fixed background model. Stored as a non-trainable buffer.
    c_ref : float, optional
        Velocity scale applied to every correction (default: ``mean(base)``,
        falling back to ``1.0`` for a zero/degenerate base). Pass ``1.0`` for
        corrections in physical m/s.
    interpolation : {"linear"}
        Prolongation kernel; only bilinear ("linear") is implemented.
    level_scales : sequence of float, optional
        Per-level multipliers on each prolonged correction. Default all ``1.0``
        (numerically neutral).
    regularization : {"l2", "none"}
        Correction-space penalty used by :meth:`regularization_loss`.
    reg_weight : float
        Uniform weight applied to every level's penalty when
        ``level_reg_weights`` is not given.
    level_reg_weights : sequence of float, optional
        Per-level penalty weights (overrides ``reg_weight``).
    staggering : {"vertex"}
        Grid-sampling convention; only vertex-centred is supported.
    """

    def __init__(
        self,
        level_grids: Sequence[Grid],
        base_interior_velocity,
        *,
        c_ref: Optional[float] = None,
        interpolation: str = "linear",
        level_scales: Optional[Sequence[float]] = None,
        regularization: str = "l2",
        reg_weight: float = 0.0,
        level_reg_weights: Optional[Sequence[float]] = None,
        staggering: str = "vertex",
    ) -> None:
        super().__init__()
        level_grids = list(level_grids)
        if len(level_grids) < 1:
            raise ValueError("level_grids must contain at least the finest grid.")
        if interpolation != "linear":
            raise ValueError(
                f"interpolation={interpolation!r} is not supported; only "
                f"vertex-centred bilinear ('linear') prolongation is implemented."
            )
        if regularization not in ("l2", "none"):
            raise ValueError(
                f"regularization must be 'l2' or 'none', got {regularization!r}."
            )

        self.staggering = _canonical_staggering(staggering)
        self.interpolation = interpolation
        self.regularization = regularization
        self.level_grids = level_grids
        self.finest_grid = level_grids[0]
        finest = self.finest_grid
        dtype, device = finest.dtype, finest.device
        fnx, fny = int(finest.interior_nx), int(finest.interior_ny)
        self._finest_interior_shape = (fnx, fny)

        # Every level must live over the same physical interior as the finest.
        for lvl, g in enumerate(level_grids):
            assert_compatible_interiors(g, finest, staggering=self.staggering)
            if g.interior_nx > fnx or g.interior_ny > fny:
                raise ValueError(
                    f"level {lvl} interior {g.interior_nx}x{g.interior_ny} is "
                    f"finer than level 0 {fnx}x{fny}; pass grids finest-first."
                )

        # Fixed (non-trainable) background model.
        if isinstance(base_interior_velocity, (int, float)):
            base = torch.full(
                (fnx, fny), float(base_interior_velocity), dtype=dtype, device=device
            )
        else:
            base = torch.as_tensor(base_interior_velocity, dtype=dtype, device=device)
            if tuple(base.shape) != (fnx, fny):
                raise ValueError(
                    f"base_interior_velocity shape {tuple(base.shape)} must match "
                    f"the finest interior {(fnx, fny)}."
                )
            base = base.detach().clone()
        self.register_buffer("base_interior", base)

        # Velocity scale that makes corrections O(1) for a well-conditioned
        # L-BFGS c-subproblem (numerically neutral for the reconstruction).
        if c_ref is None:
            c_ref_val = float(base.mean())
            if not math.isfinite(c_ref_val) or c_ref_val == 0.0:
                c_ref_val = 1.0
        else:
            c_ref_val = float(c_ref)
        self.register_buffer(
            "c_ref", torch.tensor(c_ref_val, dtype=dtype, device=device)
        )

        # One trainable, zero-initialised correction per level.
        self.corrections = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        (int(g.interior_nx), int(g.interior_ny)),
                        dtype=dtype,
                        device=device,
                    )
                )
                for g in level_grids
            ]
        )

        if level_scales is None:
            scales = [1.0] * len(level_grids)
        else:
            if len(level_scales) != len(level_grids):
                raise ValueError("level_scales length must equal the number of levels.")
            scales = [float(s) for s in level_scales]
        self.register_buffer(
            "level_scales", torch.tensor(scales, dtype=dtype, device=device)
        )

        if level_reg_weights is None:
            self._level_reg_weights = [float(reg_weight)] * len(level_grids)
        else:
            if len(level_reg_weights) != len(level_grids):
                raise ValueError(
                    "level_reg_weights length must equal the number of levels."
                )
            self._level_reg_weights = [float(w) for w in level_reg_weights]

    # -- introspection ----------------------------------------------------- #
    @property
    def num_levels(self) -> int:
        return len(self.corrections)

    @property
    def level_parameters(self) -> nn.ParameterList:
        """The trainable per-level correction tensors (finest first)."""
        return self.corrections

    # -- reconstruction ---------------------------------------------------- #
    def _prolonged_contribution(self, level: int) -> torch.Tensor:
        """``level_scales[level] · P_l(δc_l)`` on the finest interior (differentiable)."""
        corr = self.corrections[level]
        g = self.level_grids[level]
        if (int(g.interior_nx), int(g.interior_ny)) == self._finest_interior_shape:
            contrib = corr
        else:
            contrib = resample_interior_field(
                corr, g, self.finest_grid, staggering=self.staggering
            )
        return self.level_scales[level] * contrib

    def reconstructed_interior(self) -> torch.Tensor:
        """Reconstructed finest-grid interior velocity (finest dtype/device).

        Differentiable with respect to every level correction. With all
        corrections zero this returns exactly ``base_interior``.
        """
        total = self.base_interior
        for level in range(self.num_levels):
            total = total + self.c_ref * self._prolonged_contribution(level)
        return total

    def forward(self) -> torch.Tensor:
        return self.reconstructed_interior()

    def full_velocity(self, pml_c: Optional[float] = None) -> torch.Tensor:
        """Reconstructed velocity padded onto the full finest grid (interior + PML).

        The PML ring is filled with the constant ``pml_c`` (default: the minimum
        of the background model), mirroring
        :meth:`odil_wave.models.VelocityModel.build_full_c`. Differentiable in
        the interior.
        """
        interior = self.reconstructed_interior()
        p = int(self.finest_grid.pml_width)
        fill = float(self.base_interior.min()) if pml_c is None else float(pml_c)
        return F.pad(interior, (p, p, p, p), mode="constant", value=fill)

    def regularization_loss(self) -> torch.Tensor:
        """Weighted correction-space penalty (zero when all corrections are zero).

        For ``regularization == 'l2'`` this is ``Σ_l w_l · mean(δc_l²)`` with the
        per-level weights ``w_l``; ``'none'`` always returns zero.
        """
        total = self.base_interior.new_zeros(())
        if self.regularization == "none":
            return total
        for level, corr in enumerate(self.corrections):
            w = self._level_reg_weights[level]
            if w != 0.0:
                total = total + w * corr.pow(2).mean()
        return total

    def native_corrections(self) -> List[torch.Tensor]:
        """Detached copies of each level's native (own-resolution) correction."""
        return [c.detach().clone() for c in self.corrections]

    def prolonged_contributions(self) -> List[torch.Tensor]:
        """Detached copies of each level's contribution on the finest interior."""
        with torch.no_grad():
            return [
                self._prolonged_contribution(level).detach().clone()
                for level in range(self.num_levels)
            ]


# --------------------------------------------------------------------------- #
# Optimiser                                                                    #
# --------------------------------------------------------------------------- #
def _seed_complex(
    wavefield: Wavefield, u_init, n_shots: int, cdtype, device
) -> torch.Tensor:
    """``(n_shots, nf, nx, ny)`` complex seed from ``u_init`` / the wavefield."""
    nf = wavefield.n_frequencies
    Nx, Ny = wavefield.grid.shape
    if u_init is None:
        seed = wavefield.amplitude.detach().clone().to(dtype=cdtype, device=device)
        if seed.ndim == 3:
            seed = seed.unsqueeze(0).expand(n_shots, -1, -1, -1)
        return seed.contiguous()
    if isinstance(u_init, (list, tuple)):
        stack = torch.stack(
            [
                w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                for w in u_init
            ]
        )
    else:
        stack = torch.as_tensor(u_init)
    stack = stack.detach().to(dtype=cdtype, device=device)
    if stack.ndim == 3:
        stack = stack.unsqueeze(0).expand(n_shots, -1, -1, -1)
    if stack.shape[0] != n_shots:
        raise ValueError(f"u_init provides {stack.shape[0]} shots, expected {n_shots}.")
    if tuple(stack.shape[1:]) != (nf, Nx, Ny):
        raise ValueError(
            f"u_init shape {tuple(stack.shape)} incompatible with "
            f"(n_shots={n_shots}, nf={nf}, nx={Nx}, ny={Ny})."
        )
    return stack.contiguous()


class MODILInversion(Optimiser):
    """Joint (u, mODIL-c) frequency-domain FWI optimiser.

    Owns a :class:`MODILVelocityParameterization` and exposes all of its level
    corrections to L-BFGS simultaneously. Every loss evaluation reconstructs the
    finest-grid velocity from the current corrections (never a stale reference),
    so the standard :class:`~odil_wave.loss.InverseLoss` is used unchanged.

    Two schedules are provided:

    * block-coordinate (default): ``u_steps`` of complex u-L-BFGS with the
      velocity fixed, then ``c_steps`` of L-BFGS over *all* level corrections at
      once with ``u`` fixed — mirroring
      :class:`~odil_wave.optimisation.LBFGSB` but with the multilevel velocity;
    * ``joint=True``: a single L-BFGS over ``[u_real, u_imag, *corrections]``,
      one closure (one finest-grid loss) per iteration.

    Parameters
    ----------
    parameterization : MODILVelocityParameterization
        Its ``finest_grid`` must match ``wavefield.grid``'s interior.
    clamp : bool
        Clamp the *final* reconstructed velocity to the grid's ``[c_min, c_max]``.
    u_init : optional
        Wavefield seed (list of per-shot complex tensors / Wavefields), e.g. a
        forward warm start. Defaults to the wavefield's current amplitude.
    joint : bool
        Use the single joint L-BFGS schedule instead of block-coordinate.
    modil_reg_weight : float
        Weight on :meth:`MODILVelocityParameterization.regularization_loss`
        added to the loss during c-updates.
    """

    _DEFAULT_OPTS = {
        "n_iter": 60,
        "u_steps": 6,
        "c_steps": 1,
        "max_iter": 15,
        "history_size": 10,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
        "c_lr": 1.0,
        "c_max_iter": 6,
        "c_history_size": 10,
        "reset_c_history": True,
    }
    _LBFGS_KEYS = frozenset(
        {
            "lr",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search_fn",
        }
    )

    def __init__(
        self,
        wavefield: Wavefield,
        loss: InverseLoss,
        parameterization: MODILVelocityParameterization,
        *,
        clamp: bool = False,
        u_init=None,
        joint: bool = False,
        modil_reg_weight: float = 0.0,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        if not isinstance(loss, InverseLoss):
            raise TypeError(
                "MODILInversion requires an InverseLoss (joint u,c inversion)."
            )
        grid = wavefield.grid
        assert_compatible_interiors(parameterization.finest_grid, grid)
        if (
            int(parameterization.finest_grid.interior_nx),
            int(parameterization.finest_grid.interior_ny),
        ) != (
            int(grid.interior_nx),
            int(grid.interior_ny),
        ):
            raise ValueError(
                "parameterization finest interior "
                f"{parameterization._finest_interior_shape} must match the "
                f"wavefield grid interior {(grid.interior_nx, grid.interior_ny)}."
            )
        self.param = parameterization
        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.u_init = u_init
        self.joint = bool(joint)
        self.modil_reg_weight = float(modil_reg_weight)
        self.opts = dict(self._DEFAULT_OPTS)
        self.opts.update(opts)

    def _split_opts(self):
        opts = dict(self.opts)
        n_iter = int(opts.pop("n_iter"))
        u_steps = int(opts.pop("u_steps", 1))
        c_steps = int(opts.pop("c_steps", 1))
        c_lr = float(opts.pop("c_lr", 1.0))
        c_max_iter = int(opts.pop("c_max_iter", opts.get("max_iter", 4)))
        c_history_size = int(opts.pop("c_history_size", opts.get("history_size", 10)))
        reset_c_history = bool(opts.pop("reset_c_history", True))
        u_torch_opts = {k: v for k, v in opts.items() if k in self._LBFGS_KEYS}
        c_torch_opts = dict(u_torch_opts)
        c_torch_opts["lr"] = c_lr
        c_torch_opts["max_iter"] = c_max_iter
        c_torch_opts["history_size"] = c_history_size
        return n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts, reset_c_history

    def _project_c(self, param: MODILVelocityParameterization) -> None:
        """Project the reconstructed velocity into ``[c_min, c_max]`` in place.

        Called after every c-step (and every joint step) when ``clamp`` is set,
        exactly mirroring :class:`LBFGSB`, which clamps its single-grid ``c``
        after each c-step. The box projection is folded into the **finest**
        level correction (level 0, identity prolongation), whose physical
        sensitivity is exactly ``c_ref · level_scales[0]``; this makes the new
        reconstruction equal ``clamp(c, c_min, c_max)`` to machine precision and
        keeps mODIL with a single level bit-compatible with ``LBFGSB``. Coarser
        corrections are untouched, so the finest lattice carries the (sharp)
        box-boundary detail. No-op when neither bound is set.
        """
        if self.c_min is None and self.c_max is None:
            return
        with torch.no_grad():
            c_int = param.reconstructed_interior()
            c_clamped = c_int.clamp(min=self.c_min, max=self.c_max)
            if torch.equal(c_clamped, c_int):
                return
            coeff = param.c_ref * param.level_scales[0]
            param.corrections[0].add_((c_clamped - c_int) / coeff)

    def minimise(
        self, on_iteration=None, **overrides
    ) -> Tuple[List[Wavefield], object]:
        self.opts.update(overrides)
        (n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts, reset_c_history) = (
            self._split_opts()
        )

        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        dtype = grid.dtype
        cdtype = self.wavefield.cdtype
        device = grid.device
        n_shots = self.loss.config.geometry.n_sources

        vm_in = self.wavefield.velocity_model
        param = self.param.to(device=device)

        u_seed = _seed_complex(self.wavefield, self.u_init, n_shots, cdtype, device)
        u_real = nn.Parameter(u_seed.real.contiguous().to(dtype=dtype))
        u_imag = nn.Parameter(u_seed.imag.contiguous().to(dtype=dtype))

        def pack_u() -> torch.Tensor:
            return torch.complex(u_real, u_imag)

        def pack_u_detached() -> torch.Tensor:
            return torch.complex(u_real.detach(), u_imag.detach())

        c_params = list(param.parameters())
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        n_u_closure = 0
        n_c_closure = 0

        def _reg_term() -> float:
            return self.modil_reg_weight

        if self.joint:
            joint_opt = torch.optim.LBFGS([u_real, u_imag, *c_params], **u_torch_opts)
            for i in range(n_iter):

                def joint_closure():
                    nonlocal n_u_closure
                    n_u_closure += 1
                    joint_opt.zero_grad()
                    amps = pack_u()
                    c_int = param.reconstructed_interior()
                    c_full = vm_in.build_full_c(c_int)
                    L = self.loss.evaluate(amps, c_full, c_int)
                    if self.modil_reg_weight != 0.0:
                        L = L + self.modil_reg_weight * param.regularization_loss()
                    L.backward()
                    return L

                loss_value = joint_opt.step(joint_closure)
                self._project_c(param)
                loss_value = self._log_iter(
                    i, n_iter, log_every, loss_value, param, vm_in, on_iteration
                )
        else:

            def make_u_optimiser():
                return torch.optim.LBFGS([u_real, u_imag], **u_torch_opts)

            def make_c_optimiser():
                return torch.optim.LBFGS(c_params, **c_torch_opts)

            u_optimiser = make_u_optimiser()
            c_optimiser = make_c_optimiser() if c_steps > 0 else None

            for i in range(n_iter):
                for _ in range(u_steps):

                    def u_closure():
                        nonlocal n_u_closure
                        n_u_closure += 1
                        u_optimiser.zero_grad()
                        amps = pack_u()
                        with torch.no_grad():
                            c_int_fixed = param.reconstructed_interior()
                        c_full = vm_in.build_full_c(c_int_fixed)
                        L = self.loss.evaluate(amps, c_full)
                        L.backward()
                        return L

                    loss_value = u_optimiser.step(u_closure)

                if c_steps > 0:
                    if reset_c_history:
                        c_optimiser = make_c_optimiser()
                    for _ in range(c_steps):

                        def c_closure():
                            nonlocal n_c_closure
                            n_c_closure += 1
                            c_optimiser.zero_grad()
                            c_int = param.reconstructed_interior()
                            c_full = vm_in.build_full_c(c_int)
                            amps_fixed = pack_u_detached()
                            L_c = self.loss.evaluate(amps_fixed, c_full, c_int)
                            if self.modil_reg_weight != 0.0:
                                L_c = (
                                    L_c
                                    + self.modil_reg_weight
                                    * param.regularization_loss()
                                )
                            L_c.backward()
                            return L_c

                        loss_value = c_optimiser.step(c_closure)
                        self._project_c(param)
                    # c changed -> u curvature history is stale.
                    u_optimiser = make_u_optimiser()

                loss_value = self._log_iter(
                    i, n_iter, log_every, loss_value, param, vm_in, on_iteration
                )

        # ---- finalize --------------------------------------------------- #
        c_int_final = param.reconstructed_interior().detach()
        if self.c_min is not None or self.c_max is not None:
            c_int_final = c_int_final.clamp(min=self.c_min, max=self.c_max)
        c_full_final = vm_in.build_full_c(c_int_final)
        vm_out = VelocityModel.from_field(grid, c_full_final, pml_c=vm_in.pml_c)

        u_final = pack_u_detached()
        outputs: List[Wavefield] = []
        for s in range(n_shots):
            wf = Wavefield(grid=grid, frequency_selection=freq, velocity_model=vm_out)
            wf.amplitude = u_final[s]
            outputs.append(wf)

        self.loss.callback.result = {
            "loss": (
                float(loss_value.detach().cpu()) if loss_value is not None else None
            ),
            "n_outer_iter": n_iter,
            "n_levels": param.num_levels,
            "schedule": "joint" if self.joint else "block_coordinate",
            "n_u_closure": n_u_closure,
            "n_c_closure": n_c_closure,
            "n_closure": n_u_closure + n_c_closure,
        }
        return outputs, self.loss.callback

    def _log_iter(self, i, n_iter, log_every, loss_value, param, vm_in, on_iteration):
        should_log = (i % log_every == 0) or (i == n_iter - 1)
        c_full_now = None
        if should_log or on_iteration is not None:
            with torch.no_grad():
                c_full_now = vm_in.build_full_c(param.reconstructed_interior().detach())
        if on_iteration is not None:
            on_iteration(i, c_full_now)
        loss_scalar = (
            float(loss_value.detach().cpu()) if loss_value is not None else None
        )
        if should_log and loss_scalar is not None:
            ratio = self.loss.pde_src_ratio()
            self.loss.callback.log(
                loss_scalar, self.loss._last_residuals, pde_src_ratio=ratio
            )
            if c_full_now is not None:
                self.loss.callback.log_c(c_full_now.cpu().numpy())
            print(
                f"Iteration: {i} | loss = {loss_scalar:.6e} | "
                f"|r_pde|/|src| = {ratio:.3e}"
            )
        return loss_value
