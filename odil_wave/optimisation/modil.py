"""Simultaneous multilevel (mODIL) velocity parameterization for FWI.

This module implements the multigrid-*decomposition* mODIL algorithm.
mODIL represents the finest-grid velocity as a
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

The same multilevel *decomposition* is available for the wavefield ``u``::

    u = Σ_l  s_l · P_l( δu_l )

where each ``δu_l`` is a trainable complex correction on level ``l`` (level 0 is
the finest, seeded with the wavefield warm start; coarser levels are
zero-initialised) and ``P_l`` prolongates a coarse full-grid wavefield to the
finest grid. With a single u-level this reproduces the ordinary single-grid
u-block exactly; extra levels give L-BFGS smooth, long-wavelength wavefield
updates it would otherwise reach only slowly.

Public API
----------
* :func:`build_grid_hierarchy` — validated factor-of-two grid hierarchy;
* :class:`MODILVelocityParameterization` — the velocity ``nn.Module``;
* :class:`MODILWavefieldParameterization` — the multilevel wavefield ``nn.Module``;
* :class:`MODILInversion` — an :class:`~odil_wave.optimisation.base.Optimiser`
  that jointly optimises the multilevel wavefield ``u`` and all velocity level
  corrections against the repository's :class:`~odil_wave.loss.InverseLoss`,
  with the c-block driven either by L-BFGS over the corrections or by an exact
  closed-form (variable-projection) ``c`` update (``c_update="closed_form"``).

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
    resample_full_wavefield,
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
    min_ppw: float = 5.0,
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
# Wavefield parameterization                                                  #
# --------------------------------------------------------------------------- #
class MODILWavefieldParameterization(nn.Module):
    """Multilevel additive parameterization of the complex wavefield.

    The reconstructed finest-grid wavefield is::

        u(x) = Σ_l  level_scales[level] · P_l( δu_l )

    with one trainable complex correction ``δu_l`` per level, stored as two real
    :class:`torch.nn.Parameter` tensors (real / imag) so ``torch.optim.LBFGS`` — a
    real-valued optimiser — drives it exactly as the single-grid u-block does.
    Level 0 (the finest) is initialised to the wavefield *seed* and prolongates
    by the identity; every coarser level is zero-initialised, so the initial
    reconstruction equals the seed exactly. ``P_l`` bilinearly prolongates a
    coarse *full*-grid (interior + PML) complex field to the finest full grid via
    :func:`~odil_wave.optimisation.grid_transfer.resample_full_wavefield`.

    This mirrors :class:`MODILVelocityParameterization` but for ``u``: the coarse
    levels supply smooth, long-wavelength wavefield updates that L-BFGS reaches in
    far fewer iterations than editing the finest grid alone, while the finest
    level still resolves sharp detail. With a single level it is bit-compatible
    with the ordinary two-tensor (``u_real`` / ``u_imag``) u-block.

    Parameters
    ----------
    level_grids : sequence of Grid, finest first
        Every grid must share the finest grid's physical interior (validated).
        Usually produced by :func:`build_grid_hierarchy`. A single-element
        sequence reproduces the ordinary single-grid u-block exactly.
    seed : complex Tensor ``(n_shots, nf, Nx, Ny)`` on the finest full grid
        Initial wavefield (e.g. a forward warm start). Stored into level 0.
    level_scales : sequence of float, optional
        Per-level multipliers on each prolonged correction. Default all ``1.0``
        (numerically neutral).
    staggering : {"vertex"}
        Grid-sampling convention; only vertex-centred is supported.
    """

    def __init__(
        self,
        level_grids: Sequence[Grid],
        seed: torch.Tensor,
        *,
        level_scales: Optional[Sequence[float]] = None,
        staggering: str = "vertex",
    ) -> None:
        super().__init__()
        level_grids = list(level_grids)
        if len(level_grids) < 1:
            raise ValueError("level_grids must contain at least the finest grid.")

        self.staggering = _canonical_staggering(staggering)
        self.level_grids = level_grids
        self.finest_grid = level_grids[0]
        finest = self.finest_grid
        dtype, device = finest.dtype, finest.device
        fNx, fNy = int(finest.nx), int(finest.ny)
        self._finest_full_shape = (fNx, fNy)

        # Every level must live over the same physical interior as the finest.
        for lvl, g in enumerate(level_grids):
            assert_compatible_interiors(g, finest, staggering=self.staggering)
            if g.interior_nx > int(finest.interior_nx) or g.interior_ny > int(
                finest.interior_ny
            ):
                raise ValueError(
                    f"level {lvl} interior {g.interior_nx}x{g.interior_ny} is finer "
                    f"than level 0 {finest.interior_nx}x{finest.interior_ny}; pass "
                    f"grids finest-first."
                )

        seed = torch.as_tensor(seed)
        if not seed.is_complex():
            raise ValueError("wavefield seed must be complex.")
        if seed.ndim != 4 or tuple(seed.shape[-2:]) != (fNx, fNy):
            raise ValueError(
                f"seed shape {tuple(seed.shape)} must be (n_shots, nf, {fNx}, {fNy}) "
                f"on the finest full grid."
            )
        seed = seed.detach()
        self.n_shots, self.nf = int(seed.shape[0]), int(seed.shape[1])

        # One trainable complex correction per level, split into real/imag so
        # L-BFGS (a real optimiser) can own them. Level 0 carries the seed;
        # coarser levels start at zero (initial reconstruction == seed).
        self.u_real = nn.ParameterList()
        self.u_imag = nn.ParameterList()
        for lvl, g in enumerate(level_grids):
            if lvl == 0:
                re = seed.real.to(dtype=dtype, device=device).contiguous()
                im = seed.imag.to(dtype=dtype, device=device).contiguous()
            else:
                shape = (self.n_shots, self.nf, int(g.nx), int(g.ny))
                re = torch.zeros(shape, dtype=dtype, device=device)
                im = torch.zeros(shape, dtype=dtype, device=device)
            self.u_real.append(nn.Parameter(re))
            self.u_imag.append(nn.Parameter(im))

        if level_scales is None:
            scales = [1.0] * len(level_grids)
        else:
            if len(level_scales) != len(level_grids):
                raise ValueError("level_scales length must equal the number of levels.")
            scales = [float(s) for s in level_scales]
        self.register_buffer(
            "level_scales", torch.tensor(scales, dtype=dtype, device=device)
        )

    # -- introspection ----------------------------------------------------- #
    @property
    def num_levels(self) -> int:
        return len(self.u_real)

    # -- reconstruction ---------------------------------------------------- #
    def _prolonged_contribution(self, level: int) -> torch.Tensor:
        """``level_scales[level] · P_l(δu_l)`` on the finest full grid (differentiable)."""
        corr = torch.complex(self.u_real[level], self.u_imag[level])
        g = self.level_grids[level]
        if (int(g.nx), int(g.ny)) == self._finest_full_shape:
            contrib = corr
        else:
            contrib = resample_full_wavefield(corr, g, self.finest_grid)
        return self.level_scales[level] * contrib

    def reconstructed_wavefield(self) -> torch.Tensor:
        """Reconstructed finest-grid wavefield ``(n_shots, nf, Nx, Ny)`` (complex).

        Differentiable with respect to every level correction. With all coarser
        corrections zero this returns exactly the seed.
        """
        total = self._prolonged_contribution(0)
        for level in range(1, self.num_levels):
            total = total + self._prolonged_contribution(level)
        return total

    def forward(self) -> torch.Tensor:
        return self.reconstructed_wavefield()

    def native_corrections(self) -> List[torch.Tensor]:
        """Detached copies of each level's native (own-resolution) complex correction."""
        return [
            torch.complex(self.u_real[level], self.u_imag[level]).detach().clone()
            for level in range(self.num_levels)
        ]

    def prolonged_contributions(self) -> List[torch.Tensor]:
        """Detached copies of each level's contribution on the finest full grid."""
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
    """Multilevel-u + mODIL-c frequency-domain FWI optimiser.

    Owns a :class:`MODILVelocityParameterization` (multilevel ``c``) and a
    :class:`MODILWavefieldParameterization` (multilevel ``u``) and exposes each
    block's level corrections to L-BFGS simultaneously. Every loss evaluation
    reconstructs both the finest-grid velocity and the finest-grid wavefield from
    the current corrections (never a stale reference), so the standard
    :class:`~odil_wave.loss.InverseLoss` is used unchanged.

    Block-coordinate schedule: ``u_steps`` of L-BFGS over *all* wavefield level
    corrections with the velocity fixed, then a c-update with ``u`` fixed —
    mirroring :class:`~odil_wave.optimisation.LBFGSB` but with multilevel ``u``
    and ``c``. The c-update is either (``c_update="lbfgs"``, default) ``c_steps``
    of L-BFGS over *all* velocity level corrections at once, or
    (``c_update="closed_form"``) a single exact per-cell variable-projection
    solve ``c* = c0 √( Σ Re(ā b) / Σ |a|² )`` (see
    :meth:`odil_wave.operator.utils.WaveEquation.c_closed_form`), optionally
    relaxed and clamped, folded back into the finest velocity correction. The
    closed form is the argmin of the loss over ``c`` for the current ``u`` — no
    c-step size, preconditioner, or line search to tune — and needs no multilevel
    c-hierarchy to work (level 0 absorbs the update).

    The wavefield hierarchy defaults to a single level (ordinary single-grid
    ``u``); pass ``u_num_levels > 1`` (or an explicit ``u_levels``) for
    multilevel ``u``.

    Parameters
    ----------
    parameterization : MODILVelocityParameterization
        Its ``finest_grid`` must match ``wavefield.grid``'s interior.
    clamp : bool
        Clamp the reconstructed velocity to the grid's ``[c_min, c_max]`` after
        every c-update (and the final model).
    u_init : optional
        Wavefield seed (list of per-shot complex tensors / Wavefields), e.g. a
        forward warm start. Defaults to the wavefield's current amplitude. Seeds
        the finest u-level.
    u_levels : sequence of Grid, optional
        Explicit finest-first wavefield hierarchy. Its finest grid must match
        ``wavefield.grid``. Overrides ``u_num_levels`` when given.
    u_num_levels : int
        Number of wavefield levels to build (via :func:`build_grid_hierarchy`)
        when ``u_levels`` is not supplied. ``1`` (default) is single-grid ``u``.
    u_coarsening_factor : int
        Coarsening factor for the auto-built wavefield hierarchy (default ``2``).
    u_level_scales : sequence of float, optional
        Per-level multipliers on the prolonged wavefield corrections.
    c_update : {"lbfgs", "closed_form"}
        Velocity block: L-BFGS over the level corrections (default) or the exact
        closed-form variable-projection update.
    c_relax : float
        Closed-form only: relaxation ``c ← (1-α) c + α c*`` (``α = c_relax``).
    c_update_every : int
        Closed-form only: apply the c-update every ``c_update_every`` outer
        iterations (default every iteration).
    illum_rel_floor : float
        Closed-form only: cells illuminated below this fraction of the peak keep
        their current velocity.
    modil_reg_weight : float
        Weight on :meth:`MODILVelocityParameterization.regularization_loss`
        added to the loss during L-BFGS c-updates.
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
        u_levels: Optional[Sequence[Grid]] = None,
        u_num_levels: int = 1,
        u_coarsening_factor: int = 2,
        u_level_scales: Optional[Sequence[float]] = None,
        c_update: str = "lbfgs",
        c_relax: float = 1.0,
        c_update_every: int = 1,
        illum_rel_floor: float = 1e-6,
        modil_reg_weight: float = 0.0,
        **opts,
    ) -> None:
        super().__init__(wavefield, loss)
        if not isinstance(loss, InverseLoss):
            raise TypeError(
                "MODILInversion requires an InverseLoss (joint u,c inversion)."
            )
        if c_update not in ("lbfgs", "closed_form"):
            raise ValueError(
                f"c_update must be 'lbfgs' or 'closed_form', got {c_update!r}."
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

        # Wavefield hierarchy (finest first, finest == wavefield.grid).
        if u_levels is not None:
            u_levels = list(u_levels)
            if len(u_levels) < 1:
                raise ValueError("u_levels must contain at least the finest grid.")
            assert_compatible_interiors(u_levels[0], grid)
            if (int(u_levels[0].nx), int(u_levels[0].ny)) != (
                int(grid.nx),
                int(grid.ny),
            ):
                raise ValueError(
                    f"u_levels finest full shape "
                    f"{(u_levels[0].nx, u_levels[0].ny)} must match the wavefield "
                    f"grid {(grid.nx, grid.ny)}."
                )
            self.u_levels = u_levels
        elif int(u_num_levels) > 1:
            self.u_levels = build_grid_hierarchy(
                grid, int(u_num_levels), int(u_coarsening_factor)
            )
        else:
            self.u_levels = [grid]
        self.u_level_scales = u_level_scales

        self.c_min = grid.c_min if clamp else None
        self.c_max = grid.c_max if clamp else None
        self.u_init = u_init
        self.c_update = c_update
        self.c_relax = float(c_relax)
        self.c_update_every = int(c_update_every)
        self.illum_rel_floor = float(illum_rel_floor)
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

    def _fold_reconstruction_to(
        self, param: MODILVelocityParameterization, target_interior: torch.Tensor
    ) -> None:
        """Adjust the finest correction so the reconstruction equals ``target``.

        The change is folded into the **finest** level correction (level 0,
        identity prolongation), whose physical sensitivity is exactly
        ``c_ref · level_scales[0]``; coarser corrections are left untouched, so
        the reconstruction becomes ``target_interior`` to machine precision while
        the finest lattice carries any newly introduced detail. Both the box
        projection (:meth:`_project_c`) and the closed-form c-update write back
        through this single mechanism.
        """
        with torch.no_grad():
            c_int = param.reconstructed_interior()
            coeff = param.c_ref * param.level_scales[0]
            param.corrections[0].add_((target_interior - c_int) / coeff)

    def _project_c(self, param: MODILVelocityParameterization) -> None:
        """Project the reconstructed velocity into ``[c_min, c_max]`` in place.

        Called after every L-BFGS c-step when ``clamp`` is set, exactly mirroring
        :class:`LBFGSB`, which clamps its single-grid ``c`` after each c-step.
        No-op when neither bound is set.
        """
        if self.c_min is None and self.c_max is None:
            return
        with torch.no_grad():
            c_int = param.reconstructed_interior()
            c_clamped = c_int.clamp(min=self.c_min, max=self.c_max)
            if torch.equal(c_clamped, c_int):
                return
        self._fold_reconstruction_to(param, c_clamped)

    def _prox_regularise(
        self, c_star: torch.Tensor, illum: torch.Tensor
    ) -> torch.Tensor:
        """Illumination-weighted proximal step applying the loss regulariser.

        Solves ``min_c 0.5 Σ w (c - c*)² + λ R(c)`` on the interior map with
        ``w = illum / mean(illum)``, mirroring
        :meth:`~odil_wave.optimisation.base.LBFGSB._prox_regularise`.
        No-op without a configured regulariser. Used only by the closed-form
        c-update.
        """
        reg = self.loss.config.regulariser
        lam = float(self.loss.config.weights.get("reg", 0.0))
        if reg is None or lam <= 0.0:
            return c_star
        w = illum / illum.mean().clamp(min=1e-30)
        c = c_star.detach().clone().requires_grad_(True)
        prox_opt = torch.optim.LBFGS(
            [c], max_iter=50, history_size=10, line_search_fn="strong_wolfe"
        )

        def prox_closure():
            with torch.enable_grad():
                prox_opt.zero_grad()
                F = 0.5 * (w * (c - c_star) ** 2).sum() + lam * reg(c)
                F.backward()
            return F

        prox_opt.step(prox_closure)
        return c.detach()

    def _closed_form_c_step(
        self,
        param: MODILVelocityParameterization,
        amps_detached: torch.Tensor,
        vm_in,
        grid: Grid,
        wave_eq,
    ) -> torch.Tensor:
        """One exact per-cell variable-projection c-update, folded into ``param``.

        Computes ``c*`` for the current (fixed) wavefield, optionally relaxes it
        against the current reconstruction (``c_relax``), regularises it and
        clamps it to ``[c_min, c_max]``, then writes the result into the finest
        velocity correction via :meth:`_fold_reconstruction_to`. Returns the loss
        at the updated ``(u, c)`` for logging.
        """
        with torch.no_grad():
            c_full_cur = vm_in.build_full_c(param.reconstructed_interior())
            c_star_full, illum_full = wave_eq.c_closed_form(
                amps_detached,
                self.loss.sources,
                c_current=c_full_cur,
                illum_rel_floor=self.illum_rel_floor,
            )
            c_star = c_star_full[grid.interior_slice]
            c_star = self._prox_regularise(c_star, illum_full[grid.interior_slice])
            alpha = self.c_relax
            if alpha != 1.0:
                c_star = (1.0 - alpha) * param.reconstructed_interior() + alpha * c_star
            if self.c_min is not None or self.c_max is not None:
                c_star = c_star.clamp(min=self.c_min, max=self.c_max)
            self._fold_reconstruction_to(param, c_star)
            c_full_new = vm_in.build_full_c(param.reconstructed_interior())
            L = self.loss.evaluate(
                amps_detached, c_full_new, param.reconstructed_interior()
            )
        return L

    def minimise(
        self, on_iteration=None, **overrides
    ) -> Tuple[List[Wavefield], object]:
        self.opts.update(overrides)
        (n_iter, u_steps, c_steps, u_torch_opts, c_torch_opts, reset_c_history) = (
            self._split_opts()
        )

        grid = self.wavefield.grid
        freq = self.wavefield.frequency_selection
        cdtype = self.wavefield.cdtype
        device = grid.device
        n_shots = self.loss.config.geometry.n_sources

        vm_in = self.wavefield.velocity_model
        wave_eq = self.loss.config.wave_eq
        param = self.param.to(device=device)
        closed_form_c = self.c_update == "closed_form"

        # ---- multilevel wavefield parameterization ---------------------- #
        u_seed = _seed_complex(self.wavefield, self.u_init, n_shots, cdtype, device)
        u_param = MODILWavefieldParameterization(
            self.u_levels, u_seed, level_scales=self.u_level_scales
        ).to(device=device)

        def pack_u() -> torch.Tensor:
            return u_param.reconstructed_wavefield()

        def pack_u_detached() -> torch.Tensor:
            return u_param.reconstructed_wavefield().detach()

        u_params = list(u_param.parameters())
        c_params = list(param.parameters())
        log_every = max(1, int(self.loss.callback.log_every))
        loss_value = None
        n_u_closure = 0
        n_c_closure = 0

        def make_u_optimiser():
            return torch.optim.LBFGS(u_params, **u_torch_opts)

        def make_c_optimiser():
            return torch.optim.LBFGS(c_params, **c_torch_opts)

        u_optimiser = make_u_optimiser()
        c_optimiser = (
            make_c_optimiser() if (c_steps > 0 and not closed_form_c) else None
        )

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

            if closed_form_c:
                do_c = self.c_update_every > 0 and (i + 1) % self.c_update_every == 0
                if do_c:
                    loss_value = self._closed_form_c_step(
                        param, pack_u_detached(), vm_in, grid, wave_eq
                    )
                    # c changed -> u curvature history is stale.
                    u_optimiser = make_u_optimiser()
            elif c_steps > 0:
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
                                + self.modil_reg_weight * param.regularization_loss()
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
        vm_out = VelocityModel.from_field(
            grid, c_full_final, pml_c=vm_in.pml_c, pml_fill=vm_in.pml_fill
        )

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
            "n_u_levels": u_param.num_levels,
            "c_update": self.c_update,
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
