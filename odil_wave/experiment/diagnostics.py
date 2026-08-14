"""Block-coordinate diagnostics for the LBFGSB direct-u optimiser.

A :class:`DiagnosticsCollector` is passed to ``LBFGSB.minimise(diagnostics=...)``
(via ``run_inverse`` when ``diagnostics.enabled``). The optimiser hands it a
per-outer snapshot of the wavefield before/after the u-block and the c-block; the
collector computes the block-diagnostic quantities on *clones* (never disturbing
the trajectory) and writes, into ``<run_dir>/diagnostics/``:

* ``scalars.csv`` — per outer: ``cos(update, -grad)`` for the u/c blocks, the
  block gradient L2 norms, the relative model error and relative PDE/data
  residuals at the end of the outer, and the scheduled ``c_lr`` /
  ``c_grad_smooth_sigma`` in effect (A4).
* ``summary.png`` — four trajectory panels built from those scalars.
* ``c_evolution.png`` — the physical interior velocity after each outer (init
  first, truth last), refreshed every outer so a partial run keeps a figure.
* ``outer_XX.png`` — (only when ``per_outer_field_maps``) per-outer term-induced
  wavefield update maps ``du_term = -H^{-1} g_term`` at several u-solve depths
  plus the c-gradient / velocity update at the exact wavefield minimiser ``u*``.

This is the library port of ``sandbox/profiling/optim_block_diag/run_diagnostics``
(+ ``diagnostics`` / ``objective`` / ``hess_split``), trimmed to exactly the four
outputs above. Term gradients are isolated through the library loss itself
(``InverseLoss.evaluate(weights_override=...)``), so the diagnostics track the
*actual* objective, and the ``-H^{-1} g`` maps reuse :class:`UBlockHessian`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _new_figure(**kwargs) -> Figure:
    """A standalone Agg-backed Figure not registered with pyplot."""
    fig = Figure(**kwargs)
    FigureCanvasAgg(fig)  # attaches itself as ``fig.canvas``
    return fig


from odil_wave.models import velocity_norm  # noqa: E402
from odil_wave.optimisation.base import (  # noqa: E402
    _c_hat_from_c_param,
    _gaussian_smooth_2d,
)
from odil_wave.optimisation.u_block_hessian import UBlockHessian  # noqa: E402

_TINY = 1e-300

# Shared two-slope velocity colour scale for c_evolution (matches the sandbox /
# VelocityModel.show(): c_min at bottom, 1600 midpoint, high velocity at top).
_C_EVOL_VMIN = 1400.0
_C_EVOL_VCENTER = 1600.0
_C_EVOL_VMAX = 3000.0


# --------------------------------------------------------------------------- #
# complex-aware inner-product / norm helpers
# --------------------------------------------------------------------------- #
def _gnorm(x: torch.Tensor) -> float:
    v = x.real.square() + x.imag.square() if x.is_complex() else x.square()
    return float(v.sum().sqrt().cpu())


def _real_inner(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.is_complex() or b.is_complex():
        return float((a.conj() * b).real.sum().cpu())
    return float((a * b).sum().cpu())


def _real_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = _gnorm(a), _gnorm(b)
    if na < _TINY or nb < _TINY:
        return float("nan")
    return _real_inner(a, b) / (na * nb)


def _row_depths(u_steps: int) -> List[int]:
    """Default per-outer u-solve depths: 1, 5, 10, ... up to u_steps."""
    steps = [1] + list(range(5, int(u_steps) + 1, 5))
    if u_steps not in steps:
        steps.append(int(u_steps))
    return sorted({s for s in steps if 1 <= s <= max(int(u_steps), 1)})


class DiagnosticsCollector:
    """Capture + render block diagnostics for one LBFGSB run (single band)."""

    def __init__(
        self,
        *,
        out_dir: Path,
        truth_velocity,
        grid,
        per_outer_field_maps: bool = True,
        u_depths: Optional[List[int]] = None,
        verify_hessian: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.grid = grid
        self.per_outer_field_maps = bool(per_outer_field_maps)
        self.u_depths_cfg = u_depths
        self.verify_hessian = bool(verify_hessian)
        self.interior_slice = grid.interior_slice
        self.interior_shape = tuple(truth_velocity.c[grid.interior_slice].shape)
        self._truth_int = (
            truth_velocity.c[grid.interior_slice].detach().clone().double()
        )

        # per-outer scalar tape
        self._cos_du: List[float] = []
        self._cos_dc: List[float] = []
        self._ug_norm: List[float] = []
        self._cg_norm: List[float] = []
        self._rel_c_end: List[float] = []
        self._rel_data_end: List[float] = []
        self._rel_pde_end: List[float] = []
        self._c_lr: List[float] = []
        self._sigma: List[float] = []
        self._init_rel: Optional[Tuple[float, float, float]] = None
        self._c_maps: List[Tuple[str, float, np.ndarray]] = []
        self._bound = False

    # -- bound by LBFGSB.minimise at the start of the run ------------------- #
    def bind(
        self,
        *,
        loss,
        wavefield,
        c_ref: float,
        c_param: str,
        build_full_c,
        u_torch_opts: dict,
        u_steps: int,
        c_torch_opts: dict,
        c_steps: int,
        c_param_bounds: Tuple[Optional[float], Optional[float]],
        u_solve: str,
    ) -> None:
        self.loss = loss
        self.wavefield = wavefield
        self.c_ref = float(c_ref)
        self.c_param = c_param
        self.build_full_c = build_full_c
        self.u_torch_opts = dict(u_torch_opts)
        self.u_steps = int(u_steps)
        self.c_torch_opts = dict(c_torch_opts)
        self.c_steps = int(c_steps)
        self.c_param_bounds = c_param_bounds
        self.u_solve = u_solve
        self.depths = (
            list(self.u_depths_cfg)
            if self.u_depths_cfg is not None
            else _row_depths(self.u_steps)
        )
        self._bound = True

    # -- coordinate helpers ------------------------------------------------- #
    def _c_full(self, c_raw: torch.Tensor) -> torch.Tensor:
        c_hat = _c_hat_from_c_param(c_raw.detach(), self.c_param)
        return self.build_full_c(c_hat * self.c_ref)

    def _c_phys_int(self, c_raw: torch.Tensor) -> torch.Tensor:
        c_hat = _c_hat_from_c_param(c_raw.detach(), self.c_param)
        return c_hat * self.c_ref

    def _term_weights(self, term: str) -> Dict[str, float]:
        if term == "pde":
            return {"pde": 1.0, "data": 0.0, "reg": 0.0}
        if term == "data":
            return {"pde": 0.0, "data": 1.0, "reg": 0.0}
        w = dict(self.loss.config.weights)
        w.setdefault("reg", 0.0)
        return w

    # -- gradients (on clones; loss.evaluate isolates the terms) ------------ #
    def _u_grad(self, u_re, u_im, c_full, c_hat, term: str) -> torch.Tensor:
        a = u_re.detach().clone().requires_grad_(True)
        b = u_im.detach().clone().requires_grad_(True)
        L = self.loss.evaluate(
            torch.complex(a, b), c_full, None, weights_override=self._term_weights(term)
        )
        ga, gb = torch.autograd.grad(L, (a, b), allow_unused=True)
        ga = torch.zeros_like(a) if ga is None else ga
        gb = torch.zeros_like(b) if gb is None else gb
        return torch.complex(ga.detach(), gb.detach())

    def _u_block_grads(self, u_re, u_im, c_raw) -> Dict[str, object]:
        c_hat = _c_hat_from_c_param(c_raw.detach(), self.c_param)
        c_full = self.build_full_c(c_hat * self.c_ref)
        w = self._term_weights("total")
        g_pde = self._u_grad(u_re, u_im, c_full, c_hat, "pde")
        g_data = self._u_grad(u_re, u_im, c_full, c_hat, "data")
        g_total = self._u_grad(u_re, u_im, c_full, c_hat, "total")
        return {
            "g_total": g_total,
            "g_pde_weighted": float(w["pde"]) * g_pde,
            "g_data_weighted": float(w["data"]) * g_data,
            "norm_total_l2": _gnorm(g_total),
        }

    def _c_block_grad(self, u_re, u_im, c_raw) -> Dict[str, object]:
        """Weighted c-block gradient (raw coordinate) with u frozen."""
        cr = c_raw.detach().clone().requires_grad_(True)
        c_hat = _c_hat_from_c_param(cr, self.c_param)
        c_full = self.build_full_c(c_hat * self.c_ref)
        u = torch.complex(u_re.detach(), u_im.detach())
        L = self.loss.evaluate(
            u, c_full, c_hat, weights_override=self._term_weights("total")
        )
        (g,) = torch.autograd.grad(L, cr, allow_unused=True)
        g = torch.zeros_like(cr) if g is None else g.detach()
        return {"g_total": g, "norm_weighted_l2": _gnorm(g)}

    # -- state (relative model error + residuals) --------------------------- #
    def _state(self, u_re, u_im, c_raw) -> Dict[str, float]:
        c_full = self._c_full(c_raw)
        u = torch.complex(u_re.detach(), u_im.detach())
        r_pde, r_data = self.loss._residuals(u, c_full)
        obs = self.loss._obs_traces()
        c_int = self._c_phys_int(c_raw).double()
        rel_c = _gnorm(c_int - self._truth_int) / max(_gnorm(self._truth_int), _TINY)
        return {
            "rel_c": rel_c,
            "rel_pde": _gnorm(r_pde) / max(_gnorm(self.loss.sources), _TINY),
            "rel_data": _gnorm(r_data) / max(_gnorm(obs), _TINY),
        }

    # -- re-run the u-block on a clone to several depths -------------------- #
    def _u_depths(self, u_re, u_im, c_raw):
        c_hat = _c_hat_from_c_param(c_raw.detach(), self.c_param)
        c_full = self.build_full_c(c_hat * self.c_ref)
        ur = u_re.detach().clone().requires_grad_(True)
        ui = u_im.detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS([ur, ui], **self.u_torch_opts)

        def closure():
            opt.zero_grad()
            L = self.loss.evaluate(torch.complex(ur, ui), c_full, c_hat)
            L.backward()
            return L

        fields: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        cur = 0
        for N in sorted(self.depths):
            while cur < N:
                opt.step(closure)
                cur += 1
            fields[N] = (ur.detach().clone(), ui.detach().clone())
        return fields

    # -- one real c-block on a clone: the dc it would drive ----------------- #
    def _fire_c_block(self, u_re, u_im, c_raw, sigma: float) -> torch.Tensor:
        cr = c_raw.detach().clone().requires_grad_(True)
        u = torch.complex(u_re.detach(), u_im.detach())
        opt = torch.optim.LBFGS([cr], **self.c_torch_opts)
        lo, hi = self.c_param_bounds

        def closure():
            opt.zero_grad()
            c_hat = _c_hat_from_c_param(cr, self.c_param)
            c_full = self.build_full_c(c_hat * self.c_ref)
            L = self.loss.evaluate(u, c_full, c_hat)
            L.backward()
            if sigma > 0.0 and cr.grad is not None:
                with torch.no_grad():
                    cr.grad.copy_(_gaussian_smooth_2d(cr.grad, sigma))
            return L

        for _ in range(self.c_steps):
            opt.step(closure)
            with torch.no_grad():
                if lo is not None or hi is not None:
                    cr.clamp_(min=lo, max=hi)
        dc_hat = _c_hat_from_c_param(cr.detach(), self.c_param) - _c_hat_from_c_param(
            c_raw.detach(), self.c_param
        )
        return dc_hat * self.c_ref

    # -- map reductions ----------------------------------------------------- #
    @staticmethod
    def _u_map(g: torch.Tensor) -> np.ndarray:
        return g.detach().abs().sum(dim=(0, 1)).cpu().numpy()

    def _c_map(self, g: torch.Tensor) -> np.ndarray:
        return g.detach().cpu().numpy().reshape(self.interior_shape)

    # ==================================================================== #
    # LBFGSB hooks
    # ==================================================================== #
    def record_init(self, c_raw, u_re, u_im) -> None:
        if not self._bound:
            return
        sd = self._state(u_re, u_im, c_raw)
        self._init_rel = (sd["rel_c"], sd["rel_data"], sd["rel_pde"])
        self._c_maps = [("init", sd["rel_c"], self._c_phys_int(c_raw).cpu().numpy())]
        self._render_c_evolution()

    def record_outer(
        self,
        i: int,
        u_before_re,
        u_before_im,
        u_after_re,
        u_after_im,
        c_raw_before,
        c_raw_after,
        *,
        c_lr: float,
        c_grad_smooth_sigma: float,
    ) -> None:
        if not self._bound:
            return
        # u-block: cos(du, -g) and ||g_u|| at the pre-u-block state.
        ug = self._u_block_grads(u_before_re, u_before_im, c_raw_before)
        du = torch.complex(u_after_re - u_before_re, u_after_im - u_before_im)
        self._cos_du.append(_real_cos(du, -ug["g_total"]))
        self._ug_norm.append(ug["norm_total_l2"])

        # c-block: cos(dc, -g_c) and ||g_c|| at the pre-c-block state (u frozen).
        cg = self._c_block_grad(u_after_re, u_after_im, c_raw_before)
        dc = c_raw_after - c_raw_before
        self._cos_dc.append(_real_cos(dc, -cg["g_total"]))
        self._cg_norm.append(cg["norm_weighted_l2"])

        # end-of-outer state.
        sd = self._state(u_after_re, u_after_im, c_raw_after)
        self._rel_c_end.append(sd["rel_c"])
        self._rel_data_end.append(sd["rel_data"])
        self._rel_pde_end.append(sd["rel_pde"])
        self._c_lr.append(float(c_lr))
        self._sigma.append(float(c_grad_smooth_sigma))

        self._c_maps.append(
            (f"outer {i + 1}", sd["rel_c"], self._c_phys_int(c_raw_after).cpu().numpy())
        )
        self._render_c_evolution()

        if self.per_outer_field_maps:
            self._render_outer_figure(
                i, u_before_re, u_before_im, c_raw_before, c_grad_smooth_sigma
            )

    def finalise(self) -> None:
        if not self._bound or not self._rel_c_end:
            return
        self._write_scalars()
        self._render_summary()

    # ==================================================================== #
    # writers / renderers
    # ==================================================================== #
    def _write_scalars(self) -> None:
        with (self.out_dir / "scalars.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "outer",
                    "cos_du_neg_g",
                    "cos_dc_neg_g",
                    "ug_norm_total_l2",
                    "cg_norm_weighted_l2",
                    "rel_c_end",
                    "rel_data_end",
                    "rel_pde_end",
                    "c_lr",
                    "c_grad_smooth_sigma",
                ]
            )
            for i in range(len(self._rel_c_end)):
                w.writerow(
                    [
                        i,
                        self._cos_du[i],
                        self._cos_dc[i],
                        self._ug_norm[i],
                        self._cg_norm[i],
                        self._rel_c_end[i],
                        self._rel_data_end[i],
                        self._rel_pde_end[i],
                        self._c_lr[i],
                        self._sigma[i],
                    ]
                )

    def _render_c_evolution(self) -> None:
        truth = self._truth_int.cpu().numpy()
        panels = list(self._c_maps) + [("truth", 0.0, truth)]
        norm = velocity_norm(_C_EVOL_VMIN, _C_EVOL_VCENTER, _C_EVOL_VMAX)
        n = len(panels)
        fig = _new_figure(figsize=(3.0 * n, 3.4))
        ax = fig.subplots(1, n, squeeze=False)
        for k, (label, rel_c, m) in enumerate(panels):
            a = ax[0, k]
            im = a.imshow(m.T, origin="lower", cmap="viridis", norm=norm)
            cbar = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            cbar.set_ticks([_C_EVOL_VMIN, _C_EVOL_VCENTER, 2000, 2500, _C_EVOL_VMAX])
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{label}\nrel_c={rel_c:.4f}", fontsize=9)
        fig.suptitle("Interior velocity c [m/s] per outer iteration", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(self.out_dir / "c_evolution.png", dpi=130)

    def _render_summary(self) -> None:
        n = len(self._rel_c_end)
        outers = np.arange(n)
        sx = np.arange(n + 1)
        irc, ird, irp = self._init_rel if self._init_rel else (np.nan, np.nan, np.nan)
        relc = np.array([irc] + self._rel_c_end)
        reld = np.array([ird] + self._rel_data_end)
        relp = np.array([irp] + self._rel_pde_end)

        fig = _new_figure(figsize=(13, 9))
        ax = fig.subplots(2, 2)
        a = ax[0, 0]
        a.plot(outers, self._cos_du, "o-", c="C0", label="u block  cos(du, -g)")
        a.plot(outers, self._cos_dc, "s-", c="C3", label="c block  cos(dc, -g)")
        a.axhline(0.0, ls=":", c="gray")
        a.set_xlabel("outer iteration")
        a.set_ylabel("cos(update, -grad)")
        a.set_title("block update alignment")
        a.legend(fontsize=9)
        a.grid(alpha=0.3)

        a = ax[0, 1]
        a.plot(sx, relc, "o-", c="C3")
        a.set_xlabel("outer (0 = init)")
        a.set_ylabel("rel_c")
        a.set_title("relative model error")
        a.grid(alpha=0.3)

        a = ax[1, 0]
        a.plot(sx, reld, "o-", c="C0", label="rel_data")
        a.plot(sx[1:], relp[1:], "s-", c="C2", label="rel_pde (skip init)")
        a.set_yscale("log")
        a.set_xlabel("outer (0 = init)")
        a.set_ylabel("relative residual")
        a.set_title("relative residuals")
        a.legend(fontsize=9)
        a.grid(alpha=0.3)

        a = ax[1, 1]
        a.plot(outers, self._ug_norm, "o-", c="C4", label="||g_u|| total (u_before)")
        a.plot(outers, self._cg_norm, "s-", c="C5", label="||g_c|| total (c_before)")
        a.set_yscale("log")
        a.set_xlabel("outer iteration")
        a.set_ylabel("gradient L2 norm")
        a.set_title("block gradient norms")
        a.legend(fontsize=9)
        a.grid(alpha=0.3)

        fig.suptitle("LBFGSB direct-u block-coordinate trajectory", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(self.out_dir / "summary.png", dpi=130)

    def _render_outer_figure(self, i, u_before_re, u_before_im, c_raw, sigma) -> None:
        c_full = self._c_full(c_raw)
        ubh = UBlockHessian(
            self.wavefield, self.loss, c_full, weights=dict(self.loss.config.weights)
        )
        if self.verify_hessian:
            chk = ubh.verify(u_before_re, u_before_im, c_full)
            print(
                f"    [diag outer {i:2d}] H-split verify rel_err={chk['rel_err']:.1e} "
                f"(alpha={chk['alpha']:.2e} beta={chk['beta']:.2e})"
            )
        u0 = torch.complex(u_before_re, u_before_im)
        exact = self.u_solve == "exact"

        def _row(u_re, u_im, label):
            """One figure row of term-split maps at the wavefield state (u_re,u_im)."""
            ug = self._u_block_grads(u_re, u_im, c_raw)
            cg = self._c_block_grad(u_re, u_im, c_raw)
            du_pde = ubh.du_from_grad(ug["g_pde_weighted"])
            du_data = ubh.du_from_grad(ug["g_data_weighted"])
            du_split = du_pde + du_data
            du_total = torch.complex(u_re, u_im) - u0
            dc_phys = self._fire_c_block(u_re, u_im, c_raw, sigma)
            return {
                "N": label,
                "panels": [
                    (
                        self._u_map(ug["g_total"]),
                        False,
                        "g (weighted total)",
                        ug["norm_total_l2"],
                    ),
                    (
                        self._u_map(du_pde),
                        False,
                        "du_pde = -H^-1 g_pde",
                        _gnorm(du_pde),
                    ),
                    (
                        self._u_map(du_data),
                        False,
                        "du_data = -H^-1 g_data",
                        _gnorm(du_data),
                    ),
                    (
                        self._u_map(du_split),
                        False,
                        "du_pde + du_data",
                        _gnorm(du_split),
                    ),
                    (
                        self._u_map(du_total),
                        False,
                        "du_total = u_N - u_0",
                        _gnorm(du_total),
                    ),
                    (
                        self._c_map(cg["g_total"]),
                        True,
                        "g_c total",
                        cg["norm_weighted_l2"],
                    ),
                    (
                        self._c_map(dc_phys),
                        True,
                        "dc [m/s]",
                        float(np.linalg.norm(dc_phys.detach().cpu().numpy())),
                    ),
                ],
            }

        # In exact mode the run reaches u* = u0 - H^{-1} g(u0) in a single direct
        # step (no iterative L-BFGS), so collapse the depth sweep to one row
        # evaluated at u* itself. Otherwise show the iterative u-solve depths.
        rows = []
        if exact:
            ug0 = self._u_block_grads(u_before_re, u_before_im, c_raw)
            u_star = u0 + ubh.du_from_grad(ug0["g_total"])
            rows.append(_row(u_star.real, u_star.imag, "exact"))
        else:
            fields = self._u_depths(u_before_re, u_before_im, c_raw)
            for N in sorted(self.depths):
                u_reN, u_imN = fields[N]
                rows.append(_row(u_reN, u_imN, N))

        # Reference bottom row: c-gradient / dc at the exact minimiser u*. Optim
        # path only — in exact mode the single row above already *is* u*, so this
        # would duplicate its g_c / dc columns.
        star_row = not exact
        if star_row:
            ug0 = self._u_block_grads(u_before_re, u_before_im, c_raw)
            u_star = u0 + ubh.du_from_grad(ug0["g_total"])
            cg_star = self._c_block_grad(u_star.real, u_star.imag, c_raw)
            gc_star_map = self._c_map(cg_star["g_total"])
            dc_star = self._fire_c_block(u_star.real, u_star.imag, c_raw, sigma)
            dc_star_map = self._c_map(dc_star)

        ncol = len(rows[0]["panels"])
        nrow = len(rows)
        gc_col = next(
            (
                c
                for c, p in enumerate(rows[0]["panels"])
                if str(p[2]).startswith("g_c total")
            ),
            ncol - 2,
        )
        dc_col = next(
            (
                c
                for c, p in enumerate(rows[0]["panels"])
                if str(p[2]).startswith("dc [m/s]")
            ),
            ncol - 1,
        )
        n_extra = 1 if star_row else 0
        fig = _new_figure(figsize=(2.9 * ncol, 2.7 * (nrow + 1)))
        gs = fig.add_gridspec(nrow + n_extra, ncol)
        ax = np.empty((nrow, ncol), dtype=object)
        for r in range(nrow):
            for c in range(ncol):
                ax[r, c] = fig.add_subplot(gs[r, c])
        for r, row in enumerate(rows):
            for c, (arr, signed, name, nrm) in enumerate(row["panels"]):
                a = ax[r, c]
                if signed:
                    v = float(np.nanpercentile(np.abs(arr), 99.5)) or 1.0
                    im = a.imshow(arr.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
                else:
                    im = a.imshow(arr.T, origin="lower", cmap="magma")
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(
                    (f"{name}\n" if r == 0 else "") + f"||.||={nrm:.1e}", fontsize=8
                )
            ax[r, 0].set_ylabel(f"N={row['N']}", fontsize=11)

        if star_row:
            a = fig.add_subplot(gs[nrow, gc_col])
            v = float(np.nanpercentile(np.abs(gc_star_map), 99.5)) or 1.0
            im = a.imshow(gc_star_map.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(
                f"g_c(u*)  [fully-minimised u]\n"
                f"||.||={cg_star['norm_weighted_l2']:.1e}",
                fontsize=8,
            )

            a = fig.add_subplot(gs[nrow, dc_col])
            dc_star_norm = float(np.linalg.norm(dc_star.detach().cpu().numpy()))
            v = float(np.nanpercentile(np.abs(dc_star_map), 99.5)) or 1.0
            im = a.imshow(dc_star_map.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(
                f"dc(u*) [m/s]  [from g_c(u*)]\n||.||={dc_star_norm:.1e}", fontsize=8
            )

        if exact:
            suptitle = (
                f"Outer {i}: exact wavefield step u* = u0 - H^-1 g(u0) "
                f"(c frozen at c_{i}); single row = term-split maps at the reached "
                f"u* (u_solve=exact, no iterative L-BFGS)"
            )
        else:
            suptitle = (
                f"Outer {i}: term-induced wavefield updates (du_term = -H^-1 "
                f"g_term) & c updates (c frozen at c_{i}); bottom: g_c at exact u* "
                f"and the dc it drives (u_solve={self.u_solve})"
            )
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(self.out_dir / f"outer_{i:02d}.png", dpi=120)
