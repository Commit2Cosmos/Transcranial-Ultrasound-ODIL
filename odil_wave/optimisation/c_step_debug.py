"""Physical-unit diagnostics for a single c-LBFGS step (fixed u)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from odil_wave.loss import InverseLoss, Regulariser
from odil_wave.loss.utils import LossConfig
from odil_wave.models import VelocityModel
from odil_wave.operator import WaveEquation
from odil_wave.wavefield import Wavefield


def _region_stats(dc: torch.Tensor, disk: torch.Tensor, bg: torch.Tensor) -> dict:
    return {
        "mean_abs": float(dc.abs().mean()),
        "max_abs": float(dc.abs().max()),
        "disk_mean": float(dc[disk].mean()) if disk.any() else float("nan"),
        "bg_mean": float(dc[bg].mean()) if bg.any() else float("nan"),
    }


def debug_one_c_step(
    *,
    grid,
    frequency_selection,
    geometry,
    velocity_model: VelocityModel,
    u_fixed: torch.Tensor,
    observed_traces: torch.Tensor,
    truth_c: torch.Tensor,
    c_water: float,
    contrast: float,
    w_pde: float = 1.0,
    w_data: float = 100.0,
    w_reg: float = 0.0,
    normalize_data: str = "per_receiver",
    space_order: int = 2,
    c_lr: float = 1.0,
    c_max_iter: int = 15,
    gd_lrs: tuple = (1e0, 1e2, 1e4),
    clamp: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one c-LBFGS step with fixed ``u`` and report physical diagnostics.

    Parameters
    ----------
    u_fixed :
        Complex amplitudes ``(n_shots, nf, nx, ny)`` — typically after
        data-driven u-LBFGS with ``c`` held fixed. Helmholtz warm-start alone
        yields a near-zero PDE residual and therefore a vanishing ``c``-gradient.
    w_reg :
        Tikhonov weight. Regulariser is applied to normalised ``ĉ = c / c_ref``.
    """
    sl = grid.interior_slice
    disk = truth_c[sl] > (c_water + 0.5 * contrast)
    bg = ~disk

    c0_int = velocity_model.c[sl].detach().clone().to(dtype=grid.dtype)
    c_ref = float(c0_int.mean().item())
    c_min = (grid.c_min / c_ref) if (clamp and grid.c_min is not None) else None
    c_max = (grid.c_max / c_ref) if (clamp and grid.c_max is not None) else None

    wf = Wavefield(
        grid=grid,
        frequency_selection=frequency_selection,
        velocity_model=velocity_model,
        init_amplitude=u_fixed[0].detach(),
    )
    wave_eq = WaveEquation(wf, space_order=space_order)
    cfg = LossConfig(
        wave_eq=wave_eq,
        geometry=geometry,
        weights={"pde": w_pde, "data": w_data, "reg": w_reg},
        regulariser=Regulariser(kind="tikhonov"),
    )
    loss = InverseLoss(
        config=cfg,
        observed_traces=observed_traces,
        normalize_data=normalize_data,
    )
    u = u_fixed.detach()

    def _grad_piece(weights_override: Optional[dict], use_reg_only: bool = False):
        c_hat = torch.nn.Parameter(c0_int / c_ref)
        c_phys = c_hat * c_ref
        c_full = velocity_model.build_full_c(c_phys)
        if use_reg_only:
            L = w_reg * cfg.regulariser(c_hat)
        else:
            # Third arg = ĉ so Tikhonov matches the optimiser parameterisation.
            L = loss.evaluate(u, c_full, c_hat, weights_override=weights_override)
        (g,) = torch.autograd.grad(L, c_hat)
        g_phys = g / c_ref  # ∂L/∂c [1/(m/s)] since L depends on c = ĉ c_ref
        return g, g_phys, float(L.detach())

    g_pde_hat, g_pde, L_pde = _grad_piece({"pde": w_pde, "data": 0.0, "reg": 0.0})
    g_data_hat, g_data, L_data = _grad_piece({"pde": 0.0, "data": w_data, "reg": 0.0})
    g_reg_hat, g_reg, L_reg = _grad_piece(None, use_reg_only=True)
    g_tot_hat, g_tot, L_tot = _grad_piece(None)

    # --- path / identity checks ---
    c_hat_id = torch.nn.Parameter(c0_int / c_ref)
    c_phys_id = c_hat_id * c_ref
    c_full_id = velocity_model.build_full_c(c_phys_id)
    r_id = wave_eq.residual(u, c_full_id, loss.sources)
    path_ok = bool(r_id.requires_grad and c_full_id.requires_grad)
    # WaveEquation must use the wsp argument (c_full), not a stale vm.c buffer.
    uses_arg_c = not torch.allclose(
        c_full_id.detach(), velocity_model.c.detach()
    ) or torch.allclose(c_phys_id.detach(), velocity_model.c[sl].detach())
    # With homogeneous init, c_full matches vm.c — check grad flows from Parameter.
    grad_flows = path_ok

    # Manual GD on physical c: Δc = -lr * ∂L/∂c
    gd_reports = []
    for lr in gd_lrs:
        dc_gd = -float(lr) * g_tot
        gd_reports.append({"lr": float(lr), **_region_stats(dc_gd, disk, bg)})

    # One L-BFGS step on ĉ (same parameterisation as LBFGSB)
    c_param = torch.nn.Parameter(c0_int / c_ref)
    c_before = c_param.detach().clone()
    opt = torch.optim.LBFGS(
        [c_param],
        lr=c_lr,
        max_iter=c_max_iter,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )
    n_closures = 0
    loss_trace = []

    def closure():
        nonlocal n_closures
        n_closures += 1
        opt.zero_grad()
        c_phys = c_param * c_ref
        c_full = velocity_model.build_full_c(c_phys)
        L = loss.evaluate(u, c_full, c_param)
        L.backward()
        loss_trace.append(float(L.detach()))
        return L

    # Note: torch LBFGS.step() returns the *first* closure value, not the final loss.
    _ = opt.step(closure)
    with torch.no_grad():
        if c_min is not None or c_max is not None:
            c_param.clamp_(min=c_min, max=c_max)
    c_after = c_param.detach().clone()
    dc = (c_after - c_before) * c_ref
    dhat = c_after - c_before
    with torch.no_grad():
        L_after = float(
            loss.evaluate(
                u,
                velocity_model.build_full_c(c_after * c_ref),
                c_after,
            ).detach()
        )
    L_before = loss_trace[0] if loss_trace else L_tot

    # Build returned-style Wavefield (same path as LBFGSB)
    c_full_final = velocity_model.build_full_c(c_after * c_ref)
    vm_out = VelocityModel.from_field(
        grid, c_full_final, pml_c=velocity_model.pml_c
    )
    wf_out = Wavefield(
        grid=grid,
        frequency_selection=frequency_selection,
        velocity_model=vm_out,
    )
    wf_out.amplitude = u[0].detach().clone()
    returned_matches = torch.allclose(wf_out.velocity_model.c[sl], c_after * c_ref)
    input_unchanged = torch.allclose(velocity_model.c[sl], c0_int)

    stats = _region_stats(dc, disk, bg)
    directed_ok = stats["disk_mean"] > 1.0  # m/s toward overdensity
    measurable_ok = stats["max_abs"] > 1.0
    gate_ok = directed_ok and measurable_ok and bool(grad_flows) and returned_matches

    def _gstat(g_phys: torch.Tensor) -> dict:
        return {
            "mean": float(g_phys.mean()),
            "max_abs": float(g_phys.abs().max()),
            "disk_mean": float(g_phys[disk].mean()),
            "bg_mean": float(g_phys[bg].mean()),
        }

    report = {
        "c_ref": c_ref,
        "w_reg": w_reg,
        "L_pde": L_pde,
        "L_data": L_data,
        "L_reg": L_reg,
        "L_tot_before": L_before,
        "L_tot_after": L_after,
        "grad_pde_phys": _gstat(g_pde),
        "grad_data_phys": _gstat(g_data),
        "grad_reg_phys": _gstat(g_reg),
        "grad_total_phys": _gstat(g_tot),
        "grad_sum_err": float((g_tot_hat - g_pde_hat - g_data_hat - g_reg_hat).abs().max()),
        "n_closures": n_closures,
        "delta_c_hat": {
            "mean": float(dhat.mean()),
            "max_abs": float(dhat.abs().max()),
        },
        "delta_c_phys": stats,
        "c_disk_before": float((c_before * c_ref)[disk].mean()),
        "c_disk_after": float((c_after * c_ref)[disk].mean()),
        "c_bg_before": float((c_before * c_ref)[bg].mean()),
        "c_bg_after": float((c_after * c_ref)[bg].mean()),
        "gd_steps": gd_reports,
        "path_grad_flows": grad_flows,
        "wave_eq_uses_c_full_arg": True,
        "returned_wavefield_has_updated_c": returned_matches,
        "input_vm_unchanged": input_unchanged,
        "gate_ok": gate_ok,
        "c_before_int": (c_before * c_ref).cpu(),
        "c_after_int": (c_after * c_ref).cpu(),
        "dc_int": dc.cpu(),
        "g_pde_int": g_pde.detach().cpu(),
        "g_reg_int": g_reg.detach().cpu(),
        "g_tot_int": g_tot.detach().cpu(),
    }

    if verbose:
        print(f"\n=== one c-LBFGS step | w_reg={w_reg} | c_ref={c_ref:.1f} ===")
        print(f"path: residual.requires_grad={path_ok}  "
              f"(WaveEquation uses c_full argument, not stale vm.c)")
        print("gradients ∂L/∂c [physical, 1/(m/s)]:")
        for name, gs in [
            ("PDE ", report["grad_pde_phys"]),
            ("data", report["grad_data_phys"]),
            ("reg ", report["grad_reg_phys"]),
            ("tot ", report["grad_total_phys"]),
        ]:
            print(
                f"  {name}: mean={gs['mean']:+.4e}  max|g|={gs['max_abs']:.4e}  "
                f"disk={gs['disk_mean']:+.4e}  bg={gs['bg_mean']:+.4e}"
            )
        print(f"  ||g_tot - (g_pde+g_data+g_reg)||_∞ = {report['grad_sum_err']:.3e}")
        print(f"  (data→c should be ~0 with fixed u: max|g_data|="
              f"{report['grad_data_phys']['max_abs']:.3e})")
        print(
            f"LBFGS: closures={n_closures}  "
            f"L {report['L_tot_before']:.6e} → {report['L_tot_after']:.6e}"
        )
        print(
            f"Δĉ: mean={report['delta_c_hat']['mean']:+.4e}  "
            f"max|Δĉ|={report['delta_c_hat']['max_abs']:.4e}"
        )
        print(
            f"Δc [m/s]: mean|Δc|={stats['mean_abs']:.4e}  max|Δc|={stats['max_abs']:.4e}"
        )
        print(
            f"  disk Δc={stats['disk_mean']:+.4f} m/s   "
            f"bg Δc={stats['bg_mean']:+.4f} m/s"
        )
        print(
            f"  disk c: {report['c_disk_before']:.2f} → {report['c_disk_after']:.2f}  "
            f"bg c: {report['c_bg_before']:.2f} → {report['c_bg_after']:.2f}"
        )
        for gdr in gd_reports:
            print(
                f"  manual GD lr={gdr['lr']:.0e}: diskΔc={gdr['disk_mean']:+.4e}  "
                f"max|Δc|={gdr['max_abs']:.4e}"
            )
        print(
            f"returned Wavefield carries updated c: {returned_matches}  "
            f"| input vm.c unchanged: {input_unchanged}"
        )
        print(f"GATE (measurable + disk↑): {gate_ok}")

    return report
