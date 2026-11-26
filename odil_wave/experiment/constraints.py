"""Constraint checks: the rules a run config must satisfy for its grid, time
step, source and transducers to resolve the problem.

:func:`check_constraints` is the entry point; :func:`format_constraint_report`
renders its result as the table ``--dry-run`` and the run log print.
:func:`~odil_wave.experiment.config.validate_config` turns failed checks into a
:class:`~odil_wave.experiment.config.ConfigError` (``error``) or config
warnings (``warn``). The optimiser health check during a run lives in
:mod:`.health`. Tolerances are module constants, not config keys; the basis
for each is ``research/discretisation-constraints.md`` and issue #124.
"""

from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal

import numpy as np
from scipy.optimize import brentq

from odil_wave.grid import time_samples
from odil_wave.operator.spatial import FD2_COEFFS
from odil_wave.models.shepp_logan import SOS_WATER

if TYPE_CHECKING:
    from .config import RunConfig

# Least Gaussian source width, in grid spacings.
MIN_SOURCE_CELLS = 1.0
# Largest k_max * sigma_s before the source low-pass filters the top frequency.
MAX_SOURCE_K_SIGMA = 1.0
# Least ring clearance to the PML, in source widths.
MIN_CLEARANCE_SIGMAS = 3.0
# Fewest PML cells per side: a stretched PML's reflection is set by its width
# in cells (#37, ``pml_check``).
MIN_PML_CELLS = 12
# Least source amplitude at a band frequency, relative to the spectrum's peak.
MIN_SOURCE_AMPLITUDE = 0.5
# Accumulated phase error allowed over the ring diameter, rad (a judgement call).
PHASE_TOLERANCE = math.pi / 4

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


@dataclass(frozen=True)
class Check:
    """One constraint check: passes iff ``value <op> threshold``.

    ``severity`` is ``"error"`` (the run is refused) or ``"warn"``;
    ``message`` explains a failure.
    """

    name: str
    quantity: str
    value: float
    op: Literal["<", "<=", ">", ">="]
    threshold: float
    severity: Literal["error", "warn"]
    message: str

    @property
    def ok(self) -> bool:
        """Whether the config satisfies the check."""
        return bool(_OPS[self.op](self.value, self.threshold))

    def to_dict(self) -> Dict[str, Any]:
        """The check as a plain dict, with ``ok``, for ``metadata.json``."""
        return {**asdict(self), "ok": self.ok}


def check_constraints(cfg: "RunConfig") -> List[Check]:
    """Every constraint check that applies to the validated config ``cfg``;
    frequency-dependent checks are made at the top band frequency."""
    g = _Geometry(cfg)
    source_width = (
        _source_width(cfg, g) if cfg.acquisition.source_spatial == "gaussian" else []
    )
    return [
        *source_width,
        _dispersion(cfg, g),
        _ring_sampling(cfg, g),
        _temporal_nyquist(g),
        _pml_width(cfg),
        _source_spectrum(cfg, g),
    ]


class _Geometry:
    """The quantities of ``cfg`` the checks share."""

    def __init__(self, cfg: "RunConfig") -> None:
        (x0, x1), (y0, y1) = cfg.grid.interior_extent
        nx, ny = cfg.grid.interior_shape
        self.dx = (x1 - x0) / (nx - 1)
        self.dy = (y1 - y0) / (ny - 1)
        self.f_max = max(
            float(f) for b in cfg.continuation.bands for f in b.frequencies_hz
        )
        self.a = cfg.acquisition.a_frac * (x1 - x0) / 2
        self.b = cfg.acquisition.b_frac * (y1 - y0) / 2
        self.R = max(self.a, self.b)
        rc = cfg.acquisition.ring_center
        cx, cy = ((x0 + x1) / 2, (y0 + y1) / 2) if rc is None else map(float, rc)
        self.clearance = min(
            cx - self.a - x0, x1 - cx - self.a, cy - self.b - y0, y1 - cy - self.b
        )
        self.k_water = 2.0 * math.pi * self.f_max / SOS_WATER
        t = cfg.time
        nt = time_samples(
            t.t_max, t.init_nt, t.cfl_safety, cfg.grid.c_max, self.dx, self.dy
        )
        self.dt = t.t_max / (nt - 1)
        # Wavelengths in water along the longest source-receiver path, 2R.
        self.n_wavelengths = 2.0 * self.R * self.f_max / SOS_WATER


def _source_width(cfg: "RunConfig", g: _Geometry) -> List[Check]:
    sigma = float(cfg.acquisition.sigma_s)
    h = max(g.dx, g.dy)
    k_max = 2.0 * math.pi * g.f_max / float(cfg.grid.c_min)
    facts = (
        f"(sigma_s/h = {sigma / h:.2f}, k_max*sigma_s = {k_max * sigma:.2f} at "
        f"{g.f_max / 1e3:g} kHz, ring clearance to the PML = "
        f"{g.clearance * 1e3:.1f} mm = {g.clearance / sigma:.1f} sigma_s)"
    )
    err = 4.0 * math.exp(-((math.pi * sigma / h) ** 2))
    return [
        Check(
            name="source_resolved",
            quantity="sigma_s / h",
            value=sigma / h,
            op=">=",
            threshold=MIN_SOURCE_CELLS,
            severity="warn",
            message=(
                f"acquisition.sigma_s={sigma:g} m is under-resolved "
                f"(< {MIN_SOURCE_CELLS} h): the "
                f"source's normalisation is off by ~{err:.0e} {facts}"
            ),
        ),
        Check(
            name="source_low_pass",
            quantity="k_max * sigma_s",
            value=k_max * sigma,
            op="<=",
            threshold=MAX_SOURCE_K_SIGMA,
            severity="warn",
            message=(
                f"acquisition.sigma_s={sigma:g} m low-pass filters the top frequency "
                f"(k_max*sigma_s > {MAX_SOURCE_K_SIGMA:g}, amplitude factor "
                f"{math.exp(-((k_max * sigma) ** 2) / 4):.2f}) {facts}"
            ),
        ),
        Check(
            name="ring_clearance",
            quantity="ring clearance to PML / sigma_s",
            value=g.clearance / sigma,
            op=">=",
            threshold=MIN_CLEARANCE_SIGMAS,
            severity="warn",
            message=(
                f"acquisition.sigma_s={sigma:g} m: the transducer ring is within "
                f"{MIN_CLEARANCE_SIGMAS:g} sigma_s of the PML {facts}"
            ),
        ),
    ]


def _dispersion(cfg: "RunConfig", g: _Geometry) -> Check:
    order = int(cfg.grid.space_order)
    err = _worst_wavenumber_error(order, g.k_water, g.dx, g.dy)
    phase = 2.0 * math.pi * g.n_wavelengths * err
    return Check(
        name="dispersion",
        quantity=f"phase error over 2R (rad), order {order}",
        value=phase,
        op="<=",
        threshold=PHASE_TOLERANCE,
        severity="warn",
        message=(
            f"numerical dispersion: the order-{order} stencil at dx = "
            f"{g.dx * 1e3:.3g} mm accumulates {phase:.2f} rad of phase error over "
            f"the ring diameter at {g.f_max / 1e3:g} kHz in water "
            f"(> {PHASE_TOLERANCE:.2f} rad; velocity bias {err * SOS_WATER:.1f} m/s); refine grid.interior_shape "
            "or raise grid.space_order"
        ),
    )


def _ring_sampling(cfg: "RunConfig", g: _Geometry) -> Check:
    n = int(cfg.acquisition.n_transducers)
    need = 2.0 * g.k_water * g.R
    return Check(
        name="ring_sampling",
        quantity="n_transducers vs 2kR",
        value=n,
        op=">",
        threshold=need,
        severity="warn",
        message=(
            f"ring sampling: {n} transducers under-sample the ring (need > 2kR = "
            f"{need:.1f} at {g.f_max / 1e3:g} kHz in water, Simonetti 2007); raise "
            "acquisition.n_transducers"
        ),
    )


def _temporal_nyquist(g: _Geometry) -> Check:
    nyquist = 1.0 / (2.0 * g.dt)
    return Check(
        name="temporal_nyquist",
        quantity="f_max (Hz) vs 1/(2 dt)",
        value=g.f_max,
        op="<",
        threshold=nyquist,
        severity="error",
        message=(
            f"temporal Nyquist: the top band frequency {g.f_max / 1e3:g} kHz is not "
            f"below 1/(2 dt) = {nyquist / 1e3:.4g} kHz (dt = {g.dt:.3g} s), so the "
            "DFT on the time axis aliases it; raise time.init_nt"
        ),
    )


def _pml_width(cfg: "RunConfig") -> Check:
    width = int(cfg.pml.width)
    return Check(
        name="pml_width",
        quantity="pml.width (cells)",
        value=width,
        op=">=",
        threshold=MIN_PML_CELLS,
        severity="warn",
        message=(
            f"PML width: {width} cells per side reflect more than "
            f"{MIN_PML_CELLS} would (``python -m odil_wave.experiment.pml_check`` "
            "measures it); raise pml.width"
        ),
    )


def _source_spectrum(cfg: "RunConfig", g: _Geometry) -> Check:
    src = cfg.source
    amp = _relative_amplitude(cfg, g.f_max)
    return Check(
        name="source_spectrum",
        quantity="source amplitude at f_max / peak",
        value=amp,
        op=">=",
        threshold=MIN_SOURCE_AMPLITUDE,
        severity="warn",
        message=(
            f"source spectrum: the {src.kind} source at f0 = {src.f0 / 1e3:g} kHz "
            f"has {amp:.2f} of its peak amplitude at the top band frequency "
            f"{g.f_max / 1e3:g} kHz (< {MIN_SOURCE_AMPLITUDE}); move source.f0 toward "
            "the bands"
        ),
    )


def _relative_amplitude(cfg: "RunConfig", f: float) -> float:
    """The source spectrum's amplitude at ``f`` (Hz) over its peak's."""
    src = cfg.source
    r = f / float(src.f0)
    if src.kind == "ricker":
        return r**2 * math.exp(1.0 - r**2)
    if src.envelope == "rectangular":
        return abs(float(np.sinc((r - 1.0) * src.n_cycles)))
    sigma_t = src.n_cycles / (6.0 * src.f0)
    return math.exp(-((2.0 * math.pi * (f - src.f0) * sigma_t) ** 2) / 2.0)


def _stencil_symbol(order: int, theta: float) -> float:
    """``-h**2`` times the symbol of the 1D ``d²/dx²`` stencil at ``theta = k h``
    (``theta**2`` for the exact derivative)."""
    c0, cs = FD2_COEFFS[order]
    return -(c0 + 2.0 * sum(c * math.cos(m * theta) for m, c in enumerate(cs, 1)))


def _worst_wavenumber_error(order: int, k: float, dx: float, dy: float) -> float:
    """Largest ``|kappa / k - 1|`` over propagation directions, where ``kappa``
    is the wavenumber of a plane wave solving the discrete Helmholtz equation
    at exact wavenumber ``k``; ``inf`` below the stencil's cutoff."""
    worst = 0.0
    for phi in np.linspace(0.0, math.pi / 2, 37):
        cx, sy = math.cos(phi), math.sin(phi)

        def residual(kappa: float) -> float:
            return (
                _stencil_symbol(order, kappa * dx * cx) / dx**2
                + _stencil_symbol(order, kappa * dy * sy) / dy**2
                - k**2
            )

        kappa_hi = math.pi / max(dx * cx, dy * sy)
        if residual(kappa_hi) < 0:
            return math.inf
        worst = max(worst, abs(brentq(residual, 0.5 * k, kappa_hi) / k - 1.0))
    return worst


def format_constraint_report(checks: List[Check]) -> str:
    """``checks`` as a table, one row per check."""
    rows = [
        (
            c.name,
            c.quantity,
            f"{c.value:.4g} {c.op} {c.threshold:.4g}",
            c.severity,
            "ok" if c.ok else "FAIL",
        )
        for c in checks
    ]
    head = ("check", "quantity", "value", "severity", "")
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    lines = ["constraint checks:"]
    for r in [head, *rows]:
        lines.append("  " + "  ".join(s.ljust(w) for s, w in zip(r, widths)).rstrip())
    return "\n".join(lines)
