"""Typed, serialisable experiment configuration for frequency-domain FWI runs.

Dependency-light (standard library + PyYAML only, no ``torch`` / ``odil_wave``
imports) so a config can be resolved and validated without importing the
numerical stack. Every field has an explicit default; a user config may omit
any field and the fully-resolved config written to ``config_resolved.yaml``
records every effective value. The dataclasses are frozen (immutable once
resolved).
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml


class ConfigError(ValueError):
    """Raised for malformed / unknown configuration entries."""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunMetaCfg:
    """Run identity and output location.

    Params:
    * ``run_id``: run directory name; blank -> auto-minted from the resolved
      config at :func:`resolve_config` time (see :func:`_make_run_id`). Set to
      pin a name.
    * ``output_root``: output directory; run dir = ``<output_root>/<run_id>``.
    * ``seed``: int RNG seed for reproducibility.
    * ``tags``: list[str] free-form labels recorded in the config.
    * ``notes``: str free-form description.
    """

    run_id: str = ""
    output_root: str = "outputs"
    seed: int = 0
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class RuntimeCfg:
    """Device / dtype / threading.

    Params:
    * ``device``: ``"cpu"`` | ``"cuda"`` | ``"mps"``.
    * ``dtype``: ``"float32"`` | ``"float64"`` (aliases ``f32``/``float``,
      ``f64``/``double``); float64 is more reproducible for the inverse solve.
    * ``torch_num_threads``: ``None`` -> torch default, else int >= 1.
    * ``deterministic``: force deterministic torch kernels.
    """

    device: str = "cpu"
    dtype: str = "float32"
    torch_num_threads: Optional[int] = None
    deterministic: bool = False


@dataclass(frozen=True)
class GridCfg:
    """Spatial + time discretisation (odil_wave.Grid).

    Params:
    * ``interior_shape``: ``[ny, nx]`` interior cells (PML added outside).
    * ``c_min`` / ``c_max``: m/s velocity bounds (``c_min`` < ``c_max``).
    * ``interior_extent``: metres, ``[[y0, y1], [x0, x1]]`` physical domain.
    * ``t_max``: s; forward time window (leapfrog observation).
    * ``init_nt``: ``None`` -> CFL-derived, else int >= 1 time steps.
    * ``pml_width``: cells; absorbing-border thickness (inverse grid).
    * ``pml_power``: PML grading polynomial order (typ. 2-4).
    * ``pml_R0``: target PML reflection coefficient (small, e.g. 1e-4 .. 1e-8).
    * ``cfl_safety``: (0, 1]; CFL fraction used to pick dt.
    * ``L0`` / ``c0``: ``None`` -> auto, else > 0 length / velocity
      non-dimensionalisation scales.
    """

    interior_shape: List[int] = field(default_factory=lambda: [125, 125])
    c_min: float = 1300.0
    c_max: float = 3100.0
    interior_extent: List[List[float]] = field(
        default_factory=lambda: [[0.0, 0.25], [0.0, 0.25]]
    )
    t_max: float = 500e-6
    init_nt: Optional[int] = 1300
    pml_width: int = 40
    pml_power: int = 3
    pml_R0: float = 1e-6
    cfl_safety: float = 0.9
    L0: Optional[float] = None
    c0: Optional[float] = None


@dataclass(frozen=True)
class SourceCfg:
    """Source wavelet (odil_wave.SourceSignal).

    Params:
    * ``kind``: ``"tone_burst"`` | ``"ricker"``.
    * ``f0``: Hz centre frequency.
    * ``amplitude``: source amplitude scale.
    * ``n_cycles``: tone_burst only; number of cycles in the burst.
    * ``envelope``: tone_burst only; ``"gaussian"`` | ``"rectangular"``.
    * ``offset``: int sample offset delaying the wavelet.
    * ``t0``: ``None`` -> auto (1/f0 for ricker), else float wavelet time
      centre [s].
    * ``dimensionless``: emit in non-dimensional units.
    """

    kind: str = "tone_burst"
    f0: float = 80e3
    amplitude: float = 1.0
    n_cycles: float = 3.0
    envelope: str = "gaussian"
    offset: int = 0
    t0: Optional[float] = None
    dimensionless: bool = True


@dataclass(frozen=True)
class AcquisitionCfg:
    """Ring source/receiver layout (odil_wave.AcquisitionGeometry).

    Params:
    * ``n_receivers`` / ``n_sources``: int >= 1 points on the ring.
    * ``a_frac`` / ``b_frac``: (0, 1]; ring semi-axes as a fraction of the
      half-extent in x / y.
    * ``ring_center``: ``"grid_center"`` (centre of the interior extent) |
      ``[x, y]`` in metres.
    * ``sigma_s``: ``None`` -> auto, else > 0 Gaussian source width in cells
      (used when ``source_spatial == "gaussian"``).
    * ``source_spatial``: ``"gaussian"`` | ``"point"``.
    """

    n_receivers: int = 64
    n_sources: int = 8
    a_frac: float = 0.9
    b_frac: float = 0.9
    ring_center: Union[str, List[float]] = "grid_center"
    sigma_s: Optional[float] = None
    source_spatial: str = "gaussian"


@dataclass(frozen=True)
class PhysicsCfg:
    """Discrete operator settings shared by forward + inverse.

    Params:
    * ``space_order``: spatial FD stencil order ``2`` | ``4`` | ``6`` | ``8``
      (higher = more accurate, costlier).
    * ``time_order``: temporal FD order (leapfrog: 2).
    * ``pml_weight``: >= 0; PML-residual weight in the PDE loss.
    """

    space_order: int = 2
    time_order: int = 2
    pml_weight: float = 1.0


@dataclass(frozen=True)
class ModelCfg:
    """A VelocityModel spec (truth or initial model).

    Params:
    * ``profile``: ``"homogeneous"`` | ``"overdensity"`` | ``"shepp_logan"`` |
      ``"shepp_logan_skull"`` | ``"skull"``.
    * ``scale``: phantom contrast scale.
    * ``base``: m/s background velocity.
    * ``contrast``: profile contrast fraction.
    * ``pml_c``: ``None`` -> derived; constant PML fill velocity used when
      ``pml_fill == "constant"``.
    * ``pml_fill``: ``"edge"`` (replicate the interior boundary outward, so c
      is continuous across the interior<->PML interface) | ``"constant"`` (pad
      with ``pml_c``).
    * ``skull_alpha``: [0, 1]; shepp_logan_skull only (skull contrast fraction).
    * ``skull_sigma``: >= 0 grid cells; shepp_logan_skull only, Gaussian soft
      edge (try 1-3).
    * ``extra``: extra profile kwargs (``threshold``, ``c_water``, ``c_skull``,
      ``center``, ``radius``, ``skull_smooth`` alias, ...).
    """

    profile: str = "shepp_logan"
    scale: float = 0.85
    base: float = 1500.0
    contrast: float = 0.4
    pml_c: Optional[float] = None
    pml_fill: str = "edge"
    skull_alpha: float = 1.0
    skull_sigma: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationCfg:
    """How the observed ("true") data is synthesised and normalised.

    Params:
    * ``method``: ``"leapfrog_fft"`` (broadband leapfrog time solve then FFT
      onto each band's bins; avoids the inverse crime) | ``"helmholtz"``
      (frequency-domain Helmholtz solve on the truth model).
    * ``normalize_data``: ``None`` | ``"none"`` | ``"per_receiver"``.
    * ``verbose``: bool.
    * ``pml_width``: ``None`` -> ``grid.pml_width``, else int >= 0
      forward-only PML thickness. When set and different from the inverse PML,
      data are generated on a separate forward grid (the PML_FWD / PML_INV
      split) and reduced to receiver traces.
    """

    method: str = "leapfrog_fft"
    normalize_data: str = "per_receiver"
    verbose: bool = False
    pml_width: Optional[int] = None


@dataclass(frozen=True)
class BandCfg:
    """One continuation stage (single-frequency if ``len(frequencies_hz) == 1``).

    Params:
    * ``frequencies_hz``: list[float > 0]; multiple frequencies in one stage are
      inverted jointly (multi-frequency stage).
    * ``n_iter``: ``None`` -> ``optimiser.n_iter``, else int >= 1 per-stage
      iteration budget.
    * ``c_grad_smooth_sigma``: ``None`` -> ``lbfgsb.c_grad_smooth_sigma``, else
      float >= 0 per-stage override (LBFGSB only; ignored by other optimisers).
    """

    frequencies_hz: List[float] = field(default_factory=lambda: [40e3])
    n_iter: Optional[int] = None
    c_grad_smooth_sigma: Optional[float] = None


@dataclass(frozen=True)
class ContinuationCfg:
    """Frequency-band continuation schedule + warm-start policy.

    Params:
    * ``warm_start``: ``"helmholtz"`` (re-solve Helmholtz on the current model
      each band to warm-start u) | ``"none"`` (start u from the wavefield seed).
    * ``bands``: ordered list[:class:`BandCfg`], solved in sequence (typically
      low-to-high frequency).
    """

    warm_start: str = "helmholtz"
    bands: List[BandCfg] = field(default_factory=lambda: [BandCfg()])


@dataclass(frozen=True)
class RegulariserCfg:
    """Regulariser selection.

    Params:
    * ``name``: ``None`` (no regulariser) | ``"tikhonov"`` | ``"tv_iso"`` |
      ``"tv_aniso"`` (aliases ``l2``/``smooth``, ``tv``/``iso``, ``aniso``).
    * ``params``: only ``"eps"`` accepted, e.g. ``{"eps": 1e-6}`` (used by the
      TV regularisers).
    """

    name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossCfg:
    """Objective weights + regularisation.

    Params:
    * ``weights``: dict with required keys ``pde`` / ``data`` (and optional
      ``reg``), each >= 0; PDE-residual / data-misfit / regulariser weights.
    * ``regulariser``: :class:`RegulariserCfg`.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {"pde": 1.0, "data": 100.0, "reg": 0.0}
    )
    regulariser: RegulariserCfg = field(default_factory=RegulariserCfg)


@dataclass(frozen=True)
class SchedulerCfg:
    """Multiplicative outer-loop scheduler for a single LBFGSB knob.

    The scheduled value starts at its base (the knob's configured value) and,
    when active, is rescaled every outer whose index is a positive multiple of
    ``every_n``. Inactive (a no-op leaving the value at its base) whenever
    ``factor == 0`` or ``every_n == 0``. Only ``pde_weight`` and
    ``c_grad_smooth_sigma`` are schedulable.

    Params:
    * ``factor``: >= 0 multiplier applied to the knob.
    * ``every_n``: >= 0 cadence in outers.
    * ``direction``: ``"increase"`` (x ``factor``) | ``"decrease"`` (/ ``factor``).
    """

    factor: float = 0.0
    every_n: int = 0
    direction: str = "increase"


@dataclass(frozen=True)
class LBFGSBCfg:
    """Params specific to the block-coordinate dual L-BFGS (LBFGSB).

    Params:
    * ``z_steps``: int >= 1; z-block substeps per outer (``u_precond == "z"``
      only).
    * ``u_precond``: ``None`` | ``"z"`` (reduced-space source-extension
      preconditioner for the wavefield block).
    * ``u_solve``: ``"optim"`` (one fresh L-BFGS u-block step) | ``"exact"``
      (replace the u-block by the exact minimiser via a direct sparse Hessian
      factorisation; incompatible with ``u_precond == "z"``).
    * ``z_optim``: ``"gd"`` (Armijo steepest descent) | ``"lbfgs"``
      (``u_precond == "z"`` only).
    * ``z_lr``: > 0; z-block initial step when ``z_optim == "gd"``.
    * ``c_lr``: model-block L-BFGS learning rate.
    * ``c_max_iter``: model-block L-BFGS max iterations.
    * ``c_history_size``: model-block L-BFGS history length.
    * ``reset_c_history``: reset the model-block L-BFGS history each outer.
    * ``c_param``: model variable ``"velocity"`` (optimise c) |
      ``"squared_slowness"`` (optimise 1/c^2; incompatible with
      ``u_precond == "z"``).
    * ``c_grad_smooth_sigma``: >= 0 Gaussian smoothing of the c-gradient in
      grid cells (0 = off; try 1-3). Schedulable.
    * ``pde_weight_schedule`` / ``c_grad_smooth_sigma_schedule``:
      :class:`SchedulerCfg` for the only two schedulable knobs.
    """

    z_steps: int = 1
    u_precond: Optional[str] = None
    u_solve: str = "optim"
    z_optim: str = "gd"
    z_lr: float = 1.0
    c_lr: float = 5.0
    c_max_iter: int = 6
    c_history_size: int = 10
    reset_c_history: bool = True
    c_param: str = "velocity"
    c_grad_smooth_sigma: float = 0.0
    pde_weight_schedule: SchedulerCfg = field(default_factory=SchedulerCfg)
    c_grad_smooth_sigma_schedule: SchedulerCfg = field(default_factory=SchedulerCfg)


@dataclass(frozen=True)
class JointODILCfg:
    """Params specific to the pure joint full-space solver (``JointFreqODIL``).

    A single joint L-BFGS over the complex wavefield block and a
    squared-slowness model latent bounded by a sigmoid.

    Params:
    * ``data_weight``: > 0 fixed run-level data weight (``pde_weight = 1``;
      never adapted during optimisation).
    * ``reg_weight``: >= 0 regulariser weight.
    * ``c_min`` / ``c_max``: ``None`` -> ``grid.c_min`` / ``grid.c_max``; set the
      squared-slowness bounds (``c_min`` < ``c_max``).
    * ``inner_max_iter``: int >= 1 L-BFGS iterations per logged outer step; keep
      >= ~5 for a genuine continuous solve (inner=1 stalls the line search).
    * ``history_size``: int >= 1 L-BFGS history length.
    * ``line_search_fn``: ``"strong_wolfe"`` | ``"none"``.
    * ``lbfgs_lr``: L-BFGS learning rate.
    * ``tolerance_grad`` / ``tolerance_change``: stopping tolerances.
    * ``eps_u`` / ``eps_data`` / ``eps_pde``: fixed-scale floors.
    * ``u_scale_factor`` / ``pde_scale_factor`` / ``data_scale_factor``: > 0
      conditioning-sweep perturbations (identity by default; change only the
      optimisation geometry, not the physical residual).
    * ``z_scale``: > 0 model-latent scale.
    * ``logit_clip``: in (0, 0.5); sigmoid latent clip.
    * ``verbose``: bool.
    """

    data_weight: float = 1.0
    reg_weight: float = 0.0
    c_min: Optional[float] = None
    c_max: Optional[float] = None
    inner_max_iter: int = 20
    history_size: int = 20
    line_search_fn: str = "strong_wolfe"
    lbfgs_lr: float = 1.0
    tolerance_grad: float = 1e-12
    tolerance_change: float = 1e-14
    eps_u: float = 1e-30
    eps_data: float = 1e-30
    eps_pde: float = 1e-30
    u_scale_factor: float = 1.0
    pde_scale_factor: float = 1.0
    data_scale_factor: float = 1.0
    z_scale: float = 1.0
    logit_clip: float = 1e-6
    verbose: bool = False


@dataclass(frozen=True)
class OptimiserCfg:
    """Optimiser selection + shared block-coordinate settings.

    Params:
    * ``name``: ``"lbfgsb"`` | ``"joint"`` (aliases resolved via
      :func:`canonical_optimiser_name`).
    * ``log_every``: int >= 1 metric-logging cadence in outers.
    * ``n_iter``: int >= 1 outer iterations (per band unless ``band.n_iter``).
    * ``u_steps``: wavefield-block L-BFGS steps per outer.
    * ``c_steps``: model-block updates per outer.
    * ``max_iter``: L-BFGS max iterations per block step.
    * ``history_size``: L-BFGS history length.
    * ``tolerance_grad`` / ``tolerance_change``: stopping tolerances.
    * ``line_search_fn``: ``"strong_wolfe"`` | ``"none"``.
    * ``clamp``: enforce [c_min, c_max] via a logistic reparam.
    * ``lbfgsb`` / ``joint``: per-optimiser sub-blocks.
    """

    name: str = "lbfgsb"
    log_every: int = 5
    n_iter: int = 20
    u_steps: int = 20
    c_steps: int = 1
    max_iter: int = 20
    history_size: int = 10
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    line_search_fn: str = "strong_wolfe"
    clamp: bool = True
    lbfgsb: LBFGSBCfg = field(default_factory=LBFGSBCfg)
    joint: JointODILCfg = field(default_factory=JointODILCfg)


@dataclass(frozen=True)
class SSIMCfg:
    """Masked SSIM settings for the ``ssim_head_roi`` metric.

    Params:
    * ``mask``: ``"head_roi"`` (intracranial mask) | ``"interior"`` (whole
      non-PML interior) | ``"none"``.
    * ``data_range``: ``"auto"`` (peak-to-peak of truth in mask) | float.
    * ``win_size``: odd int >= 3; SSIM window size.
    * ``gaussian_weights``: Gaussian vs uniform SSIM window.
    * ``sigma``: Gaussian window std (used when ``gaussian_weights``).
    """

    mask: str = "head_roi"
    data_range: Union[str, float] = "auto"
    win_size: int = 7
    gaussian_weights: bool = False
    sigma: float = 1.5


@dataclass(frozen=True)
class MetricsCfg:
    """Per-iteration metric settings.

    Params:
    * ``ssim``: :class:`SSIMCfg`.
    * ``log_c_stats``: log per-iteration velocity statistics.
    """

    ssim: SSIMCfg = field(default_factory=SSIMCfg)
    log_c_stats: bool = True


@dataclass(frozen=True)
class DiagnosticsCfg:
    """Block-coordinate diagnostics (LBFGSB only; off by default).

    When ``enabled`` the run captures per-outer block-diagnostic quantities from
    inside ``LBFGSB.minimise`` (a no-op when disabled) and writes, into
    ``<run_dir>/diagnostics/``:

    * ``scalars.csv`` — per-outer cos(update,-grad) for the u/c blocks, block
      gradient L2 norms, relative model error and relative PDE/data residuals,
      plus the scheduled ``c_lr`` / ``c_grad_smooth_sigma`` in effect.
    * ``summary.png`` — the trajectory panels built from those scalars.
    * ``c_evolution.png`` — physical interior velocity after each outer.
    * ``outer_XX.png`` — per-outer wavefield term-induced update maps at several
      u-solve depths + the c-gradient and velocity update at the exact wavefield
      minimiser; only when ``per_outer_field_maps``.

    Params:
    * ``enabled`` / ``per_outer_field_maps``: bool toggles.
    * ``u_depths``: ``None`` -> auto ``1, 5, 10, ..., u_steps``, else list of
      int >= 1 u-solve depths (rows of ``outer_XX``).
    * ``verify_hessian``: add the sparse-H vs autograd-Hvp correctness gate.
    """

    enabled: bool = False
    per_outer_field_maps: bool = True
    u_depths: Optional[List[int]] = None
    verify_hessian: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Top-level fully-resolved experiment configuration."""

    run: RunMetaCfg = field(default_factory=RunMetaCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    source: SourceCfg = field(default_factory=SourceCfg)
    acquisition: AcquisitionCfg = field(default_factory=AcquisitionCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    truth: ModelCfg = field(default_factory=ModelCfg)
    init: ModelCfg = field(
        default_factory=lambda: ModelCfg(profile="shepp_logan_skull")
    )
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    continuation: ContinuationCfg = field(default_factory=ContinuationCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    optimiser: OptimiserCfg = field(default_factory=OptimiserCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)
    diagnostics: DiagnosticsCfg = field(default_factory=DiagnosticsCfg)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Return the config as a nested plain-dict tree."""
        return _asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Build a :class:`RunConfig` from a plain-dict tree ``data``."""
        return _build(cls, data or {})

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write the config to ``path`` as YAML; return the path."""
        return _dump_yaml(self.to_dict(), path)

    def to_json(self, path: Union[str, Path]) -> Path:
        """Write the config to ``path`` as JSON; return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return p

    def run_dir(self) -> Path:
        """Return the run output directory ``<output_root>/<run_id>``."""
        return Path(self.run.output_root) / self.run.run_id


# --------------------------------------------------------------------------- #
# Reflective (de)serialisation over the frozen dataclasses
# --------------------------------------------------------------------------- #
def _dataclass_of(tp: Any) -> Optional[type]:
    """Return the dataclass for ``tp`` or ``Optional[dataclass]``, else None."""
    if is_dataclass(tp):
        return tp
    if get_origin(tp) is Union:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1 and is_dataclass(non_none[0]):
            return non_none[0]
    return None


def _list_elt_dataclass(tp: Any) -> Optional[type]:
    """Return the element dataclass if ``tp`` is ``List[dataclass]``, else None."""
    if get_origin(tp) in (list, List):
        args = get_args(tp)
        if args and is_dataclass(args[0]):
            return args[0]
    return None


def _build(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively construct dataclass ``cls`` from the plain dict ``data``.

    Recurses into nested dataclass fields and lists of dataclasses; fields
    absent from ``data`` keep their dataclass default. Raises :class:`ConfigError`
    on a non-mapping input or unknown keys.
    """
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ConfigError(
            f"expected a mapping for {cls.__name__}, got {type(data).__name__}"
        )
    hints = get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"valid keys: {sorted(field_names)}"
        )
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # keep the dataclass default
        val = data[f.name]
        tp = hints[f.name]
        nested = _dataclass_of(tp)
        elt = _list_elt_dataclass(tp)
        if nested is not None and isinstance(val, dict):
            kwargs[f.name] = _build(nested, val)
        elif elt is not None and isinstance(val, list):
            kwargs[f.name] = [_build(elt, v) for v in val]
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _asdict(obj: Any) -> Any:
    """Recursively convert a dataclass tree (and nested lists/dicts) to plain
    Python containers."""
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #
class _ConfigYamlLoader(yaml.SafeLoader):
    """SafeLoader that also parses unsigned-exponent floats like ``80e3``."""


# PyYAML's implicit float resolver requires an explicit exponent sign.
_ConfigYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][0-9]+$"),
    list("-+0123456789."),
)


def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the YAML or JSON config file at ``path`` into a plain dict."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.load(text, Loader=_ConfigYamlLoader)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {p} must contain a mapping at the top level")
    return data


def _dump_yaml(obj: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Atomically write ``obj`` to ``path`` as block-style YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(obj, sort_keys=False, default_flow_style=False, width=100)
    _atomic_write(p, text)
    return p


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (write tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Merge / override / resolve
# --------------------------------------------------------------------------- #
def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_override_value(raw: str) -> Any:
    """Parse a CLI override value: JSON first (typed), else bare string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def apply_overrides(
    data: Dict[str, Any], overrides: Optional[List[str]]
) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` dotted overrides to a copy of the config dict ``data``."""
    out = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must be of the form dotted.key=value")
        dotted, raw = item.split("=", 1)
        parts = [p for p in dotted.split(".") if p]
        if not parts:
            raise ConfigError(f"override {item!r} has an empty key")
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = _parse_override_value(raw)
    return out


_OPTIMISER_ALIASES = {
    "lbfgsb": "lbfgsb",
    "lbfgs-b": "lbfgsb",
    "l-bfgs-b": "lbfgsb",
    "alt": "lbfgsb",
    "joint": "joint",
    "pure_joint": "joint",
    "pure-joint": "joint",
    "joint_odil": "joint",
    "joint-odil": "joint",
    "jointfreqodil": "joint",
}


def canonical_optimiser_name(name: str) -> str:
    """Normalise an optimiser ``name`` (or alias) to ``lbfgsb`` / ``joint``."""
    key = str(name).strip().lower()
    if key not in _OPTIMISER_ALIASES:
        raise ConfigError(
            f"unknown optimiser {name!r}; expected one of "
            f"{sorted(set(_OPTIMISER_ALIASES.values()))} (or an alias)."
        )
    return _OPTIMISER_ALIASES[key]


_REGULARISER_ALIASES = {
    "tikhonov": "tikhonov",
    "l2": "tikhonov",
    "smoothness": "tikhonov",
    "smooth": "tikhonov",
    "tv": "tv_iso",
    "tv_iso": "tv_iso",
    "tv_isotropic": "tv_iso",
    "iso": "tv_iso",
    "tv_aniso": "tv_aniso",
    "tv_anisotropic": "tv_aniso",
    "aniso": "tv_aniso",
}


def canonical_regulariser_name(name: str) -> str:
    """Normalise a regulariser ``name`` (or alias) to
    ``tikhonov`` / ``tv_iso`` / ``tv_aniso``."""
    key = str(name).strip().lower()
    if key not in _REGULARISER_ALIASES:
        raise ConfigError(
            f"unknown regulariser {name!r}; expected one of "
            f"{sorted(set(_REGULARISER_ALIASES.values()))} (or an alias)."
        )
    return _REGULARISER_ALIASES[key]


def default_config_dict() -> Dict[str, Any]:
    """The fully-resolved default configuration as a plain dict."""
    return RunConfig().to_dict()


def resolve_config(
    user_config: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
) -> RunConfig:
    """Resolve a (possibly partial) user config into a complete :class:`RunConfig`."""
    merged = deep_merge(default_config_dict(), user_config or {})
    merged = apply_overrides(merged, overrides)

    # Normalise optimiser name in the dict so it is validated and recorded.
    opt = merged.setdefault("optimiser", {})
    opt["name"] = canonical_optimiser_name(opt.get("name", "lbfgsb"))

    # Mint a run_id if none was supplied.
    run = merged.setdefault("run", {})
    if not run.get("run_id"):
        run["run_id"] = _make_run_id(merged)

    return RunConfig.from_dict(merged)


def _make_run_id(merged: Dict[str, Any]) -> str:
    """Build a descriptive run directory name from a merged config dict.

    Format::

        <DDMMYY_HHMMSS>_<problem>_<grid>_<optimiser>[_<precond>]_<f0>_<bands>

    e.g. ``280726_133521_shepp-logan_125x125_lbfgsb_80_40-50-60-70`` — where
    ``problem`` is the truth profile, ``grid`` the interior shape, ``f0`` and
    ``bands`` are frequencies in kHz (bands joined by ``-``; multiple
    frequencies within one continuation stage joined by ``+``). The ``precond``
    token (``smooth``) appears only when the LBFGSB c-gradient smoothing is
    active (``c_grad_smooth_sigma`` > 0); it is omitted otherwise.
    """
    ts = time.strftime("%d%m%y_%H%M%S", time.localtime())

    problem = _slug(str((merged.get("truth", {}) or {}).get("profile", "model")))

    shape = (merged.get("grid", {}) or {}).get("interior_shape", []) or []
    grid_size = "x".join(str(int(n)) for n in shape) or "grid"

    opt = merged.get("optimiser", {}) or {}
    optimiser = canonical_optimiser_name(opt.get("name", "lbfgsb"))
    precond = _precond_token(optimiser, opt)

    f0 = _khz_token((merged.get("source", {}) or {}).get("f0"))
    bands = _bands_token((merged.get("continuation", {}) or {}).get("bands", []) or [])

    parts = [ts, problem, grid_size, optimiser, precond, f0, bands]
    return "_".join(p for p in parts if p)


def _slug(text: str) -> str:
    """Lowercase slug with runs of non-alphanumerics collapsed to ``-``."""
    return re.sub(r"[^0-9a-zA-Z]+", "-", str(text)).strip("-").lower() or "x"


def _khz_token(hz: Any) -> str:
    """Render a frequency in Hz as a kHz token (``80e3`` -> ``80``, ``1.5e3``
    -> ``1p5``); ``na`` when unparseable."""
    try:
        khz = float(hz) / 1e3
    except (TypeError, ValueError):
        return "na"
    if abs(khz - round(khz)) < 1e-9:
        return str(int(round(khz)))
    return ("%g" % khz).replace(".", "p")


def _bands_token(bands: List[Any]) -> str:
    """Join per-stage kHz tokens: stages by ``-``, frequencies within one
    stage by ``+`` (``[[40e3], [70e3, 80e3]]`` -> ``40-70+80``)."""
    stage_tokens = []
    for b in bands:
        freqs = (b or {}).get("frequencies_hz", []) or []
        stage_tokens.append("+".join(_khz_token(f) for f in freqs) if freqs else "na")
    return "-".join(stage_tokens) or "na"


def _precond_token(optimiser: str, opt: Dict[str, Any]) -> str:
    """The active c-gradient smoothing token (``smooth``), or ``""`` when none."""
    if optimiser == "lbfgsb":
        lb = opt.get("lbfgsb", {}) or {}
        try:
            if float(lb.get("c_grad_smooth_sigma", 0.0)) > 0.0:
                return "smooth"
        except (TypeError, ValueError):
            return ""
    return ""


def validate_config(cfg: "RunConfig") -> List[str]:
    """Validate a resolved config ``cfg``.

    Raises :class:`ConfigError` on hard errors; returns a list of non-fatal
    warning strings.
    """
    warnings: List[str] = []

    name = canonical_optimiser_name(cfg.optimiser.name)
    if cfg.optimiser.log_every < 1:
        raise ConfigError("optimiser.log_every must be >= 1")
    if cfg.optimiser.n_iter < 1:
        raise ConfigError("optimiser.n_iter must be >= 1")

    if name == "lbfgsb":
        lb = cfg.optimiser.lbfgsb
        if lb.u_precond not in (None, "z"):
            raise ConfigError(
                f"optimiser.lbfgsb.u_precond must be null or 'z', "
                f"got {lb.u_precond!r}"
            )
        if str(lb.z_optim).lower() not in ("gd", "lbfgs"):
            raise ConfigError(
                f"optimiser.lbfgsb.z_optim must be 'gd' or 'lbfgs', "
                f"got {lb.z_optim!r}"
            )
        if lb.z_lr <= 0:
            raise ConfigError("optimiser.lbfgsb.z_lr must be > 0")
        if lb.z_steps < 1:
            raise ConfigError("optimiser.lbfgsb.z_steps must be >= 1")
        if str(lb.c_param).lower() not in ("velocity", "squared_slowness"):
            raise ConfigError(
                "optimiser.lbfgsb.c_param must be 'velocity' or "
                f"'squared_slowness', got {lb.c_param!r}"
            )
        if str(lb.c_param).lower() == "squared_slowness":
            if lb.u_precond == "z":
                raise ConfigError(
                    "optimiser.lbfgsb.c_param='squared_slowness' is not "
                    "supported with u_precond='z' yet"
                )
        if str(lb.u_solve).lower() not in ("optim", "exact"):
            raise ConfigError(
                f"optimiser.lbfgsb.u_solve must be 'optim' or 'exact', "
                f"got {lb.u_solve!r}"
            )
        if str(lb.u_solve).lower() == "exact" and lb.u_precond == "z":
            raise ConfigError(
                "optimiser.lbfgsb.u_solve='exact' is incompatible with "
                "u_precond='z' (the exact solve replaces the direct u block)"
            )
        if lb.c_grad_smooth_sigma < 0:
            raise ConfigError(
                "optimiser.lbfgsb.c_grad_smooth_sigma must be >= 0, "
                f"got {lb.c_grad_smooth_sigma}"
            )
        for sname, sch in (
            ("pde_weight_schedule", lb.pde_weight_schedule),
            ("c_grad_smooth_sigma_schedule", lb.c_grad_smooth_sigma_schedule),
        ):
            if sch.factor < 0:
                raise ConfigError(
                    f"optimiser.lbfgsb.{sname}.factor must be >= 0, "
                    f"got {sch.factor}"
                )
            if sch.every_n < 0:
                raise ConfigError(
                    f"optimiser.lbfgsb.{sname}.every_n must be >= 0, "
                    f"got {sch.every_n}"
                )
            if str(sch.direction).lower() not in ("increase", "decrease"):
                raise ConfigError(
                    f"optimiser.lbfgsb.{sname}.direction must be 'increase' "
                    f"or 'decrease', got {sch.direction!r}"
                )

    if name == "joint":
        jo = cfg.optimiser.joint
        if jo.data_weight <= 0:
            raise ConfigError("optimiser.joint.data_weight must be > 0")
        if jo.inner_max_iter < 1:
            raise ConfigError("optimiser.joint.inner_max_iter must be >= 1")
        if jo.history_size < 1:
            raise ConfigError("optimiser.joint.history_size must be >= 1")
        if jo.z_scale <= 0:
            raise ConfigError("optimiser.joint.z_scale must be > 0")
        if not (0.0 < jo.logit_clip < 0.5):
            raise ConfigError("optimiser.joint.logit_clip must be in (0, 0.5)")
        for fkey in ("u_scale_factor", "pde_scale_factor", "data_scale_factor"):
            if getattr(jo, fkey) <= 0:
                raise ConfigError(f"optimiser.joint.{fkey} must be > 0")
        if jo.c_min is not None and jo.c_max is not None and jo.c_min >= jo.c_max:
            raise ConfigError("optimiser.joint.c_min must be < c_max")
        if str(jo.line_search_fn).lower() not in ("strong_wolfe", "none"):
            raise ConfigError(
                "optimiser.joint.line_search_fn must be 'strong_wolfe' or 'none'"
            )
        if cfg.observation.method != "helmholtz":
            warnings.append(
                "optimiser.joint is designed for the inverse-crime baseline "
                "(observation.method='helmholtz' with matched forward/inverse "
                f"PML); got observation.method={cfg.observation.method!r}."
            )

    if not cfg.continuation.bands:
        raise ConfigError("continuation.bands must contain at least one band")
    for bi, band in enumerate(cfg.continuation.bands):
        if not band.frequencies_hz:
            raise ConfigError(f"band {bi} has no frequencies")
        if any(float(f) <= 0 for f in band.frequencies_hz):
            raise ConfigError(f"band {bi} has a non-positive frequency")
        if band.n_iter is not None and band.n_iter < 1:
            raise ConfigError(f"band {bi} n_iter must be >= 1 when set")
        if band.c_grad_smooth_sigma is not None:
            if float(band.c_grad_smooth_sigma) < 0:
                raise ConfigError(
                    f"band {bi} c_grad_smooth_sigma must be >= 0, "
                    f"got {band.c_grad_smooth_sigma}"
                )
            if name != "lbfgsb":
                warnings.append(
                    f"band {bi} sets c_grad_smooth_sigma but "
                    f"optimiser.name={cfg.optimiser.name!r} has no c-gradient "
                    "smoothing; it will be ignored."
                )

    if cfg.observation.method not in ("leapfrog_fft", "helmholtz"):
        raise ConfigError(
            f"observation.method must be 'leapfrog_fft' or 'helmholtz', "
            f"got {cfg.observation.method!r}"
        )
    if cfg.observation.pml_width is not None and int(cfg.observation.pml_width) < 0:
        raise ConfigError(
            f"observation.pml_width must be >= 0 when set, "
            f"got {cfg.observation.pml_width}"
        )
    if cfg.observation.normalize_data not in (None, "none", "per_receiver"):
        raise ConfigError(
            f"observation.normalize_data must be null, 'none' or 'per_receiver', "
            f"got {cfg.observation.normalize_data!r}"
        )
    if cfg.continuation.warm_start not in ("helmholtz", "none"):
        raise ConfigError(
            f"continuation.warm_start must be 'helmholtz' or 'none', "
            f"got {cfg.continuation.warm_start!r}"
        )
    if cfg.runtime.dtype.lower() not in (
        "float32",
        "float",
        "f32",
        "float64",
        "double",
        "f64",
    ):
        raise ConfigError(f"runtime.dtype {cfg.runtime.dtype!r} is not supported")

    for key in ("pde", "data"):
        if key not in cfg.loss.weights:
            raise ConfigError(f"loss.weights is missing required key {key!r}")

    reg = cfg.loss.regulariser
    if reg is not None and reg.name is not None:
        canonical_regulariser_name(reg.name)  # raises ConfigError on a bad name
        unknown = set(reg.params or {}) - {"eps"}
        if unknown:
            raise ConfigError(
                f"loss.regulariser.params has unknown key(s) {sorted(unknown)}; "
                "the only accepted param is 'eps'."
            )
        if float(cfg.loss.weights.get("reg", 0.0)) <= 0.0:
            warnings.append(
                f"loss.regulariser.name={reg.name!r} is set but loss.weights['reg'] "
                "is <= 0, so the regulariser has no effect."
            )

    ws = cfg.metrics.ssim.win_size
    if ws < 3 or ws % 2 == 0:
        raise ConfigError(f"metrics.ssim.win_size must be odd and >= 3, got {ws}")
    if cfg.metrics.ssim.mask not in ("head_roi", "interior", "none"):
        raise ConfigError(
            f"metrics.ssim.mask must be head_roi/interior/none, "
            f"got {cfg.metrics.ssim.mask!r}"
        )

    if (
        cfg.truth.profile not in ("shepp_logan", "shepp_logan_skull")
        and cfg.metrics.ssim.mask == "head_roi"
    ):
        warnings.append(
            f"truth profile {cfg.truth.profile!r} has no head mask; "
            "ssim_head_roi will be recorded as null."
        )

    for label, model in (("truth", cfg.truth), ("init", cfg.init)):
        _validate_model_cfg(label, model, warnings)

    diag = cfg.diagnostics
    if diag.enabled:
        if name != "lbfgsb":
            warnings.append(
                f"diagnostics.enabled is set but optimiser.name="
                f"{cfg.optimiser.name!r}; block diagnostics are only produced "
                "for the lbfgsb optimiser and will be skipped."
            )
        if diag.u_depths is not None:
            if not diag.u_depths:
                raise ConfigError("diagnostics.u_depths must be non-empty when set")
            if any(int(d) < 1 for d in diag.u_depths):
                raise ConfigError("diagnostics.u_depths values must be >= 1")

    return warnings


def _validate_model_cfg(label: str, model: "ModelCfg", warnings: List[str]) -> None:
    """Validate skull-smoothing / profile knobs on a truth or init ``model``."""
    if model.skull_alpha < 0.0 or model.skull_alpha > 1.0:
        raise ConfigError(
            f"{label}.skull_alpha must be in [0, 1]; got {model.skull_alpha}"
        )
    if model.skull_sigma < 0.0:
        raise ConfigError(f"{label}.skull_sigma must be >= 0; got {model.skull_sigma}")
    extra = model.extra or {}
    if "skull_smooth" in extra:
        try:
            smooth = float(extra["skull_smooth"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{label}.extra.skull_smooth must be a number; "
                f"got {extra['skull_smooth']!r}"
            ) from exc
        if smooth < 0.0:
            raise ConfigError(f"{label}.extra.skull_smooth must be >= 0; got {smooth}")
    if model.profile != "shepp_logan_skull" and (
        model.skull_alpha != 1.0
        or model.skull_sigma != 0.0
        or "skull_smooth" in extra
        or "skull_alpha" in extra
        or "skull_sigma" in extra
    ):
        warnings.append(
            f"{label}.profile={model.profile!r} ignores skull_alpha / "
            "skull_sigma (they apply only to shepp_logan_skull)."
        )
