"""Typed, serialisable run configuration for frequency-domain FWI runs.

Dependency-light (standard library + PyYAML only, no ``torch`` / ``odil_wave``
imports) so a config can be resolved and validated without importing the
numerical stack.

The dataclass defaults below are the **default config**, the only one: there is
no default YAML file (ADR 0001). A run config names only the values it
changes; :func:`resolve_config` applies it (and any CLI overrides) to these
defaults, and the resolved config written to ``config_resolved.yaml`` records
every value. ``python main.py --print-defaults`` prints the default config.
The docstrings here are the single place each field is documented. The
dataclasses are frozen (immutable once resolved).
"""

from __future__ import annotations

import copy
import json
import math
import re
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

from .atomic import write_text_atomic


class ConfigError(ValueError):
    """Raised for malformed / unknown configuration entries."""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunMetaCfg:
    """Run seed and free-form labels. The run's directory is not a config value:
    the entry point picks it (see ``odil_wave.experiment.rundir``).

    Params:
    * ``seed``: int RNG seed for reproducibility.
    * ``tags``: list[str] free-form labels recorded in the config.
    * ``notes``: str free-form description.
    """

    seed: int = 0
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class RuntimeCfg:
    """Device / dtype / threading / logging.

    Params:
    * ``device``: ``"cpu"`` | ``"cuda"`` | ``"mps"``.
    * ``dtype``: ``"float32"`` | ``"float64"`` (aliases ``f32``/``float``,
      ``f64``/``double``); float64 is more reproducible for the inverse solve.
    * ``torch_num_threads``: ``None`` -> torch default, else int >= 1.
    * ``deterministic``: force deterministic torch kernels.
    * ``verbose``: print Helmholtz-solve and per-iteration optimiser progress.
    """

    device: str = "cpu"
    dtype: str = "float32"
    torch_num_threads: Optional[int] = None
    deterministic: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class GridCfg:
    """Spatial discretisation of the interior (odil_wave.Grid).

    Params:
    * ``interior_shape``: ``[nx, ny]`` interior grid **nodes** along x then y
      (spacing ``dx = (x1 - x0) / (nx - 1)``); the PML adds ``pml.width``
      nodes on every side.
    * ``interior_extent``: metres, ``[[x0, x1], [y0, y1]]`` physical extent of
      the interior, x first (same axis order as ``interior_shape``).
    * ``c_min`` / ``c_max``: m/s velocity bounds (``c_min`` < ``c_max``). They
      also bound the inverted model: the joint solver's squared-slowness latent
      maps onto ``[1/c_max**2, 1/c_min**2]``. ``c_max`` sets the CFL time step
      when ``time.init_nt`` is null.
    * ``space_order``: spatial FD stencil order ``2`` | ``4`` | ``6`` | ``8``
      | ``10`` (higher = less numerical dispersion, costlier).
    """

    interior_shape: List[int] = field(default_factory=lambda: [125, 125])
    interior_extent: List[List[float]] = field(
        default_factory=lambda: [[0.0, 0.25], [0.0, 0.25]]
    )
    c_min: float = 1300.0
    c_max: float = 3100.0
    space_order: int = 2


@dataclass(frozen=True)
class PmlCfg:
    """Absorbing layer (PML sponge) around the interior.

    Params:
    * ``width``: int >= 0 nodes per side; PML thickness of the inversion grid.
    * ``power``: PML grading polynomial order (typ. 2-4).
    * ``R0``: target PML reflection coefficient (small, e.g. 1e-4 .. 1e-8).
    * ``damping_scale``: >= 0 factor multiplying the operator's PML damping
      terms (``sigma`` in the frequency-domain residual); ``1`` is the designed
      absorption, ``0`` disables absorption (reflecting borders).
    * ``observation_width``: ``None`` -> same as ``width``: observations are
      generated on the inversion grid with the inversion operator (the inverse
      crime; the true model fits the data exactly). An int >= 0 different from
      ``width`` opts into a deliberate modelling error: observations are
      synthesised on a separate grid with that PML width and reduced to
      receiver traces.
    """

    width: int = 40
    power: int = 3
    R0: float = 1e-6
    damping_scale: float = 1.0
    observation_width: Optional[int] = None


@dataclass(frozen=True)
class TimeCfg:
    """Time axis; it fixes the FFT bins the band frequencies snap to.

    Params:
    * ``t_max``: s; time window (with ``init_nt`` sets ``dt`` and the FFT
      frequency resolution ``1 / t_max``).
    * ``init_nt``: ``None`` -> CFL-derived from ``grid.c_max``, else int >= 2
      time samples.
    * ``cfl_safety``: (0, 1]; CFL fraction used to pick ``dt`` when
      ``init_nt`` is null.
    """

    t_max: float = 500e-6
    init_nt: Optional[int] = 1300
    cfl_safety: float = 0.9


@dataclass(frozen=True)
class SourceCfg:
    """Source wavelet (odil_wave.SourceSignal).

    Params:
    * ``kind``: ``"tone_burst"`` | ``"ricker"``.
    * ``f0``: Hz centre frequency.
    * ``amplitude``: source amplitude scale.
    * ``n_cycles``: tone_burst only; number of cycles in the burst.
    * ``envelope``: tone_burst only; ``"gaussian"`` | ``"rectangular"``.
    * ``dimensionless``: true -> the physical waveform amplitude is
      ``amplitude * 2 * f0**2`` (lands the wavefield near O(1)); false -> it is
      ``amplitude``.
    """

    kind: str = "tone_burst"
    f0: float = 80e3
    amplitude: float = 1.0
    n_cycles: float = 3.0
    envelope: str = "gaussian"
    dimensionless: bool = True


@dataclass(frozen=True)
class AcquisitionCfg:
    """Ring transducer layout (odil_wave.AcquisitionGeometry).

    Params:
    * ``n_transducers``: int >= 2 transducers evenly spaced on the ring. Each
      transmits one shot, recorded by the other ``n_transducers - 1``.
    * ``a_frac`` / ``b_frac``: (0, 1]; ring semi-axes as a fraction of the
      interior half-extent in x / y.
    * ``ring_center``: ``"grid_center"`` (centre of the interior extent) |
      ``[x, y]`` in metres.
    * ``sigma_s``: > 0 metres; width of the Gaussian source density
      ``exp(-r**2 / sigma_s**2) / (pi sigma_s**2)``, i.e. a Gaussian with
      standard deviation ``sigma_s / sqrt(2)`` (used when
      ``source_spatial == "gaussian"``). The default 3.4 mm is just under
      ``1 / k_max`` = 3.45 mm at 60 kHz in 1300 m/s (1.69 h on the default grid). The
      config warns when ``sigma_s < 1.5 h`` (under-resolved), when
      ``k_max * sigma_s > 1`` for the top band frequency (the source low-pass
      filters it) and when the ring is within ``3 sigma_s`` of the PML.
    * ``source_spatial``: ``"gaussian"`` | ``"point"``; either is a density
      that integrates to 1 (the point source is ``1 / (dx dy)``).
    """

    n_transducers: int = 64
    a_frac: float = 0.9
    b_frac: float = 0.9
    ring_center: Union[str, List[float]] = "grid_center"
    sigma_s: float = 3.4e-3
    source_spatial: str = "gaussian"


@dataclass(frozen=True)
class ModelCfg:
    """A VelocityModel spec (truth or initial model).

    Params:
    * ``profile``: ``"homogeneous"`` | ``"overdensity"`` | ``"shepp_logan"`` |
      ``"shepp_logan_skull"``.
    * ``scale``: shepp_logan / shepp_logan_skull; fraction of the interior the
      phantom occupies.
    * ``base``: m/s background velocity (homogeneous / overdensity).
    * ``contrast``: overdensity only; m/s added to ``base`` inside the anomaly.
    * ``pml_c``: ``None`` -> derived (background for the Shepp-Logan
      profiles, else the field minimum); constant velocity of the PML ring.
    * ``skull_alpha``: [0, 1]; shepp_logan_skull only (skull contrast fraction).
    * ``skull_sigma``: >= 0 grid cells; shepp_logan_skull only, Gaussian soft
      edge (try 1-3).
    * ``extra``: other profile kwargs: ``radius`` / ``center`` (overdensity),
      ``threshold`` / ``c_water`` / ``c_skull`` / ``soft_intercept`` /
      ``soft_slope`` / ``interior_value`` (Shepp-Logan profiles).
    """

    profile: str = "shepp_logan"
    scale: float = 0.85
    base: float = 1500.0
    contrast: float = 0.4
    pml_c: Optional[float] = None
    skull_alpha: float = 1.0
    skull_sigma: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BandCfg:
    """One continuation stage (single-frequency if ``len(frequencies_hz) == 1``).

    Params:
    * ``frequencies_hz``: list[float > 0]; multiple frequencies in one stage are
      inverted jointly (multi-frequency stage).
    * ``n_iter``: ``None`` -> ``optimiser.n_iter``, else int >= 1 per-stage
      iteration budget.
    """

    frequencies_hz: List[float] = field(default_factory=lambda: [60e3])
    n_iter: Optional[int] = None


@dataclass(frozen=True)
class ContinuationCfg:
    """Frequency-band continuation schedule.

    Every band warm-starts ``u`` with a Helmholtz solve on the current model.

    Params:
    * ``bands``: ordered list[:class:`BandCfg`], solved in sequence (typically
      low-to-high frequency).
    """

    bands: List[BandCfg] = field(default_factory=lambda: [BandCfg()])


@dataclass(frozen=True)
class RegulariserCfg:
    """Regulariser on the interior velocity; active iff ``loss.weights.reg > 0``.

    Params:
    * ``name``: ``"tv_iso"`` | ``"tikhonov"`` | ``"tv_aniso"``.
    * ``eps``: > 0 smoothing floor inside the isotropic-TV square root (used
      by ``tv_iso`` only).
    """

    name: str = "tv_iso"
    eps: float = 1e-8


@dataclass(frozen=True)
class LossCfg:
    """Objective weights and regularisation.

    Params:
    * ``weights``: the only place the joint objective weights are set, applied
      as ``Phi = pde * pde_term + data * data_term + reg * R(c)``. Keys ``pde``,
      ``data`` and ``reg`` (all required); each >= 0 and ``data`` > 0.
      ``reg`` > 0 activates the regulariser.
    * ``regulariser``: :class:`RegulariserCfg`.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {"pde": 1.0, "data": 1.0, "reg": 0.0}
    )
    regulariser: RegulariserCfg = field(default_factory=RegulariserCfg)


@dataclass(frozen=True)
class OptimiserCfg:
    """Joint full-space L-BFGS solver (``JointFreqODIL``) settings.

    A single L-BFGS over the complex wavefield block and a squared-slowness
    model latent bounded by a sigmoid onto ``[grid.c_min, grid.c_max]``. The
    objective weights are not set here: they come from ``loss.weights``.

    Params:
    * ``n_iter``: int >= 1 outer iterations (per band unless ``band.n_iter``).
    * ``log_every``: int >= 1 metric-logging cadence in outers.
    * ``inner_max_iter``: int >= 1 L-BFGS iterations per logged outer step; keep
      >= ~5 for a genuine continuous solve (inner=1 stalls the line search).
    * ``history_size``: int >= 1 L-BFGS history length.
    * ``line_search_fn``: ``"strong_wolfe"`` | ``"none"``.
    * ``lbfgs_lr``: L-BFGS learning rate.
    * ``tolerance_grad`` / ``tolerance_change``: L-BFGS stopping tolerances.
    """

    n_iter: int = 20
    log_every: int = 5
    inner_max_iter: int = 20
    history_size: int = 20
    line_search_fn: str = "strong_wolfe"
    lbfgs_lr: float = 1.0
    tolerance_grad: float = 1e-12
    tolerance_change: float = 1e-14


@dataclass(frozen=True)
class SSIMCfg:
    """Settings of the tape's ``ssim_brain`` metric.

    The scoring rule is fixed (Gaussian window, ``use_sample_covariance=False``,
    data range = peak-to-peak of the truth over the brain region, map averaged
    over the brain region eroded by the window radius).

    Params:
    * ``sigma``: > 0 Gaussian window std in cells. The window spans
      ``2 * round(3.5 * sigma) + 1`` cells (``sigma = 1.5``: 11, Wang 2004)
      and must fit in the interior.
    """

    sigma: float = 1.5


def ssim_window_size(sigma: float) -> int:
    """Width in cells of the Gaussian SSIM window of std ``sigma``.

    Matches ``skimage.metrics.structural_similarity``, which truncates the
    Gaussian at 3.5 standard deviations either side of the centre cell.
    """
    return 2 * int(3.5 * float(sigma) + 0.5) + 1


# Truth profiles that define a brain and a skull region.
_REGION_PROFILES = ("shepp_logan", "shepp_logan_skull")


@dataclass(frozen=True)
class MetricsCfg:
    """Per-iteration metric settings.

    Params:
    * ``ssim``: :class:`SSIMCfg`.
    """

    ssim: SSIMCfg = field(default_factory=SSIMCfg)


@dataclass(frozen=True)
class RunConfig:
    """Top-level config; its defaults are the default config."""

    run: RunMetaCfg = field(default_factory=RunMetaCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    pml: PmlCfg = field(default_factory=PmlCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    source: SourceCfg = field(default_factory=SourceCfg)
    acquisition: AcquisitionCfg = field(default_factory=AcquisitionCfg)
    truth: ModelCfg = field(default_factory=ModelCfg)
    init: ModelCfg = field(
        default_factory=lambda: ModelCfg(profile="shepp_logan_skull")
    )
    continuation: ContinuationCfg = field(default_factory=ContinuationCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    optimiser: OptimiserCfg = field(default_factory=OptimiserCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)

    def band_n_iter(self, band: BandCfg) -> int:
        """``band``'s outer iterations: its ``n_iter``, else ``optimiser.n_iter``."""
        return int(band.n_iter if band.n_iter is not None else self.optimiser.n_iter)

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
        write_text_atomic(p, json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return p


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


# Objective-weight keys accepted in ``loss.weights``.
_WEIGHT_KEYS = ("pde", "data", "reg")


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
    text = yaml.safe_dump(obj, sort_keys=False, default_flow_style=False, width=100)
    write_text_atomic(p, text)
    return p


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


# Accepted values of the enum-like fields. config.py stays torch-free, so these
# mirror the library: WaveEquation's stencil orders, the Regulariser kinds and
# the VelocityModel profiles / profile kwargs.
_SPACE_ORDERS = (2, 4, 6, 8, 10)
_LINE_SEARCHES = ("strong_wolfe", "none")
_REGULARISERS = ("tv_iso", "tikhonov", "tv_aniso")
_PROFILES = ("homogeneous", "overdensity", "shepp_logan", "shepp_logan_skull")
# Profile kwargs accepted in ``truth.extra`` / ``init.extra``.
_EXTRA_KEYS = (
    "radius",
    "center",
    "threshold",
    "c_water",
    "c_skull",
    "soft_intercept",
    "soft_slope",
    "interior_value",
)


def default_config_dict() -> Dict[str, Any]:
    """The fully-resolved default configuration as a plain dict."""
    return RunConfig().to_dict()


def resolve_config(
    user_config: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
) -> RunConfig:
    """Resolve a run config ``user_config`` and dotted ``overrides`` onto
    the default config into a complete :class:`RunConfig`."""
    merged = deep_merge(default_config_dict(), user_config or {})
    merged = apply_overrides(merged, overrides)
    return RunConfig.from_dict(merged)


def validate_config(cfg: "RunConfig") -> List[str]:
    """Validate a resolved config ``cfg``.

    Raises :class:`ConfigError` on hard errors; returns a list of non-fatal
    warning strings.
    """
    warnings: List[str] = []

    opt = cfg.optimiser
    for key in ("n_iter", "log_every", "inner_max_iter", "history_size"):
        if getattr(opt, key) < 1:
            raise ConfigError(f"optimiser.{key} must be >= 1")
    if str(opt.line_search_fn).lower() not in _LINE_SEARCHES:
        raise ConfigError("optimiser.line_search_fn must be 'strong_wolfe' or 'none'")

    if not cfg.continuation.bands:
        raise ConfigError("continuation.bands must contain at least one band")
    for bi, band in enumerate(cfg.continuation.bands):
        if not band.frequencies_hz:
            raise ConfigError(f"band {bi} has no frequencies")
        if any(float(f) <= 0 for f in band.frequencies_hz):
            raise ConfigError(f"band {bi} has a non-positive frequency")
        if band.n_iter is not None and band.n_iter < 1:
            raise ConfigError(f"band {bi} n_iter must be >= 1 when set")

    if cfg.acquisition.n_transducers < 2:
        raise ConfigError(
            f"acquisition.n_transducers must be >= 2, "
            f"got {cfg.acquisition.n_transducers}"
        )

    sigma = cfg.acquisition.sigma_s
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ConfigError(
            f"acquisition.sigma_s must be a length > 0 in metres, got {sigma!r}"
        )
    if cfg.acquisition.source_spatial == "gaussian":
        warnings.extend(_source_width_warnings(cfg))

    if cfg.grid.space_order not in _SPACE_ORDERS:
        raise ConfigError(
            f"grid.space_order must be one of {list(_SPACE_ORDERS)}, "
            f"got {cfg.grid.space_order!r}"
        )
    pml = cfg.pml
    if int(pml.width) < 0:
        raise ConfigError(f"pml.width must be >= 0, got {pml.width}")
    if float(pml.damping_scale) < 0:
        raise ConfigError(f"pml.damping_scale must be >= 0, got {pml.damping_scale}")
    if pml.observation_width is not None and int(pml.observation_width) < 0:
        raise ConfigError(
            f"pml.observation_width must be >= 0 when set, got {pml.observation_width}"
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

    loss = cfg.loss
    weights = loss.weights
    for key in _WEIGHT_KEYS:
        if key not in weights:
            raise ConfigError(f"loss.weights is missing required key {key!r}")
    unknown = set(weights) - set(_WEIGHT_KEYS)
    if unknown:
        raise ConfigError(
            f"loss.weights has unknown key(s) {sorted(unknown)}; "
            f"valid keys: {list(_WEIGHT_KEYS)}"
        )
    for key, val in weights.items():
        if float(val) < 0:
            raise ConfigError(f"loss.weights.{key} must be >= 0, got {val}")
    if float(weights["data"]) <= 0:
        raise ConfigError(f"loss.weights.data must be > 0, got {weights['data']}")
    if loss.regulariser.name not in _REGULARISERS:
        raise ConfigError(
            f"loss.regulariser.name must be one of {list(_REGULARISERS)}, "
            f"got {loss.regulariser.name!r}"
        )

    # The tape scores the brain and skull regions of the truth model.
    if cfg.truth.profile not in _REGION_PROFILES:
        raise ConfigError(
            f"truth.profile {cfg.truth.profile!r} has no brain or skull region "
            f"for the tape's model-error metrics; use one of {list(_REGION_PROFILES)}"
        )
    sigma = float(cfg.metrics.ssim.sigma)
    if sigma <= 0:
        raise ConfigError(f"metrics.ssim.sigma must be > 0, got {sigma}")
    win = ssim_window_size(sigma)
    if win > min(cfg.grid.interior_shape):
        raise ConfigError(
            f"metrics.ssim.sigma={sigma} gives a {win}-cell SSIM window, larger "
            f"than the interior {list(cfg.grid.interior_shape)}"
        )

    for label, model in (("truth", cfg.truth), ("init", cfg.init)):
        _validate_model_cfg(label, model, warnings)

    return warnings


def _source_width_warnings(cfg: "RunConfig") -> List[str]:
    """Warnings for a Gaussian ``acquisition.sigma_s`` that is under-resolved,
    low-pass filters the top band frequency, or reaches into the PML; each
    names ``sigma/h``, ``k_max sigma`` and the ring's clearance to the PML."""
    sigma = float(cfg.acquisition.sigma_s)
    (x0, x1), (y0, y1) = cfg.grid.interior_extent
    nx, ny = cfg.grid.interior_shape
    h = max((x1 - x0) / (nx - 1), (y1 - y0) / (ny - 1))
    f_max = max(float(f) for b in cfg.continuation.bands for f in b.frequencies_hz)
    k_max = 2.0 * math.pi * f_max / float(cfg.grid.c_min)
    rc = cfg.acquisition.ring_center
    cx, cy = ((x0 + x1) / 2, (y0 + y1) / 2) if isinstance(rc, str) else map(float, rc)
    a = cfg.acquisition.a_frac * (x1 - x0) / 2
    b = cfg.acquisition.b_frac * (y1 - y0) / 2
    clearance = min(cx - a - x0, x1 - cx - a, cy - b - y0, y1 - cy - b)
    facts = (
        f"(sigma_s/h = {sigma / h:.2f}, k_max*sigma_s = {k_max * sigma:.2f} at "
        f"{f_max / 1e3:g} kHz, ring clearance to the PML = {clearance * 1e3:.1f} mm "
        f"= {clearance / sigma:.1f} sigma_s)"
    )
    out = []
    if sigma < 1.5 * h:
        err = 4.0 * math.exp(-((math.pi * sigma / h) ** 2))
        out.append(
            f"acquisition.sigma_s={sigma:g} m is under-resolved (< 1.5 h): the "
            f"source's normalisation is off by ~{err:.0e} {facts}"
        )
    if k_max * sigma > 1.0:
        out.append(
            f"acquisition.sigma_s={sigma:g} m low-pass filters the top frequency "
            f"(k_max*sigma_s > 1, amplitude factor "
            f"{math.exp(-((k_max * sigma) ** 2) / 4):.2f}) {facts}"
        )
    if clearance < 3.0 * sigma:
        out.append(
            f"acquisition.sigma_s={sigma:g} m: the transducer ring is within "
            f"3 sigma_s of the PML {facts}"
        )
    return out


def _validate_model_cfg(label: str, model: "ModelCfg", warnings: List[str]) -> None:
    """Validate the profile, skull knobs and ``extra`` kwargs of a truth or init
    ``model``."""
    if model.profile not in _PROFILES:
        raise ConfigError(
            f"{label}.profile must be one of {list(_PROFILES)}, got {model.profile!r}"
        )
    if model.skull_alpha < 0.0 or model.skull_alpha > 1.0:
        raise ConfigError(
            f"{label}.skull_alpha must be in [0, 1]; got {model.skull_alpha}"
        )
    if model.skull_sigma < 0.0:
        raise ConfigError(f"{label}.skull_sigma must be >= 0; got {model.skull_sigma}")
    unknown = set(model.extra or {}) - set(_EXTRA_KEYS)
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {label}.extra: {sorted(unknown)}; "
            f"valid keys: {sorted(_EXTRA_KEYS)}"
        )
    if model.profile != "shepp_logan_skull" and (
        model.skull_alpha != 1.0 or model.skull_sigma != 0.0
    ):
        warnings.append(
            f"{label}.profile={model.profile!r} ignores skull_alpha / "
            "skull_sigma (they apply only to shepp_logan_skull)."
        )
