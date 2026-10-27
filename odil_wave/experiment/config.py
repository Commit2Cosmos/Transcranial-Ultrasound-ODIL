"""Typed, serialisable experiment configuration for frequency-domain FWI runs.

Dependency-light (standard library + PyYAML only, no ``torch`` / ``odil_wave``
imports) so a config can be resolved and validated without importing the
numerical stack.

The dataclass defaults below are the **default config**, the only one: there is
no default YAML file (ADR 0001). An experiment config names only the values it
changes; :func:`resolve_config` applies it (and any CLI overrides) to these
defaults, and the resolved config written to ``config_resolved.yaml`` records
every value. ``python main.py --print-defaults`` prints the default config.
The docstrings here are the single place each field is documented. The
dataclasses are frozen (immutable once resolved).
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
      config at :func:`resolve_config` time as
      ``<DDMMYY_HHMMSS>_<problem>_<grid>_<f0>_<bands>`` (see
      :func:`_make_run_id`). Set to pin a name.
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
    """Ring source/receiver layout (odil_wave.AcquisitionGeometry).

    Params:
    * ``n_receivers``: int >= 1 receivers evenly spaced on the ring.
    * ``n_sources``: int >= 1 requested sources, subsampled from the receiver
      positions (the active count may be smaller; the step is floored at 1).
    * ``a_frac`` / ``b_frac``: (0, 1]; ring semi-axes as a fraction of the
      interior half-extent in x / y.
    * ``ring_center``: ``"grid_center"`` (centre of the interior extent) |
      ``[x, y]`` in metres.
    * ``sigma_s``: ``None`` -> derived from the grid spacing and ``f0``, else
      > 0 metres; width of the Gaussian source profile
      ``exp(-r**2 / sigma_s**2)``, i.e. a Gaussian with standard deviation
      ``sigma_s / sqrt(2)`` (used when ``source_spatial == "gaussian"``).
    * ``source_spatial``: ``"gaussian"`` | ``"point"``.
    """

    n_receivers: int = 64
    n_sources: int = 64
    a_frac: float = 0.9
    b_frac: float = 0.9
    ring_center: Union[str, List[float]] = "grid_center"
    sigma_s: Optional[float] = None
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
    """Resolve an experiment config ``user_config`` and dotted ``overrides`` onto
    the default config into a complete :class:`RunConfig`."""
    merged = deep_merge(default_config_dict(), user_config or {})
    merged = apply_overrides(merged, overrides)

    # Mint a run_id if none was supplied.
    run = merged.setdefault("run", {})
    if not run.get("run_id"):
        run["run_id"] = _make_run_id(merged)

    return RunConfig.from_dict(merged)


def _make_run_id(merged: Dict[str, Any]) -> str:
    """Build a descriptive run directory name from a merged config dict.

    Format::

        <DDMMYY_HHMMSS>_<problem>_<grid>_<f0>_<bands>

    e.g. ``280726_133521_shepp-logan_125x125_80_40-50-60-70`` — where
    ``problem`` is the truth profile, ``grid`` the interior shape, ``f0`` and
    ``bands`` are frequencies in kHz (bands joined by ``-``; multiple
    frequencies within one continuation stage joined by ``+``).
    """
    ts = time.strftime("%d%m%y_%H%M%S", time.localtime())

    problem = _slug(str((merged.get("truth", {}) or {}).get("profile", "model")))

    shape = (merged.get("grid", {}) or {}).get("interior_shape", []) or []
    grid_size = "x".join(str(int(n)) for n in shape) or "grid"

    f0 = _khz_token((merged.get("source", {}) or {}).get("f0"))
    bands = _bands_token((merged.get("continuation", {}) or {}).get("bands", []) or [])

    parts = [ts, problem, grid_size, f0, bands]
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
            f"pml.observation_width must be >= 0 when set, "
            f"got {pml.observation_width}"
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

    return warnings


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
