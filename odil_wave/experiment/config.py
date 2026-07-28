"""Typed, serialisable experiment configuration for frequency-domain FWI runs.

This module is deliberately dependency-light (standard library + PyYAML only,
*no* ``torch``/``odil_wave`` imports) so that ``--dry-run`` and the config
tests resolve and validate a configuration without touching the numerical
stack.

Design
------
* Every effective value has an explicit default here, taken from
  ``sandbox/odil_2d_freq_domain.ipynb`` (``SETUP_SHEPP``) and the constructor
  defaults of the ``odil_wave`` classes. A user config may omit any field; the
  omitted value falls back to the default baked in below. The *fully resolved*
  configuration written to ``config_resolved.yaml`` therefore contains every
  value the run actually used, including code-derived defaults and the default
  blocks for every supported optimiser (lbfgsb / cf / modil).
* Configs round-trip through YAML/JSON. Numeric scientific notation without an
  explicit exponent sign (e.g. ``80e3``) is parsed as a float, matching the
  Python literals used in the notebook (plain PyYAML would treat ``80e3`` as a
  string).
* ``--override a.b.c=value`` applies dotted-key overrides; ``value`` is parsed
  as JSON (so ``40e3``, ``true``, ``null``, ``[1,2]`` are typed correctly),
  falling back to a bare string.

The dataclasses are frozen: once resolved, a config is immutable.
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
# Physical constants mirrored from odil_wave.models.velocity_models so this
# module stays torch-free. Keep in sync with that module.
# --------------------------------------------------------------------------- #
_SOS_WATER = 1500.0
_SHEPP_PHANTOM_SCALE = 0.90


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunMetaCfg:
    """Run identity and output location.

    ``run_id`` (the run directory name) is minted automatically at
    :func:`resolve_config` time when left blank, from the resolved config — see
    :func:`_make_run_id` for the format. Set it explicitly to pin a name.
    """

    run_id: str = ""  # resolved at resolve_config time if blank
    output_root: str = "outputs"
    seed: int = 0
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class RuntimeCfg:
    """Device / dtype / threading."""

    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    dtype: str = "float32"  # "float32" | "float64"
    torch_num_threads: Optional[int] = None
    deterministic: bool = False


@dataclass(frozen=True)
class GridCfg:
    """Spatial + time discretisation (odil_wave.Grid)."""

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
    """Source wavelet (odil_wave.SourceSignal)."""

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
    """Ring source/receiver layout (odil_wave.AcquisitionGeometry)."""

    n_receivers: int = 64
    n_sources: int = 8
    a_frac: float = 0.9
    b_frac: float = 0.9
    # "grid_center" -> centre of the interior extent, resolved in build_problem.
    ring_center: Union[str, List[float]] = "grid_center"
    sigma_s: Optional[float] = None
    source_spatial: str = "gaussian"


@dataclass(frozen=True)
class PhysicsCfg:
    """Discrete operator settings shared by forward + inverse."""

    space_order: int = 2
    time_order: int = 2
    pml_weight: float = 1.0


@dataclass(frozen=True)
class ModelCfg:
    """A VelocityModel spec (truth or initial model)."""

    profile: str = "shepp_logan"
    scale: float = 0.85
    base: float = _SOS_WATER
    contrast: float = 0.4
    pml_c: Optional[float] = None
    # extra profile_kwargs (threshold, c_water, c_skull, center, radius, ...)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationCfg:
    """How the observed ("true") data is synthesised and normalised."""

    # "leapfrog_fft": broadband leapfrog time solve then FFT onto each band's
    #   bins (avoids the inverse crime; notebook default).
    # "helmholtz":    frequency-domain Helmholtz solve on the truth model.
    method: str = "leapfrog_fft"
    normalize_data: str = "per_receiver"  # None | "none" | "per_receiver" | "global"
    verbose: bool = False


@dataclass(frozen=True)
class BandCfg:
    """One continuation stage. Single-frequency if ``len == 1``.

    ``source_offsets`` selects one or more rotated source octets on the
    receiver ring (see :func:`odil_wave.geometry.source_ring_indices`); the
    default ``[0]`` is the historical single-octet layout. ``source_schedule``
    controls how multiple offsets are run (mirrors
    :class:`odil_wave.optimisation.frequency_continuation.FrequencyBand`):

      * ``"joint"`` (default): all offsets active together in this one stage.
      * ``"sequential"``: run one octet after another (same frequencies),
        carrying ``c`` between offsets — expands to one stage per offset.
      * ``"cyclic"``: alternate offsets one octet per outer step — expands to
        ``n_iter`` single-iteration stages cycling through the offsets.

    ``sequential`` / ``cyclic`` expand into several concrete stages at run time
    (see :func:`expand_band_schedule`); ``joint`` is a single stage.
    """

    frequencies_hz: List[float] = field(default_factory=lambda: [40e3])
    # per-band optimiser iteration budget; None -> optimiser.n_iter.
    n_iter: Optional[int] = None
    source_offsets: List[int] = field(default_factory=lambda: [0])
    source_schedule: str = "joint"  # "joint" | "sequential" | "cyclic"


@dataclass(frozen=True)
class ContinuationCfg:
    """Frequency-band continuation schedule + warm-start policy."""

    # "helmholtz": re-solve Helmholtz on the current model each band to warm
    #   start u (notebook policy). "none": start u from the wavefield seed.
    warm_start: str = "helmholtz"
    bands: List[BandCfg] = field(default_factory=lambda: [BandCfg()])


@dataclass(frozen=True)
class RegulariserCfg:
    """Regulariser selection (None name -> no regulariser)."""

    name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossCfg:
    """Objective weights + regularisation."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {"pde": 1.0, "data": 100.0, "reg": 0.0}
    )
    regulariser: RegulariserCfg = field(default_factory=RegulariserCfg)


@dataclass(frozen=True)
class PrecondCfg:
    """c-gradient preconditioner (LBFGSB c-block only)."""

    c_precond: bool = False
    c_precond_type: str = "energy"  # "energy" | "gaussian"
    c_precond_sigma: float = 2.0
    c_precond_stab: float = 1e-2


@dataclass(frozen=True)
class LBFGSBCfg:
    """Fields specific to the block-coordinate dual L-BFGS (LBFGSB)."""

    z_steps: int = 1
    u_precond: Optional[str] = None  # None | "z"
    # z-block controls (only used when u_precond == "z").
    z_optim: str = "gd"  # "gd" (Armijo steepest descent) | "lbfgs"
    z_lr: float = 1.0  # Armijo initial step when z_optim == "gd"
    c_lr: float = 5.0
    c_max_iter: int = 6
    c_history_size: int = 10
    reset_c_history: bool = True
    # c-block update rule. "lbfgs" (default): the L-BFGS c-step above (with the
    # optional c-gradient preconditioner). "closed_form": the exact per-cell
    # variable-projection update (as in the ``cf`` optimiser) instead of L-BFGS,
    # driven by ``c_update_every`` / ``c_relax`` / ``illum_rel_floor`` and
    # available for both the direct and the ``u_precond='z'`` wavefield blocks.
    c_update: str = "lbfgs"  # "lbfgs" | "closed_form"
    c_update_every: int = 1  # closed_form only
    c_relax: float = 1.0  # closed_form only
    illum_rel_floor: float = 1e-6  # closed_form only
    early_stop_rtol: float = 0.0
    early_stop_min_iter: int = 0
    early_stop_patience: int = 3
    debug_c: bool = False
    precond: PrecondCfg = field(default_factory=PrecondCfg)


@dataclass(frozen=True)
class CFCfg:
    """Fields specific to the ``cf`` optimiser: LBFGSB with a closed-form
    (variable-projection) c-update (``c_update='closed_form'``)."""

    c_update_every: int = 2
    c_relax: float = 1.0
    illum_rel_floor: float = 1e-6


@dataclass(frozen=True)
class MODILCfg:
    """Fields specific to simultaneous multilevel inversion (MODILInversion)."""

    num_levels: int = 2
    coarsening_factor: int = 2
    min_ppw: float = 3.0
    c_update: str = "closed_form"  # "lbfgs" | "closed_form"
    c_update_every: int = 2
    c_relax: float = 1.0
    c_lr: float = 5.0
    c_max_iter: int = 6
    c_history_size: int = 10
    u_num_levels: int = 1
    u_coarsening_factor: int = 2
    modil_reg_weight: float = 0.0
    illum_rel_floor: float = 1e-6


@dataclass(frozen=True)
class OptimiserCfg:
    """Optimiser selection + shared block-coordinate settings.

    ``name`` is one of ``lbfgsb`` / ``cf`` / ``modil`` (aliases resolved in
    :func:`canonical_optimiser_name`). The three sub-blocks are always present
    in the resolved config so a run records the defaults for every optimiser,
    not only the selected one.
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
    cf: CFCfg = field(default_factory=CFCfg)
    modil: MODILCfg = field(default_factory=MODILCfg)


@dataclass(frozen=True)
class SSIMCfg:
    """Masked SSIM settings for the ``ssim_head_roi`` metric."""

    # "head_roi": intracranial mask (VelocityModel.head_mask); "interior":
    # whole non-PML interior; "none": no masking.
    mask: str = "head_roi"
    data_range: Union[str, float] = "auto"  # "auto" -> peak-to-peak of truth in mask
    win_size: int = 7
    gaussian_weights: bool = False
    sigma: float = 1.5


@dataclass(frozen=True)
class MetricsCfg:
    """Per-iteration metric settings."""

    ssim: SSIMCfg = field(default_factory=SSIMCfg)
    log_c_stats: bool = True


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

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        return _build(cls, data or {})

    def to_yaml(self, path: Union[str, Path]) -> Path:
        return _dump_yaml(self.to_dict(), path)

    def to_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return p

    def run_dir(self) -> Path:
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
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- #
# YAML helpers (float scientific-notation friendly)
# --------------------------------------------------------------------------- #
class _ConfigYamlLoader(yaml.SafeLoader):
    """SafeLoader that also parses unsigned-exponent floats like ``80e3``."""


# PyYAML's implicit float resolver requires an explicit exponent sign
# (``8.0e+4``). This resolver additionally accepts ``80e3`` / ``1.3e3`` / ``5e4``
# (unsigned positive exponent) as floats, matching the notebook's Python
# literals. Plain integers and dotted floats are unaffected.
_ConfigYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][0-9]+$"),
    list("-+0123456789."),
)


def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML or JSON config file into a plain dict."""
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
    """Recursively merge ``overlay`` onto a copy of ``base``.

    Mappings merge key-by-key; every other value (including lists such as the
    band schedule) is replaced wholesale.
    """
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
    """Apply ``a.b.c=value`` dotted overrides to a config dict (in place copy)."""
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
    "cf": "cf",
    "closed_form": "cf",
    "closed-form": "cf",
    "closedform": "cf",
    "modil": "modil",
}


def canonical_optimiser_name(name: str) -> str:
    """Normalise an optimiser name to ``lbfgsb`` / ``cf`` / ``modil``."""
    key = str(name).strip().lower()
    if key not in _OPTIMISER_ALIASES:
        raise ConfigError(
            f"unknown optimiser {name!r}; expected one of "
            f"{sorted(set(_OPTIMISER_ALIASES.values()))} (or an alias)."
        )
    return _OPTIMISER_ALIASES[key]


# Regulariser kinds accepted by :class:`odil_wave.loss.Regulariser`, plus the
# aliases understood in configs. Kept torch-free here so ``--dry-run`` can
# validate a regulariser name without importing the numerical stack; the actual
# object is built in ``experiment.runner._build_regulariser``.
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
    """Normalise a regulariser name to ``tikhonov`` / ``tv_iso`` / ``tv_aniso``."""
    key = str(name).strip().lower()
    if key not in _REGULARISER_ALIASES:
        raise ConfigError(
            f"unknown regulariser {name!r}; expected one of "
            f"{sorted(set(_REGULARISER_ALIASES.values()))} (or an alias)."
        )
    return _REGULARISER_ALIASES[key]


_SOURCE_SCHEDULES = ("joint", "sequential", "cyclic")


def expand_band_schedule(
    bands: List["BandCfg"], default_n_iter: int
) -> List["BandCfg"]:
    """Expand ``sequential`` / ``cyclic`` source schedules into concrete stages.

    Mirrors
    :func:`odil_wave.optimisation.frequency_continuation._expand_source_schedule`
    at the :class:`BandCfg` level. Each returned stage is a plain ``"joint"``
    band with a single resolved ``source_offsets`` list, so the runner's flat
    band loop (which already carries ``c`` from one stage to the next) realises
    the sequential / cyclic semantics. ``joint`` bands (and any band with a
    single offset) pass through unchanged.
    """
    stages: List[BandCfg] = []
    for band in bands:
        schedule = str(band.source_schedule).lower()
        offs = list(band.source_offsets)
        n_iter = int(default_n_iter if band.n_iter is None else band.n_iter)
        if schedule == "sequential" and len(offs) > 1:
            for off in offs:
                stages.append(
                    BandCfg(
                        frequencies_hz=list(band.frequencies_hz),
                        n_iter=n_iter,
                        source_offsets=[off],
                        source_schedule="joint",
                    )
                )
        elif schedule == "cyclic" and len(offs) > 1:
            for k in range(n_iter):
                stages.append(
                    BandCfg(
                        frequencies_hz=list(band.frequencies_hz),
                        n_iter=1,
                        source_offsets=[offs[k % len(offs)]],
                        source_schedule="joint",
                    )
                )
        else:
            stages.append(band)
    return stages


def default_config_dict() -> Dict[str, Any]:
    """The fully-resolved default configuration as a plain dict."""
    return RunConfig().to_dict()


def resolve_config(
    user_config: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
) -> RunConfig:
    """Resolve a (possibly partial) user config into a complete :class:`RunConfig`.

    Steps: start from the full defaults, deep-merge the user config, apply
    dotted overrides, normalise the optimiser name, and mint a ``run_id`` if the
    user left it blank. The result is a frozen, fully-resolved config in which
    every effective value (including code-derived defaults and the default
    blocks for all supported optimisers) is explicit.
    """
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

    e.g. ``280726_133521_shepp-logan_125x125_cf_80_40-50-60-70`` — where
    ``problem`` is the truth profile, ``grid`` the interior shape, ``f0`` and
    ``bands`` are frequencies in kHz (bands joined by ``-``; multiple
    frequencies within one continuation stage joined by ``+``). The
    ``precond`` token appears only when a c-gradient preconditioner is active
    (LBFGSB ``c_precond``); it is omitted otherwise.
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
    """The active c-gradient preconditioner token, or ``""`` when none.

    Only LBFGSB exposes a c-gradient preconditioner (``c_precond`` with type
    ``energy``/``gaussian``); ``cf``/``modil`` have none, so the slot is empty.
    """
    if optimiser == "lbfgsb":
        pc = (opt.get("lbfgsb", {}) or {}).get("precond", {}) or {}
        if pc.get("c_precond"):
            return _slug(str(pc.get("c_precond_type", "precond")))
    return ""


def validate_config(cfg: "RunConfig") -> List[str]:
    """Validate a resolved config. Raises :class:`ConfigError` on hard errors;
    returns a list of non-fatal warning strings.
    """
    warnings: List[str] = []

    name = canonical_optimiser_name(cfg.optimiser.name)  # raises on bad name
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
        if str(lb.c_update).lower() not in ("lbfgs", "closed_form"):
            raise ConfigError(
                f"optimiser.lbfgsb.c_update must be 'lbfgs' or 'closed_form', "
                f"got {lb.c_update!r}"
            )
        if str(lb.c_update).lower() == "closed_form":
            if lb.c_update_every < 1:
                raise ConfigError(
                    "optimiser.lbfgsb.c_update_every must be >= 1 for closed_form"
                )
            if lb.precond.c_precond:
                warnings.append(
                    "optimiser.lbfgsb.c_update='closed_form' ignores the "
                    "c-gradient preconditioner (precond.c_precond)."
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
        if not band.source_offsets:
            raise ConfigError(f"band {bi} source_offsets must be non-empty")
        if str(band.source_schedule).lower() not in _SOURCE_SCHEDULES:
            raise ConfigError(
                f"band {bi} source_schedule must be one of "
                f"{list(_SOURCE_SCHEDULES)}, got {band.source_schedule!r}"
            )

    if cfg.observation.method not in ("leapfrog_fft", "helmholtz"):
        raise ConfigError(
            f"observation.method must be 'leapfrog_fft' or 'helmholtz', "
            f"got {cfg.observation.method!r}"
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

    if cfg.optimiser.name == "modil" and cfg.metrics.ssim.mask == "head_roi":
        if cfg.truth.profile not in ("shepp_logan", "shepp_logan_skull"):
            warnings.append(
                "ssim mask 'head_roi' but truth profile has no head mask; "
                "ssim_head_roi will be null."
            )
    if (
        cfg.truth.profile not in ("shepp_logan", "shepp_logan_skull")
        and cfg.metrics.ssim.mask == "head_roi"
    ):
        warnings.append(
            f"truth profile {cfg.truth.profile!r} has no head mask; "
            "ssim_head_roi will be recorded as null."
        )
    return warnings
