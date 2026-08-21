import pytest

from odil_wave.experiment import (
    ConfigError,
    RunConfig,
    apply_overrides,
    build_problem,
    canonical_optimiser_name,
    deep_merge,
    default_config_dict,
    load_config_file,
    resolve_config,
    validate_config,
)
from odil_wave.experiment.config import canonical_regulariser_name


TINY_GRID = {
    "runtime": {"dtype": "float64"},
    "grid": {
        "interior_shape": [16, 16],
        "interior_extent": [[0.0, 0.05], [0.0, 0.05]],
        "c_min": 1400.0,
        "c_max": 1700.0,
        "t_max": 6e-5,
        "init_nt": 120,
        "pml_width": 4,
    },
    "source": {"kind": "tone_burst", "f0": 40e3, "n_cycles": 2.0},
    "acquisition": {"n_receivers": 8, "n_sources": 4, "source_spatial": "point"},
    "physics": {"space_order": 2},
    "truth": {
        "profile": "overdensity",
        "base": 1500.0,
        "contrast": 100.0,
        "pml_fill": "edge",
        "extra": {"radius": 0.012},
    },
    "init": {"profile": "homogeneous", "base": 1500.0, "pml_fill": "edge"},
    "observation": {"method": "helmholtz", "normalize_data": "none", "pml_width": None},
    "continuation": {"warm_start": "helmholtz", "bands": [{"frequencies_hz": [40e3]}]},
    "metrics": {"ssim": {"mask": "none"}},
}


# --------------------------------------------------------------------------- #
# Resolution / round-trip
# --------------------------------------------------------------------------- #
def test_resolve_config_mints_run_id_and_canonicalises():
    """A partial config resolves fully, mints a run id and canonicalises names."""
    cfg = resolve_config({"optimiser": {"name": "l-bfgs-b"}})
    assert isinstance(cfg, RunConfig)
    assert cfg.run.run_id  # auto-minted, non-empty
    assert cfg.optimiser.name == "lbfgsb"


def test_resolve_config_overlays_user_fields():
    """User fields override the defaults; untouched fields keep their defaults."""
    cfg = resolve_config({"grid": {"interior_shape": [16, 16]}})
    assert cfg.grid.interior_shape == [16, 16]
    assert cfg.grid.pml_width == 40  # default preserved


def test_runconfig_dict_roundtrip():
    """to_dict / from_dict is a faithful round-trip."""
    cfg = resolve_config(TINY_GRID)
    cfg2 = RunConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()


def test_runconfig_yaml_roundtrip(tmp_path):
    """A config written to YAML reloads to the same resolved config."""
    cfg = resolve_config(TINY_GRID)
    path = cfg.to_yaml(tmp_path / "cfg.yaml")
    loaded = load_config_file(path)
    assert resolve_config(loaded).to_dict() == cfg.to_dict()


def test_unknown_key_raises():
    """Unknown config keys are rejected during construction."""
    with pytest.raises(ConfigError):
        RunConfig.from_dict({"grid": {"not_a_field": 1}})


# --------------------------------------------------------------------------- #
# Merge / override / canonicalisation helpers
# --------------------------------------------------------------------------- #
def test_deep_merge_nested():
    """deep_merge recursively overlays without dropping sibling keys."""
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 9}})
    assert merged == {"a": {"b": 1, "c": 9}}


def test_apply_overrides_typed():
    """Dotted overrides parse JSON-typed values into the nested dict."""
    out = apply_overrides(
        default_config_dict(), ["optimiser.n_iter=7", "grid.c_min=1234.5"]
    )
    assert out["optimiser"]["n_iter"] == 7
    assert isinstance(out["optimiser"]["n_iter"], int)
    assert out["grid"]["c_min"] == 1234.5


def test_apply_overrides_bad_form_raises():
    """An override without '=' is rejected."""
    with pytest.raises(ConfigError):
        apply_overrides({}, ["not_an_assignment"])


def test_canonical_name_aliases():
    """Optimiser / regulariser aliases normalise, unknown names raise."""
    assert canonical_optimiser_name("pure_joint") == "joint"
    assert canonical_optimiser_name("alt") == "lbfgsb"
    assert canonical_regulariser_name("l2") == "tikhonov"
    assert canonical_regulariser_name("tv") == "tv_iso"
    with pytest.raises(ConfigError):
        canonical_optimiser_name("nope")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_config_accepts_default():
    """The default config validates and returns a (possibly empty) warning list."""
    warnings = validate_config(resolve_config({}))
    assert isinstance(warnings, list)


@pytest.mark.parametrize(
    "bad",
    [
        {"optimiser": {"n_iter": 0}},
        {"optimiser": {"log_every": 0}},
        {"continuation": {"bands": []}},
        {"continuation": {"bands": [{"frequencies_hz": [-1.0]}]}},
        {"metrics": {"ssim": {"win_size": 4}}},
        {"observation": {"method": "nope"}},
        {"runtime": {"dtype": "float16"}},
        {"optimiser": {"name": "lbfgsb", "lbfgsb": {"u_precond": "bad"}}},
    ],
)
def test_validate_config_rejects_bad(bad):
    """Representative malformed configs raise ConfigError."""
    with pytest.raises(ConfigError):
        validate_config(resolve_config(bad))


# --------------------------------------------------------------------------- #
# Problem construction
# --------------------------------------------------------------------------- #
def test_build_problem_assembles_pieces():
    """build_problem constructs the grid, source and truth/init velocity models."""
    problem = build_problem(resolve_config(TINY_GRID))
    assert problem.grid.interior_shape == (16, 16)
    assert problem.truth_velocity.c_max == pytest.approx(1600.0)  # base + contrast
    assert problem.init_velocity.c_max == pytest.approx(1500.0)  # homogeneous
    # overdensity truth has no head mask.
    assert problem.head_mask is None


def test_make_band_produces_observed_data():
    """make_band builds the frequency selection, geometry and observed wavefields."""
    problem = build_problem(resolve_config(TINY_GRID))
    band = problem.make_band([40e3])
    assert band.freq.n_frequencies == 1
    assert len(band.observed_wfs) == problem.cfg.acquisition.n_sources
    # not PML-split, so full wavefields (no reduced traces).
    assert band.observed_traces is None


def test_warm_start_toggles_with_config():
    """warm_start solves Helmholtz wavefields, or returns None when disabled."""
    problem = build_problem(resolve_config(TINY_GRID))
    band = problem.make_band([40e3])
    wfs = problem.warm_start(problem.init_velocity, band)
    assert wfs is not None and len(wfs) == problem.cfg.acquisition.n_sources

    none_cfg = deep_merge(TINY_GRID, {"continuation": {"warm_start": "none"}})
    problem2 = build_problem(resolve_config(none_cfg))
    assert (
        problem2.warm_start(problem2.init_velocity, problem2.make_band([40e3])) is None
    )
