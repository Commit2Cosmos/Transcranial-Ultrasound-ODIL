import pytest

from odil_wave import InverseLoss, LossConfig, WaveEquation, Wavefield
import torch
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
        "extra": {"radius": 0.012},
    },
    "init": {"profile": "homogeneous", "base": 1500.0},
    "observation": {"normalize_data": "none", "pml_width": None},
    "continuation": {"warm_start": "helmholtz", "bands": [{"frequencies_hz": [40e3]}]},
    "metrics": {"ssim": {"mask": "none"}},
}


# --------------------------------------------------------------------------- #
# Resolution / round-trip
# --------------------------------------------------------------------------- #
def test_resolve_config_mints_run_id_and_canonicalises():
    """A partial config resolves fully, mints a run id and canonicalises names."""
    cfg = resolve_config({"optimiser": {"name": "joint-odil"}})
    assert isinstance(cfg, RunConfig)
    assert cfg.run.run_id  # auto-minted, non-empty
    assert cfg.optimiser.name == "joint"


def test_default_optimiser_is_joint():
    """With no optimiser given, the resolved config uses the joint solver."""
    cfg = resolve_config({})
    assert cfg.optimiser.name == "joint"
    assert "_joint_" in cfg.run.run_id


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
    assert canonical_optimiser_name("joint_odil") == "joint"
    assert canonical_regulariser_name("l2") == "tikhonov"
    assert canonical_regulariser_name("tv") == "tv_iso"
    for name in ("nope", "lbfgsb", "l-bfgs-b", "lbfgs-b", "alt"):
        with pytest.raises(ConfigError):
            canonical_optimiser_name(name)


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
        {"runtime": {"dtype": "float16"}},
        {"optimiser": {"joint": {"inner_max_iter": 0}}},
    ],
)
def test_validate_config_rejects_bad(bad):
    """Representative malformed configs raise ConfigError."""
    with pytest.raises(ConfigError):
        validate_config(resolve_config(bad))


@pytest.mark.parametrize(
    "removed",
    [
        {"optimiser": {"name": "lbfgsb"}},
        {"optimiser": {"lbfgsb": {}}},
        {"optimiser": {"u_steps": 5}},
        {"optimiser": {"clamp": True}},
        {"observation": {"method": "leapfrog_fft"}},
        {"physics": {"time_order": 2}},
        {"truth": {"pml_fill": "constant"}},
        {"init": {"pml_fill": "edge"}},
        {"diagnostics": {"enabled": False}},
        {
            "continuation": {
                "bands": [{"frequencies_hz": [40e3], "c_grad_smooth_sigma": 1.0}]
            }
        },
    ],
)
def test_removed_settings_are_rejected(removed):
    """lbfgsb / leapfrog / PML-fill-mode settings no longer resolve."""
    with pytest.raises(ConfigError):
        resolve_config(removed)


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


def test_observed_data_satisfy_inverse_operator_on_truth():
    """Observations are a Helmholtz solve on the truth model (inverse crime): the
    inversion's own PDE residual vanishes at (observed u, truth c)."""
    problem = build_problem(resolve_config(TINY_GRID))
    band = problem.make_band([40e3])
    wf = Wavefield(problem.grid, band.freq, velocity_model=problem.truth_velocity)
    loss = InverseLoss(
        observed_wavefield=band.observed_wfs,
        config=LossConfig(
            WaveEquation(
                wf, space_order=problem.space_order, pml_weight=problem.pml_weight
            ),
            band.geom,
            weights={"pde": 1.0, "data": 1.0},
        ),
    )
    amp = torch.stack([w.amplitude for w in band.observed_wfs])
    loss.evaluate(amp, problem.truth_velocity.c)
    assert loss._last_residuals["pde_src_ratio"] < 1e-8
    assert loss._last_residuals["data_rms"] == 0.0


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


# --------------------------------------------------------------------------- #
# Objective weights (loss.weights is the single source)
# --------------------------------------------------------------------------- #
def test_default_loss_weights():
    """Default objective weights are pde 1, data 1, reg 0."""
    assert resolve_config({}).loss.weights == {"pde": 1.0, "data": 1.0, "reg": 0.0}


@pytest.mark.parametrize("key", ["data_weight", "reg_weight"])
def test_joint_weight_keys_are_rejected(key):
    """optimiser.joint.{data,reg}_weight moved to loss.weights and no longer resolve."""
    with pytest.raises(ConfigError, match="loss.weights"):
        resolve_config({"optimiser": {"joint": {key: 1.0}}})


@pytest.mark.parametrize(
    "weights",
    [
        {"pde": -1.0},
        {"data": -1.0},
        {"data": 0.0},
        {"reg": -1e-3},
        {"regs": 1.0},
    ],
)
def test_bad_loss_weights_rejected(weights):
    """Negative weights, data <= 0 and unknown weight keys raise ConfigError."""
    with pytest.raises(ConfigError):
        validate_config(resolve_config({"loss": {"weights": weights}}))


def test_pde_weight_zero_is_allowed():
    """pde = 0 is a valid (data-only) weighting."""
    validate_config(resolve_config({"loss": {"weights": {"pde": 0.0}}}))


@pytest.mark.parametrize("reg, warns", [(0.0, True), (1e-3, False)])
def test_inactive_regulariser_warning(reg, warns):
    """A named regulariser with loss.weights.reg <= 0 warns that it has no effect."""
    cfg = resolve_config(
        {"loss": {"weights": {"reg": reg}, "regulariser": {"name": "tikhonov"}}}
    )
    warned = any("no effect" in w for w in validate_config(cfg))
    assert warned is warns
