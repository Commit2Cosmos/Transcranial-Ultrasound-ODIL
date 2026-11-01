import json
from pathlib import Path

import pytest

from odil_wave import InverseLoss, LossConfig, WaveEquation, Wavefield
import torch
from odil_wave.experiment import (
    ConfigError,
    RunConfig,
    apply_overrides,
    build_problem,
    deep_merge,
    default_config_dict,
    load_config_file,
    resolve_config,
    validate_config,
)

# The config shape fixed by ADR 0001 / issue #39: block -> field names, in order.
EXPECTED_SHAPE = {
    "run": ["seed", "tags", "notes"],
    "runtime": ["device", "dtype", "torch_num_threads", "deterministic", "verbose"],
    "grid": ["interior_shape", "interior_extent", "c_min", "c_max", "space_order"],
    "pml": ["width", "power", "R0", "damping_scale", "observation_width"],
    "time": ["t_max", "init_nt", "cfl_safety"],
    "source": ["kind", "f0", "amplitude", "n_cycles", "envelope", "dimensionless"],
    "acquisition": [
        "n_transducers",
        "a_frac",
        "b_frac",
        "ring_center",
        "sigma_s",
        "source_spatial",
    ],
    "truth": [
        "profile",
        "scale",
        "base",
        "contrast",
        "pml_c",
        "skull_alpha",
        "skull_sigma",
        "extra",
    ],
    "init": [
        "profile",
        "scale",
        "base",
        "contrast",
        "pml_c",
        "skull_alpha",
        "skull_sigma",
        "extra",
    ],
    "continuation": ["bands"],
    "loss": ["weights", "regulariser"],
    "optimiser": [
        "n_iter",
        "log_every",
        "inner_max_iter",
        "history_size",
        "line_search_fn",
        "lbfgs_lr",
        "tolerance_grad",
        "tolerance_change",
    ],
    "metrics": ["ssim"],
}


# --------------------------------------------------------------------------- #
# Default config / resolution / round-trip
# --------------------------------------------------------------------------- #
def test_default_config_matches_the_documented_shape():
    """RunConfig() has exactly the blocks and fields of the documented shape."""
    d = RunConfig().to_dict()
    assert list(d) == list(EXPECTED_SHAPE)
    for block, names in EXPECTED_SHAPE.items():
        assert list(d[block]) == names, block
    assert list(d["continuation"]["bands"][0]) == ["frequencies_hz", "n_iter"]
    assert list(d["loss"]["weights"]) == ["pde", "data", "reg"]
    assert list(d["loss"]["regulariser"]) == ["name", "eps"]
    assert list(d["metrics"]["ssim"]) == ["sigma"]


def test_resolve_empty_config_gives_the_default_values():
    """resolve_config({}) is the default config: 64 transducers, one 60 kHz band,
    and observations on the inversion grid."""
    cfg = resolve_config({})
    assert cfg.acquisition.n_transducers == 64
    assert [b.frequencies_hz for b in cfg.continuation.bands] == [[60e3]]
    assert cfg.pml.observation_width is None


def test_resolve_config_overlays_user_fields():
    """User fields override the defaults; untouched fields keep their defaults."""
    cfg = resolve_config({"grid": {"interior_shape": [16, 16]}})
    assert cfg.grid.interior_shape == [16, 16]
    assert cfg.pml.width == 40  # default preserved


def test_default_runconfig_dict_roundtrip():
    """RunConfig() round-trips through to_dict / from_dict."""
    assert RunConfig.from_dict(RunConfig().to_dict()) == RunConfig()


def test_default_runconfig_yaml_roundtrip(tmp_path):
    """RunConfig() written to YAML reloads to the same config."""
    path = RunConfig().to_yaml(tmp_path / "cfg.yaml")
    assert RunConfig.from_dict(load_config_file(path)) == RunConfig()


def test_resolved_config_yaml_roundtrip(tmp_path, tiny_run_config):
    """A resolved config written to YAML reloads to the same config."""
    cfg = resolve_config(tiny_run_config)
    path = cfg.to_yaml(tmp_path / "cfg.yaml")
    assert resolve_config(load_config_file(path)) == cfg


def test_unknown_key_raises():
    """Unknown config keys are rejected during construction."""
    with pytest.raises(ConfigError):
        RunConfig.from_dict({"grid": {"not_a_field": 1}})


@pytest.mark.parametrize(
    "removed, key",
    [
        ({"optimiser": {"name": "joint"}}, "name"),
        ({"optimiser": {"joint": {}}}, "joint"),
        ({"optimiser": {"z_scale": 10.0}}, "z_scale"),
        ({"optimiser": {"eps_u": 1e-30}}, "eps_u"),
        ({"optimiser": {"c_min": 1400.0}}, "c_min"),
        ({"optimiser": {"verbose": True}}, "verbose"),
        ({"metrics": {"log_c_stats": True}}, "log_c_stats"),
        ({"loss": {"regulariser": {"params": {"eps": 1e-6}}}}, "params"),
        ({"physics": {"space_order": 2}}, "physics"),
        ({"observation": {"normalize_data": "none"}}, "observation"),
        ({"loss": {"normalize_data": "none"}}, "normalize_data"),
        ({"continuation": {"warm_start": "helmholtz"}}, "warm_start"),
        ({"grid": {"pml_width": 10}}, "pml_width"),
        ({"grid": {"t_max": 1e-4}}, "t_max"),
        ({"grid": {"L0": 1.0}}, "L0"),
        ({"source": {"offset": 0}}, "offset"),
        ({"source": {"t0": None}}, "t0"),
        ({"acquisition": {"n_receivers": 64}}, "n_receivers"),
        ({"acquisition": {"n_sources": 64}}, "n_sources"),
    ],
)
def test_removed_keys_raise_naming_the_key(removed, key):
    """Removed / moved keys are plain unknown keys: ConfigError names them."""
    with pytest.raises(ConfigError, match=rf"unknown key.*'{key}'") as exc_info:
        resolve_config(removed)
    assert "moved" not in str(exc_info.value)


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_transducers_is_rejected(n):
    """A ring needs another transducer to record each shot."""
    with pytest.raises(ConfigError, match="n_transducers"):
        validate_config(resolve_config({"acquisition": {"n_transducers": n}}))


@pytest.mark.parametrize(
    "override, key",
    [
        ("loss.normalize_data=none", "normalize_data"),
        ("continuation.warm_start=helmholtz", "warm_start"),
    ],
)
def test_removed_keys_as_cli_overrides_raise_naming_the_key(override, key):
    """A CLI override naming a removed key is rejected as an unknown key."""
    with pytest.raises(ConfigError, match=rf"unknown key.*'{key}'"):
        resolve_config(apply_overrides(default_config_dict(), [override]))


# --------------------------------------------------------------------------- #
# Merge / override helpers
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


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_config_accepts_default():
    """The default config validates without warnings."""
    assert validate_config(resolve_config({})) == []


@pytest.mark.parametrize("order", [2, 4, 6, 8, 10])
def test_supported_space_orders_validate(order):
    """Every supported stencil order (including 10) is accepted."""
    validate_config(resolve_config({"grid": {"space_order": order}}))


@pytest.mark.parametrize(
    "bad",
    [
        {"optimiser": {"n_iter": 0}},
        {"optimiser": {"log_every": 0}},
        {"optimiser": {"inner_max_iter": 0}},
        {"optimiser": {"history_size": 0}},
        {"optimiser": {"line_search_fn": "backtracking"}},
        {"continuation": {"bands": []}},
        {"continuation": {"bands": [{"frequencies_hz": [-1.0]}]}},
        {"metrics": {"ssim": {"sigma": 0.0}}},
        {"metrics": {"ssim": {"win_size": 7}}},
        {"runtime": {"dtype": "float16"}},
        {"grid": {"space_order": 3}},
        {"grid": {"space_order": 12}},
        {"pml": {"width": -1}},
        {"pml": {"damping_scale": -1.0}},
        {"pml": {"observation_width": -1}},
        {"loss": {"regulariser": {"name": None}}},
        {"loss": {"regulariser": {"name": "tv"}}},
        {"loss": {"regulariser": {"name": "l2"}}},
        {"truth": {"profile": "skull"}},
        {"init": {"profile": "skull"}},
        {"init": {"extra": {"skull_smooth": 1.0}}},
    ],
)
def test_validate_config_rejects_bad(bad):
    """Representative malformed configs raise ConfigError."""
    with pytest.raises(ConfigError):
        validate_config(resolve_config(bad))


@pytest.mark.parametrize(
    "removed",
    [
        {"optimiser": {"lbfgsb": {}}},
        {"optimiser": {"u_steps": 5}},
        {"optimiser": {"clamp": True}},
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
    """lbfgsb / PML-fill-mode / diagnostics settings no longer resolve."""
    with pytest.raises(ConfigError):
        resolve_config(removed)


# --------------------------------------------------------------------------- #
# Problem construction
# --------------------------------------------------------------------------- #
def test_build_problem_assembles_pieces(tiny_run_config):
    """build_problem constructs the grid, source and truth/init velocity models."""
    problem = build_problem(resolve_config(tiny_run_config))
    assert problem.grid.interior_shape == (20, 20)
    assert problem.grid.pml_width == 6
    assert problem.truth_velocity.c_max == pytest.approx(1700.0)  # the skull
    assert problem.init_velocity.c_max == pytest.approx(1500.0)  # homogeneous
    # the Shepp-Logan truth has a head mask on the interior.
    assert problem.head_mask.shape == (20, 20) and problem.head_mask.any()


def test_make_band_produces_observed_data(tiny_run_config):
    """make_band builds the frequency selection, geometry and observed wavefields."""
    problem = build_problem(resolve_config(tiny_run_config))
    band = problem.make_band([40e3])
    assert band.freq.n_frequencies == 1
    assert len(band.observed_wfs) == problem.cfg.acquisition.n_transducers
    # not PML-split, so full wavefields (no reduced traces).
    assert band.observed_traces is None


def test_pml_split_traces_skip_the_transmitting_transducer(tiny_run_config):
    """Reduced observations hold N - 1 traces per shot, one per other transducer."""
    split = deep_merge(tiny_run_config, {"pml": {"observation_width": 12}})
    problem = build_problem(resolve_config(split))
    band = problem.make_band([40e3])
    n = problem.cfg.acquisition.n_transducers
    assert band.observed_traces.shape == (n, 1, n - 1)


def test_observed_data_satisfy_inverse_operator_on_truth(tiny_run_config):
    """Observations are a Helmholtz solve on the truth model (inverse crime): the
    inversion's own PDE residual vanishes at (observed u, truth c)."""
    problem = build_problem(resolve_config(tiny_run_config))
    band = problem.make_band([40e3])
    wf = Wavefield(problem.grid, band.freq, velocity_model=problem.truth_velocity)
    loss = InverseLoss(
        observed_wavefield=band.observed_wfs,
        config=LossConfig(
            WaveEquation(
                wf,
                space_order=problem.space_order,
                damping_scale=problem.damping_scale,
            ),
            band.geom,
            weights={"pde": 1.0, "data": 1.0, "reg": 0.0},
            regulariser=None,
        ),
    )
    amp = torch.stack([w.amplitude for w in band.observed_wfs])
    loss.evaluate(amp, problem.truth_velocity.c)
    assert loss.last_residuals["pde_src_ratio"] < 1e-8
    assert loss.last_residuals["data_rms"] == 0.0


def _amplitudes(wavefields) -> torch.Tensor:
    """Stack per-shot wavefields to ``(n_shots, nf, nx, ny)``."""
    return torch.stack([w.amplitude for w in wavefields])


def test_observed_data_are_reciprocal(tiny_run_config):
    """d[s -> r] ~= d[r -> s]: guards the per-shot receiver bookkeeping."""
    problem = build_problem(resolve_config(tiny_run_config))
    band = problem.make_band([40e3])
    traces = band.geom.extract_observations(_amplitudes(band.observed_wfs))
    n = band.geom.n_transducers
    d = torch.full((n, n), float("nan"), dtype=traces.dtype)
    for s in range(n):
        d[s, band.geom.receivers[s]] = traces[s, 0]
    off = ~torch.eye(n, dtype=torch.bool)
    rel = (d - d.T)[off].abs().max() / d[off].abs().max()
    # Measured 1.1e-15 on this point-source problem; an index slip is O(1).
    assert float(rel) < 1e-10


def _data_misfit_at_truth(run_config) -> float:
    """Relative receiver misfit between the observations and a Helmholtz solve
    on the true model on the inversion grid."""
    problem = build_problem(resolve_config(run_config))
    band = problem.make_band([40e3])
    observed = band.observed_traces
    if observed is None:
        observed = band.geom.extract_observations(_amplitudes(band.observed_wfs))
    predicted = band.geom.extract_observations(
        _amplitudes(problem.warm_start(problem.truth_velocity, band))
    )
    return float((predicted - observed).abs().norm() / observed.abs().norm())


def test_default_observation_width_commits_the_inverse_crime(tiny_run_config):
    """With the default pml.observation_width the true model fits the data."""
    assert _data_misfit_at_truth(tiny_run_config) < 1e-10


def test_different_observation_width_is_a_modelling_error(tiny_run_config):
    """A different observation PML width means the true model no longer fits."""
    split = deep_merge(tiny_run_config, {"pml": {"observation_width": 12}})
    assert _data_misfit_at_truth(split) > 1e-6


def test_warm_start_solves_one_wavefield_per_shot(tiny_run_config):
    """warm_start always solves Helmholtz wavefields, one per shot."""
    problem = build_problem(resolve_config(tiny_run_config))
    band = problem.make_band([40e3])
    wfs = problem.warm_start(problem.init_velocity, band)
    assert len(wfs) == problem.cfg.acquisition.n_transducers


# --------------------------------------------------------------------------- #
# Objective weights (loss.weights is the single source)
# --------------------------------------------------------------------------- #
def test_default_loss_weights():
    """Default objective weights are pde 1, data 1, reg 0."""
    assert resolve_config({}).loss.weights == {"pde": 1.0, "data": 1.0, "reg": 0.0}


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


@pytest.mark.parametrize("missing", ["pde", "data", "reg"])
def test_every_loss_weight_is_required(missing):
    """loss.weights must name pde, data and reg (no implicit reg = 0)."""
    weights = {k: 1.0 for k in ("pde", "data", "reg") if k != missing}
    with pytest.raises(ConfigError, match=missing):
        validate_config(resolve_config({}, [f"loss.weights={json.dumps(weights)}"]))


def test_pde_weight_zero_is_allowed():
    """pde = 0 is a valid (data-only) weighting."""
    validate_config(resolve_config({"loss": {"weights": {"pde": 0.0}}}))


@pytest.mark.parametrize("reg", [0.0, 1e-3])
def test_regulariser_choice_never_warns(reg):
    """The regulariser is inactive iff loss.weights.reg == 0; no warning either way."""
    cfg = resolve_config(
        {"loss": {"weights": {"reg": reg}, "regulariser": {"name": "tikhonov"}}}
    )
    assert validate_config(cfg) == []


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_shipped_run_configs_validate(path):
    """Every run config in configs/ resolves and validates cleanly."""
    assert validate_config(resolve_config(load_config_file(path))) == []
