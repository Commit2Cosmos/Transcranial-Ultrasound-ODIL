import json
from pathlib import Path

import numpy as np
import pytest

import torch
from odil_wave.experiment import (
    ConfigError,
    RunConfig,
    apply_overrides,
    check_constraints,
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
    "pml": ["width", "power", "R0", "damping_scale"],
    "time": ["t_max", "init_nt", "cfl_safety"],
    "source": ["kind", "f0", "amplitude", "n_cycles", "envelope"],
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
        "skull_alpha",
        "skull_sigma",
        "extra",
    ],
    "init": [
        "profile",
        "scale",
        "base",
        "contrast",
        "skull_alpha",
        "skull_sigma",
        "extra",
    ],
    "observation": [
        "mode",
        "refine",
        "space_order",
        "time_symbols",
        "pml_thickness_factor",
        "truth",
        "noise",
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
        "model_preconditioner",
        "wavefield_preconditioner",
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
    assert list(d["observation"]["noise"]) == ["snr_db", "seed"]


def test_resolve_empty_config_gives_the_default_values():
    """resolve_config({}) is the default config: 64 transducers, one 80 kHz band,
    a homogeneous initial model and inverse-crime observations."""
    cfg = resolve_config({})
    assert cfg.acquisition.n_transducers == 64
    assert [b.frequencies_hz for b in cfg.continuation.bands] == [[80e3]]
    assert cfg.init.profile == "homogeneous"
    assert cfg.observation.mode == "inverse_crime"


def test_resolve_config_overlays_user_fields():
    """User fields override the defaults; untouched fields keep their defaults."""
    cfg = resolve_config({"grid": {"interior_shape": [16, 16]}})
    assert cfg.grid.interior_shape == [16, 16]
    assert cfg.pml.width == 15


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
        ({"observation": {"normalize_data": "none"}}, "normalize_data"),
        ({"pml": {"observation_width": 12}}, "observation_width"),
        ({"loss": {"normalize_data": "none"}}, "normalize_data"),
        ({"continuation": {"warm_start": "helmholtz"}}, "warm_start"),
        ({"grid": {"pml_width": 10}}, "pml_width"),
        ({"grid": {"t_max": 1e-4}}, "t_max"),
        ({"grid": {"L0": 1.0}}, "L0"),
        ({"source": {"offset": 0}}, "offset"),
        ({"source": {"t0": None}}, "t0"),
        ({"acquisition": {"n_receivers": 64}}, "n_receivers"),
        ({"acquisition": {"n_sources": 64}}, "n_sources"),
        ({"truth": {"pml_c": None}}, "pml_c"),
        ({"init": {"pml_c": 1500.0}}, "pml_c"),
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
def test_validate_config_warns_on_the_defaults_failed_constraint_checks():
    """The default config validates; it fails only the dispersion and ring
    sampling constraint checks (#124), which warn."""
    warnings = validate_config(resolve_config({}))
    assert len(warnings) == 2
    assert warnings[0].startswith("numerical dispersion")
    assert warnings[1].startswith("ring sampling")


def test_a_failed_error_constraint_check_refuses_the_config():
    """dt = 500 us / 49 puts 1/(2 dt) = 49 kHz below the 80 kHz band."""
    with pytest.raises(ConfigError, match="temporal Nyquist"):
        validate_config(resolve_config({}, ["time.init_nt=50"]))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        # sigma_s / h = 0.50 on the default grid (h = 2.016 mm)
        (["acquisition.sigma_s=1e-3"], ["under-resolved"]),
        # k_max sigma_s = 1.93 at 80 kHz in 1300 m/s; a small ring keeps clear
        (
            [
                "acquisition.sigma_s=5e-3",
                "acquisition.a_frac=0.5",
                "acquisition.b_frac=0.5",
            ],
            ["low-pass"],
        ),
        # ring 6.25 mm from the PML < 3 sigma_s = 7.5 mm
        (["acquisition.a_frac=0.95"], ["within 3 sigma_s"]),
        # a point source has no width
        (["acquisition.sigma_s=1e-3", "acquisition.source_spatial=point"], []),
    ],
)
def test_source_width_warnings(overrides, expected):
    """Each Gaussian source-width warning fires in its condition only, naming
    sigma_s/h, k_max*sigma_s and the ring clearance."""
    warnings = validate_config(resolve_config({}, overrides))
    warnings = [w for w in warnings if w.startswith("acquisition.sigma_s")]
    assert len(warnings) == len(expected)
    for w, phrase in zip(warnings, expected):
        assert phrase in w
        assert "sigma_s/h =" in w and "k_max*sigma_s =" in w and "clearance" in w


@pytest.mark.parametrize("h_frac, warns", [(0.99, True), (1.0, False), (1.2, False)])
def test_under_resolved_bound_is_one_grid_spacing(h_frac, warns):
    """The under-resolved warning fires below 1.0 h (h = 2.016 mm on the
    default grid) and not at or above it."""
    h = 0.25 / 124  # the default grid spacing: 0.25 m over 125 nodes
    sigma = h_frac * h
    warnings = validate_config(resolve_config({}, [f"acquisition.sigma_s={sigma!r}"]))
    under = [w for w in warnings if "under-resolved" in w]
    assert len(under) == warns
    assert all("(< 1.0 h)" in w for w in under)


@pytest.mark.parametrize("value", ["null", "0", "-1e-3"])
def test_sigma_s_must_be_a_positive_length(value):
    with pytest.raises(ConfigError, match="sigma_s"):
        validate_config(resolve_config({}, [f"acquisition.sigma_s={value}"]))


@pytest.mark.parametrize("value", ["0", "-1e-4"])
def test_regulariser_eps_must_be_positive(value):
    with pytest.raises(ConfigError, match="regulariser.eps"):
        validate_config(resolve_config({}, [f"loss.regulariser.eps={value}"]))


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
        {"optimiser": {"model_preconditioner": "diagonal"}},
        {"optimiser": {"wavefield_preconditioner": "jacobi"}},
        {
            "optimiser": {"model_preconditioner": "pde_diagonal"},
            "loss": {"weights": {"pde": 0.0}},
        },
        {"continuation": {"bands": []}},
        {"continuation": {"bands": [{"frequencies_hz": [-1.0]}]}},
        {"metrics": {"ssim": {"sigma": 0.0}}},
        {"metrics": {"ssim": {"win_size": 7}}},
        {"runtime": {"dtype": "float16"}},
        {"grid": {"space_order": 3}},
        {"grid": {"space_order": 12}},
        {"pml": {"width": -1}},
        {"pml": {"damping_scale": -1.0}},
        {"observation": {"mode": "crime"}},
        {"observation": {"refine": 2}},
        {"observation": {"refine": 0}},
        {"observation": {"space_order": 3}},
        {"observation": {"time_symbols": "rk4"}},
        {"observation": {"pml_thickness_factor": 0.5}},
        {"observation": {"truth": "fine"}},
        {"observation": {"noise": {"seed": -1}}},
        {"observation": {"noise": {"snr_db": "30"}}},
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


def test_make_band_holds_traces_true_wavefields_and_warm_start(tiny_run_config):
    """The band holds N - 1 observed traces per shot, and one true wavefield and
    one warm start per shot on the inversion grid."""
    problem = build_problem(resolve_config(tiny_run_config))
    band = problem.make_band([40e3], problem.init_velocity)
    n = problem.cfg.acquisition.n_transducers
    assert band.frequency_selection.n_frequencies == 1
    assert band.observed_traces.shape == (n, 1, n - 1)
    assert len(band.true_wavefields) == n
    assert band.true_wavefields[0].grid is problem.grid
    assert band.u_init.shape == (n, 1, *problem.grid.shape)


@pytest.mark.parametrize("mode, kept", [("none", False), ("initial_model", True)])
def test_make_band_keeps_factors_for_the_wavefield_preconditioner(
    tiny_run_config, mode, kept
):
    cfg = deep_merge(tiny_run_config, {"optimiser": {"wavefield_preconditioner": mode}})
    problem = build_problem(resolve_config(cfg))
    band = problem.make_band([40e3], problem.init_velocity)
    assert (band.factors is not None) == kept


def test_observed_data_satisfy_inverse_operator_on_truth(tiny_run_config):
    """Observations are a Helmholtz solve on the truth model (inverse crime): the
    inversion's own PDE residual vanishes at (true wavefield, truth c)."""
    problem = build_problem(resolve_config(tiny_run_config))
    band = problem.make_band([40e3], problem.truth_velocity)
    amp = _amplitudes(band.true_wavefields)
    r_pde = band.wave_eq.residual(amp, problem.truth_velocity.c, band.sources)
    rms = lambda x: float(x.abs().square().mean().sqrt())  # noqa: E731
    assert rms(r_pde) < 1e-8 * rms(band.sources)
    r_data = band.geometry.extract_observations(amp) - band.observed_traces
    assert torch.equal(r_data, torch.zeros_like(r_data))


def _amplitudes(wavefields) -> torch.Tensor:
    """Stack per-shot wavefields to ``(n_shots, nf, nx, ny)``."""
    return torch.stack([w.amplitude for w in wavefields])


def test_observed_data_are_reciprocal(tiny_run_config):
    """d[s -> r] ~= d[r -> s]: guards the per-shot receiver bookkeeping."""
    problem = build_problem(resolve_config(tiny_run_config))
    band = problem.make_band([40e3], problem.init_velocity)
    traces = band.observed_traces
    n = band.geometry.n_transducers
    d = torch.full((n, n), float("nan"), dtype=traces.dtype)
    for s in range(n):
        d[s, band.geometry.receivers[s]] = traces[s, 0]
    off = ~torch.eye(n, dtype=torch.bool)
    rel = (d - d.T)[off].abs().max() / d[off].abs().max()
    # Measured 1.1e-15 on this point-source problem; an index slip is O(1).
    assert float(rel) < 1e-10


def _data_misfit_at_truth(run_config) -> float:
    """Relative receiver misfit between the observations and the warm start of
    a band started from the true model on the inversion grid."""
    problem = build_problem(resolve_config(run_config))
    band = problem.make_band([40e3], problem.truth_velocity)
    observed = band.observed_traces
    predicted = band.geometry.extract_observations(band.u_init)
    return float((predicted - observed).abs().norm() / observed.abs().norm())


def test_default_observation_mode_commits_the_inverse_crime(tiny_run_config):
    """With the default observation block the true model fits the data."""
    assert _data_misfit_at_truth(tiny_run_config) < 1e-10


# The independent observation path set up to reproduce the inversion's own
# discretisation: it guards the plumbing.
PLUMBING = {
    "observation": {
        "mode": "independent",
        "refine": 1,
        "space_order": 2,
        "time_symbols": "leapfrog",
        "truth": "inversion",
    }
}


def test_independent_observations_matching_the_inversion_reproduce_its_data(
    tiny_run_config,
):
    crime = build_problem(resolve_config(tiny_run_config)).observe([40e3])
    plumbing = build_problem(
        resolve_config(deep_merge(tiny_run_config, PLUMBING))
    ).observe([40e3])
    assert torch.equal(plumbing.traces, crime.traces)


INDEPENDENT = {"observation": {"mode": "independent"}}


@pytest.mark.parametrize("pml_thickness_factor", [1.0, 1.5])
def test_independent_observations_keep_the_transducer_positions(
    tiny_run_config, pml_thickness_factor
):
    run = deep_merge(
        tiny_run_config,
        {
            "observation": {
                "mode": "independent",
                "pml_thickness_factor": pml_thickness_factor,
            }
        },
    )
    problem = build_problem(resolve_config(run))
    geom = problem.observe([40e3]).geometry
    fine = problem.observation.observation_geometry(geom)
    assert fine.grid.dx == pytest.approx(geom.grid.dx / 3)
    for s in range(geom.n_transducers):
        assert fine.src_position(s) == pytest.approx(geom.src_position(s), abs=1e-12)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def test_independent_observations_converge_with_refinement(tiny_run_config):
    """Refining the observation grid from 3x to 5x moves the data far less than
    the inversion's own discretisation is off from them."""

    def traces(observation):
        run = deep_merge(tiny_run_config, {"observation": observation})
        return build_problem(resolve_config(run)).observe([40e3]).traces

    r3 = traces({"mode": "independent"})
    r5 = traces({"mode": "independent", "refine": 5})
    # Measured 2.3e-3 (r3 vs r5) and 2.2e-2 (inverse crime vs r5).
    assert _rel(r3, r5) < 5e-3
    assert _rel(traces({}), r5) > 1e-2


def test_independent_observations_of_a_homogeneous_truth_match_the_greens_function(
    tiny_run_config,
):
    """In a homogeneous medium the traces are the physical wavefield of the
    Gaussian source: ``S(ω) (-i/4) H0⁽²⁾(kr) exp(-k²σ²/4) / c²``."""
    from scipy.special import hankel2

    sigma, c = 4e-3, 1500.0
    run = deep_merge(
        tiny_run_config,
        {
            "truth": {"profile": "homogeneous", "base": c},
            "acquisition": {"source_spatial": "gaussian", "sigma_s": sigma},
            "observation": {"mode": "independent"},
        },
    )
    problem = build_problem(resolve_config(run))
    obs = problem.observe([40e3])
    geom = obs.geometry
    freq = geom.frequency_selection
    k = float(freq.omega[0]) / c
    spectrum = complex(problem.source.spectrum(freq)[0])
    pos = torch.tensor([geom.src_position(s) for s in range(geom.n_shots)])
    r = torch.linalg.vector_norm(pos[geom.receivers] - pos[:, None], dim=-1).numpy()
    green = spectrum * (-0.25j) * hankel2(0, k * r) * np.exp(-((k * sigma) ** 2) / 4)
    expected = torch.from_numpy(green / c**2)
    # Measured 3.4e-3; the inverse crime's own traces are 2.3e-2 off.
    assert _rel(obs.traces[:, 0], expected) < 1e-2


def _observe(run_config, frequencies_hz, observation):
    run = deep_merge(run_config, {"observation": observation})
    return build_problem(resolve_config(run)).observe(frequencies_hz)


def test_noise_sits_at_the_requested_snr_per_frequency(tiny_run_config):
    clean = _observe(tiny_run_config, [30e3, 40e3], {})
    noisy = _observe(tiny_run_config, [30e3, 40e3], {"noise": {"snr_db": 20.0}})
    assert noisy.noise_floor == pytest.approx(0.1)
    for f in range(2):
        # 56 complex traces per frequency: the realised level is within ~15%.
        assert _rel(noisy.traces[:, f], clean.traces[:, f]) == pytest.approx(
            0.1, rel=0.3
        )
    # Under the inverse crime the truth's own data misfit is the noise.
    assert noisy.rel_data_residual_truth == pytest.approx(0.1, rel=0.3)
    assert clean.noise_floor is None
    assert clean.rel_data_residual_truth == 0.0


def test_noise_is_seeded_per_frequency(tiny_run_config):
    noise = {"noise": {"snr_db": 20.0, "seed": 3}}
    pair = _observe(tiny_run_config, [30e3, 40e3], noise).traces
    alone = _observe(tiny_run_config, [40e3], noise).traces
    reseeded = _observe(
        tiny_run_config, [40e3], {"noise": {"snr_db": 20.0, "seed": 4}}
    ).traces
    # A frequency shared by two bands gets the same realisation.
    assert torch.equal(pair[:, 1], alone[:, 0])
    assert _rel(reseeded, alone) > 0.05


def test_independent_observations_report_their_modelling_error(tiny_run_config):
    obs = _observe(tiny_run_config, [40e3], {"mode": "independent"})
    # The inversion's own solve at the truth is ~2% off the independent data.
    assert 1e-2 < obs.rel_data_residual_truth < 5e-2
    assert obs.solve_s > 0


def test_native_truth_is_a_different_model_from_the_prolonged_one(
    tiny_run_config,
):
    """Rebuilding the phantom on the observation grid changes the data (11% on
    the tiny config), while the tape keeps scoring the inversion-grid truth."""
    native = deep_merge(
        tiny_run_config, {"observation": {"mode": "independent", "truth": "native"}}
    )
    a = _observe(tiny_run_config, [40e3], {"mode": "independent"}).traces
    b = build_problem(resolve_config(native))
    assert _rel(b.observe([40e3]).traces, a) > 5e-2
    crime = build_problem(resolve_config(tiny_run_config))
    assert torch.equal(b.truth_velocity.c, crime.truth_velocity.c)


def test_independent_observations_warn_about_a_point_source():
    run = {
        "observation": {"mode": "independent"},
        "acquisition": {"source_spatial": "point"},
    }
    assert any("point" in w for w in validate_config(resolve_config(run)))


def test_inverse_crime_warns_about_independent_only_keys():
    warnings = validate_config(resolve_config({"observation": {"refine": 5}}))
    assert any("observation.refine" in w for w in warnings)


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


def test_model_preconditioner_defaults_to_none_and_accepts_pde_diagonal():
    assert resolve_config({}).optimiser.model_preconditioner == "none"
    validate_config(
        resolve_config({"optimiser": {"model_preconditioner": "pde_diagonal"}})
    )


def test_wavefield_preconditioner_defaults_to_none_and_accepts_initial_model():
    assert resolve_config({}).optimiser.wavefield_preconditioner == "none"
    validate_config(
        resolve_config({"optimiser": {"wavefield_preconditioner": "initial_model"}})
    )


def test_pde_weight_zero_is_allowed():
    """pde = 0 is a valid (data-only) weighting."""
    validate_config(resolve_config({"loss": {"weights": {"pde": 0.0}}}))


@pytest.mark.parametrize("reg", [0.0, 1e-3])
def test_regulariser_choice_never_warns(reg):
    """The regulariser is inactive iff loss.weights.reg == 0; no warning either way."""
    cfg = resolve_config(
        {"loss": {"weights": {"reg": reg}, "regulariser": {"name": "tikhonov"}}}
    )
    assert validate_config(cfg) == validate_config(resolve_config({}))


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_shipped_run_configs_validate(path):
    """Every run config in configs/ resolves and validates, warning only for
    failed constraint checks (the default fails two until #138)."""
    cfg = resolve_config(load_config_file(path))
    failed = [c.message for c in check_constraints(cfg) if not c.ok]
    assert validate_config(cfg) == failed
