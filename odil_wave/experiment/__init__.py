from .config import (
    ConfigError,
    DiagnosticsCfg,
    RunConfig,
    apply_overrides,
    canonical_optimiser_name,
    deep_merge,
    default_config_dict,
    load_config_file,
    resolve_config,
    validate_config,
)
from .diagnostics import DiagnosticsCollector
from .plots import (
    LiveVelocityView,
    animate_bands,
    load_band_velocities,
    load_config,
    load_final_velocity,
    load_loss_tape,
    load_metrics,
    plot_run_history,
    plot_velocity_recovery,
    rebuild_problem,
)
from .problem import BandContext, Problem, build_problem, grid_summary
from .recorder import FIELDNAMES, RecordingTape, RunRecorder
from .runner import (
    BandResult,
    RunResult,
    run_frequency_band,
    run_inverse,
    run_inverse_lbfgsb,
)

__all__ = [
    # config
    "ConfigError",
    "RunConfig",
    "DiagnosticsCfg",
    "resolve_config",
    "validate_config",
    "load_config_file",
    "default_config_dict",
    "apply_overrides",
    "deep_merge",
    "canonical_optimiser_name",
    # diagnostics
    "DiagnosticsCollector",
    # problem
    "Problem",
    "BandContext",
    "build_problem",
    "grid_summary",
    # recorder
    "RunRecorder",
    "RecordingTape",
    "FIELDNAMES",
    # plots / analysis
    "LiveVelocityView",
    "animate_bands",
    "load_band_velocities",
    "load_config",
    "load_final_velocity",
    "load_loss_tape",
    "load_metrics",
    "plot_run_history",
    "plot_velocity_recovery",
    "rebuild_problem",
    # runner
    "RunResult",
    "BandResult",
    "run_inverse",
    "run_inverse_lbfgsb",
    "run_frequency_band",
]
