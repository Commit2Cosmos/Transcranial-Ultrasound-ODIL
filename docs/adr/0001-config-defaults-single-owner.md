# Default config lives in code; every parameter has exactly one owner

The default config is defined only by the typed `RunConfig` dataclass defaults; there is no default YAML file. Run configs name only the values they change, and `main.py --print-defaults` prints the full default config. We chose this over keeping a commented default YAML as the source of truth because two copies of the defaults (YAML and dataclasses) had already drifted apart, and the dataclasses already give typing and validation.

Each parameter is owned by exactly one layer. A parameter that appears in `RunConfig` is a required keyword-only argument of the library constructor it feeds (`Grid`, `SourceSignal`, `AcquisitionGeometry`, `JointFreqODIL`, ...), so the library holds no competing default. A numerical-method detail that is not in `RunConfig` (e.g. `eps_u`, `z_scale`, `logit_clip`) keeps its library default and is never exposed in the config. A shared constants module imported by both layers was rejected: it still leaves two places that look like defaults, and readers have to follow the constant to find the value.

## Consequences

- Library callers (tests, the low-level example notebook) must pass every configured parameter explicitly.
- No backwards compatibility: configs or run directories that use removed or renamed keys fail with an unknown-key error.
