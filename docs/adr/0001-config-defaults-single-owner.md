# Default config lives in code; every parameter has exactly one owner

The default config is defined only in code; there is no default YAML file. Run configs name only the values they change, and `main.py --print-defaults` prints the full default config. We chose this over keeping a commented default YAML as the source of truth because two copies of the defaults (YAML and dataclasses) had already drifted apart, and the dataclasses already give typing and validation.

Each parameter is owned by exactly one layer. A configured parameter is a **required** keyword-only argument of the library constructor it feeds iff no one value serves most calls (it defines the problem: grid size and extent, velocity bounds, `t_max`, source kind and `f0`, transducer count, stencil order, iteration budget, objective weights, regulariser); `RunConfig`'s default for it is the default problem's value. Otherwise it is **defaulted**: the library signature owns the literal, and `RunConfig` reads it with `library_default(cls, name)`. A parameter threaded through several constructors is owned by its innermost consumer, and outer signatures read it the same way (e.g. `damping_scale` from `WaveEquation`). A numerical-method detail that is not in `RunConfig` (e.g. `eps_u`, `z_scale`, `logit_clip`) keeps its library default; promoting one into the config means adding a `RunConfig` field that reads it. The config represents a value as the library does (`ring_center: null` means the interior centre; `Grid` takes `device` / `dtype` names).

Rejected alternatives:

- **A shared constants module** imported by both layers: it still leaves two places that look like defaults, and readers have to follow the constant to find the value.
- **A defaults YAML** read by both layers: untyped, hides the values from `help()`, and is the second copy this ADR removed.
- **The literal in both layers plus a parity test**: two copies again; the test only reports the drift.

## Consequences

- Library callers (tests, the low-level example notebook) pass only the parameters that define their problem, plus deliberate overrides.
- `odil_wave.experiment.config` imports the library to read its defaults.
