# odil-wave

Frequency-domain acoustic **full-waveform inversion (FWI)** for transcranial
ultrasound, built on **ODIL** (*Optimizing a DIscrete Loss*). Instead of
eliminating the wavefield with a forward solve and an adjoint, ODIL keeps the
discrete wavefield `u` as an optimisation variable beside the velocity model
`c`, and minimises one objective: a soft PDE penalty, the data misfit at the
receivers and an optional regulariser. A ring of transducers images a
Shepp-Logan head phantom from synthetic data.

## Install

```bash
uv sync --extra dev        # or: pip install -e ".[dev]"
pre-commit install
```

Python ≥ 3.12; dependencies are in `pyproject.toml`, locked in `uv.lock`.

## Run

From the repository root:

```bash
python main.py --print-defaults                     # the default config, as YAML
python main.py -c configs/example_multiband.yaml    # a run config
python main.py --help                               # every flag
```

The **default config** is the `RunConfig` dataclass defaults in
`odil_wave/experiment/config.py`, whose docstrings document every field; a run
config names only the values it changes (ADR 0001). Runs, experiments and
their outputs: [`odil_wave/experiment/README.md`](odil_wave/experiment/README.md).
A walkthrough using the library directly:
`notebooks/odil_wave_inverse_example.ipynb`.

## Test

```bash
pytest -n auto
```

CI runs the tests and the pre-commit hooks (`.github/workflows/tests.yml`).

## Where things live

| path | contents |
|---|---|
| `odil_wave/grid`, `source`, `geometry`, `models`, `wavefield` | The discretised problem: grid and PML, source wavelets, the transducer ring, velocity models, wavefields |
| `odil_wave/operator` | Discrete physics: the wave-equation residual, Laplacian stencils, the Helmholtz solver |
| `odil_wave/band.py` | One band of the inversion, as the joint solve receives it |
| `odil_wave/optimisation` | `JointFreqODIL`, the joint L-BFGS solve, and its preconditioners |
| `odil_wave/loss`, `metrics` | Regularisers, the loss-history plot, SSIM / error metrics |
| `odil_wave/experiment` | The reproducible run layer behind `main.py` |
| `configs/` | Run configs, experiment specs and check baselines ([README](configs/README.md)) |
| `docs/experiments/` | The experiment bank: one record per experiment |
| `docs/adr/` | Architecture decisions |
| `research/` | Literature notes behind open questions |
| `tests/` | The test suite, one file per module or feature |

## Further reading

- [`GLOSSARY.md`](GLOSSARY.md): the domain terms used in code and docs.
- [`docs/adr/`](docs/adr/): why the code is shaped the way it is.
- [`CODING_STANDARDS.md`](CODING_STANDARDS.md): review standards and what a merge-ready PR is.
- [`docs/experiments/`](docs/experiments/README.md): how experiments are recorded.
