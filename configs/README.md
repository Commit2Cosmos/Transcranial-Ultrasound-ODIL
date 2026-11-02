# configs

- Top-level `*.yaml` are run configs: each must resolve and validate
  (`tests/test_experiments.py::test_shipped_run_configs_validate`).
- `experiments/`: experiment specs (`docs/experiments/README.md`).
- `checks/`: baselines for deterministic checks that are not run configs: the
  balance check's `balance.yaml`, and `tiny.yaml`, the test suite's tiny run
  and the run comparison's default (`python -m odil_wave.experiment.compare`).
