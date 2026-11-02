# configs

- Top-level `*.yaml` are run configs: each must resolve and validate
  (`tests/test_experiments.py::test_shipped_run_configs_validate`).
- `experiments/`: experiment specs (`docs/experiments/README.md`).
- `checks/`: baselines for deterministic checks that are not run configs, e.g.
  the balance check's `balance.yaml`.
