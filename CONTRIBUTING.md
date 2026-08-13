# Contributing experiments

This repository is a research codebase with frozen statistical claims. Changes
to methods, data, baselines, or evaluation must follow
[`papers/PROJECT_MANAGEMENT.md`](papers/PROJECT_MANAGEMENT.md).

Before opening a pull request:

1. Register a falsifiable experiment ID or link an existing GitHub issue.
2. Keep pilot, selection, and confirmation seeds disjoint.
3. Never use held-out target values for geometry construction, normalization,
   hyperparameter selection, calibration, or early stopping.
4. Preserve per-seed results and negative outcomes.
5. Run:

   ```bash
   PYTHONPATH=.python-packages:src python3 -m pytest -q
   PYTHONPATH=.python-packages:src python3 -m compileall -q src experiments
   ```

6. State the paper claim, figure, table, limitation, or reviewer risk affected.

Large data, `runs/`, checkpoints, and local environments must not be committed.
Publish dataset split manifests and checksums instead.
