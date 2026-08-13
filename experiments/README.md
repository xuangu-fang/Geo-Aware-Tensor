# Experiment operations

Research states, naming, gates, and milestones are defined in
[`papers/PROJECT_MANAGEMENT.md`](../papers/PROJECT_MANAGEMENT.md). Every new
experiment must be registered in `experiments/registry.json` before frozen
confirmation.

## Local environment

The current machine uses repository-local packages because system Python is
externally managed:

```bash
PYTHONPATH=.python-packages:src python3 -m pytest -q
```

`.python-packages/`, `runs/`, downloaded datasets, and checkpoints are ignored
by Git. On the current A100/CUDA-driver environment, recreate dependencies with
`python3 -m pip install --target .python-packages -r requirements-cu128.txt`;
do not commit the local 6+GB CUDA environment. The generic package metadata
keeps Torch below 2.9 because newer CUDA-13 wheels do not initialize with the
current 12.9-capable driver.

## Current reproducible entry points

```bash
# Paper A operator CP/Tucker
PYTHONPATH=.python-packages:src python3 experiments/run_tensor_bayes.py --help

# Paper B conditional/paired/envelope tensors
PYTHONPATH=.python-packages:src python3 experiments/paper_b_tensor_run.py --help

# Three-round aggregation and phase diagrams
PYTHONPATH=.python-packages:src python3 experiments/analyze_longterm_iterations.py

# Completed irregular-boundary NO-GO gates
PYTHONPATH=.python-packages:src python3 experiments/build_irregular_elliptic_dataset.py
PYTHONPATH=.python-packages:src python3 experiments/run_irregular_elliptic_paper_a.py
PYTHONPATH=.python-packages:src python3 experiments/run_irregular_elliptic_paper_b.py

# Official NeuralOperator 2.0 FNO/TFNO baseline and frozen aggregation
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_the_well_official_fno.py --help
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/analyze_the_well_official_fno.py
```

## Required output

Every paper-facing run must preserve:

- full CLI/configuration;
- dataset version and split manifest;
- per-seed metrics, not only averages;
- model parameter count, elapsed time, and peak GPU memory;
- pilot/selection/confirmation seed role;
- failure examples and negative results.
