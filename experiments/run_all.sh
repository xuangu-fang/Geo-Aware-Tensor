#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GEO_TENSOR_PYTHON:-/home/ubuntu/project/yanjiu/.venv/bin/python}"
export PYTHONPATH="${REPO_DIR}/src"
cd "${REPO_DIR}"

"${PYTHON_BIN}" experiments/run_poc.py --dataset synthetic_boundary \
  --models cp,inr,neural_cp,spectral_cp,wrong_spectral_cp,bayesian_spectral_tensor,geo_nft \
  --ratios 0.005,0.01,0.05 --masks random --seeds 0,1,2 --rank 4 --hidden 32 \
  --steps 1600 --reg-weight 0.05 --noise-std 0.1 --output runs/final_boundary

"${PYTHON_BIN}" experiments/run_poc.py --dataset synthetic_boundary \
  --models inr,neural_cp,spectral_cp,wrong_spectral_cp,bayesian_spectral_tensor,geo_nft \
  --ratios 0.05 --masks periodic_gap --seeds 0,1,2 --rank 4 --hidden 32 \
  --steps 1600 --reg-weight 0.05 --noise-std 0.1 --output runs/final_boundary_gap

"${PYTHON_BIN}" experiments/run_poc.py --dataset synthetic_wave \
  --models cp,inr,neural_cp,spectral_cp,wrong_spectral_cp,bayesian_spectral_tensor,geo_nft \
  --ratios 0.005,0.01,0.05 --masks random --seeds 0,1,2 --rank 4 --hidden 32 \
  --steps 1800 --reg-weight 0.05 --noise-std 0.1 --output runs/final_synthetic

"${PYTHON_BIN}" experiments/run_poc.py --dataset active_matter \
  --models cp,inr,neural_cp,spectral_cp,bayesian_spectral_tensor,geo_nft \
  --ratios 0.005,0.01,0.05 --masks random --seeds 0,1,2 --rank 8 --hidden 64 \
  --steps 2200 --reg-weight 0.01 --noise-std 0.1 --output runs/final_active

"${PYTHON_BIN}" experiments/run_poc.py --dataset realpde_cylinder \
  --models cp,inr,neural_cp,spectral_cp,bayesian_spectral_tensor,geo_nft \
  --ratios 0.01,0.05 --masks random --seeds 0,1,2 --rank 8 --hidden 64 \
  --steps 2200 --reg-weight 0.01 --noise-std 0.1 --output runs/final_realpde

"${PYTHON_BIN}" experiments/aggregate_results.py runs/final_boundary \
  runs/final_boundary_gap runs/final_synthetic runs/final_active runs/final_realpde \
  --output reports/results
