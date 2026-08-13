#!/usr/bin/env python3
"""One-seed Paper-A operator-Tucker harness on The Well validation data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from geoaware.tensor_bayes import OperatorBayesianCP, OperatorBayesianTucker
from geoaware.the_well_pilot import fixed_random_mask, load_the_well_case


def one_dimensional_basis(edge_weight: np.ndarray, modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(edge_weight)+1
    operator = np.zeros((n, n), dtype=np.float64)
    ids = np.arange(n-1)
    operator[ids, ids] += edge_weight
    operator[ids+1, ids+1] += edge_weight
    operator[ids, ids+1] -= edge_weight
    operator[ids+1, ids] -= edge_weight
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    return (torch.from_numpy(eigenvectors[:, :modes]).float(),
            torch.from_numpy(eigenvalues[:modes]).float())


def geometry_bases(speed: np.ndarray, time_count: int,
                   modes: tuple[int, int, int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    time_weight = np.ones(time_count-1, dtype=np.float64)
    conductance = np.maximum(speed.astype(np.float64), 2e-3)**2
    x_edges = np.sqrt(conductance[:-1]*conductance[1:]).mean(axis=1)
    y_edges = np.sqrt(conductance[:, :-1]*conductance[:, 1:]).mean(axis=0)
    pairs = [one_dimensional_basis(time_weight, modes[0]),
             one_dimensional_basis(x_edges, modes[1]),
             one_dimensional_basis(y_edges, modes[2])]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def metrics(target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
            held_out: torch.Tensor) -> dict:
    truth = target[held_out]
    prediction = mean[held_out]
    uncertainty = std[held_out].clamp_min(1e-7)
    error = prediction-truth
    rmse = error.square().mean().sqrt()
    return {
        "rmse": float(rmse),
        "nrmse_std": float(rmse/truth.std().clamp_min(1e-8)),
        "vrmse": float(rmse/truth.square().mean().sqrt().clamp_min(1e-8)),
        "coverage95": float((error.abs() <= 1.96*uncertainty).float().mean()),
        "width95": float(3.92*uncertainty.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/the_well_acoustic_64x64/validation/trajectory_000.npz"))
    parser.add_argument("--wrong-data", type=Path,
                        default=Path("data/the_well_acoustic_64x64/validation/trajectory_001.npz"))
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/the_well_paper_a_harness.json"))
    args = parser.parse_args()
    inputs, target_np = load_the_well_case(args.data)
    wrong_inputs, _ = load_the_well_case(args.wrong_data)
    target = torch.from_numpy(target_np)
    observed_np = fixed_random_mask(target_np.shape, args.ratio, args.seed)
    observed = torch.from_numpy(observed_np)
    held_out = ~observed
    observed_indices = torch.from_numpy(np.argwhere(observed_np)).long()
    all_indices = torch.cartesian_prod(
        torch.arange(target.shape[0]), torch.arange(target.shape[1]),
        torch.arange(target.shape[2]))
    center = target[observed].mean()
    scale = target[observed].std().clamp_min(1e-6)
    y = (target[observed]-center)/scale

    spectral_modes = (40, 32, 32)
    correct_basis, correct_eigen = geometry_bases(
        inputs["speed_of_sound"], target.shape[0], spectral_modes)
    wrong_basis, wrong_eigen = geometry_bases(
        wrong_inputs["speed_of_sound"], target.shape[0], spectral_modes)
    flat_speed = np.ones_like(inputs["speed_of_sound"])
    flat_basis, flat_eigen = geometry_bases(flat_speed, target.shape[0], spectral_modes)
    configurations = {
        "operator_tucker": ("tucker", correct_basis, correct_eigen),
        "wrong_operator_tucker": ("tucker", wrong_basis, wrong_eigen),
        "flat_operator_tucker": ("tucker", flat_basis, flat_eigen),
        "operator_cp": ("cp", correct_basis, correct_eigen),
    }
    rows = []
    for name, (kind, basis, eigenvalues) in configurations.items():
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
        if kind == "tucker":
            model = OperatorBayesianTucker(basis, eigenvalues, ranks=(6, 10, 10),
                                           power=1.5, device="cuda")
        else:
            # Rank 14 approximately matches the learned parameter count of the
            # 6x10x10 Tucker configuration.
            model = OperatorBayesianCP(basis, eigenvalues, rank=14, power=1.5,
                                       device="cuda")
        started = time.perf_counter()
        model.fit(observed_indices, y, steps=args.steps, lr=3e-3,
                  reg_weight=2e-3, seed=args.seed)
        prediction = model.predict(all_indices)
        mean = prediction.mean.reshape(target.shape)*scale+center
        std = prediction.std.reshape(target.shape)*scale
        row = {
            "model": name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds": time.perf_counter()-started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "effective_rank": prediction.effective_rank,
            "held_out_metrics": metrics(target, mean, std, held_out),
            "observed_metrics": metrics(target, mean, std, observed),
        }
        rows.append(row)
        print(f"{name}: held={row['held_out_metrics']['nrmse_std']:.4f} "
              f"observed={row['observed_metrics']['nrmse_std']:.4f} "
              f"time={row['elapsed_seconds']:.1f}s", flush=True)
        del model, prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "experiment_id": "A-METHOD-R5-WELLMAZE-01-RANDOM-HARNESS",
        "status": "SMOKE",
        "config": {**vars(args), "spectral_modes": spectral_modes,
                   "tucker_ranks": [6, 10, 10], "cp_rank": 14},
        "data_split": "validation only",
        "shape": list(target.shape),
        "observed": int(observed.sum()),
        "realized_ratio": float(observed.float().mean()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
