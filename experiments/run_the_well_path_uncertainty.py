#!/usr/bin/env python3
"""Validation-only stress test for uncertain intrinsic path geometry.

The robust variant marginalizes the paired CP predictor over a small posterior
on path length.  With ``path_sigma=0`` it is exactly the original paired model.
No test split is exposed by this script because this is a formulation-selection
experiment, not a new confirmation result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from geoaware.the_well_pilot import fixed_random_mask
from run_the_well_paper_b_harness import (
    PairedPhaseCP,
    case_features,
    evaluate,
    point_batch,
)


class PathMarginalizedPairedPhaseCP(PairedPhaseCP):
    """Paired CP mean under a discrete Gaussian path-length posterior."""

    def __init__(self, path_sigma: float, hidden: int = 64):
        super().__init__(hidden=hidden)
        # Four symmetric normal quantiles; normalized weights sum to one.
        self.path_sigma = float(path_sigma)
        self.register_buffer("quadrature_offsets", torch.tensor(
            [-1.5104176, -.4527800, .4527800, 1.5104176]))

    def forward(self, geometry, time_features_value, space):
        if self.path_sigma == 0.:
            return super().forward(geometry, time_features_value, space)
        predictions = []
        for offset in self.quadrature_offsets:
            perturbed = space.clone()
            perturbed[:, 5] = (space[:, 5] + offset*self.path_sigma).clamp_min(0.)
            predictions.append(super().forward(
                geometry, time_features_value, perturbed))
        return torch.stack(predictions).mean(0)


def perturb_case(case, sigma: float, random_seed: int):
    """Add a fixed, spatially correlated error to the estimated path map."""
    inputs, target, descriptor, spatial, euclidean = case
    if sigma == 0.:
        return case
    generator = np.random.default_rng(random_seed)
    error = gaussian_filter(generator.standard_normal(spatial.shape[:2]), sigma=2.)
    error = (error-error.mean())/(error.std()+1e-8)
    corrupted = spatial.copy()
    corrupted[..., 5] = np.maximum(0., corrupted[..., 5] + sigma*error)
    return inputs, target, descriptor, corrupted, euclidean


def fit_and_evaluate(name, factory, train_cases, evaluation_cases, args, device):
    training_batches = []
    for case_index, (inputs, target, descriptor, spatial, _) in enumerate(train_cases):
        mask = fixed_random_mask(target.shape, args.ratio, args.seed+case_index)
        indices = np.argwhere(mask)
        training_batches.append((
            point_batch(descriptor, spatial, inputs["query_times"], indices),
            target[mask],
        ))
    normalizer = np.std(np.concatenate([batch[1] for batch in training_batches]))+1e-6
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats()
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    generator = np.random.default_rng(args.seed)
    started = time.perf_counter(); model.train(); history = []
    for step in range(args.steps):
        case_index = int(generator.integers(len(training_batches)))
        features, target_values = training_batches[case_index]
        chosen = generator.integers(len(target_values), size=args.batch_size)
        tensors = [torch.from_numpy(value[chosen]).to(device) for value in features]
        target = torch.from_numpy(target_values[chosen]).to(device)
        prediction = model(*tensors)
        loss = ((prediction-target)/normalizer).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % max(1, args.steps//10) == 0 or step == args.steps-1:
            history.append({"step": step, "loss": float(loss.detach())})
    model.eval(); case_metrics = []
    for case in evaluation_cases:
        prediction = evaluate(model, case, device)
        target = torch.from_numpy(case[1]); error = prediction-target
        case_metrics.append({
            "nrmse_std": float(error.square().mean().sqrt()/target.std().clamp_min(1e-8)),
            "vrmse": float(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-8)),
        })
    return {
        "model": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter()-started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "evaluation_macro_nrmse": float(np.mean([row["nrmse_std"] for row in case_metrics])),
        "evaluation_macro_vrmse": float(np.mean([row["vrmse"] for row in case_metrics])),
        "case_metrics": case_metrics,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/the_well_acoustic_64x64"))
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--evaluation-cases", type=int, default=16)
    parser.add_argument("--time-limit", type=int, default=40)
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--path-sigma", type=float, required=True,
                        help="Std. dev. in normalized intrinsic-distance units.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_train = [case_features(args.data/"train"/f"trajectory_{i:03d}.npz",
                                 args.time_limit) for i in range(args.train_cases)]
    clean_validation = [case_features(
        args.data/"validation"/f"trajectory_{i:03d}.npz", args.time_limit)
        for i in range(args.evaluation_cases)]
    noisy_train = [perturb_case(case, args.path_sigma, 10000+args.seed*1000+i)
                   for i, case in enumerate(clean_train)]
    noisy_validation = [perturb_case(case, args.path_sigma, 20000+args.seed*1000+i)
                        for i, case in enumerate(clean_validation)]
    rows = []
    for name, factory, train, validation in [
        ("oracle_clean_paired", PairedPhaseCP, clean_train, clean_validation),
        ("naive_noisy_paired", PairedPhaseCP, noisy_train, noisy_validation),
        ("path_marginalized_paired",
         lambda: PathMarginalizedPairedPhaseCP(args.path_sigma),
         noisy_train, noisy_validation),
    ]:
        row = fit_and_evaluate(name, factory, train, validation, args, device)
        rows.append(row)
        print(f"{name}: validation macro NRMSE={row['evaluation_macro_nrmse']:.6f}",
              flush=True)
    payload = {
        "experiment_id": "B-WELL-EARLY40-PATH-UNCERTAINTY-VALIDATION",
        "status": "FORMULATION_SELECTION_VALIDATION_ONLY",
        "geometry_error": (
            "Fixed Gaussian-correlated error (correlation width 2 grid cells) "
            "added to each normalized intrinsic source-distance map."),
        "config": vars(args),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
