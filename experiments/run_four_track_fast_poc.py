#!/usr/bin/env python3
"""Shared fast POC for the two new tracks on unseen irregular domains.

Train geometries use the 24x24 background mesh and validation/test geometries
use 32x32.  Only a fixed low-ratio entry mask from each training geometry is
seen.  This simultaneously probes sparse reconstruction, unseen shapes, holes,
and cross-resolution evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from geoaware.domain_kernels import matern_domain_kernel_sections
from geoaware.functional_tucker import (
    DomainKernelFunctionalTucker,
    GeometryConditionedNeuralFunctionalCP,
    GeometryConditionedNeuralFunctionalTucker,
    scalar_parameter_features,
    sdf_query_features,
)
from run_irregular_elliptic_paper_b import case_payload, fixed_mask


class JointSDFINR(nn.Module):
    """Monolithic neural regression baseline with exactly the same inputs."""

    def __init__(self, hidden: int = 96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        geometry = case["descriptor"].to(indices.device)[None].expand(len(indices), -1)
        parameter = scalar_parameter_features(case, indices)
        spatial = sdf_query_features(case, indices, use_sdf=True)
        return self.network(torch.cat([geometry, parameter, spatial], 1)).squeeze(1)


def load_case(path: Path, representation: str, modes: int, seed: int) -> dict:
    case = case_payload(path, representation, modes, seed)
    case["domain_kernel_features"] = matern_domain_kernel_sections(
        case["basis"], case["eigenvalues"], case["source_nodes"])
    return case


def all_indices(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.cartesian_prod(*[torch.arange(length) for length in shape])


@torch.no_grad()
def predict_case(model: nn.Module, case: dict, device: torch.device,
                 chunk: int = 65536) -> torch.Tensor:
    indices = all_indices(tuple(case["target"].shape))
    predictions = []
    for start in range(0, len(indices), chunk):
        predictions.append(model.forward_case(
            case, indices[start:start + chunk].to(device)).cpu())
    return torch.cat(predictions).reshape(case["target"].shape)


def model_factories() -> dict[str, tuple[callable, str]]:
    """Frozen POC configurations; validation data never choose model size."""
    return {
        "domain_kernel_functional_tucker": (
            lambda: DomainKernelFunctionalTucker(
                kernel_channels=5, ranks=(6, 8, 12), hidden=64), "correct"),
        "topology_erased_kernel_tucker": (
            lambda: DomainKernelFunctionalTucker(
                kernel_channels=5, ranks=(6, 8, 12), hidden=64), "bbox"),
        "sdf_neural_functional_tucker": (
            lambda: GeometryConditionedNeuralFunctionalTucker(
                ranks=(6, 8, 16), hidden=64, use_sdf=True), "correct"),
        "coordinate_neural_functional_tucker": (
            lambda: GeometryConditionedNeuralFunctionalTucker(
                ranks=(6, 8, 16), hidden=64, use_sdf=False), "correct"),
        "sdf_neural_functional_cp": (
            lambda: GeometryConditionedNeuralFunctionalCP(
                rank=24, hidden=64, use_sdf=True), "correct"),
        "joint_sdf_inr": (lambda: JointSDFINR(hidden=96), "correct"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--split", type=Path, default=Path(
        "experiments/dataset_splits/irregular_boundary_wave_smoke.json"))
    parser.add_argument("--evaluation-split", choices=("validation", "test"),
                        default="validation")
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path(
        "papers/four_tracks/results/new_tracks_validation_seed0.json"))
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    evaluation_names = split[f"{args.evaluation_split}_geometries"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for model_index, (model_name, (factory, representation)) in enumerate(
            model_factories().items()):
        train = [load_case(args.data / f"{name}_r24.npz", representation,
                           args.modes, 3000 + args.seed + index)
                 for index, name in enumerate(split["train_geometries"])]
        evaluation = [load_case(args.data / f"{name}_r32.npz", representation,
                                args.modes, 4000 + args.seed + index)
                      for index, name in enumerate(evaluation_names)]
        observed_batches = []
        for case_index, case in enumerate(train):
            mask = fixed_mask(case["target"].shape, args.ratio,
                              args.seed + case_index)
            observed_batches.append((
                case,
                torch.from_numpy(np.argwhere(mask)).long(),
                case["target"][mask],
            ))
        all_observed = torch.cat([values for _, _, values in observed_batches])
        center = all_observed.mean()
        scale = all_observed.std().clamp_min(1e-6)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                      weight_decay=2e-5)
        generator = np.random.default_rng(args.seed)
        best = (float("inf"), None)
        started = time.perf_counter()
        model.train()
        for _ in range(args.steps):
            case, indices, values = observed_batches[
                int(generator.integers(len(observed_batches)))]
            chosen = torch.from_numpy(generator.integers(
                len(indices), size=args.batch_size)).long()
            query = indices[chosen].to(device)
            target = ((values[chosen] - center) / scale).to(device)
            prediction = model.forward_case(case, query)
            loss = (prediction - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            score = float(loss.detach())
            if score < best[0]:
                best = (score, copy.deepcopy(model.state_dict()))
        model.load_state_dict(best[1])
        model.eval()

        case_metrics = []
        for case in evaluation:
            prediction = predict_case(model, case, device) * scale + center
            target = case["target"]
            error = prediction - target
            boundary = case["boundary"][None, None, :].expand_as(target)
            case_metrics.append({
                "case": case["name"],
                "nrmse": float(error.square().mean().sqrt()
                               / target.std().clamp_min(1e-8)),
                "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                        / target[boundary].std().clamp_min(1e-8)),
            })
        row = {
            "model": model_name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_loss": best[0],
            "evaluation_macro_nrmse": float(np.mean(
                [metric["nrmse"] for metric in case_metrics])),
            "evaluation_boundary_nrmse": float(np.mean(
                [metric["boundary_nrmse"] for metric in case_metrics])),
            "case_metrics": case_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(f"{model_name}: nrmse={row['evaluation_macro_nrmse']:.4f} "
              f"boundary={row['evaluation_boundary_nrmse']:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # A constant predictor is an absolute-skill check, not a trainable model.
    reference_cases = [load_case(args.data / f"{name}_r32.npz", "correct",
                                 args.modes, 9000 + index)
                       for index, name in enumerate(evaluation_names)]
    reference_metrics = []
    for case in reference_cases:
        target = case["target"]
        error = center - target
        boundary = case["boundary"][None, None, :].expand_as(target)
        reference_metrics.append({
            "case": case["name"],
            "nrmse": float(error.square().mean().sqrt()
                           / target.std().clamp_min(1e-8)),
            "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                    / target[boundary].std().clamp_min(1e-8)),
        })
    rows.append({
        "model": "observed_global_mean",
        "parameters": 0,
        "evaluation_macro_nrmse": float(np.mean(
            [metric["nrmse"] for metric in reference_metrics])),
        "evaluation_boundary_nrmse": float(np.mean(
            [metric["boundary_nrmse"] for metric in reference_metrics])),
        "case_metrics": reference_metrics,
        "elapsed_seconds": 0.,
    })

    result = {
        "experiment_id": (
            f"FOUR-TRACK-POC-{args.evaluation_split.upper()}-"
            f"{100 * args.ratio:.0f}BP-SEED{args.seed}"),
        "status": "FAST_POC_NOT_FINAL_EVIDENCE",
        "protocol": {
            "train_resolution": 24,
            "evaluation_resolution": 32,
            "train_geometries": split["train_geometries"],
            "evaluation_geometries": evaluation_names,
            "observation_ratio": args.ratio,
            "configuration_rule": "all architectures frozen before evaluation",
        },
        "config": vars(args),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str),
                           encoding="utf-8")


if __name__ == "__main__":
    main()

