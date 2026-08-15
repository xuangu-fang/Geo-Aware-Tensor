#!/usr/bin/env python3
"""Track-4 R1: geometry-NO-conditioned CP under one-percent supervision.

The frozen test specifications are recorded in the manifest, but this script
deliberately refuses to materialize or evaluate test fields.
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

from geoaware.geometry_no_data import case_from_spec, write_protocol_manifest
from geoaware.geometry_no_tensor import (
    BoundaryOperatorFunctionalCP,
    CoordinateSDFFunctionalCP,
    CoordinateSDFPlusGeometryNOCP,
    GeometryNODenseHead,
    GeometryNOFunctionalCP,
    RankModulatedCoordinateCP,
    boundary_token_bundle,
    handcrafted_geometry_descriptor,
)


MODEL_FACTORIES = {
    "geometry_no_cp": lambda: GeometryNOFunctionalCP(
        rank=20, width=24, modes=8, masked=True, geometry_inputs="full"),
    "sdf_only_no_cp": lambda: GeometryNOFunctionalCP(
        rank=20, width=24, modes=8, masked=True, geometry_inputs="sdf_only"),
    "unmasked_geometry_no_cp": lambda: GeometryNOFunctionalCP(
        rank=20, width=24, modes=8, masked=False, geometry_inputs="full"),
    "geometry_no_dense_head": lambda: GeometryNODenseHead(
        latent=20, width=24, modes=8, masked=True, geometry_inputs="full"),
    "coordinate_sdf_cp": lambda: CoordinateSDFFunctionalCP(rank=20, hidden=48),
    "coordinate_only_cp": lambda: CoordinateSDFFunctionalCP(
        rank=20, hidden=48, use_sdf=False),
    "masked_geometry_no_residual": lambda: CoordinateSDFPlusGeometryNOCP(
        rank=20, hidden=48, width=24, modes=8, masked=True,
        initial_residual_gate=.01),
    "unmasked_geometry_no_residual": lambda: CoordinateSDFPlusGeometryNOCP(
        rank=20, hidden=48, width=24, modes=8, masked=False,
        initial_residual_gate=.01),
    "boundary_integral_cp": lambda: BoundaryOperatorFunctionalCP(
        rank=20, hidden=48, operator="integral", initial_gate=.05),
    "boundary_pooled_cp": lambda: BoundaryOperatorFunctionalCP(
        rank=20, hidden=48, operator="pooled", initial_gate=.05),
    "boundary_integral_outer_only_cp": lambda: BoundaryOperatorFunctionalCP(
        rank=20, hidden=48, operator="integral_outer_only", initial_gate=.05),
    "boundary_integral_wrong_type_cp": lambda: BoundaryOperatorFunctionalCP(
        rank=20, hidden=48, operator="integral_wrong_type", initial_gate=.05),
    "boundary_integral_no_sdf_cp": lambda: BoundaryOperatorFunctionalCP(
        rank=20, hidden=48, operator="integral", initial_gate=.05,
        use_sdf=False),
    "descriptor_rank_gated_cp": lambda: RankModulatedCoordinateCP(
        rank=20, hidden=48, conditioning="descriptor", set_width=16),
    "boundary_deepsets_rank_gated_cp": lambda: RankModulatedCoordinateCP(
        rank=20, hidden=48, conditioning="boundary", set_width=16),
    "wrong_boundary_rank_gated_cp": lambda: RankModulatedCoordinateCP(
        rank=20, hidden=48, conditioning="wrong_boundary", set_width=16),
}

DEFAULT_R1_MODELS = [
    "geometry_no_cp",
    "sdf_only_no_cp",
    "unmasked_geometry_no_cp",
    "geometry_no_dense_head",
    "coordinate_sdf_cp",
]


def fixed_observations(case: dict, ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(case["target"].shape)
    generator = np.random.default_rng(seed)
    count = max(1, int(round(ratio*np.prod(shape))))
    flat = generator.choice(np.prod(shape), count, replace=False)
    source, parameter, node = np.unravel_index(flat, shape)
    indices = torch.from_numpy(np.stack([source, parameter, node], 1)).long()
    values = case["target"][source, parameter, node]
    return indices, values


def load_split(manifest: dict, split: str, cache: Path,
               limit: int | None = None) -> list[dict]:
    if split == "test":
        raise ValueError("R1 forbids reading or materializing the frozen test split")
    specs = manifest["splits"][split]
    if limit is not None:
        specs = specs[:limit]
    cases = []
    cache.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(specs):
        path = cache / f"domain_{spec['id']}_r{manifest['resolution']}.pt"
        if path.exists():
            case = torch.load(path, map_location="cpu", weights_only=False)
        else:
            case = case_from_spec(spec, manifest["resolution"])
            torch.save(case, path)
        if "boundary_tokens" not in case:
            case["boundary_tokens"] = torch.from_numpy(boundary_token_bundle(
                case["mask"].numpy(), case["geometry"].numpy()))
        case["geometry_descriptor"] = torch.from_numpy(
            handcrafted_geometry_descriptor(case["mask"].numpy(),
                                             case["geometry"].numpy()))
        cases.append(case)
        if index == 0 or index + 1 == len(specs):
            print(f"loaded {split}: {index+1}/{len(specs)}", flush=True)
    # A deterministic cyclic mismatch is a causal control for whether a set
    # embedding exploits the boundary paired with this particular solution.
    for index, case in enumerate(cases):
        case["wrong_boundary_tokens"] = cases[(index+1) % len(cases)][
            "boundary_tokens"]
    return cases


@torch.no_grad()
def complete_observed_mse(model: nn.Module, observed: list[tuple],
                          center: torch.Tensor, scale: torch.Tensor,
                          device: torch.device) -> float:
    total, count = 0., 0
    model.eval()
    for case, indices, values in observed:
        prediction = model.forward_case(case, indices.to(device))
        target = ((values-center)/scale).to(device)
        total += float((prediction-target).square().sum())
        count += len(indices)
    model.train()
    return total/max(1, count)


@torch.no_grad()
def evaluate_case(model: nn.Module, case: dict, center: torch.Tensor,
                  scale: torch.Tensor, device: torch.device) -> dict:
    shape = tuple(case["target"].shape)
    indices = torch.cartesian_prod(*[torch.arange(length) for length in shape])
    prediction = model.forward_case(case, indices.to(device)).cpu()*scale + center
    prediction = prediction.reshape(shape)
    error = prediction-case["target"]
    boundary = case["boundary"][None, None].expand_as(error)
    return {
        "case": case["name"],
        "holes": len(case["spec"]["holes"]),
        "nrmse": float(error.square().mean().sqrt()
                       / case["target"].std().clamp_min(1e-8)),
        "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                / case["target"][boundary].std().clamp_min(1e-8)),
    }


def aggregate(metrics: list[dict]) -> dict:
    return {
        "macro_nrmse": float(np.mean([row["nrmse"] for row in metrics])),
        "macro_boundary_nrmse": float(np.mean(
            [row["boundary_nrmse"] for row in metrics])),
        "cases": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(
        "experiments/dataset_splits/track4_geometry_no_r1.json"))
    parser.add_argument("--cache", type=Path,
                        default=Path("data/track4_geometry_no_r1"))
    parser.add_argument("--output", type=Path, default=Path(
        "papers/four_tracks/results/track4_geometry_no_r1_seed0.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--cases-per-step", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_FACTORIES),
                        default=DEFAULT_R1_MODELS)
    args = parser.parse_args()

    if not 0 < args.ratio <= .1:
        raise ValueError("this sparse-label protocol requires ratio in (0, .1]")
    manifest = write_protocol_manifest(args.manifest)
    train = load_split(manifest, "train", args.cache, args.train_limit)
    id_validation = load_split(manifest, "id_validation", args.cache,
                               args.validation_limit)
    topology_validation = load_split(
        manifest, "topology_ood_validation", args.cache, args.validation_limit)
    observed = []
    for case_index, case in enumerate(train):
        indices, values = fixed_observations(
            case, args.ratio, 100_000*args.seed+case_index)
        observed.append((case, indices, values))
    all_values = torch.cat([values for _, _, values in observed])
    center, scale = all_values.mean(), all_values.std().clamp_min(1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for model_index, model_name in enumerate(args.models):
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = MODEL_FACTORIES[model_name]().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                      weight_decay=1e-5)
        # Every model receives the exact same sequence of cases.  This matters
        # when only 400 optimizer updates are available.
        generator = np.random.default_rng(70_000*args.seed+17)
        best = (float("inf"), None, -1)
        start = time.perf_counter()
        for step in range(args.steps):
            chosen_cases = generator.integers(len(observed),
                                              size=args.cases_per_step)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for case_index in chosen_cases:
                case, indices, values = observed[int(case_index)]
                prediction = model.forward_case(case, indices.to(device))
                target = ((values-center)/scale).to(device)
                losses.append((prediction-target).square().mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            if step % 50 == 0 or step+1 == args.steps:
                score = complete_observed_mse(
                    model, observed, center, scale, device)
                if score < best[0]:
                    best = (score, copy.deepcopy(model.state_dict()), step+1)
        model.load_state_dict(best[1])
        model.eval()
        id_metrics = [evaluate_case(model, case, center, scale, device)
                      for case in id_validation]
        topology_metrics = [evaluate_case(model, case, center, scale, device)
                            for case in topology_validation]
        row = {
            "model": model_name,
            "parameters": sum(parameter.numel()
                              for parameter in model.parameters()),
            "best_complete_observed_mse": best[0],
            "checkpoint_step": best[2],
            "id_validation": aggregate(id_metrics),
            "topology_ood_validation": aggregate(topology_metrics),
            "elapsed_seconds": time.perf_counter()-start,
        }
        if hasattr(model, "residual_gate"):
            row["learned_residual_gate"] = float(model.residual_gate.detach().cpu())
        if hasattr(model, "operator_gate"):
            row["learned_operator_gate"] = float(model.operator_gate.detach().cpu())
        if hasattr(model, "modulation_scale"):
            row["learned_modulation_scale"] = float(
                model.modulation_scale.detach().cpu())
        rows.append(row)
        print(f"{model_name}: ID={row['id_validation']['macro_nrmse']:.4f} "
              f"2-hole={row['topology_ood_validation']['macro_nrmse']:.4f}",
              flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mean_rows = {}
    for split_name, cases in (("id_validation", id_validation),
                              ("topology_ood_validation", topology_validation)):
        metrics = []
        for case in cases:
            error = center-case["target"]
            boundary = case["boundary"][None, None].expand_as(error)
            metrics.append({
                "case": case["name"],
                "holes": len(case["spec"]["holes"]),
                "nrmse": float(error.square().mean().sqrt()
                               / case["target"].std().clamp_min(1e-8)),
                "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                        / case["target"][boundary].std().clamp_min(1e-8)),
            })
        mean_rows[split_name] = aggregate(metrics)
    rows.append({"model": "observed_global_mean", "parameters": 0, **mean_rows})

    result = {
        "experiment_id": f"TRACK4-GEOMETRY-NO-R1-SEED{args.seed}",
        "status": "VALIDATION_ONLY_EARLY_POC",
        "decision_rule": "fixed 400-step budget; no test fields materialized or read",
        "protocol": {
            "train_geometries": len(train),
            "id_validation_geometries": len(id_validation),
            "topology_ood_validation_geometries": len(topology_validation),
            "train_holes": "0/1",
            "topology_ood_holes": 2,
            "observation_ratio": args.ratio,
            "observed_labels": sum(len(indices) for _, indices, _ in observed),
            "steps": args.steps,
            "cases_per_step": args.cases_per_step,
            "test_read": False,
            "batch_schedule": "identical across model variants",
            "manifest_sha256": manifest["spec_sha256"],
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
