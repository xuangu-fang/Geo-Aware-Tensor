#!/usr/bin/env python3
"""Paper-A method-matched gate on smooth irregular-domain elliptic tensors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from geoaware.tensor_bayes import OperatorBayesianCP, OperatorBayesianTucker
from run_irregular_boundary_paper_a import (
    cosine_basis,
    fixed_mask,
    graph_basis,
    rectangle_basis,
)


def mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, output_dim))


class CoordinateConditionalCP(nn.Module):
    """Strong functional CP baseline with optional boundary-distance input."""
    def __init__(self, rank=10, hidden=48, use_sdf=True):
        super().__init__(); self.use_sdf = use_sdf
        self.parameter_factor = mlp(2, rank, hidden)
        self.space_factor = mlp(7, rank, hidden)
        self.weight = nn.Parameter(torch.ones(rank)/math.sqrt(rank))

    def forward(self, indices, parameter_values, coords, boundary_distance, source_xy):
        source, parameter, node = indices.T
        values = parameter_values[parameter]
        parameter_feature = torch.stack([torch.log(values), values], 1)
        xy = coords[node]; src = source_xy[source]
        distance = torch.linalg.vector_norm(xy-src, dim=1, keepdim=True)
        sdf = boundary_distance[node, None] if self.use_sdf else torch.zeros_like(distance)
        spatial_feature = torch.cat([xy, sdf, src, distance, torch.ones_like(distance)], 1)
        return (self.parameter_factor(parameter_feature)*self.space_factor(spatial_feature)
                *self.weight).sum(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/irregular_boundary_elliptic"))
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--space-modes", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/irregular_elliptic_paper_a_1pct.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data/"manifest.json").read_text())
    names = sorted({case["geometry"]["name"] for case in manifest["cases"]})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for geometry_index, name in enumerate(names):
        payload = np.load(args.data/f"{name}_r24.npz")
        target = torch.from_numpy(payload["field"].astype(np.float32))
        shape = target.shape
        observed_np = fixed_mask(shape, args.ratio, args.seed+geometry_index)
        observed = torch.from_numpy(observed_np)
        indices = torch.from_numpy(np.argwhere(observed_np)).long()
        center = target[observed].mean(); scale = target[observed].std().clamp_min(1e-6)
        values = (target[observed]-center)/scale
        initial = torch.zeros_like(target); initial[observed] = values
        source_basis = torch.eye(shape[0]); source_eigen = torch.zeros(shape[0])
        parameter_basis, parameter_eigen = cosine_basis(shape[1], shape[1])
        correct_basis, correct_eigen = graph_basis(payload, args.space_modes)
        flat_basis, flat_eigen = rectangle_basis(payload["coordinates"], args.space_modes)
        permutation = torch.randperm(shape[2], generator=torch.Generator().manual_seed(2000+args.seed))
        configs = {
            "correct_tucker": ("tucker", correct_basis, correct_eigen),
            "wrong_boundary_tucker": ("tucker", correct_basis[permutation], correct_eigen),
            "topology_erased_tucker": ("tucker", flat_basis, flat_eigen),
            "correct_cp": ("cp", correct_basis, correct_eigen),
            "wrong_boundary_cp": ("cp", correct_basis[permutation], correct_eigen),
            "topology_erased_cp": ("cp", flat_basis, flat_eigen),
        }
        all_indices = torch.cartesian_prod(*[torch.arange(n) for n in shape])
        boundary_nodes = torch.from_numpy(payload["boundary_mask"].astype(bool)[
            tuple(payload["grid_indices"].T)])
        held = ~observed; boundary_eval = held & boundary_nodes[None, None, :]
        for model_name, (kind, spatial_basis, spatial_eigen) in configs.items():
            torch.manual_seed(args.seed)
            basis = [source_basis, parameter_basis, spatial_basis]
            eigenvalues = [source_eigen, parameter_eigen, spatial_eigen]
            if kind == "tucker":
                model = OperatorBayesianTucker(
                    basis, eigenvalues, ranks=(3, 6, 8), power=1.5, device=device)
            else:
                # Rank 10 is slightly larger in factor capacity while retaining
                # the CP super-diagonal core restriction.
                model = OperatorBayesianCP(
                    basis, eigenvalues, rank=10, power=1.5, device=device)
            started = time.perf_counter()
            model.fit(indices, values, steps=args.steps, lr=3e-3,
                      reg_weight=2e-3, seed=args.seed, initial_tensor=initial)
            prediction = model.predict(all_indices).mean.reshape(shape)*scale+center
            error = prediction[held]-target[held]
            row = {
                "geometry": name,
                "model": model_name,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "held_nrmse": float(error.square().mean().sqrt()/target[held].std().clamp_min(1e-8)),
                "boundary_nrmse": float((prediction[boundary_eval]-target[boundary_eval]).square().mean().sqrt()
                                          / target[boundary_eval].std().clamp_min(1e-8)),
                "elapsed_seconds": time.perf_counter()-started,
            }
            rows.append(row)
            print(f"{name} {model_name}: held={row['held_nrmse']:.4f} "
                  f"boundary={row['boundary_nrmse']:.4f}", flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        parameter_values = torch.from_numpy(payload["diffusivities"].astype(np.float32)).to(device)
        coords = torch.from_numpy(payload["coordinates"].astype(np.float32)).to(device)
        boundary_distance = torch.from_numpy(payload["boundary_distance"].astype(np.float32)).to(device)
        boundary_distance /= torch.quantile(boundary_distance, .95).clamp_min(1e-8)
        source_xy = torch.from_numpy(payload["source_xy"].astype(np.float32)).to(device)
        for model_name, use_sdf in (("sdf_coordinate_cp", True), ("coordinate_cp", False)):
            torch.manual_seed(args.seed)
            model = CoordinateConditionalCP(use_sdf=use_sdf).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
            query = indices.to(device); observed_values = values.to(device)
            started = time.perf_counter(); best = (float("inf"), None)
            for _ in range(args.steps):
                prediction = model(query, parameter_values, coords, boundary_distance, source_xy)
                loss = (prediction-observed_values).square().mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                score = float(loss.detach())
                if score < best[0]:
                    best = (score, {key: value.detach().cpu().clone()
                                    for key, value in model.state_dict().items()})
            model.load_state_dict(best[1]); model.eval()
            predictions = []
            with torch.no_grad():
                for start in range(0, len(all_indices), 65536):
                    predictions.append(model(
                        all_indices[start:start+65536].to(device), parameter_values,
                        coords, boundary_distance, source_xy).cpu())
            prediction = torch.cat(predictions).reshape(shape)*scale+center
            error = prediction[held]-target[held]
            row = {
                "geometry": name,
                "model": model_name,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "held_nrmse": float(error.square().mean().sqrt()/target[held].std().clamp_min(1e-8)),
                "boundary_nrmse": float((prediction[boundary_eval]-target[boundary_eval]).square().mean().sqrt()
                                          / target[boundary_eval].std().clamp_min(1e-8)),
                "elapsed_seconds": time.perf_counter()-started,
            }
            rows.append(row)
            print(f"{name} {model_name}: held={row['held_nrmse']:.4f} "
                  f"boundary={row['boundary_nrmse']:.4f}", flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = {}
    for model_name in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model_name]
        summary[model_name] = {
            "macro_nrmse": float(np.mean([row["held_nrmse"] for row in selected])),
            "macro_boundary_nrmse": float(np.mean([row["boundary_nrmse"] for row in selected])),
            "wins_vs_correct_tucker": sum(
                row["held_nrmse"] < next(candidate["held_nrmse"] for candidate in rows
                                           if candidate["geometry"] == row["geometry"]
                                           and candidate["model"] == "correct_tucker")
                for row in selected) if model_name != "correct_tucker" else None,
        }
    result = {
        "experiment_id": f"A-IRREGULAR-ELLIPTIC-{100*args.ratio:.0f}BP-SEED{args.seed}",
        "status": "METHOD_MATCHED_GATE",
        "config": vars(args),
        "tensor_semantics": ["source", "diffusivity", "irregular-domain node"],
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
