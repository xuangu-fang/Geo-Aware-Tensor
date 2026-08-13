#!/usr/bin/env python3
"""Paper-B cross-boundary/cross-resolution smoke using the frozen phase CP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch

from run_the_well_paper_b_harness import JointINR, NeuralCP, PairedPhaseCP, time_features


def fixed_mask(shape, ratio, seed):
    generator = np.random.default_rng(seed)
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    mask[generator.choice(len(mask), max(1, int(round(ratio*len(mask)))), replace=False)] = True
    return mask.reshape(shape)


def load_case(path: Path):
    payload = np.load(path)
    coords = payload["coordinates"].astype(np.float32)
    edges = payload["undirected_edges"].astype(np.int64)
    speed = payload["material_speed"].astype(np.float32)
    spacing = 2/(int(payload["fluid_mask"].shape[0])-1)
    edge_speed = .5*(speed[edges[:, 0]]+speed[edges[:, 1]])
    weights = spacing/np.maximum(edge_speed, 1e-3)
    graph = sp.coo_matrix((np.r_[weights, weights],
                           (np.r_[edges[:, 0], edges[:, 1]],
                            np.r_[edges[:, 1], edges[:, 0]])),
                          shape=(len(coords), len(coords))).tocsr()
    requested_source = payload["source_xy"].astype(np.float32)
    source_node = int(np.argmin(np.sum((coords-requested_source)**2, axis=1)))
    intrinsic = dijkstra(graph, directed=False, indices=source_node).astype(np.float32)
    euclidean = np.linalg.norm(coords-coords[source_node], axis=1).astype(np.float32)
    distance_scale = np.quantile(intrinsic[np.isfinite(intrinsic)], .95)+1e-8
    intrinsic /= distance_scale; euclidean /= distance_scale
    boundary_grid = payload["boundary_mask"].astype(bool)
    boundary = boundary_grid[tuple(payload["grid_indices"].T)].astype(np.float32)
    boundary_distance = payload["boundary_distance"].astype(np.float32)
    boundary_distance /= np.quantile(boundary_distance, .95)+1e-8
    fluid_fraction = float(payload["fluid_mask"].mean())
    boundary_fraction = float(boundary_grid.sum()/max(1, payload["fluid_mask"].sum()))
    hole_proxy = float(np.mean(boundary_distance < .12))
    descriptor = np.asarray([
        fluid_fraction, boundary_fraction, coords[:, 0].mean(), coords[:, 1].mean(),
        hole_proxy, requested_source[0], requested_source[1]], dtype=np.float32)
    # Keep the exact seven-dimensional interface used by the frozen The-Well
    # phase CP. Index 5 is the only correct/wrong path coordinate.
    spatial = np.stack([
        coords[:, 0], coords[:, 1], boundary_distance, speed, boundary,
        intrinsic, np.full(len(coords), fluid_fraction, np.float32)], axis=1)
    return {
        "name": path.stem,
        "target": payload["field"].astype(np.float32),
        "times": payload["record_times"].astype(np.float32),
        "descriptor": descriptor,
        "spatial": spatial,
        "wrong_distance": euclidean,
        "boundary": boundary.astype(bool),
    }


def point_batch(case, indices, wrong=False):
    time_index, node_index = indices.T
    spatial = case["spatial"][node_index].copy()
    if wrong:
        spatial[:, 5] = case["wrong_distance"][node_index]
    descriptor = np.broadcast_to(case["descriptor"],
                                 (len(indices), len(case["descriptor"]))).copy()
    return descriptor, time_features(case["times"])[time_index], spatial


@torch.no_grad()
def evaluate(model, case, device, wrong=False, chunk=65536):
    shape = case["target"].shape
    indices = np.indices(shape).reshape(2, -1).T
    predictions = []
    for start in range(0, len(indices), chunk):
        batch = point_batch(case, indices[start:start+chunk], wrong)
        tensors = [torch.from_numpy(value).to(device) for value in batch]
        predictions.append(model(*tensors).cpu())
    return torch.cat(predictions).reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/irregular_boundary_wave"))
    parser.add_argument("--split", type=Path,
                        default=Path("experiments/dataset_splits/irregular_boundary_wave_smoke.json"))
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/irregular_boundary_paper_b_smoke.json"))
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    train = [load_case(args.data/f"{name}_r24_s{source}.npz")
             for name in split["train_geometries"] for source in range(2)]
    # Configuration selection sees the validation geometry only. The hole-domain
    # test files remain untouched in this smoke.
    validation = [load_case(args.data/f"{name}_r32_s{source}.npz")
                  for name in split["validation_geometries"] for source in range(2)]
    batches = []
    for case_index, case in enumerate(train):
        mask = fixed_mask(case["target"].shape, args.ratio, args.seed+case_index)
        indices = np.argwhere(mask)
        batches.append((point_batch(case, indices), point_batch(case, indices, True),
                        case["target"][mask]))
    normalizer = np.std(np.concatenate([batch[2] for batch in batches]))+1e-6
    factories = {"paired_phase_cp": PairedPhaseCP, "wrong_distance_cp": PairedPhaseCP,
                 "neural_cp": NeuralCP, "joint_inr": JointINR}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, factory in factories.items():
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
        generator = np.random.default_rng(args.seed)
        started = time.perf_counter(); model.train()
        for _ in range(args.steps):
            case_index = int(generator.integers(len(batches)))
            correct, wrong, target = batches[case_index]
            chosen = generator.integers(len(target), size=args.batch_size)
            features = wrong if name == "wrong_distance_cp" else correct
            tensors = [torch.from_numpy(value[chosen]).to(device) for value in features]
            y = torch.from_numpy(target[chosen]).to(device)
            loss = ((model(*tensors)-y)/normalizer).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        model.eval(); metrics = []
        for case in validation:
            prediction = evaluate(model, case, device, wrong=name == "wrong_distance_cp")
            target = torch.from_numpy(case["target"])
            error = prediction-target
            boundary = torch.from_numpy(case["boundary"])[None].expand_as(target)
            metrics.append({
                "case": case["name"],
                "nrmse": float(error.square().mean().sqrt()/target.std().clamp_min(1e-8)),
                "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                        / target[boundary].std().clamp_min(1e-8)),
            })
        row = {
            "model": name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "validation_macro_nrmse": float(np.mean([x["nrmse"] for x in metrics])),
            "validation_boundary_nrmse": float(np.mean([x["boundary_nrmse"] for x in metrics])),
            "case_metrics": metrics,
            "elapsed_seconds": time.perf_counter()-started,
        }
        rows.append(row)
        print(f"{name}: validation={row['validation_macro_nrmse']:.4f} "
              f"boundary={row['validation_boundary_nrmse']:.4f}", flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result = {
        "experiment_id": "B-IRREGULAR-BOUNDARY-SMOKE-01",
        "status": "SMOKE_VALIDATION_ONLY",
        "config": vars(args),
        "train_cases": [case["name"] for case in train],
        "validation_cases": [case["name"] for case in validation],
        "test_geometries_read": [],
        "target_normalizer": float(normalizer),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
