#!/usr/bin/env python3
"""One-seed Paper-B phase-factor harness on unseen The Well geometry."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch import nn

from geoaware.the_well_pilot import fixed_random_mask, load_the_well_case


def mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, output_dim))


def time_features(times: np.ndarray) -> np.ndarray:
    normalized = (times-times.min())/(times.max()-times.min())
    bands = np.asarray([1., 2., 4., 8.], dtype=np.float32)*math.pi
    return np.concatenate([normalized[:, None], np.sin(normalized[:, None]*bands),
                           np.cos(normalized[:, None]*bands)], axis=1).astype(np.float32)


def source_distances(inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    initial = inputs["initial_pressure"]
    source = np.abs(initial) > .12*float(np.max(np.abs(initial)))
    nx, ny = initial.shape
    ids = np.arange(nx*ny).reshape(nx, ny)
    speed = np.maximum(inputs["speed_of_sound"].astype(np.float64), 2e-3)
    edge_i, edge_j, edge_w = [], [], []
    for a, b in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        mean_speed = 2/(1/speed.reshape(-1)[a.ravel()]+1/speed.reshape(-1)[b.ravel()])
        edge_i.extend(a.ravel()); edge_j.extend(b.ravel()); edge_w.extend(1/mean_speed)
    edge_i = np.asarray(edge_i); edge_j = np.asarray(edge_j); edge_w = np.asarray(edge_w)
    graph = sp.coo_matrix((np.r_[edge_w, edge_w],
                           (np.r_[edge_i, edge_j], np.r_[edge_j, edge_i])),
                          shape=(nx*ny, nx*ny)).tocsr()
    intrinsic = dijkstra(graph, directed=False, indices=np.flatnonzero(source),
                         min_only=True).reshape(nx, ny)
    intrinsic /= np.quantile(intrinsic[np.isfinite(intrinsic)], .95)+1e-8
    weights = np.abs(initial)*source
    grid_x, grid_y = np.meshgrid(inputs["x"], inputs["y"], indexing="ij")
    cx = float(np.sum(grid_x*weights)/(weights.sum()+1e-8))
    cy = float(np.sum(grid_y*weights)/(weights.sum()+1e-8))
    euclidean = np.sqrt((grid_x-cx)**2+(grid_y-cy)**2)
    euclidean /= np.quantile(euclidean, .95)+1e-8
    return intrinsic.astype(np.float32), euclidean.astype(np.float32)


def case_features(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    inputs, target = load_the_well_case(path)
    intrinsic, euclidean = source_distances(inputs)
    density = np.log10(np.maximum(inputs["density"], 1.)) / 6.
    initial_scale = np.std(inputs["initial_pressure"])+1e-6
    spatial = np.stack(list(np.meshgrid(inputs["x"], inputs["y"], indexing="ij")) + [
        density, inputs["speed_of_sound"], inputs["initial_pressure"]/initial_scale,
        intrinsic, (inputs["density"] > 1e5).astype(np.float32)], axis=-1).astype(np.float32)
    descriptor = np.asarray([
        np.mean(inputs["density"] > 1e5), np.mean(inputs["speed_of_sound"]),
        np.std(inputs["speed_of_sound"]), np.mean(inputs["initial_pressure"]),
        np.std(inputs["initial_pressure"]), np.max(np.abs(inputs["initial_pressure"])),
        np.mean(np.abs(inputs["initial_pressure"]) > .12*np.max(np.abs(inputs["initial_pressure"])))
    ], dtype=np.float32)
    return inputs, target, descriptor, spatial, euclidean


class NeuralCP(nn.Module):
    def __init__(self, rank=48, hidden=64):
        super().__init__(); self.rank = rank
        self.geometry = mlp(7, rank, hidden); self.time = mlp(9, rank, hidden)
        self.space = mlp(7, rank, hidden); self.weight = nn.Parameter(torch.ones(rank)/math.sqrt(rank))

    def forward(self, geometry, time, space):
        return (self.geometry(geometry)*self.time(time)*self.space(space)*self.weight).sum(1)


class JointINR(nn.Module):
    def __init__(self, hidden=96):
        super().__init__(); self.network = mlp(7+9+7, 1, hidden)

    def forward(self, geometry, time, space):
        return self.network(torch.cat([geometry, time, space], 1)).squeeze(1)


class PairedPhaseCP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.register_buffer("bands", torch.tensor([math.pi, 2*math.pi, 4*math.pi, 8*math.pi]))
        self.register_buffer("speeds", torch.tensor([.5, 1., 2.]))
        self.components = 4*len(self.bands)*len(self.speeds)
        self.geometry = mlp(7, self.components, hidden)
        self.time_amplitude = mlp(1, self.components, hidden)
        self.space_amplitude = mlp(7, self.components, hidden)
        self.weight = nn.Parameter(torch.ones(self.components)/math.sqrt(self.components))

    def forward(self, geometry, time_features_value, space):
        normalized_time = time_features_value[:, :1]
        distance = space[:, 5:6]
        kd = (distance[:, :, None]*self.bands[None, :, None]).expand(
            -1, -1, len(self.speeds)).reshape(len(space), -1)
        kt = (normalized_time[:, :, None]*self.bands[None, :, None]*
              self.speeds[None, None, :]).reshape(len(space), -1)
        sd, cd, st, ct = torch.sin(kd), torch.cos(kd), torch.sin(kt), torch.cos(kt)
        spatial_carrier = torch.stack([cd, sd, cd, sd], -1).reshape(len(space), -1)
        temporal_carrier = torch.stack([ct, st, st, ct], -1).reshape(len(space), -1)
        return (self.geometry(geometry)*self.time_amplitude(normalized_time)*temporal_carrier*
                self.space_amplitude(space)*spatial_carrier*self.weight).sum(1)


def point_batch(descriptor: np.ndarray, spatial: np.ndarray, times: np.ndarray,
                indices: np.ndarray, wrong_distance: np.ndarray | None = None):
    ti, xi, yi = indices.T
    selected_space = spatial[xi, yi].copy()
    if wrong_distance is not None:
        selected_space[:, 5] = wrong_distance[xi, yi]
    return (np.broadcast_to(descriptor, (len(indices), len(descriptor))).copy(),
            time_features(times)[ti], selected_space)


@torch.no_grad()
def evaluate(model, case, device, wrong=False, chunk=65536):
    inputs, target, descriptor, spatial, euclidean = case
    all_indices = np.indices(target.shape).reshape(3, -1).T
    predictions = []
    for start in range(0, len(all_indices), chunk):
        batch = point_batch(descriptor, spatial, inputs["query_times"],
                            all_indices[start:start+chunk], euclidean if wrong else None)
        tensors = [torch.from_numpy(value).to(device) for value in batch]
        predictions.append(model(*tensors).cpu())
    return torch.cat(predictions).reshape(target.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/the_well_acoustic_64x64"))
    parser.add_argument("--train-cases", type=int, default=8)
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/the_well_paper_b_harness.json"))
    args = parser.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cases = [case_features(args.data/"train"/f"trajectory_{i:03d}.npz")
                   for i in range(args.train_cases)]
    validation_cases = [case_features(args.data/"validation"/f"trajectory_{i:03d}.npz")
                        for i in range(2)]
    training_batches = []
    for case_index, (inputs, target, descriptor, spatial, euclidean) in enumerate(train_cases):
        mask = fixed_random_mask(target.shape, args.ratio, args.seed+case_index)
        indices = np.argwhere(mask)
        features = point_batch(descriptor, spatial, inputs["query_times"], indices)
        wrong_features = point_batch(descriptor, spatial, inputs["query_times"], indices, euclidean)
        training_batches.append((features, wrong_features, target[mask]))
    normalizer = np.std(np.concatenate([batch[2] for batch in training_batches]))+1e-6
    factories = {"paired_phase_cp": PairedPhaseCP, "wrong_distance_cp": PairedPhaseCP,
                 "neural_cp": NeuralCP, "joint_inr": JointINR}
    rows = []
    for name, factory in factories.items():
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
        generator = np.random.default_rng(args.seed)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model.train()
        for _ in range(args.steps):
            case_index = int(generator.integers(len(training_batches)))
            correct, wrong, target_values = training_batches[case_index]
            chosen = generator.integers(len(target_values), size=args.batch_size)
            feature_values = wrong if name == "wrong_distance_cp" else correct
            tensors = [torch.from_numpy(value[chosen]).to(device) for value in feature_values]
            y = torch.from_numpy(target_values[chosen]).to(device)
            prediction = model(*tensors)
            loss=((prediction-y)/normalizer).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        model.eval(); case_metrics=[]
        for case in validation_cases:
            prediction = evaluate(model, case, device, wrong=name == "wrong_distance_cp")
            target = torch.from_numpy(case[1])
            error = prediction-target
            case_metrics.append({
                "nrmse_std": float(error.square().mean().sqrt()/target.std().clamp_min(1e-8)),
                "vrmse": float(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-8)),
            })
        rows.append({
            "model": name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds": time.perf_counter()-started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "validation_macro_nrmse": float(np.mean([x["nrmse_std"] for x in case_metrics])),
            "validation_macro_vrmse": float(np.mean([x["vrmse"] for x in case_metrics])),
            "case_metrics": case_metrics,
        })
        print(f"{name}: validation macro NRMSE={rows[-1]['validation_macro_nrmse']:.4f}", flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    payload = {
        "experiment_id": "B-METHOD-R5-WELLMAZE-01-RANDOM-HARNESS",
        "status": "SMOKE", "data_split": "8 train trajectories; 2 validation; test untouched",
        "config": vars(args), "target_normalizer": float(normalizer), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
