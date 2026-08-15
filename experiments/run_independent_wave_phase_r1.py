#!/usr/bin/env python3
"""Locked validation-only R1 for the phase-factorized wave track.

This is deliberately a sparse-*training-label*, zero-shot geometry-transfer
task.  No validation observations are consumed and the locked test geometries
are never opened.  Checkpoint selection uses the loss on all observed training
labels, never validation metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch import nn

from geoaware.neural_tensor import paired_phase_carriers
from geoaware.phase_wave_protocol import absolute_wave_gate


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/independent_wave_smoke"
SPLIT = ROOT / "experiments/dataset_splits/independent_wave_smoke.json"
TRAIN_RESOLUTION = 24
VALIDATION_RESOLUTION = 32
SOURCES = (0, 1)
RATIO = 0.01
STEPS = 800
LEARNING_RATE = 2e-3


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, output_dim),
    )


def time_features(times: np.ndarray) -> np.ndarray:
    """Features for amplitudes; column zero is physical time in [0, 2]."""
    angular = 2 * math.pi * np.asarray([1., 2., 4., 8.], np.float32)
    return np.concatenate([
        times[:, None], np.sin(times[:, None] * angular),
        np.cos(times[:, None] * angular),
    ], axis=1).astype(np.float32)


def fixed_mask(shape: tuple[int, ...], ratio: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    flat[generator.choice(len(flat), max(1, round(ratio * len(flat))),
                          replace=False)] = True
    return flat.reshape(shape)


def load_case(path: Path) -> dict:
    with np.load(path) as payload:
        coords = payload["coordinates"].astype(np.float32)
        edges = payload["undirected_edges"].astype(np.int64)
        speed = payload["material_speed"].astype(np.float32)
        source_xy = payload["source_xy"].astype(np.float32)
        source_node = int(np.argmin(np.sum((coords-source_xy)**2, axis=1)))
        spacing = 2. / (payload["fluid_mask"].shape[0]-1)
        edge_speed = .5 * (speed[edges[:, 0]] + speed[edges[:, 1]])
        travel_weight = spacing / np.maximum(edge_speed, 1e-4)
        graph = sp.coo_matrix(
            (np.r_[travel_weight, travel_weight],
             (np.r_[edges[:, 0], edges[:, 1]],
              np.r_[edges[:, 1], edges[:, 0]])),
            shape=(len(coords), len(coords)),
        ).tocsr()
        travel_time = dijkstra(graph, directed=False,
                               indices=source_node).astype(np.float32)
        euclidean_time = (
            np.linalg.norm(coords-coords[source_node], axis=1)
            / max(float(np.mean(speed)), 1e-4)
        ).astype(np.float32)
        sdf = payload["signed_distance"].astype(np.float32)
        descriptor = np.asarray([
            float(payload["fluid_mask"].mean()), float(np.mean(sdf)),
            float(np.std(sdf)), float(np.mean(speed)), float(np.std(speed)),
            float(source_xy[0]), float(source_xy[1]),
        ], np.float32)
        relative = coords-coords[source_node]
        spatial = np.column_stack([
            coords, sdf, speed, travel_time, relative,
        ]).astype(np.float32)
        wrong_spatial = spatial.copy()
        wrong_spatial[:, 4] = euclidean_time
        return {
            "name": path.stem,
            "target": payload["field"].astype(np.float32),
            "times": payload["record_times"].astype(np.float32),
            "descriptor": descriptor,
            "spatial": spatial,
            "wrong_spatial": wrong_spatial,
            "boundary": sdf < 1.75 * spacing,
            "source_xy": source_xy,
        }


def point_features(case: dict, indices: np.ndarray,
                   wrong: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_index, node_index = indices.T
    geometry = np.broadcast_to(
        case["descriptor"], (len(indices), len(case["descriptor"]))).copy()
    temporal = time_features(case["times"])[time_index]
    spatial = (case["wrong_spatial"] if wrong else case["spatial"])[node_index]
    return geometry, temporal, spatial


class FunctionalCP(nn.Module):
    """Ordinary source-conditioned functional CP with no fixed phase carrier."""

    def __init__(self, rank: int = 36, hidden: int = 64):
        super().__init__()
        self.geometry = mlp(7, rank, hidden)
        self.time = mlp(9, rank, hidden)
        self.space = mlp(7, rank, hidden)
        self.weight = nn.Parameter(torch.ones(rank)/math.sqrt(rank))

    def forward(self, geometry, temporal, spatial):
        return (self.geometry(geometry) * self.time(temporal)
                * self.space(spatial) * self.weight).sum(1)


class JointINR(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.network = mlp(7+9+7, 1, hidden)

    def forward(self, geometry, temporal, spatial):
        return self.network(torch.cat([geometry, temporal, spatial], 1)).squeeze(1)


class SourceConditionedPairedPhase(nn.Module):
    """Physical-time paired CP for the independent Ricker-wave solver.

    Travel time already integrates inverse material speed, so the ideal
    characteristic is ``travel_time - t``.  The small speed grid tolerates
    discretization and reflected-path mismatch.  The 5 Hz source frequency is
    generator metadata; lower/upper side bands cover its Ricker bandwidth.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.register_buffer(
            "bands", 2*math.pi*torch.tensor([2.5, 5., 7.5]))
        self.register_buffer("speeds", torch.tensor([.85, 1., 1.15]))
        self.components = 4*len(self.bands)*len(self.speeds)
        self.geometry = mlp(7, self.components, hidden)
        self.time_amplitude = mlp(9, self.components, hidden)
        self.space_amplitude = mlp(7, self.components, hidden)
        self.weight = nn.Parameter(
            torch.ones(self.components)/math.sqrt(self.components))

    def forward(self, geometry, temporal, spatial):
        # temporal[:,0] is physical time; spatial[:,4] is physical travel time.
        space_carrier, time_carrier = paired_phase_carriers(
            spatial[:, 4:5], temporal[:, 0:1], self.bands, self.speeds)
        return (self.geometry(geometry)
                * self.time_amplitude(temporal) * time_carrier
                * self.space_amplitude(spatial) * space_carrier
                * self.weight).sum(1)


def tensorize(features, device):
    return tuple(torch.from_numpy(value).to(device) for value in features)


@torch.no_grad()
def predict(model, case: dict, device: torch.device,
            wrong: bool = False, chunk: int = 65536) -> torch.Tensor:
    indices = np.indices(case["target"].shape).reshape(2, -1).T
    output = []
    model.eval()
    for start in range(0, len(indices), chunk):
        batch = point_features(case, indices[start:start+chunk], wrong=wrong)
        output.append(model(*tensorize(batch, device)).cpu())
    return torch.cat(output).reshape(case["target"].shape)


def case_metrics(case: dict, prediction: torch.Tensor) -> dict:
    target = torch.from_numpy(case["target"])
    error = prediction-target
    scale = target.std().clamp_min(1e-8)
    late = torch.from_numpy(case["times"] >= 1.)[:, None].expand_as(target)
    boundary = torch.from_numpy(case["boundary"])[None].expand_as(target)
    return {
        "case": case["name"],
        "nrmse": float(error.square().mean().sqrt()/scale),
        "vrmse": float(error.square().mean().sqrt()
                       / target.square().mean().sqrt().clamp_min(1e-8)),
        "late_nrmse": float(error[late].square().mean().sqrt()
                            / target[late].std().clamp_min(1e-8)),
        "boundary_nrmse": float(error[boundary].square().mean().sqrt()
                                / target[boundary].std().clamp_min(1e-8)),
        "mse": float(error.square().mean()),
        "target_variance": float(target.var(unbiased=False)),
    }


def aggregate_metrics(cases: list[dict], predictions: list[torch.Tensor]) -> dict:
    target = torch.cat([torch.from_numpy(case["target"]).flatten()
                        for case in cases])
    prediction = torch.cat([value.flatten() for value in predictions])
    rmse = (prediction-target).square().mean().sqrt()
    return {
        "global_nrmse": float(rmse/target.std().clamp_min(1e-8)),
        "global_vrmse": float(rmse/target.square().mean().sqrt().clamp_min(1e-8)),
    }


def fit_full_observed(model, correct_features, wrong_features, target,
                      target_scale, wrong, steps, device):
    model.to(device)
    selected = wrong_features if wrong else correct_features
    x = tensorize(selected, device)
    y = torch.from_numpy(target).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    best_loss = math.inf
    best_step = -1
    best_state = None
    history = []
    for step in range(steps):
        model.train()
        prediction = model(*x)
        loss = ((prediction-y)/target_scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        # The serialized state and the selection loss must refer to the same
        # post-update parameters.  This extra full-observation forward is cheap
        # at 2,222 labels and keeps the checkpoint contract exact.
        model.eval()
        with torch.no_grad():
            selection_loss = ((model(*x)-y)/target_scale).square().mean()
        value = float(selection_loss)
        if value < best_loss:
            best_loss = value
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
        if step % 100 == 0 or step == steps-1:
            history.append({"step": step, "all_observed_train_loss": value})
    model.load_state_dict(best_state)
    return {
        "best_step": best_step,
        "best_all_observed_train_loss": best_loss,
        "history": history,
    }


def run(seed: int, output: Path, steps: int = STEPS) -> None:
    split = json.loads(SPLIT.read_text())
    train_paths = [
        DATA/f"{geometry}_r{TRAIN_RESOLUTION}_s{source}.npz"
        for geometry in split["train_geometries"] for source in SOURCES
    ]
    validation_paths = [
        DATA/f"{geometry}_r{VALIDATION_RESOLUTION}_s{source}.npz"
        for geometry in split["validation_geometries"] for source in SOURCES
    ]
    # The locked test geometry names are recorded but never converted to paths
    # or opened in this selection run.
    train = [load_case(path) for path in train_paths]
    validation = [load_case(path) for path in validation_paths]
    observed_correct, observed_wrong, observed_target = [], [], []
    realized = []
    for case_index, case in enumerate(train):
        mask = fixed_mask(case["target"].shape, RATIO, seed+1009*case_index)
        indices = np.argwhere(mask)
        observed_correct.append(point_features(case, indices, wrong=False))
        observed_wrong.append(point_features(case, indices, wrong=True))
        observed_target.append(case["target"][mask])
        realized.append(float(mask.mean()))
    correct = tuple(np.concatenate([value[i] for value in observed_correct])
                    for i in range(3))
    wrong = tuple(np.concatenate([value[i] for value in observed_wrong])
                  for i in range(3))
    target = np.concatenate(observed_target).astype(np.float32)
    scale = max(float(np.std(target)), 1e-6)
    train_mean = float(np.mean(target))
    factories = {
        "ordinary_functional_cp": FunctionalCP,
        "joint_inr": JointINR,
        "paired_phase_cp": SourceConditionedPairedPhase,
        "wrong_euclidean_phase": SourceConditionedPairedPhase,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    predictions_by_model = {}
    training = {}
    started = time.perf_counter()
    for name, factory in factories.items():
        seed_all(seed)
        model = factory()
        wrong_model = name == "wrong_euclidean_phase"
        training[name] = fit_full_observed(
            model, correct, wrong, target, scale, wrong_model, steps, device)
        predictions = [predict(model, case, device, wrong=wrong_model)
                       for case in validation]
        predictions_by_model[name] = predictions
        metrics = [case_metrics(case, prediction)
                   for case, prediction in zip(validation, predictions)]
        rows.append({
            "model": name,
            "parameters": sum(p.numel() for p in model.parameters()),
            **aggregate_metrics(validation, predictions),
            "macro_nrmse": float(np.mean([item["nrmse"] for item in metrics])),
            "macro_late_nrmse": float(np.mean(
                [item["late_nrmse"] for item in metrics])),
            "macro_boundary_nrmse": float(np.mean(
                [item["boundary_nrmse"] for item in metrics])),
            "case_metrics": metrics,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, value in (("zero", 0.), ("train_observed_mean", train_mean)):
        predictions = [torch.full_like(torch.from_numpy(case["target"]), value)
                       for case in validation]
        predictions_by_model[name] = predictions
        metrics = [case_metrics(case, prediction)
                   for case, prediction in zip(validation, predictions)]
        rows.append({
            "model": name, "parameters": 0,
            **aggregate_metrics(validation, predictions),
            "macro_nrmse": float(np.mean([item["nrmse"] for item in metrics])),
            "macro_late_nrmse": float(np.mean(
                [item["late_nrmse"] for item in metrics])),
            "macro_boundary_nrmse": float(np.mean(
                [item["boundary_nrmse"] for item in metrics])),
            "case_metrics": metrics,
        })
    trivial_nrmse = min(
        row["global_nrmse"] for row in rows
        if row["model"] in ("zero", "train_observed_mean"))
    for row in rows:
        if row["model"] not in ("zero", "train_observed_mean"):
            row["absolute_gate"] = absolute_wave_gate(
                row["global_nrmse"], trivial_nrmse).to_dict()
    result = {
        "experiment_id": "TRACK2-INDEPENDENT-WAVE-R1-LOCKED-VALIDATION",
        "status": "VALIDATION_ONLY",
        "task_semantics": (
            "1% sparse training-target supervision; source-conditioned "
            "zero-shot unseen-geometry and 24-to-32 resolution transfer; "
            "no validation context"),
        "checkpoint_rule": "minimum loss on all observed training labels",
        "config": {
            "seed": seed, "ratio": RATIO, "steps": steps,
            "learning_rate": LEARNING_RATE,
            "train_resolution": TRAIN_RESOLUTION,
            "validation_resolution": VALIDATION_RESOLUTION,
            "source_indices": list(SOURCES),
            "bands_hz": [2.5, 5., 7.5],
            "characteristic_speed_multipliers": [.85, 1., 1.15],
        },
        "train_cases": [case["name"] for case in train],
        "validation_cases": [case["name"] for case in validation],
        "locked_test_geometries": split["test_geometries"],
        "test_files_read": [],
        "n_observed_training_labels": int(len(target)),
        "realized_training_ratio_range": [min(realized), max(realized)],
        "target_scale_from_observed_train_only": scale,
        "strongest_trivial_global_nrmse": trivial_nrmse,
        "rows": rows,
        "training": training,
        "elapsed_seconds": time.perf_counter()-started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({row["model"]: {
        "global_nrmse": row["global_nrmse"],
        "late_nrmse": row["macro_late_nrmse"],
        "gate": row.get("absolute_gate", {}).get("passed"),
    } for row in rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="Engineering smoke override; scientific lock is 800.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.output, args.steps)


if __name__ == "__main__":
    main()
