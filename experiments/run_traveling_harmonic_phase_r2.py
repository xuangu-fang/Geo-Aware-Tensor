#!/usr/bin/env python3
"""Locked R2 decision run for the wave-specific paired-phase track.

This is a validation-only, 1%-train-label, zero-shot geometry-transfer test on
an independently generated *true traveling harmonic*.  Generator frequencies
are irrational and are not read by the learner.  The paired model starts from
a broad off-grid trainable dictionary, shared unchanged by its Euclidean-path
control.  Failure of the absolute gate stops/downgrades Track 2.
"""

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

from geoaware.neural_tensor import paired_phase_carriers
from geoaware.phase_wave_protocol import absolute_wave_gate
from geoaware.traveling_harmonic_generator import generate_traveling_harmonic

# Reuse only protocol/training utilities and the two ordinary baselines.  The
# R2 target generator remains in a learner-free module above.
from run_independent_wave_phase_r1 import (
    DATA,
    LEARNING_RATE,
    RATIO,
    SOURCES,
    SPLIT,
    TRAIN_RESOLUTION,
    VALIDATION_RESOLUTION,
    FunctionalCP,
    JointINR,
    aggregate_metrics,
    case_metrics,
    fit_full_observed,
    fixed_mask,
    mlp,
    point_features,
    predict,
    seed_all,
)


STEPS = 500
# Chosen before looking at generator frequencies.  These are broad, off-grid
# initial centers and remain trainable; no generator metadata enters the model.
INITIAL_BANDS_HZ = (1.25, 2.75, 4.25, 5.75, 7.25, 8.75)


def load_case(path: Path) -> dict:
    """Read geometry/operator metadata, not the reflected-wave target field."""
    with np.load(path) as payload:
        coords = payload["coordinates"].astype(np.float32)
        edges = payload["undirected_edges"].astype(np.int64)
        speed = payload["material_speed"].astype(np.float32)
        source_xy = payload["source_xy"].astype(np.float32)
        source_node = int(np.argmin(np.sum((coords - source_xy) ** 2, axis=1)))
        spacing = 2.0 / (payload["fluid_mask"].shape[0] - 1)
        edge_speed = 0.5 * (speed[edges[:, 0]] + speed[edges[:, 1]])
        travel_weight = spacing / np.maximum(edge_speed, 1e-4)
        graph = sp.coo_matrix(
            (
                np.r_[travel_weight, travel_weight],
                (
                    np.r_[edges[:, 0], edges[:, 1]],
                    np.r_[edges[:, 1], edges[:, 0]],
                ),
            ),
            shape=(len(coords), len(coords)),
        ).tocsr()
        travel_time = dijkstra(graph, directed=False, indices=source_node).astype(
            np.float32
        )
        euclidean_time = (
            np.linalg.norm(coords - coords[source_node], axis=1)
            / max(float(np.mean(speed)), 1e-4)
        ).astype(np.float32)
        sdf = payload["signed_distance"].astype(np.float32)
        times = payload["record_times"].astype(np.float32)
        descriptor = np.asarray(
            [
                float(payload["fluid_mask"].mean()),
                float(np.mean(sdf)),
                float(np.std(sdf)),
                float(np.mean(speed)),
                float(np.std(speed)),
                float(source_xy[0]),
                float(source_xy[1]),
            ],
            np.float32,
        )
        relative = coords - coords[source_node]
        spatial = np.column_stack(
            [coords, sdf, speed, travel_time, relative]
        ).astype(np.float32)
        wrong_spatial = spatial.copy()
        wrong_spatial[:, 4] = euclidean_time
        target = generate_traveling_harmonic(
            travel_time, times, coords, source_xy
        )
        return {
            "name": path.stem,
            "target": target,
            "times": times,
            "descriptor": descriptor,
            "spatial": spatial,
            "wrong_spatial": wrong_spatial,
            "boundary": sdf < 1.75 * spacing,
            "source_xy": source_xy,
        }


class TrainablePairedPhase(nn.Module):
    """Paired CP with a broad trainable frequency dictionary and known τ-t."""

    def __init__(self, hidden: int = 72):
        super().__init__()
        self.bands = nn.Parameter(
            2 * math.pi * torch.tensor(INITIAL_BANDS_HZ, dtype=torch.float32)
        )
        self.components = 4 * len(INITIAL_BANDS_HZ)
        self.geometry = mlp(7, self.components, hidden)
        self.time_amplitude = mlp(9, self.components, hidden)
        self.space_amplitude = mlp(7, self.components, hidden)
        self.weight = nn.Parameter(
            torch.ones(self.components) / math.sqrt(self.components)
        )

    def forward(self, geometry, temporal, spatial):
        # A single known physical characteristic τ-t.  Only frequency is fit.
        speed = self.bands.new_ones(1)
        space_carrier, time_carrier = paired_phase_carriers(
            spatial[:, 4:5], temporal[:, 0:1], self.bands, speed
        )
        return (
            self.geometry(geometry)
            * self.time_amplitude(temporal)
            * time_carrier
            * self.space_amplitude(spatial)
            * space_carrier
            * self.weight
        ).sum(1)


def run(seed: int, output: Path, steps: int = STEPS) -> None:
    split = json.loads(SPLIT.read_text())
    train_paths = [
        DATA / f"{geometry}_r{TRAIN_RESOLUTION}_s{source}.npz"
        for geometry in split["train_geometries"]
        for source in SOURCES
    ]
    validation_paths = [
        DATA / f"{geometry}_r{VALIDATION_RESOLUTION}_s{source}.npz"
        for geometry in split["validation_geometries"]
        for source in SOURCES
    ]
    train = [load_case(path) for path in train_paths]
    validation = [load_case(path) for path in validation_paths]

    observed_correct, observed_wrong, observed_target = [], [], []
    realized = []
    for case_index, case in enumerate(train):
        mask = fixed_mask(case["target"].shape, RATIO, seed + 1009 * case_index)
        indices = np.argwhere(mask)
        observed_correct.append(point_features(case, indices, wrong=False))
        observed_wrong.append(point_features(case, indices, wrong=True))
        observed_target.append(case["target"][mask])
        realized.append(float(mask.mean()))
    correct = tuple(
        np.concatenate([value[index] for value in observed_correct])
        for index in range(3)
    )
    wrong = tuple(
        np.concatenate([value[index] for value in observed_wrong])
        for index in range(3)
    )
    target = np.concatenate(observed_target).astype(np.float32)
    scale = max(float(np.std(target)), 1e-6)
    train_mean = float(np.mean(target))
    factories = {
        "ordinary_functional_cp": FunctionalCP,
        "joint_inr": JointINR,
        "paired_travel_time_phase": TrainablePairedPhase,
        "wrong_euclidean_phase": TrainablePairedPhase,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, training = [], {}
    started = time.perf_counter()
    for name, factory in factories.items():
        seed_all(seed)
        model = factory()
        wrong_model = name == "wrong_euclidean_phase"
        training[name] = fit_full_observed(
            model, correct, wrong, target, scale, wrong_model, steps, device
        )
        predictions = [
            predict(model, case, device, wrong=wrong_model) for case in validation
        ]
        metrics = [
            case_metrics(case, prediction)
            for case, prediction in zip(validation, predictions)
        ]
        row = {
            "model": name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            **aggregate_metrics(validation, predictions),
            "macro_nrmse": float(np.mean([item["nrmse"] for item in metrics])),
            "macro_late_nrmse": float(
                np.mean([item["late_nrmse"] for item in metrics])
            ),
            "macro_boundary_nrmse": float(
                np.mean([item["boundary_nrmse"] for item in metrics])
            ),
            "case_metrics": metrics,
        }
        if isinstance(model, TrainablePairedPhase):
            row["learned_bands_hz"] = (
                model.bands.detach().cpu().numpy() / (2 * math.pi)
            ).tolist()
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name, value in (("zero", 0.0), ("train_observed_mean", train_mean)):
        predictions = [
            torch.full_like(torch.from_numpy(case["target"]), value)
            for case in validation
        ]
        metrics = [
            case_metrics(case, prediction)
            for case, prediction in zip(validation, predictions)
        ]
        rows.append(
            {
                "model": name,
                "parameters": 0,
                **aggregate_metrics(validation, predictions),
                "macro_nrmse": float(np.mean([item["nrmse"] for item in metrics])),
                "macro_late_nrmse": float(
                    np.mean([item["late_nrmse"] for item in metrics])
                ),
                "macro_boundary_nrmse": float(
                    np.mean([item["boundary_nrmse"] for item in metrics])
                ),
                "case_metrics": metrics,
            }
        )

    trivial = min(
        row["global_nrmse"]
        for row in rows
        if row["model"] in ("zero", "train_observed_mean")
    )
    for row in rows:
        if row["model"] not in ("zero", "train_observed_mean"):
            row["absolute_gate"] = absolute_wave_gate(
                row["global_nrmse"], trivial
            ).to_dict()
    result = {
        "experiment_id": "TRACK2-TRAVELING-HARMONIC-R2-LOCKED-VALIDATION",
        "status": "VALIDATION_ONLY",
        "decision_rule": (
            "STOP/DOWNGRADE if paired phase fails the pre-registered absolute gate; "
            "do not read test files"
        ),
        "task_semantics": (
            "1% sparse training-target supervision; source-conditioned zero-shot "
            "unseen-geometry and 24-to-32 transfer; no validation context"
        ),
        "generator_independence": (
            "learner-free analytic τ-t generator; irrational generator frequencies "
            "are not imported by or used to initialize any model"
        ),
        "checkpoint_rule": "minimum loss on all observed training labels",
        "config": {
            "seed": seed,
            "ratio": RATIO,
            "steps": steps,
            "learning_rate": LEARNING_RATE,
            "train_resolution": TRAIN_RESOLUTION,
            "validation_resolution": VALIDATION_RESOLUTION,
            "source_indices": list(SOURCES),
            "learner_initial_bands_hz": list(INITIAL_BANDS_HZ),
            "learner_characteristic": "physical travel time minus time",
        },
        "train_cases": [case["name"] for case in train],
        "validation_cases": [case["name"] for case in validation],
        "locked_test_geometries": split["test_geometries"],
        "test_files_read": [],
        "n_observed_training_labels": int(len(target)),
        "realized_training_ratio_range": [min(realized), max(realized)],
        "target_scale_from_observed_train_only": scale,
        "strongest_trivial_global_nrmse": trivial,
        "rows": rows,
        "training": training,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                row["model"]: {
                    "global_nrmse": row["global_nrmse"],
                    "gate": row.get("absolute_gate", {}).get("passed"),
                    "learned_bands_hz": row.get("learned_bands_hz"),
                }
                for row in rows
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.output, args.steps)


if __name__ == "__main__":
    main()
