#!/usr/bin/env python3
"""Observed-only zero and time-scaled persistence sanity baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from geoaware.the_well_pilot import fixed_random_mask
from run_the_well_official_unet import load_case


ROOT = Path("papers/longterm_results")
DATA = Path("data/the_well_acoustic_64x64")
HORIZON = 40


def fit_time_scaled_persistence(cases, seed):
    coefficients = []
    masks = [torch.from_numpy(fixed_random_mask(
        case["target"].shape, .01, seed+index)) for index, case in enumerate(cases)]
    for time_index in range(HORIZON):
        x_values, y_values = [], []
        for case, mask in zip(cases, masks):
            selected = mask[time_index]
            x_values.append(case["base"][2][selected])
            y_values.append(case["target"][time_index][selected])
        x = torch.cat(x_values); y = torch.cat(y_values)
        design = torch.stack([x, torch.ones_like(x)], 1).double()
        eye = torch.eye(2, dtype=torch.double)*1e-6
        coefficients.append(torch.linalg.solve(
            design.T@design+eye, design.T@y.double()).float())
    return torch.stack(coefficients), masks


def evaluate(cases, coefficients):
    metrics = {"zero": [], "time_scaled_persistence": []}
    for case in cases:
        target = case["target"]
        initial = case["base"][2]
        persistence = (coefficients[:, 0, None, None]*initial[None]
                       + coefficients[:, 1, None, None])
        for name, prediction in (("zero", torch.zeros_like(target)),
                                 ("time_scaled_persistence", persistence)):
            error = prediction-target
            metrics[name].append(float(
                error.square().mean().sqrt()/target.std().clamp_min(1e-8)))
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def main():
    train = [load_case(DATA/"train"/f"trajectory_{index:03d}.npz", HORIZON)
             for index in range(64)]
    validation = [load_case(DATA/"validation"/f"trajectory_{index:03d}.npz", HORIZON)
                  for index in range(16)]
    test = [load_case(DATA/"test"/f"trajectory_{index:03d}.npz", HORIZON)
            for index in range(32)]
    records = []
    for seed in [0, *range(10, 20)]:
        coefficients, masks = fit_time_scaled_persistence(train, seed)
        split = "validation" if seed == 0 else "test"
        result = evaluate(validation if seed == 0 else test, coefficients)
        records.append({
            "seed": seed, "split": split, **result,
            "realized_ratio": float(sum(mask.sum() for mask in masks)
                                    /sum(mask.numel() for mask in masks)),
            "coefficients": coefficients.tolist(),
        })
    test_records = [row for row in records if row["split"] == "test"]
    payload = {
        "experiment_id": "B-WELL-EARLY40-SANITY-BASELINES",
        "status": "POSTHOC_SANITY_BASELINES",
        "protocol_note": (
            "The persistence baseline fits one slope and intercept per query time "
            "using only the same 1% observed training targets."),
        "validation_seed0": records[0],
        "test_summary": {
            name+"_mean": float(np.mean([row[name] for row in test_records]))
            for name in ("zero", "time_scaled_persistence")
        } | {
            name+"_std": float(np.std([row[name] for row in test_records], ddof=1))
            for name in ("zero", "time_scaled_persistence")
        },
        "records": records,
    }
    (ROOT/"the_well_sanity_baselines.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["test_summary"], indent=2))


if __name__ == "__main__":
    main()
