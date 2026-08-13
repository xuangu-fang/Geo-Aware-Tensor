#!/usr/bin/env python3
"""Aggregate the three-seed early-horizon Paper-B selection pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ["paired_phase_cp", "wrong_distance_cp", "neural_cp", "joint_inr"]
LABELS = ["paired phase CP", "wrong path", "neural CP", "joint INR"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("runs/B-METHOD-R6-WELLMAZE-EARLY40"))
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/the_well_early40_selection.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("papers/longterm_results/the_well_early40_selection.png"))
    args = parser.parse_args()
    records = []
    for path in sorted(args.input.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = {row["model"]: row for row in payload["rows"]}
        ratio = float(payload["config"]["ratio"]); seed = int(payload["config"]["seed"])
        records.append({"ratio": ratio, "seed": seed,
                        **{model: rows[model].get(
                            "evaluation_macro_nrmse", rows[model].get("validation_macro_nrmse"))
                           for model in MODELS}})
    ratios = sorted({record["ratio"] for record in records})
    summary = {}
    for ratio in ratios:
        selected = [record for record in records if record["ratio"] == ratio]
        entry = {"seeds": [record["seed"] for record in selected], "models": {}}
        for model in MODELS:
            values = np.asarray([record[model] for record in selected])
            entry["models"][model] = {"mean": float(values.mean()),
                                      "std": float(values.std(ddof=1)),
                                      "values": values.tolist()}
        paired = np.asarray([record["paired_phase_cp"] for record in selected])
        entry["paired_differences"] = {}
        for baseline in MODELS[1:]:
            base = np.asarray([record[baseline] for record in selected])
            improvement = (base-paired)/base
            entry["paired_differences"][baseline] = {
                "nrmse_difference_baseline_minus_paired": (base-paired).tolist(),
                "relative_improvement_mean": float(improvement.mean()),
                "all_seeds_favor_paired": bool(np.all(paired < base)),
            }
        summary[str(ratio)] = entry
    output = {
        "experiment_id": "B-METHOD-R6-WELLMAZE-EARLY40-SELECTION",
        "status": "SELECTED",
        "selection_only": True,
        "test_split_used": False,
        "fixed_horizon_future_frames": 40,
        "observation_ratios": ratios,
        "records": records,
        "summary": summary,
        "freeze_candidate": {
            "ratio": .01,
            "reason": "lowest tested observation ratio and all three paired comparisons favor the method",
            "required_next_step": "freeze config, then evaluate fresh confirmation seeds and untouched test split"
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    fig, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for model, label in zip(MODELS, LABELS):
        means = [summary[str(r)]["models"][model]["mean"] for r in ratios]
        stds = [summary[str(r)]["models"][model]["std"] for r in ratios]
        axis.errorbar(np.asarray(ratios)*100, means, yerr=stds, marker="o", capsize=3,
                      linewidth=1.8, label=label)
    axis.set_xlabel("Observation ratio (%)")
    axis.set_ylabel("Validation macro NRMSE (lower is better)")
    axis.set_title("The Well early causal horizon: 3 selection seeds")
    axis.grid(alpha=.25); axis.legend(frameon=False, ncol=2)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
