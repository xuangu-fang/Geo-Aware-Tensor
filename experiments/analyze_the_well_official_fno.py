#!/usr/bin/env python3
"""Aggregate frozen Paper-B paired phase CP versus official FNO baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path,
                        default=Path("papers/longterm_results/the_well_early40_confirmation.json"))
    parser.add_argument("--fno-dir", type=Path, default=Path("papers/longterm_results"))
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/the_well_official_fno_confirmation.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("papers/longterm_results/the_well_official_fno_confirmation.png"))
    args = parser.parse_args()
    paired_payload = json.loads(args.paired.read_text())
    paired_by_seed = {int(row["seed"]): float(row["paired_phase_cp"])
                      for row in paired_payload["records"]}
    rows = []
    for seed in sorted(paired_by_seed):
        fno_path = args.fno_dir/f"the_well_official_fno_test_seed{seed}.json"
        payload = json.loads(fno_path.read_text())
        rows.append({"seed": seed, "paired_phase_cp": paired_by_seed[seed],
                     "official_fno": float(payload["evaluation_macro_nrmse"])})
    paired = np.asarray([row["paired_phase_cp"] for row in rows])
    fno = np.asarray([row["official_fno"] for row in rows])
    statistic, p_value = wilcoxon(paired, fno, alternative="less")
    validation_phase = json.loads(
        (args.fno_dir/"the_well_early40_paired_validation_seed0.json").read_text())
    phase_row = next(row for row in validation_phase["rows"]
                     if row["model"] == "paired_phase_cp")
    fno_selection = json.loads(
        (args.fno_dir/"the_well_official_fno_selection.json").read_text())
    result = {
        "experiment_id": "B-WELL-EARLY40-OFFICIAL-FNO-CONFIRMATION",
        "status": "POSTHOC_BASELINE_CONFIRMATION",
        "important_protocol_note": (
            "Paired-phase configuration and its seeds were frozen before FNO was added; "
            "FNO architecture was selected on validation and test seeds match the existing confirmation."),
        "library": {"name": "neuraloperator", "version": "2.0.0"},
        "validation_seed0": {
            "paired_phase_cp": phase_row["evaluation_macro_nrmse"],
            "official_fno": fno_selection["evaluation_macro_nrmse"],
        },
        "test_summary": {
            "paired_phase_cp_mean": float(paired.mean()),
            "paired_phase_cp_std": float(paired.std()),
            "official_fno_mean": float(fno.mean()),
            "official_fno_std": float(fno.std()),
            "paired_wins": int(np.sum(paired < fno)),
            "seeds": len(rows),
            "paired_minus_fno_mean": float(np.mean(paired-fno)),
            "relative_improvement_percent": float(100*(fno.mean()-paired.mean())/fno.mean()),
            "one_sided_wilcoxon_statistic": float(statistic),
            "one_sided_wilcoxon_p": float(p_value),
        },
        "efficiency_seed0_validation": {
            "paired_parameters": phase_row["parameters"],
            "fno_parameters": fno_selection["parameters"],
            "parameter_ratio_fno_over_paired": float(fno_selection["parameters"]
                                                      /phase_row["parameters"]),
            "paired_elapsed_seconds": phase_row["elapsed_seconds"],
            "fno_elapsed_seconds": fno_selection["elapsed_seconds"],
            "paired_peak_gpu_bytes": phase_row["peak_gpu_bytes"],
            "fno_peak_gpu_bytes": fno_selection["peak_gpu_bytes"],
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    x = np.arange(len(rows)); width = .36
    fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    axis.bar(x-width/2, paired, width, label="Paired phase CP", color="#2c7fb8")
    axis.bar(x+width/2, fno, width, label="Official FNO 2.0", color="#f28e2b")
    axis.axhline(1., color="black", linewidth=.8, linestyle="--")
    axis.set_xticks(x, [str(row["seed"]) for row in rows])
    axis.set_xlabel("Fresh confirmation seed"); axis.set_ylabel("Test macro NRMSE")
    axis.set_title("The Well early-40, 1% sparse supervision, 32 test geometries")
    axis.legend()
    axis.text(.01, .02, f"paired wins {result['test_summary']['paired_wins']}/{len(rows)}, "
              f"one-sided Wilcoxon p={p_value:.4g}", transform=axis.transAxes, fontsize=9)
    fig.savefig(args.figure, dpi=180); plt.close(fig)
    print(json.dumps(result["test_summary"], indent=2))


if __name__ == "__main__":
    main()
