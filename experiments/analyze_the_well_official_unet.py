#!/usr/bin/env python3
"""Aggregate frozen Paper-B paired phase CP versus The Well classic U-Net."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path("papers/longterm_results")


def main():
    paired_payload = json.loads(
        (ROOT/"the_well_early40_confirmation.json").read_text())
    paired_by_seed = {int(row["seed"]): float(row["paired_phase_cp"])
                      for row in paired_payload["records"]}
    rows = []
    for seed in sorted(paired_by_seed):
        payload = json.loads(
            (ROOT/f"the_well_official_unet_test_seed{seed}.json").read_text())
        rows.append({"seed": seed, "paired_phase_cp": paired_by_seed[seed],
                     "the_well_unet": float(payload["evaluation_macro_nrmse"])})
    paired = np.asarray([row["paired_phase_cp"] for row in rows])
    unet = np.asarray([row["the_well_unet"] for row in rows])
    statistic, p_value = wilcoxon(paired, unet, alternative="less")
    validation_phase = json.loads(
        (ROOT/"the_well_early40_paired_validation_seed0.json").read_text())
    phase_row = next(row for row in validation_phase["rows"]
                     if row["model"] == "paired_phase_cp")
    selection = json.loads(
        (ROOT/"the_well_official_unet_selection.json").read_text())
    result = {
        "experiment_id": "B-WELL-EARLY40-THE-WELL-UNET-CONFIRMATION",
        "status": "REJECTED_ABSOLUTE_EFFECTIVENESS",
        "interpretation": (
            "All methods have NRMSE approximately one. The paired p-value is "
            "retained for audit only and is not positive paper evidence."),
        "protocol_note": (
            "The paired-phase method was frozen before adding this baseline. "
            "The Well 1.2 UNetClassic architecture is locally adapted and retrained "
            "under the identical 1% early-40 protocol."),
        "library": selection["library"],
        "validation_seed0": {
            "paired_phase_cp": phase_row["evaluation_macro_nrmse"],
            "the_well_unet": selection["evaluation_macro_nrmse"],
        },
        "test_summary": {
            "paired_phase_cp_mean": float(paired.mean()),
            "paired_phase_cp_std": float(paired.std(ddof=1)),
            "the_well_unet_mean": float(unet.mean()),
            "the_well_unet_std": float(unet.std(ddof=1)),
            "paired_wins": int(np.sum(paired < unet)),
            "seeds": len(rows),
            "relative_improvement_percent": float(
                100*(unet.mean()-paired.mean())/unet.mean()),
            "one_sided_wilcoxon_statistic": float(statistic),
            "one_sided_wilcoxon_p": float(p_value),
        },
        "efficiency_seed0_validation": {
            "paired_parameters": phase_row["parameters"],
            "unet_parameters": selection["parameters"],
            "parameter_ratio_unet_over_paired": float(
                selection["parameters"]/phase_row["parameters"]),
            "paired_elapsed_seconds": phase_row["elapsed_seconds"],
            "unet_elapsed_seconds": selection["elapsed_seconds"],
            "paired_peak_gpu_bytes": phase_row["peak_gpu_bytes"],
            "unet_peak_gpu_bytes": selection["peak_gpu_bytes"],
        },
        "rows": rows,
    }
    (ROOT/"the_well_official_unet_confirmation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    x = np.arange(len(rows)); width = .36
    fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    axis.bar(x-width/2, paired, width, label="Paired phase CP", color="#2c7fb8")
    axis.bar(x+width/2, unet, width, label="The Well U-Net", color="#59a14f")
    axis.axhline(1., color="black", linewidth=.8, linestyle="--")
    axis.set_xticks(x, [str(row["seed"]) for row in rows])
    axis.set_xlabel("Fresh confirmation seed"); axis.set_ylabel("Test macro NRMSE")
    axis.set_title("Rejected The Well early-40 stress test: all methods near NRMSE 1")
    axis.legend()
    axis.text(.01, .02, f"paired wins {result['test_summary']['paired_wins']}/{len(rows)}, "
              f"one-sided Wilcoxon p={p_value:.4g}", transform=axis.transAxes,
              fontsize=9)
    fig.savefig(ROOT/"the_well_official_unet_confirmation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result["test_summary"], indent=2))


if __name__ == "__main__":
    main()
