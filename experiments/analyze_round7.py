#!/usr/bin/env python3
"""Aggregate the publication-focused seventh-round experiments."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geoaware.statistics import paired_seed_summary


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"papers/longterm_results"


def load_run(name):
    return json.loads((ROOT/"runs"/name/"results.json").read_text())["results"]


def model_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("model", row.get("strategy"))].append(row["metrics"]["nrmse"])
    return {name: {"mean": float(np.mean(values)),
                   "sd": float(np.std(values, ddof=1)) if len(values)>1 else 0.,
                   "n": len(values), "values": values}
            for name, values in grouped.items()}


def paired(rows, proposed, baseline):
    values = defaultdict(dict)
    for row in rows:
        name = row.get("model", row.get("strategy"))
        values[name][int(row["seed"])] = float(row["metrics"]["nrmse"])
    return paired_seed_summary(values[proposed], values[baseline])


def main():
    block = (load_run("longterm_a_r4_block_2pct")
             + load_run("longterm_a_r4_block_2pct_confirm"))
    noise = (load_run("longterm_a_r4_noise30_2pct")
             + load_run("longterm_a_r4_noise30_2pct_confirm"))
    active = load_run("longterm_a_r5_core_iv_validation")
    path_rows = []
    for seed in range(3):
        payload = json.loads((OUT/f"the_well_path_uncertainty_sigma006_seed{seed}.json").read_text())
        for row in payload["rows"]:
            path_rows.append({"seed": seed, "model": row["model"],
                              "nrmse": row["evaluation_macro_nrmse"]})
    path_summary = {}
    for name in {row["model"] for row in path_rows}:
        values = [row["nrmse"] for row in path_rows if row["model"] == name]
        path_summary[name] = {"mean": float(np.mean(values)),
                              "sd": float(np.std(values, ddof=1)),
                              "values": values}
    unet = json.loads((OUT/"the_well_official_unet_confirmation.json").read_text())
    fno = json.loads((OUT/"the_well_official_fno_confirmation.json").read_text())
    sanity = json.loads((OUT/"the_well_sanity_baselines.json").read_text())
    payload = {
        "paper_a_block_2pct": {
            "summary": model_summary(block),
            "geo_vs_flat": paired(block, "geo_btucker", "flat_geo_gp"),
            "geo_vs_cp": paired(block, "geo_btucker", "geo_bcp_noard"),
            "geo_vs_wrong": paired(block, "geo_btucker", "wrong_btucker"),
        },
        "paper_a_noise30_2pct": {
            "summary": model_summary(noise),
            "geo_vs_flat": paired(noise, "geo_btucker", "flat_geo_gp"),
            "geo_vs_cp": paired(noise, "geo_btucker", "geo_bcp_noard"),
            "geo_vs_wrong": paired(noise, "geo_btucker", "wrong_btucker"),
        },
        "paper_a_core_iv_negative": {
            "summary": model_summary(active),
            "correct_iv_vs_random": paired(active, "correct_core_iv", "random"),
            "wrong_iv_vs_random": paired(active, "wrong_core_iv", "random"),
        },
        "paper_b_path_noise_validation": path_summary,
        "paper_b_unet_confirmation": unet["test_summary"],
        "paper_b_fno_confirmation": fno["test_summary"],
        "paper_b_sanity": sanity["test_summary"],
    }
    (OUT/"round7_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    models = ["geo_btucker", "geo_bcp_noard", "flat_geo_gp", "wrong_btucker"]
    labels = ["Geo\nTucker", "Geo CP", "Flat GP", "Wrong\nTucker"]
    colors = ["#2c7fb8", "#59a14f", "#7f7f7f", "#e15759"]
    for axis, rows, title in ((axes[0], block, "A: 2% + missing block"),
                              (axes[1], noise, "A: 2% + 30% noise")):
        stats = model_summary(rows)
        means = [stats[name]["mean"] for name in models]
        sds = [stats[name]["sd"] for name in models]
        axis.bar(np.arange(4), means, yerr=sds, color=colors, capsize=3)
        axis.set_xticks(np.arange(4), labels, fontsize=8)
        axis.set_ylabel("Held-out NRMSE"); axis.set_title(title)
    b_labels = ["Paired\nphase CP", "FNO", "U-Net", "Scaled\npersistence", "Zero"]
    b_values = [fno["test_summary"]["paired_phase_cp_mean"],
                fno["test_summary"]["official_fno_mean"],
                unet["test_summary"]["the_well_unet_mean"],
                sanity["test_summary"]["time_scaled_persistence_mean"],
                sanity["test_summary"]["zero_mean"]]
    axes[2].bar(np.arange(5), b_values,
                color=["#2c7fb8", "#f28e2b", "#59a14f", "#bab0ac", "#7f7f7f"])
    axes[2].axhline(1., color="black", linestyle="--", linewidth=.8)
    axes[2].set_xticks(np.arange(5), b_labels, fontsize=7)
    axes[2].set_ylabel("Test macro NRMSE"); axes[2].set_title("B: The Well, 1%, early-40")
    axes[2].set_ylim(.988, 1.008)
    fig.savefig(OUT/"round7_headline.png", dpi=190); plt.close(fig)

    def fmt(row): return f"{row['mean']:.3f}±{row['sd']:.3f}"
    block_s = payload["paper_a_block_2pct"]["summary"]
    noise_s = payload["paper_a_noise30_2pct"]["summary"]
    active_s = payload["paper_a_core_iv_negative"]["summary"]
    lines = [
        "# 第七轮：发表导向证据表", "",
        "## Paper A：固定方法的压力测试", "",
        "| Setting | Geo-BTucker | Geo-CP | Flat GP | Wrong Tucker |",
        "|---|---:|---:|---:|---:|",
        f"| 2% + center block missing | {fmt(block_s['geo_btucker'])} | {fmt(block_s['geo_bcp_noard'])} | {fmt(block_s['flat_geo_gp'])} | {fmt(block_s['wrong_btucker'])} |",
        f"| 2% + 30% obs. noise | {fmt(noise_s['geo_btucker'])} | {fmt(noise_s['geo_bcp_noard'])} | {fmt(noise_s['flat_geo_gp'])} | {fmt(noise_s['wrong_btucker'])} |",
        "", "## Paper A：主动采样负结果", "",
        "| Correct core-IV | Wrong core-IV | Random |",
        "|---:|---:|---:|",
        f"| {fmt(active_s['correct_core_iv'])} | {fmt(active_s['wrong_core_iv'])} | {fmt(active_s['random'])} |",
        "", "## Paper B：The Well 强/简单基线", "",
        "| Method | Test macro NRMSE | Paired wins |",
        "|---|---:|---:|",
        f"| Paired phase CP | {unet['test_summary']['paired_phase_cp_mean']:.5f}±{unet['test_summary']['paired_phase_cp_std']:.5f} | — |",
        f"| Official FNO 2.0 | {fno['test_summary']['official_fno_mean']:.5f}±{fno['test_summary']['official_fno_std']:.5f} | 8/10 |",
        f"| The Well U-Net | {unet['test_summary']['the_well_unet_mean']:.5f}±{unet['test_summary']['the_well_unet_std']:.5f} | 9/10 |",
        f"| time-scaled persistence | {sanity['test_summary']['time_scaled_persistence_mean']:.5f}±{sanity['test_summary']['time_scaled_persistence_std']:.5f} | descriptive |",
        f"| zero | {sanity['test_summary']['zero_mean']:.5f} | descriptive |",
        "", "完整逐 seed 数值和 paired tests 见 `round7_summary.json`。",
    ]
    (OUT/"ROUND7_TABLES.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    tracked = [
        ROOT/"src/geoaware/well_baselines.py",
        ROOT/"experiments/run_the_well_official_unet.py",
        ROOT/"experiments/analyze_the_well_official_unet.py",
        ROOT/"experiments/run_the_well_sanity_baselines.py",
        ROOT/"experiments/run_the_well_path_uncertainty.py",
        ROOT/"experiments/run_tensor_core_iv_acquisition.py",
        ROOT/"experiments/analyze_round7.py",
        ROOT/"papers/paper_a/DRAFT_TUCKER.md",
        ROOT/"papers/paper_b/DRAFT.md",
        OUT/"round7_summary.json",
    ]
    manifest = {
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in tracked},
        "environment": {
            "python": "/home/ubuntu/project/yanjiu/.venv/bin/python",
            "torch": torch_version(),
            "neuraloperator": "2.0.0",
            "the_well_architecture_source": "1.2.0",
        },
    }
    (OUT/"round7_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote round7 summary, table, figure, and manifest")


def torch_version():
    import torch
    return torch.__version__


if __name__ == "__main__":
    main()
