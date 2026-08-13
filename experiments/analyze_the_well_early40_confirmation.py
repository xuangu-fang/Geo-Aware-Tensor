#!/usr/bin/env python3
"""Freeze the fresh-seed, untouched-test Paper-B confirmation statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


MODELS = ["paired_phase_cp", "wrong_distance_cp", "neural_cp", "joint_inr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("runs/B-METHOD-R6-WELLMAZE-EARLY40-CONFIRM"))
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/the_well_early40_confirmation.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("papers/longterm_results/the_well_early40_confirmation.png"))
    parser.add_argument("--sanity", type=Path,
                        default=Path("papers/longterm_results/the_well_sanity_baselines.json"))
    args = parser.parse_args()
    records=[]
    for path in sorted(args.input.glob("*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        payload=json.loads(path.read_text()); rows={row["model"]:row for row in payload["rows"]}
        records.append({"seed":int(payload["config"]["seed"]),
                        **{model:rows[model]["evaluation_macro_nrmse"] for model in MODELS}})
    if [record["seed"] for record in records] != list(range(10,20)):
        raise RuntimeError("confirmation requires exactly fresh seeds 10--19")
    values={model:np.asarray([record[model] for record in records]) for model in MODELS}
    summary={model:{"mean":float(array.mean()),"std":float(array.std(ddof=1)),
                    "values":array.tolist()} for model,array in values.items()}
    comparisons={}
    paired=values["paired_phase_cp"]
    for baseline in MODELS[1:]:
        differences=values[baseline]-paired
        statistic,pvalue=wilcoxon(differences,alternative="greater")
        comparisons[baseline]={
            "difference_baseline_minus_paired":differences.tolist(),
            "mean_difference":float(differences.mean()),
            "relative_improvement":float(differences.mean()/values[baseline].mean()),
            "paired_wins":int(np.sum(differences>0)),
            "n_pairs":len(differences),
            "one_sided_wilcoxon_statistic":float(statistic),
            "one_sided_wilcoxon_pvalue":float(pvalue),
        }
    sanity=json.loads(args.sanity.read_text())["test_summary"]
    zero=float(sanity["zero_mean"])
    persistence=float(sanity["time_scaled_persistence_mean"])
    paired_mean=float(summary["paired_phase_cp"]["mean"])
    absolute_effect={
        "decision":"REJECTED_ABSOLUTE_EFFECTIVENESS",
        "approx_explained_variance":1-paired_mean**2,
        "mse_skill_vs_zero":1-(paired_mean/zero)**2,
        "mse_skill_vs_time_scaled_persistence":1-(paired_mean/persistence)**2,
    }
    result={
        "experiment_id":"B-METHOD-R6-WELLMAZE-EARLY40-CONFIRM",
        "status":"REJECTED",
        "configuration_frozen_before_confirmation":True,
        "selection_seeds":[0,1,2],"confirmation_seeds":list(range(10,20)),
        "test_split_used_only_for_confirmation":True,
        "test_geometries":32,"train_geometries":64,"future_horizon_frames":40,
        "observation_ratio":.01,"summary":summary,"paired_comparisons":comparisons,
        "records":records,"absolute_effectiveness":absolute_effect,
        "scope_note":"Sparse-supervised cross-geometry early-horizon field regression; not long-horizon forecasting.",
        "limitation":"All methods have NRMSE approximately one. Small paired differences are not evidence of useful reconstruction."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))

    fig,axis=plt.subplots(figsize=(7.2,4.2),constrained_layout=True)
    offsets=np.linspace(-.24,.24,len(MODELS)); x=np.arange(len(records))
    for offset,model in zip(offsets,MODELS):
        axis.scatter(x+offset,values[model],s=28,label=model.replace("_"," "))
    axis.axhline(1.,color="black",linestyle="--",linewidth=1,alpha=.6)
    axis.set_xticks(x,[str(record["seed"]) for record in records]);axis.set_xlabel("Fresh confirmation seed")
    axis.set_ylabel("Test macro NRMSE");axis.set_title("Rejected early-horizon stress test: all methods near NRMSE 1")
    axis.grid(axis="y",alpha=.25);axis.legend(frameon=False,ncol=2)
    fig.savefig(args.figure,dpi=180);plt.close(fig)
    print(json.dumps({"summary":summary,"paired_comparisons":comparisons},indent=2))


if __name__=="__main__":main()
