#!/usr/bin/env python3
"""Aggregate the post-feedback three-round A/B iteration campaign."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geoaware.statistics import paired_seed_summary


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers" / "longterm_results"


def load_a(directory: str):
    path = ROOT / "runs" / directory / "results.json"
    return json.loads(path.read_text())["results"] if path.exists() else []


def load_b(directory: str):
    rows = []
    for path in sorted((ROOT / "runs" / directory).glob("seed_*.json")):
        payload = json.loads(path.read_text())
        config = payload["config"]
        for key, values in payload["summary"].items():
            split, model = key.split("/")
            if split == "unseen":
                rows.append({"seed": config["seed"], "ratio": config["ratio"],
                             "mismatch": config.get("mismatch", 1.), "model": model,
                             **values})
    return rows


def summary(rows, key="nrmse"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row["metrics"][key] if "metrics" in row else row[key])
    return {model: {"mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                    "n": len(values)} for model, values in grouped.items()}


def paired(rows, proposed, baseline, key="nrmse"):
    values = defaultdict(dict)
    for row in rows:
        value = row["metrics"][key] if "metrics" in row else row[key]
        values[row["model"]][row["seed"]] = value
    return paired_seed_summary(values[proposed], values[baseline])


def phase_a():
    rows = []
    for ratio_tag, ratio in (("r01", .01), ("r02", .02)):
        for eps_tag, eps in (("0", 0.), ("_25", .25), ("_5", .5), ("_75", .75), ("1", 1.)):
            directory = f"longterm_phase_a_{ratio_tag}_e{eps_tag}"
            for row in load_a(directory):
                row = dict(row); row["mismatch"] = eps; row["phase_ratio"] = ratio
                rows.append(row)
    return rows


def phase_b():
    rows = []
    for ratio_tag in ("_005", "_01", "_02"):
        for eps_tag in ("0", "_5", "1"):
            rows += load_b(f"longterm_phase_b_r{ratio_tag}_e{eps_tag}")
    return rows


def phase_table(rows, ratios, mismatches, models, nested_metrics=False):
    table = {}
    for ratio in ratios:
        for eps in mismatches:
            subset = [r for r in rows if abs((r.get("phase_ratio", r["ratio"]))-ratio)<1e-9
                      and abs(r["mismatch"]-eps)<1e-9]
            table[f"ratio={ratio:g}|mismatch={eps:g}"] = summary(subset)
    return table


def heatmap(rows, ratios, mismatches, models, path, title):
    fig, axes = plt.subplots(1, len(models), figsize=(4.1*len(models), 3.5), squeeze=False)
    matrices=[]
    for model in models:
        matrix = np.full((len(ratios), len(mismatches)), np.nan)
        for i, ratio in enumerate(ratios):
            for j, eps in enumerate(mismatches):
                vals=[]
                for r in rows:
                    rr=r.get("phase_ratio",r["ratio"])
                    if r["model"]==model and abs(rr-ratio)<1e-9 and abs(r["mismatch"]-eps)<1e-9:
                        vals.append(r["metrics"]["nrmse"] if "metrics" in r else r["nrmse"])
                if vals: matrix[i,j]=np.mean(vals)
        matrices.append(matrix)
    vmin=min(np.nanmin(x) for x in matrices);vmax=max(np.nanmax(x) for x in matrices)
    for ax,model,matrix in zip(axes[0],models,matrices):
        image=ax.imshow(matrix,aspect="auto",vmin=vmin,vmax=vmax,cmap="viridis_r")
        for i in range(len(ratios)):
            for j in range(len(mismatches)):
                if np.isfinite(matrix[i,j]): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=8)
        ax.set_xticks(range(len(mismatches)),mismatches);ax.set_yticks(range(len(ratios)),ratios)
        ax.set_xlabel("format mismatch");ax.set_ylabel("observation ratio");ax.set_title(model)
        fig.colorbar(image,ax=ax,fraction=.046)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=190);plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a2 = (load_a("longterm_a_r3_tucker_2pct_confirm") +
          load_a("longterm_a_r3_tucker_2pct_confirm_more"))
    a1 = (load_a("longterm_a_r3_tucker_1pct_confirm") +
          load_a("longterm_a_r3_tucker_1pct_confirm_more"))
    b_learned = load_b("longterm_b_r2_nested")
    b_rbf = load_b("longterm_b_r3_rbf_seed0") + load_b("longterm_b_r3_rbf")
    pa, pb = phase_a(), phase_b()
    payload = {
        "paper_a_confirm": {
            "1pct": summary(a1), "2pct": summary(a2),
            "1pct_paired_vs_flat": paired(a1,"geo_btucker","flat_geo_gp"),
            "2pct_paired_vs_cp": paired(a2,"geo_btucker","geo_bcp_noard"),
            "2pct_paired_vs_flat": paired(a2,"geo_btucker","flat_geo_gp"),
            "2pct_paired_vs_wrong": paired(a2,"geo_btucker","wrong_btucker"),
        },
        "paper_b_envelope_selection": {
            "learned_q2": summary(b_learned), "rbf_tucker": summary(b_rbf),
            "learned_q2_vs_ipnf": paired(b_learned,"envelope_cp","ipnf"),
            "rbf_tucker_vs_ipnf": paired(b_rbf,"envelope_tucker","ipnf"),
        },
        "phase_a": phase_table(pa,[.01,.02],[0,.25,.5,.75,1],
                               ["geo_btucker","geo_bcp_noard","flat_geo_gp"]),
        "phase_b": phase_table(pb,[.005,.01,.02],[0,.5,1],
                               ["paired_cp","envelope_cp","tensor_tucker","ipnf"]),
    }
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2))
    heatmap(pa,[.01,.02],[0,.25,.5,.75,1],
            ["geo_btucker","geo_bcp_noard","flat_geo_gp"],OUT/"phase_a.png",
            "Paper A: operator format mismatch × observation ratio")
    heatmap(pb,[.005,.01,.02],[0,.5,1],
            ["paired_cp","envelope_cp","tensor_tucker","ipnf"],OUT/"phase_b.png",
            "Paper B: envelope mismatch × observation ratio")
    def fmt(q): return f"{q['mean']:.3f}±{q['sd']:.3f}"
    lines=["# 三轮持续迭代统计","","## Paper A 新 seed 确认","",
           "| 观测率 | Geo-BTucker | Geo-CP | Flat GP | Wrong BTucker | Discrete BTucker |",
           "|---:|---:|---:|---:|---:|---:|",
           f"| 1% | {fmt(payload['paper_a_confirm']['1pct']['geo_btucker'])} | {fmt(payload['paper_a_confirm']['1pct']['geo_bcp_noard'])} | {fmt(payload['paper_a_confirm']['1pct']['flat_geo_gp'])} | {fmt(payload['paper_a_confirm']['1pct']['wrong_btucker'])} | — |",
           f"| 2% | {fmt(payload['paper_a_confirm']['2pct']['geo_btucker'])} | {fmt(payload['paper_a_confirm']['2pct']['geo_bcp_noard'])} | {fmt(payload['paper_a_confirm']['2pct']['flat_geo_gp'])} | {fmt(payload['paper_a_confirm']['2pct']['wrong_btucker'])} | {fmt(payload['paper_a_confirm']['2pct']['discrete_btucker'])} |",
           "","## Paper B phase-envelope selection","",
           "| 版本 | Envelope | Paired CP | IP-NF | 结论 |","|---|---:|---:|---:|---|",
           f"| learned Q=2 | {fmt(payload['paper_b_envelope_selection']['learned_q2']['envelope_cp'])} | {fmt(payload['paper_b_envelope_selection']['learned_q2']['paired_cp'])} | {fmt(payload['paper_b_envelope_selection']['learned_q2']['ipnf'])} | 不进入主方法 |",
           f"| fixed-RBF Tucker | {fmt(payload['paper_b_envelope_selection']['rbf_tucker']['envelope_tucker'])} | {fmt(payload['paper_b_envelope_selection']['rbf_tucker']['paired_cp'])} | {fmt(payload['paper_b_envelope_selection']['rbf_tucker']['ipnf'])} | 不进入主方法 |",
           "","## Paper A phase diagram（两 seed exploratory）","",
           "| observation | mismatch | Geo-BTucker | Geo-CP | Flat GP |","|---:|---:|---:|---:|---:|"]
    for ratio in (.01,.02):
        for eps in (0,.25,.5,.75,1):
            q=payload["phase_a"][f"ratio={ratio:g}|mismatch={eps:g}"]
            lines.append(f"| {ratio:g} | {eps:g} | {q['geo_btucker']['mean']:.3f} | {q['geo_bcp_noard']['mean']:.3f} | {q['flat_geo_gp']['mean']:.3f} |")
    lines += ["","## Paper B phase diagram（两 seed exploratory）","",
              "| observation | mismatch | Paired CP | Envelope CP | Geo-Tucker | IP-NF |",
              "|---:|---:|---:|---:|---:|---:|"]
    for ratio in (.005,.01,.02):
        for eps in (0,.5,1):
            q=payload["phase_b"][f"ratio={ratio:g}|mismatch={eps:g}"]
            lines.append(f"| {ratio:g} | {eps:g} | {q['paired_cp']['mean']:.3f} | {q['envelope_cp']['mean']:.3f} | {q['tensor_tucker']['mean']:.3f} | {q['ipnf']['mean']:.3f} |")
    lines += ["","完整 paired statistics 与逐 seed 数值见 `summary.json`。"]
    (OUT/"TABLES.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
