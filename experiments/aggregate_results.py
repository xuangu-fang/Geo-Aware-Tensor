#!/usr/bin/env python3
"""Aggregate per-run JSON into audit-friendly CSV, JSON, Markdown and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("nrmse", "relative_l2", "rmse", "mae", "observed_rmse", "coverage_95",
           "gaussian_nll", "uncertainty_error_spearman")


def load_rows(inputs):
    rows = []
    for root in inputs:
        for path in Path(root).glob("*.json"):
            if path.name == "results.json":
                continue
            row = json.loads(path.read_text())
            if "metrics" in row:
                rows.append(row)
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["dataset"], row["mask"], float(row["ratio"]), row["model"])
        groups.setdefault(key, []).append(row)
    out = []
    for (dataset, mask, ratio, model), items in sorted(groups.items()):
        record = {"dataset": dataset, "mask": mask, "ratio": ratio, "model": model,
                  "seeds": len(items), "parameters": round(mean(
                      float(x["metrics"].get("parameters", 0) or
                            x["metrics"].get("spectral_summary", [{}])[0].get("features", 0))
                      for x in items))}
        for metric in METRICS:
            values = [float(x["metrics"][metric]) for x in items if x["metrics"].get(metric) is not None]
            if values:
                record[f"{metric}_mean"] = mean(values)
                record[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out.append(record)
    return out


def write_csv(rows, path):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_markdown(rows, path):
    lines = ["# Aggregated POC results", "",
             "Metrics are evaluated only on unobserved entries. Values are mean ± sample std across seeds.", ""]
    datasets = sorted({r["dataset"] for r in rows})
    for dataset in datasets:
        lines += [f"## {dataset}", "",
                  "| mask | obs | model | NRMSE ↓ | RelL2 ↓ | coverage95 | params |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        selected = [r for r in rows if r["dataset"] == dataset]
        for r in selected:
            def fmt(metric):
                if f"{metric}_mean" not in r: return "—"
                return f"{r[f'{metric}_mean']:.4f} ± {r[f'{metric}_std']:.4f}"
            lines.append(f"| {r['mask']} | {100*r['ratio']:.2g}% | {r['model']} | "
                         f"{fmt('nrmse')} | {fmt('relative_l2')} | {fmt('coverage_95')} | "
                         f"{r['parameters']:,} |")
        lines.append("")
        for mask, ratio in sorted({(r["mask"], r["ratio"]) for r in selected}):
            candidates = [r for r in selected if r["mask"] == mask and r["ratio"] == ratio]
            best = min(candidates, key=lambda x: x.get("nrmse_mean", float("inf")))
            lines.append(f"- Best at {mask}, {100*ratio:.2g}%: `{best['model']}` "
                         f"(NRMSE {best.get('nrmse_mean', float('nan')):.4f}).")
        lines.append("")
    path.write_text("\n".join(lines))


def plot_curves(rows, path):
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False,
                             constrained_layout=True)
    for ax, dataset in zip(axes[0], datasets):
        selected = [r for r in rows if r["dataset"] == dataset and r["mask"] == "random"]
        for model in sorted({r["model"] for r in selected}):
            mr = sorted([r for r in selected if r["model"] == model], key=lambda x: x["ratio"])
            ax.errorbar([100*x["ratio"] for x in mr], [x["nrmse_mean"] for x in mr],
                        yerr=[x["nrmse_std"] for x in mr], marker="o", capsize=3, label=model)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("observation ratio (%)"); ax.set_ylabel("held-out NRMSE")
        ax.set_title(dataset); ax.grid(True, which="both", alpha=0.25)
    axes[0, -1].legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("reports/results"))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    raw = load_rows(args.inputs); summary = aggregate(raw)
    (args.output / "aggregate.json").write_text(json.dumps(summary, indent=2))
    write_csv(summary, args.output / "aggregate.csv")
    write_markdown(summary, args.output / "RESULTS.md")
    plot_curves(summary, args.output / "observation_scaling.png")
    print(f"aggregated {len(raw)} runs into {len(summary)} groups at {args.output}")


if __name__ == "__main__":
    main()
