#!/usr/bin/env python3
"""Aggregate Paper-B runs at the seed level and draw the primary figure."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from geoaware.statistics import paired_seed_summary

METRICS = ["nrmse", "high_band_nrmse", "boundary_nrmse", "shadow_nrmse"]

def main():
    p=argparse.ArgumentParser();p.add_argument("inputs",nargs="+");p.add_argument("--output",required=True);a=p.parse_args()
    files=[]
    for x in a.inputs: files += glob.glob(str(Path(x)/"seed_*.json")) if Path(x).is_dir() else glob.glob(x)
    seed_values=defaultdict(lambda:defaultdict(dict)); configs=[]
    for f in sorted(set(files)):
        payload=json.load(open(f));configs.append(payload["config"]); seed=int(payload["config"]["seed"])
        rows=[r for r in payload["rows"] if r["split"]=="unseen_geometry"]
        for model in sorted(set(r["model"] for r in rows)):
            rr=[r for r in rows if r["model"]==model]
            for metric in METRICS: seed_values[metric][model][seed]=float(np.mean([r[metric] for r in rr]))
    baselines=[x for x in ["wrong_geometry","siren","rff","neural_cp"] if x in seed_values["nrmse"]]
    stats={m:{b:paired_seed_summary(seed_values[m]["graph_adapter"],seed_values[m][b]) for b in baselines} for m in METRICS}
    summary={m:{model:{"mean":float(np.mean(list(v.values()))),"std":float(np.std(list(v.values()),ddof=1)),"seeds":v} for model,v in models.items()} for m,models in seed_values.items()}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps({"files":sorted(set(files)),"configs":configs,"summary":summary,"paired":stats},indent=2))
    models=["graph_adapter","wrong_geometry","siren","rff","neural_cp"]
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for ax,metric,title in zip(axes,["nrmse","high_band_nrmse"],["Held-out NRMSE","High-band NRMSE"]):
        means=[summary[metric][m]["mean"] for m in models]; std=[summary[metric][m]["std"] for m in models]
        ax.bar(range(len(models)),means,yerr=std,color=["#176b87","#a7a7a7","#d28e42","#d28e42","#d28e42"],capsize=3)
        ax.set_xticks(range(len(models)),["Intrinsic\nphase","Euclidean\nphase","SIREN","RFF","Neural CP"],rotation=20)
        ax.set_title(title);ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(out/"cross_resolution.png",dpi=180);plt.close(fig)
    print(json.dumps(stats,indent=2))
if __name__=="__main__":main()
