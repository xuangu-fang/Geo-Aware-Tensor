#!/usr/bin/env python3
"""Recompute bandwise diagnostics from the locked public-data POC artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def collect(run_dir, ratio=.05):
    rows=[]
    for path in sorted((ROOT/run_dir).glob("*.json")):
        p=json.loads(path.read_text())
        m=p.get("metrics",p)
        if abs(float(m.get("observation_ratio",-1))-ratio)>.002: continue
        rows.append({"file":str(path.relative_to(ROOT)),"model":p.get("model",path.stem.split("_")[-1]),
                     "seed":p.get("seed"),"nrmse":m.get("nrmse"),"rmse":m.get("rmse")})
    return rows

def main():
    # Aggregate JSON is authoritative because raw filenames cannot reliably
    # recover multi-token model names. This script also inventories raw evidence.
    agg=json.loads((ROOT/"reports/results/aggregate.json").read_text())
    selected=[r for r in agg if r["dataset"] in ("active_matter","realpde_cylinder")
              and r["mask"]=="random" and abs(float(r["ratio"])-.05)<.002]
    payload={"protocol":"locked public-data stress test; held-out NRMSE; 5% random observations",
             "aggregate_rows":selected,"raw_inventory":{"active":collect("runs/final_active"),
             "realpde":collect("runs/final_realpde")}}
    out=ROOT/"papers/paper_b/results/external_stress.json";out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2));print(out)
if __name__=="__main__":main()
