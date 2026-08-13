#!/usr/bin/env python3
"""Aggregate the tensor-refocus campaign with paired seed statistics."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from geoaware.statistics import paired_seed_summary

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"papers/paper_a/tensor_results"
def load(name): return json.loads((ROOT/"runs"/name/"results.json").read_text())["results"]
def group(rows):
 d=defaultdict(list)
 for r in rows:
  for metric in ("nrmse","nll","coverage95","width95","selective_gain50"):
   d[(r["mask"],r["ratio"],r["model"],metric)].append(r["metrics"][metric])
 return {" | ".join(map(str,k)):{"mean":float(np.mean(v)),"sd":float(np.std(v,ddof=1)) if len(v)>1 else 0,"n":len(v)} for k,v in d.items()}
def paired(rows,a,b,metric):
 x={r["seed"]:r["metrics"][metric] for r in rows if r["model"]==a}; y={r["seed"]:r["metrics"][metric] for r in rows if r["model"]==b}
 # lower is better; selective gain is converted to risk for the shared utility
 if metric=="selective_gain50":
  original_x=x; original_y=y
  result=paired_seed_summary({k:-v for k,v in x.items()},{k:-v for k,v in y.items()})
  result["proposed_mean"]=float(np.mean(list(original_x.values())))
  result["baseline_mean"]=float(np.mean(list(original_y.values())))
  return result
 return paired_seed_summary(x,y)
def elapsed(rows):
 d=defaultdict(list)
 for r in rows: d[r["model"]].append(r["elapsed_seconds"])
 return {k:{"mean_seconds":float(np.mean(v)),"sd_seconds":float(np.std(v,ddof=1))} for k,v in d.items()}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 fresh=load("paper_a_tensor_round4_confirm_random"); gap=load("paper_a_tensor_round4_confirm_gap")
 exploratory2=load("paper_a_tensor_round4_confirm_2pct")
 confirm2=load("paper_a_tensor_final_confirm_2pct")
 public=load("paper_a_tensor_active_smoke"); tucker=load("paper_a_tensor_tucker_smoke")
 rows=fresh+gap+confirm2
 comparisons={}
 for metric in ("nrmse","nll","coverage95","width95","selective_gain50"):
  for base in ("wrong_bcp","discrete_bcp","flat_geo_gp"):
   # Coverage is summarized descriptively rather than lower-is-better testing.
   if metric!="coverage95": comparisons[f"random_geo_vs_{base}_{metric}"]=paired(fresh,"geo_bcp_noard",base,metric)
  for base in ("wrong_bcp","discrete_bcp","flat_geo_gp"):
   if metric!="coverage95": comparisons[f"2pct_geo_vs_{base}_{metric}"]=paired(confirm2,"geo_bcp_noard",base,metric)
 payload={"tables":group(rows),"comparisons":comparisons,
          "final_2pct_efficiency":{"elapsed":elapsed(confirm2),
              "predictive_coefficients":{"geo_bcp_noard":256,"wrong_bcp":256,
                                         "discrete_bcp":680,"flat_geo_gp":512}},
          "exploratory_ratio_selection":group(exploratory2),"public":group(public),"tucker":group(tucker)}
 (OUT/"summary.json").write_text(json.dumps(payload,indent=2))
 lines=["# Tensor-refocus results","","Fresh confirmation; mean ± sample SD over seeds.","",
        "| mask | ratio | model | NRMSE | NLL | cov95 | selective gain |","|---|---:|---|---:|---:|---:|---:|"]
 for mask,rr in (("random",fresh),("periodic_gap",gap),("random_2pct",confirm2)):
  for model in ("geo_bcp_noard","wrong_bcp","discrete_bcp","flat_geo_gp"):
   q=[r for r in rr if r["model"]==model]
   vals=lambda m:f"{np.mean([r['metrics'][m] for r in q]):.3f}±{np.std([r['metrics'][m] for r in q],ddof=1):.3f}"
   ratio=.02 if mask=="random_2pct" else .005
   lines.append(f"| {mask} | {ratio} | {model} | {vals('nrmse')} | {vals('nll')} | {vals('coverage95')} | {vals('selective_gain50')} |")
 lines += ["","## Exact paired seed tests",""]
 for k,v in comparisons.items():
  if "_nll" in k:
   lines.append(f"- `{k}`: proposed={v['proposed_mean']:.4f}, baseline={v['baseline_mean']:.4f}, absolute difference={v['proposed_mean']-v['baseline_mean']:.4f}, p={v['two_sided_paired_permutation_p']:.4f}. Relative percentages are omitted because NLL can be negative.")
  else:
   lines.append(f"- `{k}`: proposed={v['proposed_mean']:.4f}, baseline={v['baseline_mean']:.4f}, improvement={100*v['relative_improvement']:.1f}% CI [{100*v['relative_improvement_ci95'][0]:.1f}, {100*v['relative_improvement_ci95'][1]:.1f}], p={v['two_sided_paired_permutation_p']:.4f}.")
 lines += ["","## Final 2% efficiency","",
           "Predictive coefficient counts: geometry/wrong BCP 256, discrete BCP 680, flat operator GP 512. "
           "Mean end-to-end time per seed (including split calibration and initialization): " +
           ", ".join(f"{k} {v['mean_seconds']:.2f}s" for k,v in elapsed(confirm2).items()) + "."]
 (OUT/"TABLES.md").write_text("\n".join(lines)+"\n")
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(9,3.5))
 models=("geo_bcp_noard","wrong_bcp","discrete_bcp","flat_geo_gp")
 labels=("Geo-Bayes-CP","Wrong geometry","Discrete Bayes CP","Flat operator GP")
 colors=("#1f77b4","#d62728","#9467bd","#7f7f7f")
 x=np.arange(len(models))
 for ax,rr,title in ((axes[0],fresh,"0.5%: uncertainty ranking"),(axes[1],confirm2,"2%: reconstruction")):
  metric="selective_gain50" if rr is fresh else "nrmse"
  means=[np.mean([r["metrics"][metric] for r in rr if r["model"]==m]) for m in models]
  stds=[np.std([r["metrics"][metric] for r in rr if r["model"]==m],ddof=1) for m in models]
  ax.bar(x,means,yerr=stds,color=colors,capsize=3); ax.set_xticks(x,labels,rotation=25,ha="right",fontsize=8)
  ax.set_ylabel("selective gain at 50%" if rr is fresh else "NRMSE"); ax.set_title(title)
 fig.tight_layout(); fig.savefig(OUT/"tensor_headline.png",dpi=190); plt.close(fig)
 # Calibration sharpness at the final ratio: being near 0.95 and left is better.
 fig,ax=plt.subplots(figsize=(5.3,3.7))
 for model,label,color in zip(models,labels,colors):
  q=[r for r in confirm2 if r["model"]==model]
  cov=np.mean([r["metrics"]["coverage95"] for r in q]); width=np.mean([r["metrics"]["width95"] for r in q])
  ax.scatter(width,cov,s=70,color=color,label=label)
 ax.axhline(.95,color="black",linestyle="--",linewidth=1,label="nominal 95%")
 ax.set_xlabel("mean 95% interval width"); ax.set_ylabel("empirical coverage")
 ax.set_title("2%: calibration versus sharpness"); ax.set_ylim(.15,1.02); ax.legend(fontsize=7)
 fig.tight_layout(); fig.savefig(OUT/"calibration_sharpness.png",dpi=190); plt.close(fig)
 tracked=[ROOT/"src/geoaware/tensor_bayes.py",ROOT/"src/geoaware/tensor_data.py",ROOT/"experiments/run_tensor_bayes.py",ROOT/"experiments/analyze_tensor_bayes.py",ROOT/"papers/paper_a/DRAFT.md",ROOT/"papers/paper_a/TENSOR_REFOCUS.md",ROOT/"papers/paper_a/TENSOR_ITERATIONS.md"]
 (OUT/"MANIFEST.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},indent=2))
if __name__=="__main__":main()
