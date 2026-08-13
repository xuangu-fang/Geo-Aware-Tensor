#!/usr/bin/env python3
"""Aggregate Paper A runs, paired statistics, and publication-ready compact tables."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import hashlib

from geoaware.statistics import paired_seed_summary


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers" / "paper_a" / "results"


def load(name):
    return json.loads((ROOT / "runs" / name / "results.json").read_text())["results"]


def at(obj, path):
    parts = path.split("."); i = 0
    while i < len(parts):
        key = parts[i]
        if key not in obj and ".".join(parts[i:]) in obj:
            obj = obj[".".join(parts[i:])]; break
        obj = obj[key]; i += 1
    return float(obj)


def grouped(rows, keys, metric):
    groups = defaultdict(list)
    for row in rows:
        try: value = at(row, metric)
        except (KeyError, TypeError): continue
        groups[tuple(row[k] for k in keys)].append(value)
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v)>1 else 0.,
                "n": len(v)} for k,v in groups.items()}


def paired(rows, filters_a, filters_b, metric="metrics.nrmse"):
    def select(filters):
        return {int(r["seed"]): at(r, metric) for r in rows
                if all(r.get(k) == v for k,v in filters.items())}
    return paired_seed_summary(select(filters_a), select(filters_b))


def markdown_table(summary, headers):
    lines = ["| " + " | ".join(headers + ["mean ± sd", "n"]) + " |",
             "|" + "---|" * (len(headers) + 2)]
    for key, value in sorted(summary.items(), key=lambda x: tuple(map(str,x[0]))):
        lines.append("| " + " | ".join(map(str,key)) +
                     f" | {value['mean']:.4f} ± {value['std']:.4f} | {value['n']} |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = load("paper_a_round1")
    r2 = load("paper_a_round2") + load("paper_a_confirm_matched")
    r3 = load("paper_a_round3_static_heterogeneous") + load("paper_a_confirm_heterogeneous")
    point_active = load("paper_a_round3_active_heterogeneous")
    sensor = load("paper_a_round3_sensor_spectral") + load("paper_a_confirm_sensor_spectral")
    real = load("paper_a_round3_real_active")

    tables = {}
    for name, rows in (("round1",r1),("round2",r2),("heterogeneous",r3)):
        tables[name+"_nrmse"] = grouped(rows,("mask","ratio","model"),"metrics.nrmse")
        tables[name+"_nll"] = grouped(rows,("mask","ratio","model"),"metrics.conditional_nll")
        tables[name+"_coverage95"] = grouped(rows,("mask","ratio","model"),"metrics.conditional_coverage.0.95")
        tables[name+"_right_nll"] = grouped(rows,("mask","ratio","model"),"metrics.regions.right_room.conditional_nll")
    tables["point_active_nrmse"] = grouped(point_active,("ratio","strategy","evaluator"),"metrics.nrmse")
    tables["sensor_nrmse"] = grouped(sensor,("requested_sensor_fraction","strategy","evaluator"),"metrics.nrmse")
    tables["real_nrmse"] = grouped(real,("ratio","model"),"metrics.nrmse")
    tables["real_cov95"] = grouped(real,("ratio","model"),"metrics.conditional_coverage.0.95")

    stats = {
        "matched_random_0.25_geo_vs_rbf": paired(r2,{"mask":"random","ratio":.0025,"model":"geo_spectral"},
                                                    {"mask":"random","ratio":.0025,"model":"rbf_gp"}),
        "matched_random_0.5_geo_vs_rbf": paired(r2,{"mask":"random","ratio":.005,"model":"geo_spectral"},
                                                   {"mask":"random","ratio":.005,"model":"rbf_gp"}),
        "hetero_imbalance_0.25_geo_vs_rbf": paired(r3,{"mask":"room_imbalance","ratio":.0025,"model":"geo_spectral"},
                                                      {"mask":"room_imbalance","ratio":.0025,"model":"rbf_gp"}),
        "hetero_imbalance_0.25_geo_vs_neural_cp": paired(r3,{"mask":"room_imbalance","ratio":.0025,"model":"geo_spectral"},
                                                            {"mask":"room_imbalance","ratio":.0025,"model":"neural_cp"}),
        "sensor_0.5_geo_iv_vs_random": paired(sensor,{"strategy":"geo_sensor_iv","evaluator":"geo_spectral","requested_sensor_fraction":.005},
                                                {"strategy":"random_sensor","evaluator":"geo_spectral","requested_sensor_fraction":.005}),
        "sensor_0.5_geo_iv_vs_wrong_iv": paired(sensor,{"strategy":"geo_sensor_iv","evaluator":"geo_spectral","requested_sensor_fraction":.005},
                                                  {"strategy":"wrong_sensor_iv","evaluator":"geo_spectral","requested_sensor_fraction":.005}),
        "point_0.5_geo_iv_vs_random": paired(point_active,{"strategy":"geo_integrated_variance","evaluator":"geo_spectral","ratio":.005},
                                               {"strategy":"random","evaluator":"geo_spectral","ratio":.005}),
    }
    serial = {name:{" | ".join(map(str,k)):v for k,v in table.items()} for name,table in tables.items()}
    (OUT/"summary.json").write_text(json.dumps({"tables":serial,"paired_statistics":stats},indent=2))
    md = ["# Paper A result tables", "", "All entries are seed mean ± sample SD.", "",
          "## Round 2 controlled matched-basis NRMSE", "",
          markdown_table(tables["round2_nrmse"],["mask","ratio","model"]), "",
          "## Round 3 heterogeneous confirmation NRMSE", "",
          markdown_table(tables["heterogeneous_nrmse"],["mask","ratio","model"]), "",
          "## Persistent-sensor acquisition on the controlled task", "",
          markdown_table(tables["sensor_nrmse"],["requested sensor fraction","strategy","evaluator"]), "",
          "## Public Active Matter stress test", "",
          markdown_table(tables["real_nrmse"],["ratio","model"]), "", "## Paired tests", ""]
    for name,s in stats.items():
        md.append(f"- `{name}`: proposed {s['proposed_mean']:.4f}, baseline {s['baseline_mean']:.4f}, "
                  f"relative improvement {100*s['relative_improvement']:.1f}% "
                  f"(bootstrap CI {100*s['relative_improvement_ci95'][0]:.1f}% to "
                  f"{100*s['relative_improvement_ci95'][1]:.1f}%), exact paired p={s['two_sided_paired_permutation_p']:.4f}.")
    (OUT/"TABLES.md").write_text("\n".join(md)+"\n")

    # Compact figures intentionally emphasize calibration and sensing, not only
    # reconstruction error.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    subset = [r for r in r3 if r["mask"] == "room_imbalance" and r["ratio"] == .0025
              and r["model"] in ("geo_spectral","wrong_geometry","rbf_gp")]
    levels = [.5,.8,.95]
    for model in ("geo_spectral","wrong_geometry","rbf_gp"):
        rs=[r for r in subset if r["model"]==model]
        axes[0].plot(levels,[np.mean([r["metrics"]["conditional_coverage"][str(q)] for r in rs]) for q in levels],
                     marker="o",label=model)
    axes[0].plot([.45,1],[.45,1],"k--",lw=1); axes[0].set(xlabel="nominal coverage",ylabel="empirical coverage",
        title="Heterogeneous 0.25%, biased rooms"); axes[0].legend(fontsize=7)
    for strategy in ("random_sensor","euclidean_maximin","geo_sensor_iv","wrong_sensor_iv"):
        means=[]
        for b in (.002,.005):
            rs=[r for r in sensor if r["strategy"]==strategy and r["evaluator"]=="geo_spectral"
                and r["requested_sensor_fraction"]==b]
            means.append(np.mean([r["metrics"]["nrmse"] for r in rs]))
        axes[1].plot([.2,.5],means,marker="o",label=strategy)
    axes[1].set(xlabel="sensor observation budget (%)",ylabel="NRMSE",
                title="Persistent-sensor design (controlled)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/"calibration_and_active.png",dpi=180); plt.close(fig)

    from geoaware.bayes_data import make_two_room_diffusion
    domain=make_two_room_diffusion(); fig, axes=plt.subplots(1,4,figsize=(11,2.8),sharex=True,sharey=True)
    chosen_seed=41
    # Plot only one row per strategy even though exploratory and confirmatory
    # manifests are concatenated above.
    for ax,strategy in zip(axes,("random_sensor","euclidean_maximin","geo_sensor_iv","wrong_sensor_iv")):
        row=next(r for r in sensor if r["seed"]==chosen_seed and r["strategy"]==strategy
                 and r["evaluator"]=="geo_spectral" and r["requested_sensor_fraction"]==.005)
        xy=domain.spatial_coordinates.numpy(); ids=np.asarray(row["sensor_ids"])
        ax.scatter(xy[:,0],xy[:,1],s=1,c="#d0d0d0"); ax.scatter(xy[ids,0],xy[ids,1],s=30,c="#d62728")
        ax.set_title(strategy.replace("_sensor","").replace("_"," "),fontsize=8); ax.set_aspect("equal")
    fig.suptitle("Five persistent sensors, seed 41"); fig.tight_layout()
    fig.savefig(OUT/"sensor_locations.png",dpi=180); plt.close(fig)

    tracked = [ROOT/"src/geoaware/bayes_data.py", ROOT/"src/geoaware/bayes_models.py",
               ROOT/"experiments/run_paper_a.py", ROOT/"experiments/run_paper_a_real.py",
               ROOT/"experiments/analyze_paper_a.py", ROOT/"papers/EVALUATION_PROTOCOL.md"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    runs={name:str((ROOT/"runs"/name/"results.json").stat().st_size) for name in
          ("paper_a_round1","paper_a_round2","paper_a_round3_static_heterogeneous",
           "paper_a_round3_active_heterogeneous","paper_a_round3_sensor_spectral",
           "paper_a_round3_real_active","paper_a_confirm_matched",
           "paper_a_confirm_heterogeneous","paper_a_confirm_sensor_spectral")}
    (OUT/"MANIFEST.json").write_text(json.dumps({"sha256":hashes,"result_bytes":runs},indent=2))


if __name__ == "__main__": main()
