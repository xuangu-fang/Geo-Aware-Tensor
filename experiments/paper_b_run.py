#!/usr/bin/env python3
"""Three-round Paper-B campaign on unseen obstacle geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from geoaware.neural_geometry import (
    CoordinateField, GatedIntrinsicResidual, IntrinsicKernelField, IntrinsicPhaseField, SensorConditionedRFF, SharedNeuralCP, SpectralTransferNet, WaveTask,
    build_obstacle_domain, make_tasks, obstacle_boundary_mask, shadow_mask,
    task_coordinate_features,
)


TRAIN_SPECS = [
    {"kind": "circle", "cx": -.10, "cy": .00, "r": .20, "name": "train_circle_0"},
    {"kind": "circle", "cx": .08, "cy": .10, "r": .16, "name": "train_circle_1"},
    {"kind": "circle", "cx": -.02, "cy": -.14, "r": .24, "name": "train_circle_2"},
    {"kind": "ellipse", "cx": .02, "cy": .02, "rx": .25, "ry": .13, "angle": .25, "name": "train_ellipse_0"},
    {"kind": "ellipse", "cx": -.08, "cy": .12, "rx": .16, "ry": .25, "angle": -.35, "name": "train_ellipse_1"},
    {"kind": "ellipse", "cx": .12, "cy": -.10, "rx": .28, "ry": .12, "angle": .60, "name": "train_ellipse_2"},
]
TEST_SPECS = [
    {"kind": "circle", "cx": .14, "cy": -.04, "r": .22, "name": "test_circle"},
    {"kind": "ellipse", "cx": -.13, "cy": -.06, "rx": .29, "ry": .14, "angle": -.65, "name": "test_ellipse"},
    {"kind": "double", "cx1": -.10, "cy1": -.20, "r1": .13,
     "cx2": .13, "cy2": .19, "r2": .11, "name": "test_double"},
]
WALL_TRAIN_SPECS = [
    {"kind":"wall","cx":x,"width":w,"door_y":y,"gap":g,"name":f"train_wall_{i}"}
    for i,(x,w,y,g) in enumerate([(-.12,.10,-.35,.24),(.05,.12,.28,.22),(-.03,.08,.05,.18),
                                  (.14,.10,-.12,.20),(-.18,.12,.38,.26),(.0,.09,-.28,.16)])]
WALL_TEST_SPECS = [
    {"kind":"wall","cx":x,"width":w,"door_y":y,"gap":g,"name":f"test_wall_{i}"}
    for i,(x,w,y,g) in enumerate([(.08,.11,.42,.15),(-.10,.09,-.05,.14),(.16,.13,-.40,.18)])]
CLOSED_TRAIN_SPECS = [
    {"kind":"wall","cx":x,"width":w,"door_y":0.,"gap":0.,"name":f"train_closed_{i}"}
    for i,(x,w) in enumerate([(-.18,.08),(-.10,.12),(-.02,.09),(.06,.11),(.14,.08),(.20,.12)])]
CLOSED_TEST_SPECS = [
    {"kind":"wall","cx":x,"width":w,"door_y":0.,"gap":0.,"name":f"test_closed_{i}"}
    for i,(x,w) in enumerate([(-.14,.10),(.02,.13),(.17,.10)])]
NARROW_TRAIN_SPECS = [
    {"kind":"wall","cx":x,"width":w,"door_y":y,"gap":g,"name":f"train_narrow_{i}"}
    for i,(x,w,y,g) in enumerate([(-.12,.10,-.55,.09),(.05,.12,.52,.08),(-.03,.08,.35,.07),
                                  (.14,.10,-.48,.09),(-.18,.12,.58,.08),(.0,.09,-.60,.07)])]
NARROW_TEST_SPECS = [
    {"kind":"wall","cx":x,"width":w,"door_y":y,"gap":g,"name":f"test_narrow_{i}"}
    for i,(x,w,y,g) in enumerate([(.08,.11,.60,.07),(-.10,.09,-.52,.08),(.16,.13,.48,.07)])]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_family(resolution, n_eigen, family="mixed"):
    train_specs, test_specs = ((WALL_TRAIN_SPECS, WALL_TEST_SPECS) if family == "wall" else
                               (CLOSED_TRAIN_SPECS, CLOSED_TEST_SPECS) if family == "closed_wall" else
                               (NARROW_TRAIN_SPECS, NARROW_TEST_SPECS) if family == "narrow_wall" else
                               (TRAIN_SPECS, TEST_SPECS))
    train = [build_obstacle_domain(x, resolution, n_eigen) for x in train_specs]
    test = [build_obstacle_domain(x, resolution, n_eigen) for x in test_specs]
    return train, test


def prediction(model, task, model_name, device):
    if isinstance(model, SensorConditionedRFF):
        return model.forward_task(task, wrong_geometry=model_name == "wrong_geometry", use_context=True).detach().cpu()
    if isinstance(model, IntrinsicPhaseField):
        return model.forward_task(task, wrong_geometry=model_name == "wrong_geometry").detach().cpu()
    if model_name.startswith("graph") or model_name == "wrong_geometry":
        return model.forward_task(task, wrong_geometry=model_name == "wrong_geometry").detach().cpu()
    x = task_coordinate_features(task).to(device)
    return model(x).detach().cpu()


def fit(model, tasks, model_name, steps, lr, device, seed):
    model.to(device); model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(seed + 712)
    obs_scale = torch.cat([t.noisy_values[t.observed] for t in tasks]).std().clamp_min(1e-6)
    # Cache all static task features on device and batch across tasks.  This
    # avoids tens of thousands of tiny Python/device transfers in the first run.
    targets = torch.cat([t.noisy_values[t.observed] for t in tasks]).to(device) / obs_scale.to(device)
    if isinstance(model, IntrinsicPhaseField):
        x_all = torch.cat([model.feature_task(t, model_name == "wrong_geometry")[t.observed]
                           for t in tasks]).to(device)
    elif isinstance(model, SensorConditionedRFF):
        x_all = torch.cat([task_coordinate_features(t)[t.observed] for t in tasks]).to(device)
    elif isinstance(model, (IntrinsicKernelField, GatedIntrinsicResidual)):
        x_all = torch.cat([model.feature_task(t, model_name == "wrong_geometry")[t.observed]
                           for t in tasks]).to(device)
    elif model_name.startswith("graph") or model_name == "wrong_geometry":
        from geoaware.neural_geometry import rectangle_spectral_features
        z_parts, task_ids, lams = [], [], []
        for ti, task in enumerate(tasks):
            ids = torch.where(task.observed)[0]
            if model_name == "wrong_geometry":
                z, lam = rectangle_spectral_features(task, len(task.domain.eigenvalues))
            else:
                z = task.domain.eigenvectors * task.domain.source_projection[None]
                lam = task.domain.eigenvalues
            z_parts.append(z[ids]); lams.append(lam)
            task_ids.append(torch.full((len(ids),), ti, dtype=torch.long))
        z_all = torch.cat(z_parts).to(device); task_ids = torch.cat(task_ids).to(device)
        lam_all = torch.stack(lams).to(device)
        times = torch.tensor([t.time for t in tasks], device=device)
    else:
        x_all = torch.cat([task_coordinate_features(t)[t.observed] for t in tasks]).to(device)
    history = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True); total = torch.zeros((), device=device)
        if isinstance(model, (IntrinsicKernelField, GatedIntrinsicResidual, SensorConditionedRFF, IntrinsicPhaseField)):
            pred = model(x_all) / obs_scale.to(device)
        elif model_name.startswith("graph") or model_name == "wrong_geometry":
            transfers = model.transfer(lam_all, times)
            pred = (z_all * transfers[task_ids]).sum(1) / obs_scale.to(device)
        else:
            pred = model(x_all) / obs_scale.to(device)
        total = (pred - targets).square().mean()
        # Mild parameter control only; no target values outside sensors are used.
        reg = sum(p.square().mean() for p in model.parameters())
        loss = total + 1e-7 * reg
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach()), "sensor_mse": float(total.detach())})
    return history


def metrics(task, pred):
    truth = task.values; eps = 1e-8
    held = ~task.observed
    global_scale = truth[held].std().clamp_min(eps)
    def nrmse(mask, local=False):
        if int(mask.sum()) < 2: return float("nan")
        denom = truth[mask].std().clamp_min(eps) if local else global_scale
        return float(torch.sqrt(torch.mean((pred[mask] - truth[mask]) ** 2)) / denom)
    phi = task.domain.eigenvectors
    # Empirical continuous projection; high band is the upper 40% of represented modes.
    err_coeff = phi.T @ (pred - truth) / task.domain.n_nodes
    true_coeff = phi.T @ truth / task.domain.n_nodes
    cut = int(0.60 * len(true_coeff))
    high_abs = torch.sqrt(torch.mean(err_coeff[cut:].square()))
    high_fraction = float(true_coeff[cut:].square().sum() / true_coeff.square().sum().clamp_min(eps))
    qualified = high_fraction >= .01
    high_rel = (float(torch.linalg.vector_norm(err_coeff[cut:]) /
                      torch.linalg.vector_norm(true_coeff[cut:]).clamp_min(eps)) if qualified else None)
    low_rel = float(torch.linalg.vector_norm(err_coeff[:cut]) / torch.linalg.vector_norm(true_coeff[:cut]).clamp_min(eps))
    return {
        "nrmse": nrmse(held), "boundary_nrmse": nrmse(held & obstacle_boundary_mask(task.domain)),
        "boundary_local_nrmse": nrmse(held & obstacle_boundary_mask(task.domain), local=True),
        "shadow_nrmse": nrmse(held & shadow_mask(task.domain)), "high_band_rel_l2": high_rel,
        "shadow_local_nrmse": nrmse(held & shadow_mask(task.domain), local=True),
        "low_band_rel_l2": low_rel, "high_band_nrmse": float(high_abs / truth.std().clamp_min(eps)),
        "high_band_energy_fraction": high_fraction, "high_band_qualified": qualified,
        "high_band_start_eigenvalue": float(task.domain.eigenvalues[cut]),
    }


def summarize(rows):
    out = {}
    for split in sorted(set(r["split"] for r in rows)):
        for model in sorted(set(r["model"] for r in rows)):
            subset = [r for r in rows if r["split"] == split and r["model"] == model]
            if not subset: continue
            out[f"{split}/{model}"] = {}
            for k in ("nrmse", "boundary_nrmse", "shadow_nrmse", "high_band_rel_l2", "low_band_rel_l2", "high_band_nrmse", "high_band_energy_fraction", "high_band_start_eigenvalue"):
                vals = [r[k] for r in subset if r[k] is not None and np.isfinite(r[k])]
                out[f"{split}/{model}"][k] = {"mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}
    return out


def run(args):
    seed_all(args.seed); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_domains, test_domains = build_family(args.train_resolution, args.n_eigen, args.geometry_family)
    if args.test_resolution != args.train_resolution:
        specs = (WALL_TEST_SPECS if args.geometry_family == "wall" else
                 CLOSED_TEST_SPECS if args.geometry_family == "closed_wall" else TEST_SPECS)
        if args.geometry_family == "narrow_wall": specs = NARROW_TEST_SPECS
        test_domains = [build_obstacle_domain(x, args.test_resolution, args.n_eigen) for x in specs]
    train_tasks = make_tasks(train_domains, args.train_times, args.ratio, args.seed, args.noise, args.mask, args.target_kind)
    # OOD masks are diagnostics only: no OOD values are used by fit().
    test_tasks = make_tasks(test_domains, args.test_times, args.ratio, args.seed + 999, args.noise, "random", args.target_kind)
    # Held-out domains use their sparse observations only as context. Metrics
    # are evaluated on their complement (few-shot geometry transfer).
    in_dim = task_coordinate_features(train_tasks[0]).shape[1]
    models = {
        "siren": CoordinateField(in_dim, args.hidden, "siren", args.seed),
        "rff": CoordinateField(in_dim, args.hidden, "rff", args.seed),
        "neural_cp": SharedNeuralCP(in_dim - 2, rank=args.rank, hidden=max(32, args.hidden // 2)),
        "wrong_geometry": (IntrinsicPhaseField(in_dim, args.hidden, args.adapter_kind == "phase") if args.adapter_kind in ("phase", "distance") else
                           SpectralTransferNet(args.hidden, phase_aligned=True) if args.adapter_kind == "spectral" else
                           SensorConditionedRFF(in_dim, args.hidden, args.seed, args.diffusion, args.context_ridge, args.context_base)),
        "graph_adapter": (IntrinsicPhaseField(in_dim, args.hidden, args.adapter_kind == "phase") if args.adapter_kind in ("phase", "distance") else
                          SpectralTransferNet(args.hidden, phase_aligned=True) if args.adapter_kind == "spectral" else
                          SensorConditionedRFF(in_dim, args.hidden, args.seed, args.diffusion, args.context_ridge, args.context_base)),
    }
    # Capacity, initialization, and residual path are exactly matched.
    models["graph_adapter"].load_state_dict(models["wrong_geometry"].state_dict())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows, histories = [], {}
    started = time.time()
    for name, model in models.items():
        seed_all(args.seed)
        histories[name] = fit(model, train_tasks, name, args.steps, args.lr, device, args.seed)
        model.eval()
        for split, tasks in (("seen_geometry", train_tasks), ("unseen_geometry", test_tasks)):
            with torch.no_grad():
                for task in tasks:
                    row = {"round": args.round, "seed": args.seed, "model": name, "split": split,
                           "task": task.name, "resolution": task.domain.resolution,
                           "observation_ratio": float(task.observed.float().mean()),
                           **metrics(task, prediction(model, task, name, device))}
                    rows.append(row)
    payload = {"config": vars(args), "elapsed_seconds": time.time() - started,
               "summary": summarize(rows), "rows": rows, "history": histories}
    (out / f"seed_{args.seed}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--ratio", type=float, default=.005); p.add_argument("--noise", type=float, default=.05)
    p.add_argument("--mask", choices=["random", "upstream", "boundary_stratified"], default="random")
    p.add_argument("--train-resolution", type=int, default=36); p.add_argument("--test-resolution", type=int, default=36)
    p.add_argument("--n-eigen", type=int, default=80); p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--hidden", type=int, default=96); p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda"); p.add_argument("--phase-aligned", action="store_true")
    p.add_argument("--target-kind", choices=["matched", "heterogeneous", "elliptic", "geodesic"], default="heterogeneous")
    p.add_argument("--diffusion", type=float, default=.008)
    p.add_argument("--context-ridge", type=float, default=.08)
    p.add_argument("--context-base", choices=["rff", "siren"], default="rff")
    p.add_argument("--adapter-kind", choices=["context", "spectral", "phase", "distance"], default="context")
    p.add_argument("--geometry-family", choices=["mixed", "wall", "closed_wall", "narrow_wall"], default="mixed")
    p.add_argument("--train-times", type=float, nargs="+", default=[.16, .24, .32, .40])
    p.add_argument("--test-times", type=float, nargs="+", default=[.20, .28, .36])
    p.add_argument("--output", required=True)
    run(p.parse_args())


if __name__ == "__main__": main()
