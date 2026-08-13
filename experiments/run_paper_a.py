#!/usr/bin/env python3
"""Paper A: extreme-sparse geometry-resolved Bayesian field experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from geoaware.bayes_data import (make_two_room_diffusion, graph_time_features,
                                 rectangular_time_features, rff_features)
from geoaware.bayes_models import (BayesianPrediction, ExactFeatureBayes, ExactRBF,
                                   uncertainty_metrics)
from geoaware.models import NeuralCP, SirenINR


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def observation_ids(data, ratio: float, kind: str, seed: int) -> torch.Tensor:
    n = len(data.flat_values); count = max(8, round(ratio * n))
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        return torch.randperm(n, generator=g)[:count]
    if kind == "room_imbalance":
        # 90% of measurements in the left room, despite equal test weighting.
        left = torch.where(data.coordinates[:, 1] < .48)[0]
        right = torch.where(data.coordinates[:, 1] > .52)[0]
        nl = min(len(left), round(.9 * count)); nr = count - nl
        return torch.cat([left[torch.randperm(len(left), generator=g)[:nl]],
                          right[torch.randperm(len(right), generator=g)[:nr]]])
    if kind == "sensor_tracks":
        nodes = torch.randperm(data.shape[1], generator=g)[:max(1, round(count / data.shape[0]))]
        ids = (torch.arange(data.shape[0])[:, None] * data.shape[1] + nodes[None, :]).flatten()
        return ids[:count]
    raise ValueError(kind)


def noisy_observations(truth: torch.Tensor, obs: torch.Tensor, noise_fraction: float,
                       seed: int) -> torch.Tensor:
    y = truth[obs].clone(); scale = y.std().clamp_min(1e-6)
    g = torch.Generator().manual_seed(seed + 9901)
    return y + torch.randn(len(y), generator=g) * noise_fraction * scale


def deterministic_fit(name: str, coords: torch.Tensor, truth: torch.Tensor, obs: torch.Tensor,
                      y: torch.Tensor, steps: int, seed: int, device: str) -> dict:
    seed_all(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if name == "siren": model = SirenINR(3, hidden=96, depth=3, omega=20.)
    elif name == "neural_cp": model = NeuralCP(3, rank=10, hidden=64)
    else: raise ValueError(name)
    model = model.to(dev); x = coords[obs].to(dev); yy = y.to(dev)
    center, scale = yy.mean(), yy.std().clamp_min(1e-6); yn = (yy - center) / scale
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    for _ in range(steps):
        pred = model(x)
        loss = (pred - yn).square().mean() + 1e-5 * model.regularization()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(coords), 8192): chunks.append(model(coords[i:i+8192].to(dev), sample=False).cpu())
    mean = torch.cat(chunks) * float(scale) + float(center)
    held = torch.ones(len(truth), dtype=torch.bool); held[obs] = False
    err = mean[held] - truth[held]
    return {"rmse": float(err.square().mean().sqrt()),
            "nrmse": float(err.square().mean().sqrt() / truth[held].std().clamp_min(1e-8)),
            "mae": float(err.abs().mean()), "parameters": sum(p.numel() for p in model.parameters())}


def fit_one(name, data, feature_sets, truth, obs, y, selector, calibrate, device):
    if name == "rbf_gp": model = ExactRBF(data.coordinates, selector, calibrate, device)
    else:
        phi, eig = feature_sets[name]
        model = ExactFeatureBayes(phi, eig, selector, calibrate, device)
    model.fit(obs, y); pred = model.predict()
    held = torch.ones(len(truth), dtype=torch.bool); held[obs] = False
    metrics = uncertainty_metrics(truth, pred, held)
    regions = {}
    for region, selected in {
        "left_room": data.coordinates[:, 1] < .48,
        "right_room": data.coordinates[:, 1] > .52,
        "near_wall": (data.coordinates[:, 1] - .5).abs() < .12,
    }.items():
        region_held = held & selected
        regions[region] = uncertainty_metrics(truth, pred, region_held)
    metrics["regions"] = regions
    return model, pred, metrics


def run_static(args, data, feature_sets, truth):
    rows = []
    models = [x for x in args.models.split(",") if x]
    for mask in args.masks.split(","):
        for ratio in [float(x) for x in args.ratios.split(",")]:
            for seed in [int(x) for x in args.seeds.split(",")]:
                obs = observation_ids(data, ratio, mask, seed)
                y = noisy_observations(truth, obs, args.noise, seed)
                for name in models:
                    started = time.perf_counter()
                    if name in ("siren", "neural_cp"):
                        metrics = deterministic_fit(name, data.coordinates, truth, obs, y,
                                                    args.steps, seed, args.device)
                        hp = {}
                    else:
                        _, pred, metrics = fit_one(name, data, feature_sets, truth, obs, y,
                                                   args.selector, not args.no_calibration, args.device)
                        hp = pred.hyperparameters
                    row = dict(experiment="static", model=name, mask=mask, ratio=ratio, seed=seed,
                               n_observed=len(obs), noise_fraction=args.noise, selector=args.selector,
                               metrics=metrics, hyperparameters=hp,
                               elapsed_seconds=time.perf_counter() - started)
                    rows.append(row)
                    print(f"static {mask} {ratio:.4g} s{seed} {name}: "
                          f"NRMSE={metrics['nrmse']:.3f} "
                          f"cov95={metrics.get('cal_coverage',{}).get('0.95','-')}", flush=True)
    return rows


def run_active(args, data, feature_sets, truth):
    rows = []; phi, eig = feature_sets["geo_spectral"]
    n = len(truth); start_n = max(8, round(args.active_start * n))
    budgets = [float(x) for x in args.active_budgets.split(",")]
    for seed in [int(x) for x in args.seeds.split(",")]:
        g = torch.Generator().manual_seed(seed + 177)
        initial = torch.randperm(n, generator=g)[:start_n]
        random_order = torch.randperm(n, generator=g)
        for strategy in ("random", "space_filling", "geo_max_variance",
                         "geo_integrated_variance", "rbf_max_variance"):
            obs = initial.clone()
            acquisition_history = [initial.tolist()]
            for budget in budgets:
                target_n = max(start_n, round(budget * n))
                while len(obs) < target_n:
                    available = torch.ones(n, dtype=torch.bool); available[obs] = False
                    candidates = torch.where(available)[0]
                    take = min(args.active_batch, target_n - len(obs))
                    if strategy == "random":
                        choice = random_order[torch.isin(random_order, obs, invert=True)][:take]
                    elif strategy == "space_filling":
                        # Greedy maximin in raw Euclidean coordinates.  This is
                        # a strong model-free design but cannot see the wall.
                        chosen = []
                        for _ in range(take):
                            remaining = candidates[~torch.isin(
                                candidates, torch.tensor(chosen, dtype=torch.long))]
                            design = torch.cat([obs, torch.tensor(chosen, dtype=torch.long)])
                            score = torch.cdist(data.coordinates[remaining], data.coordinates[design]).amin(1)
                            chosen.append(int(remaining[torch.argmax(score)]))
                        choice = torch.tensor(chosen, dtype=torch.long)
                    else:
                        y = noisy_observations(truth, obs, args.noise, seed)
                        if strategy == "rbf_max_variance":
                            model = ExactRBF(data.coordinates, args.selector, True, args.device).fit(obs, y)
                            score = model.predict().raw_std[candidates]
                        else:
                            model = ExactFeatureBayes(phi, eig, args.selector, True, args.device).fit(obs, y)
                        if strategy == "geo_max_variance":
                            score = model.predict().raw_std[candidates]
                        elif strategy == "geo_integrated_variance":
                            # A deterministic pool keeps global variance
                            # reduction feasible without peeking at targets.
                            pool_n = min(args.active_pool, len(candidates))
                            pool_gen = torch.Generator().manual_seed(seed + len(obs) * 37)
                            pool = candidates[torch.randperm(len(candidates), generator=pool_gen)[:pool_n]]
                            score = model.integrated_variance_scores(pool)
                            candidates = pool
                        choice = candidates[torch.topk(score, take).indices]
                    obs = torch.cat([obs, choice])
                    acquisition_history.append(choice.tolist())
                y = noisy_observations(truth, obs, args.noise, seed)
                # Acquisition quality is assessed with both common evaluators,
                # separating the design from its coupled reconstruction model.
                for eval_model in ("geo_spectral", "rbf_gp"):
                    _, pred, metrics = fit_one(eval_model, data, feature_sets, truth, obs, y,
                                               args.selector, True, args.device)
                    row = dict(experiment="active", strategy=strategy, evaluator=eval_model,
                               coupled=((strategy.startswith("geo_") and eval_model == "geo_spectral")
                                        or (strategy == "rbf_max_variance" and eval_model == "rbf_gp")),
                               ratio=budget, seed=seed, n_observed=len(obs), metrics=metrics,
                               hyperparameters=pred.hyperparameters,
                               observation_ids=obs.tolist(),
                               observation_coordinates=data.coordinates[obs].tolist(),
                               acquisition_history=acquisition_history)
                    rows.append(row)
                    print(f"active {strategy}/{eval_model} {budget:.4g} s{seed}: "
                          f"NRMSE={metrics['nrmse']:.3f}", flush=True)
    return rows


def run_sensor_active(args, data, feature_sets, truth):
    """Place persistent spatial sensors that observe the entire trajectory."""
    rows = []; n_time, n_nodes = data.shape
    start_sensors = max(1, round(args.active_start * n_nodes))
    budgets = [float(x) for x in args.active_budgets.split(",")]
    strategies = ("random_sensor", "euclidean_maximin", "geo_sensor_iv", "wrong_sensor_iv")
    for seed in [int(x) for x in args.seeds.split(",")]:
        g = torch.Generator().manual_seed(seed + 177)
        initial = torch.randperm(n_nodes, generator=g)[:start_sensors]
        random_order = torch.randperm(n_nodes, generator=g)
        for strategy in strategies:
            sensors = initial.clone(); acquisition_history = [initial.tolist()]
            for budget in budgets:
                target = max(start_sensors, round(budget * n_nodes))
                while len(sensors) < target:
                    candidates = torch.where(~torch.isin(torch.arange(n_nodes), sensors))[0]
                    take = min(args.active_batch, target - len(sensors))
                    if strategy == "random_sensor":
                        choice = random_order[~torch.isin(random_order, sensors)][:take]
                    elif strategy == "euclidean_maximin":
                        chosen = []
                        for _ in range(take):
                            remaining = candidates[~torch.isin(
                                candidates, torch.tensor(chosen, dtype=torch.long))]
                            design = torch.cat([sensors, torch.tensor(chosen, dtype=torch.long)])
                            score = torch.cdist(data.spatial_coordinates[remaining],
                                                data.spatial_coordinates[design]).amin(1)
                            chosen.append(int(remaining[torch.argmax(score)]))
                        choice = torch.tensor(chosen, dtype=torch.long)
                    else:
                        obs = (torch.arange(n_time)[:, None] * n_nodes + sensors[None, :]).flatten()
                        y = noisy_observations(truth, obs, args.noise, seed)
                        feature_name = "geo_spectral" if strategy == "geo_sensor_iv" else "wrong_geometry"
                        phi, eig = feature_sets[feature_name]
                        model = ExactFeatureBayes(phi, eig, args.selector, True, args.device).fit(obs, y)
                        # Exact grouped IV: installing one sensor reveals its
                        # whole time trace, and temporal redundancy is accounted
                        # for by conditioning jointly on the group.
                        groups = torch.arange(n_time)[:, None] * n_nodes + candidates[None, :]
                        score = model.group_integrated_variance_scores(groups.T.contiguous())
                        choice = candidates[torch.topk(score, take).indices]
                    sensors = torch.cat([sensors, choice]); acquisition_history.append(choice.tolist())
                obs = (torch.arange(n_time)[:, None] * n_nodes + sensors[None, :]).flatten()
                y = noisy_observations(truth, obs, args.noise, seed)
                for eval_model in ("geo_spectral", "rbf_gp"):
                    _, pred, metrics = fit_one(eval_model, data, feature_sets, truth, obs, y,
                                               args.selector, True, args.device)
                    row = dict(experiment="sensor_active", strategy=strategy, evaluator=eval_model,
                               ratio=float(len(obs) / len(truth)), requested_sensor_fraction=budget,
                               seed=seed, n_sensors=len(sensors), n_observed=len(obs), metrics=metrics,
                               hyperparameters=pred.hyperparameters, sensor_ids=sensors.tolist(),
                               sensor_coordinates=data.spatial_coordinates[sensors].tolist(),
                               acquisition_history=acquisition_history)
                    rows.append(row)
                    print(f"sensor {strategy}/{eval_model} {budget:.4g} s{seed}: "
                          f"NRMSE={metrics['nrmse']:.3f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ratios", default="0.001,0.0025,0.005")
    p.add_argument("--masks", default="random,room_imbalance")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--models", default="geo_spectral,wrong_geometry,rff,rbf_gp,siren,neural_cp")
    p.add_argument("--selector", choices=["evidence", "loo"], default="loo")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--max-features", type=int, default=1440)
    p.add_argument("--noise", type=float, default=.10)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", choices=["spectral", "heterogeneous"], default="spectral")
    p.add_argument("--active", action="store_true")
    p.add_argument("--active-mode", choices=["points", "sensors"], default="points")
    p.add_argument("--active-start", type=float, default=.001)
    p.add_argument("--active-budgets", default="0.002,0.0035,0.005")
    p.add_argument("--active-batch", type=int, default=8)
    p.add_argument("--active-pool", type=int, default=2048)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    seed_all(123); data = make_two_room_diffusion(variant=args.task); truth = data.flat_values
    feature_sets = {
        "geo_spectral": graph_time_features(data, args.max_features),
        "wrong_geometry": rectangular_time_features(data, args.max_features),
        "rff": rff_features(data, args.max_features),
    }
    if args.active and args.active_mode == "sensors":
        rows = run_sensor_active(args, data, feature_sets, truth)
    elif args.active:
        rows = run_active(args, data, feature_sets, truth)
    else:
        rows = run_static(args, data, feature_sets, truth)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": vars(args), "dataset": {"name": data.name, "shape": data.shape,
                "n_total": len(truth), "description": data.description}, "results": rows}
    (args.output / "results.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__": main()
