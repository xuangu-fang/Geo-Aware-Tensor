#!/usr/bin/env python3
"""Official NeuralOperator FNO/TFNO baselines for the frozen early-40 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from neuralop.models import FNO, TFNO

from geoaware.the_well_pilot import fixed_random_mask, load_the_well_case


def load_case(path: Path, horizon: int):
    inputs, target = load_the_well_case(path)
    target = target[:horizon].astype(np.float32)
    density = (np.log10(np.maximum(inputs["density"], 1.))/6.).astype(np.float32)
    speed = inputs["speed_of_sound"].astype(np.float32)
    speed = (speed-speed.mean())/(speed.std()+1e-6)
    initial = inputs["initial_pressure"].astype(np.float32)
    initial /= initial.std()+1e-6
    solid = (inputs["density"] > 1e5).astype(np.float32)
    base = np.stack([density, speed, initial, solid], 0)
    times = inputs["query_times"][:horizon].astype(np.float32)
    times = (times-times.min())/(times.max()-times.min()+1e-8)
    return {"name": path.stem, "base": torch.from_numpy(base),
            "times": torch.from_numpy(times), "target": torch.from_numpy(target)}


def model_input(case, time_indices):
    base = case["base"][None].expand(len(time_indices), -1, -1, -1)
    time = case["times"][time_indices, None, None, None].expand(
        -1, 1, base.shape[-2], base.shape[-1])
    return torch.cat([base, time], 1)


@torch.no_grad()
def evaluate(model, cases, scale, device, batch_size):
    metrics = []
    model.eval()
    for case in cases:
        predictions = []
        for start in range(0, len(case["times"]), batch_size):
            time_indices = torch.arange(start, min(start+batch_size, len(case["times"])))
            prediction = model(model_input(case, time_indices).to(device)).squeeze(1).cpu()*scale
            predictions.append(prediction)
        prediction = torch.cat(predictions)
        target = case["target"]
        error = prediction-target
        metrics.append({
            "case": case["name"],
            "nrmse_std": float(error.square().mean().sqrt()/target.std().clamp_min(1e-8)),
            "vrmse": float(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-8)),
        })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/the_well_acoustic_64x64"))
    parser.add_argument("--model", choices=["fno", "tfno"], default="fno")
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--evaluation-cases", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/the_well_official_fno_selection.json"))
    args = parser.parse_args()
    if args.evaluation_split == "test" and args.seed < 10:
        raise SystemExit("test confirmation requires a fresh seed >=10")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = [load_case(args.data/"train"/f"trajectory_{index:03d}.npz", args.horizon)
             for index in range(args.train_cases)]
    evaluation = [load_case(args.data/args.evaluation_split/f"trajectory_{index:03d}.npz", args.horizon)
                  for index in range(args.evaluation_cases)]
    masks = [torch.from_numpy(fixed_random_mask(case["target"].shape, args.ratio,
                                                args.seed+index))
             for index, case in enumerate(train)]
    observed_values = torch.cat([case["target"][mask] for case, mask in zip(train, masks)])
    target_scale = observed_values.std().clamp_min(1e-6)
    valid_pairs = [(case_index, time_index)
                   for case_index, mask in enumerate(masks)
                   for time_index in range(args.horizon) if bool(mask[time_index].any())]
    common = dict(n_modes=(args.modes, args.modes), in_channels=5, out_channels=1,
                  hidden_channels=args.hidden_channels, n_layers=4,
                  domain_padding=.0625)
    model = (FNO(**common) if args.model == "fno" else
             TFNO(**common, rank=.25)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = np.random.default_rng(args.seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    history = []; started = time.perf_counter(); model.train()
    for step in range(args.steps):
        chosen = generator.integers(len(valid_pairs), size=args.batch_size)
        inputs, targets, supervision = [], [], []
        for pair_index in chosen:
            case_index, time_index = valid_pairs[int(pair_index)]
            case = train[case_index]
            inputs.append(model_input(case, torch.tensor([time_index]))[0])
            targets.append(case["target"][time_index])
            supervision.append(masks[case_index][time_index])
        x = torch.stack(inputs).to(device)
        y = torch.stack(targets).to(device)/target_scale.to(device)
        mask = torch.stack(supervision).to(device)
        prediction = model(x).squeeze(1)
        loss = (prediction[mask]-y[mask]).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step(); scheduler.step()
        if step % max(1, args.steps//10) == 0 or step == args.steps-1:
            history.append({"step": step, "loss": float(loss.detach()),
                            "learning_rate": scheduler.get_last_lr()[0]})
    metrics = evaluate(model, evaluation, target_scale, device, args.batch_size)
    result = {
        "experiment_id": f"B-WELL-EARLY40-OFFICIAL-{args.model.upper()}-{args.evaluation_split.upper()}",
        "status": "SELECTION" if args.evaluation_split == "validation" else "POSTHOC_CONFIRMATION",
        "library": {"name": "neuraloperator", "version": "2.0.0"},
        "config": vars(args),
        "training_values_read": int(sum(mask.sum() for mask in masks)),
        "realized_ratio": float(sum(mask.sum() for mask in masks)
                                /sum(mask.numel() for mask in masks)),
        "target_scale_from_observed_only": float(target_scale),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter()-started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "evaluation_macro_nrmse": float(np.mean([item["nrmse_std"] for item in metrics])),
        "evaluation_macro_vrmse": float(np.mean([item["vrmse"] for item in metrics])),
        "case_metrics": metrics,
        "history": history,
    }
    print(f"{args.model}: {args.evaluation_split} macro NRMSE="
          f"{result['evaluation_macro_nrmse']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
