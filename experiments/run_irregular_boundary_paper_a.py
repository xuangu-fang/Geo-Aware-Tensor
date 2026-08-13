#!/usr/bin/env python3
"""Lightweight Paper-A gate on genuinely irregular outer boundaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from geoaware.tensor_bayes import OperatorBayesianCP, OperatorBayesianTucker


def csr(payload, prefix):
    return sp.csr_matrix((payload[f"{prefix}_data"], payload[f"{prefix}_indices"],
                          payload[f"{prefix}_indptr"]),
                         shape=tuple(payload[f"{prefix}_shape"].tolist())).astype(np.float64)


def graph_basis(payload, modes: int):
    operator = csr(payload, "geometry_operator")
    values, vectors = spla.eigsh(operator, k=min(modes, operator.shape[0]-2),
                                 sigma=-1e-5, which="LM", tol=1e-7)
    order = np.argsort(values); values, vectors = values[order], vectors[:, order]
    positive = values[values > 1e-6]
    scale = positive[0] if len(positive) else 1.
    return (torch.from_numpy(vectors.astype(np.float32)),
            torch.from_numpy((np.maximum(values, 0)/scale).astype(np.float32)))


def rectangle_basis(coords: np.ndarray, modes: int):
    xy = (coords+1)/2
    pairs = []
    side = int(math.ceil(math.sqrt(modes)))+2
    for i in range(side):
        for j in range(side):
            pairs.append((i*i+j*j, i, j))
    pairs.sort(); pairs = pairs[:modes]
    columns = []
    for _, i, j in pairs:
        column = np.cos(math.pi*i*xy[:, 0])*np.cos(math.pi*j*xy[:, 1])
        column /= np.sqrt(np.mean(column**2))+1e-8
        columns.append(column)
    values = np.asarray([p[0] for p in pairs], np.float32)
    positive = values[values > 0]
    values /= positive[0] if len(positive) else 1.
    return torch.from_numpy(np.stack(columns, 1).astype(np.float32)), torch.from_numpy(values)


def cosine_basis(length: int, modes: int):
    grid = torch.arange(length).float()[:, None]
    frequency = torch.arange(modes).float()[None]
    basis = torch.cos(math.pi*(grid+.5)*frequency/length)
    basis[:, 0] /= math.sqrt(2.); basis *= math.sqrt(2./length)
    values = 4*torch.sin(math.pi*torch.arange(modes).float()/(2*length)).square()
    positive = values[values > 1e-8]
    return basis, values/(positive[0] if len(positive) else 1.)


def fixed_mask(shape, ratio, seed):
    generator = np.random.default_rng(seed)
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    mask[generator.choice(len(mask), max(1, int(round(ratio*len(mask)))), replace=False)] = True
    return mask.reshape(shape)


def run_case(path: Path, ratio: float, seed: int, steps: int, modes: int):
    payload = np.load(path)
    fields = torch.from_numpy(payload["field"].astype(np.float32))
    # Tensor modes are source x time x irregular-domain node. The companion
    # source case is loaded outside this function and injected by the caller.
    return payload, fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/irregular_boundary_wave"))
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--space-modes", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/irregular_boundary_paper_a_smoke.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data/"manifest.json").read_text())
    names = sorted({c["geometry"]["name"] for c in manifest["cases"]})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []
    for geometry_index, name in enumerate(names):
        paths = [args.data/f"{name}_r24_s{s}.npz" for s in range(2)]
        loaded = [run_case(path, args.ratio, args.seed, args.steps, args.space_modes)
                  for path in paths]
        payload = loaded[0][0]
        target = torch.stack([item[1] for item in loaded], 0)
        shape = target.shape
        observed_np = fixed_mask(shape, args.ratio, args.seed+geometry_index)
        observed = torch.from_numpy(observed_np)
        indices = torch.from_numpy(np.argwhere(observed_np)).long()
        center = target[observed].mean(); scale = target[observed].std().clamp_min(1e-6)
        y = (target[observed]-center)/scale
        initial = torch.zeros_like(target); initial[observed] = y
        source_basis = torch.eye(2); source_eigen = torch.zeros(2)
        time_basis, time_eigen = cosine_basis(shape[1], 20)
        correct_basis, correct_eigen = graph_basis(payload, args.space_modes)
        flat_basis, flat_eigen = rectangle_basis(payload["coordinates"], args.space_modes)
        permutation = torch.randperm(shape[2], generator=torch.Generator().manual_seed(1000+args.seed))
        wrong_basis = correct_basis[permutation]
        configs = {
            "correct_tucker": ("tucker", correct_basis, correct_eigen),
            "wrong_boundary_tucker": ("tucker", wrong_basis, correct_eigen),
            "topology_erased_tucker": ("tucker", flat_basis, flat_eigen),
            "correct_cp": ("cp", correct_basis, correct_eigen),
        }
        all_indices = torch.cartesian_prod(*[torch.arange(n) for n in shape])
        for model_name, (kind, space_basis, space_eigen) in configs.items():
            torch.manual_seed(args.seed)
            basis = [source_basis, time_basis, space_basis]
            eigen = [source_eigen, time_eigen, space_eigen]
            if kind == "tucker":
                model = OperatorBayesianTucker(basis, eigen, ranks=(2, 6, 8),
                                               power=1.5, device=device)
            else:
                model = OperatorBayesianCP(basis, eigen, rank=8, power=1.5, device=device)
            started = time.perf_counter()
            model.fit(indices, y, steps=args.steps, lr=3e-3, reg_weight=2e-3,
                      seed=args.seed, initial_tensor=initial)
            prediction = model.predict(all_indices)
            mean = prediction.mean.reshape(shape)*scale+center
            held = ~observed
            error = mean[held]-target[held]
            boundary_nodes = torch.from_numpy(payload["boundary_mask"].astype(bool)[
                tuple(payload["grid_indices"].T)])
            boundary_eval = held & boundary_nodes[None, None, :]
            row = {
                "geometry": name,
                "model": model_name,
                "n_nodes": shape[2],
                "parameters": sum(p.numel() for p in model.parameters()),
                "held_nrmse": float(error.square().mean().sqrt()/target[held].std().clamp_min(1e-8)),
                "boundary_nrmse": float((mean[boundary_eval]-target[boundary_eval]).square().mean().sqrt()
                                          / target[boundary_eval].std().clamp_min(1e-8)),
                "elapsed_seconds": time.perf_counter()-started,
            }
            all_rows.append(row)
            print(f"{name} {model_name}: held={row['held_nrmse']:.4f} "
                  f"boundary={row['boundary_nrmse']:.4f}", flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = {}
    for model_name in sorted({row["model"] for row in all_rows}):
        selected = [row for row in all_rows if row["model"] == model_name]
        summary[model_name] = {
            "macro_nrmse": float(np.mean([row["held_nrmse"] for row in selected])),
            "macro_boundary_nrmse": float(np.mean([row["boundary_nrmse"] for row in selected])),
        }
    result = {"experiment_id": "A-IRREGULAR-BOUNDARY-SMOKE-01", "status": "SMOKE",
              "config": vars(args), "geometry_count": len(names), "summary": summary,
              "rows": all_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
