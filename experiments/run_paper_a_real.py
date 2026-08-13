#!/usr/bin/env python3
"""External-data UQ stress test on The Well Active Matter."""

from __future__ import annotations

import argparse, json
from pathlib import Path

import torch

from geoaware.bases import BasisSpec, evaluate_basis
from geoaware.bayes_models import ExactFeatureBayes, ExactRBF, uncertainty_metrics
from geoaware.data import load_active_matter


def product_features(coords, specs, max_features):
    bases, eigs = [], []
    for d, spec in enumerate(specs):
        p, e = evaluate_basis(coords[:, d], spec); bases.append(p); eigs.append(e)
    combos = torch.cartesian_prod(*[torch.arange(len(e)) for e in eigs])
    joint = sum(eigs[d][combos[:, d]] for d in range(len(eigs)))
    keep = torch.argsort(joint)[:min(max_features, len(joint))]; combos = combos[keep]
    phi = torch.ones(len(coords), len(combos))
    for d, p in enumerate(bases): phi *= p[:, combos[:, d]]
    phi /= phi.square().mean(0, keepdim=True).sqrt().clamp_min(1e-8)
    # Separate temporal and spatial regularity.
    spectral = torch.stack([eigs[0][combos[:, 0]],
                            sum(eigs[d][combos[:, d]] for d in range(1, len(eigs)))], 1)
    return phi, spectral


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ratios", default=".001,.0025,.005"); p.add_argument("--seeds", default="41,42,43,44,45")
    p.add_argument("--max-features", type=int, default=768); p.add_argument("--noise", type=float, default=.10)
    p.add_argument("--device", default="cuda"); args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = load_active_matter(spatial_stride=2); coords = data.flat_coordinates(); truth = data.values.flatten()
    correct = product_features(coords, data.basis_specs, args.max_features)
    wrong_specs = tuple(BasisSpec("neumann", s.n_frequencies, "wrong-rectangle") for s in data.basis_specs)
    wrong = product_features(coords, wrong_specs, args.max_features)
    rows = []
    for ratio in map(float, args.ratios.split(",")):
        for seed in map(int, args.seeds.split(",")):
            g = torch.Generator().manual_seed(seed); nobs = max(8, round(ratio * len(truth)))
            obs = torch.randperm(len(truth), generator=g)[:nobs]
            noise_g = torch.Generator().manual_seed(seed + 9901)
            y = truth[obs] + torch.randn(nobs, generator=noise_g) * args.noise * truth[obs].std()
            held = torch.ones(len(truth), dtype=torch.bool); held[obs] = False
            for name, (phi, eig) in {"geo_spectral": correct, "wrong_geometry": wrong}.items():
                model = ExactFeatureBayes(phi, eig, "loo", True, args.device).fit(obs, y)
                pred = model.predict(); metrics = uncertainty_metrics(truth, pred, held)
                rows.append(dict(model=name, ratio=ratio, seed=seed, n_observed=nobs,
                                 metrics=metrics, hyperparameters=pred.hyperparameters))
                print(name, ratio, seed, metrics["nrmse"], metrics["conditional_coverage"]["0.95"], flush=True)
            model = ExactRBF(coords, "loo", True, args.device).fit(obs, y)
            pred = model.predict(); metrics = uncertainty_metrics(truth, pred, held)
            rows.append(dict(model="rbf_gp", ratio=ratio, seed=seed, n_observed=nobs,
                             metrics=metrics, hyperparameters=pred.hyperparameters))
    (args.output / "results.json").write_text(json.dumps({
        "dataset": data.name, "source": data.source, "shape": data.shape,
        "arguments": vars(args), "results": rows}, indent=2, default=str))


if __name__ == "__main__": main()
