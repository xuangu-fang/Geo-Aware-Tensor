#!/usr/bin/env python3
"""Build and audit the irregular outer-boundary wave gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geoaware.irregular_domain_solver import generate_irregular_dataset


def csr(payload, prefix):
    shape = tuple(payload[f"{prefix}_shape"].tolist())
    return sp.csr_matrix((payload[f"{prefix}_data"], payload[f"{prefix}_indices"],
                          payload[f"{prefix}_indptr"]), shape=shape)


def audit(output: Path, manifest: dict) -> dict:
    symmetry, min_eigen, finite, source_inside = [], [], [], []
    node_ranges, boundary_ranges, late_energy = [], [], []
    for case in manifest["cases"]:
        payload = np.load(output/case["file"])
        operator = csr(payload, "geometry_operator").astype(np.float64)
        skew = operator-operator.T
        symmetry.append(float(np.max(np.abs(skew.data))) if skew.nnz else 0.)
        # Shift-invert is reliable for the exact Neumann null mode; ``which=SA``
        # can stop at the first positive mode on these differently sized graphs.
        min_eigen.append(float(spla.eigsh(operator, k=1, sigma=-1e-5, which="LM",
                                          return_eigenvectors=False, tol=1e-7)[0]))
        field = payload["field"]
        finite.append(bool(np.isfinite(field).all() and field.std() > 1e-8))
        grid_index = payload["grid_indices"][case["simulation"]["source_node"]]
        source_inside.append(bool(payload["fluid_mask"][tuple(grid_index)]))
        node_ranges.append(len(payload["coordinates"]))
        boundary_ranges.append(int(payload["boundary_mask"].sum()))
        late = field[len(field)//2:]
        late_energy.append(float(np.sum(late**2)/(np.sum(field**2)+1e-12)))
    result = {
        "n_cases": len(manifest["cases"]),
        "n_geometries": len({x["geometry"]["name"] for x in manifest["cases"]}),
        "resolutions": sorted({x["resolution"] for x in manifest["cases"]}),
        "all_finite_non_degenerate": all(finite),
        "all_sources_inside": all(source_inside),
        "max_operator_symmetry_error": max(symmetry),
        "min_operator_eigenvalue": min(min_eigen),
        "node_count_range": [min(node_ranges), max(node_ranges)],
        "boundary_node_count_range": [min(boundary_ranges), max(boundary_ranges)],
        "late_energy_fraction_range": [min(late_energy), max(late_energy)],
    }
    if not result["all_finite_non_degenerate"] or not result["all_sources_inside"]:
        raise RuntimeError(f"field/source audit failed: {result}")
    if result["max_operator_symmetry_error"] > 1e-7 or result["min_operator_eigenvalue"] < -1e-5:
        raise RuntimeError(f"operator audit failed: {result}")
    return result


def plot(output: Path, manifest: dict, destination: Path):
    examples = {}
    for case in manifest["cases"]:
        if case["resolution"] == max(x["resolution"] for x in manifest["cases"]) and case["source_index"] == 0:
            examples[case["geometry"]["name"]] = case
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), constrained_layout=True)
    for axis, (name, case) in zip(axes.flat, examples.items()):
        payload = np.load(output/case["file"])
        image = np.full(payload["fluid_mask"].shape, np.nan, dtype=np.float32)
        image[payload["fluid_mask"].astype(bool)] = payload["field"][-1]
        scale = np.nanpercentile(np.abs(image), 98)
        axis.imshow(image.T, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set_title(name.replace("_", " "))
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle("Irregular outer-boundary wave gate: post-interaction fields")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/irregular_boundary_wave"))
    parser.add_argument("--summary", type=Path,
                        default=Path("papers/dataset_gates/irregular_boundary_wave_summary.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("papers/dataset_gates/irregular_boundary_wave.png"))
    args = parser.parse_args()
    manifest = generate_irregular_dataset(args.output)
    result = audit(args.output, manifest)
    plot(args.output, manifest, args.figure)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
