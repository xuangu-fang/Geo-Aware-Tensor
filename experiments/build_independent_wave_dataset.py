#!/usr/bin/env python3
"""Build and audit the independent multi-geometry wave smoke dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geoaware.independent_wave_solver import generate_wave_dataset


def _csr(payload, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (payload[f"{prefix}_data"], payload[f"{prefix}_indices"],
         payload[f"{prefix}_indptr"]),
        shape=tuple(payload[f"{prefix}_shape"]),
    )


def audit_cases(data_dir: Path, manifest: dict) -> dict:
    symmetry_errors = []
    smallest_eigenvalues = []
    post_scattering_energy_fractions = []
    for case in manifest["cases"]:
        with np.load(data_dir/case["file"]) as payload:
            field = payload["field"]
            if not np.isfinite(field).all() or float(field.std()) <= 1e-6:
                raise RuntimeError(f"non-finite or degenerate field: {case['file']}")
            if not bool(payload["fluid_mask"][tuple(payload["grid_indices"][case["simulation"]["source_node"]])]):
                raise RuntimeError(f"source is not in fluid domain: {case['file']}")
            for prefix in ("geometry_operator", "wave_operator"):
                operator = _csr(payload, prefix).astype(np.float64)
                skew = operator-operator.T
                symmetry_errors.append(float(np.max(np.abs(skew.data))) if skew.nnz else 0.)
                smallest_eigenvalues.append(float(spla.eigsh(
                    operator, k=1, which="SA", return_eigenvectors=False, tol=1e-5)[0]))
            post = payload["record_times"] >= 1.
            total_energy = float(np.sum(field**2)) + 1e-12
            post_scattering_energy_fractions.append(float(np.sum(field[post]**2))/total_energy)
    audit = {
        "all_fields_finite_and_nondegenerate": True,
        "all_sources_in_fluid": True,
        "max_operator_symmetry_error": max(symmetry_errors),
        "min_operator_eigenvalue": min(smallest_eigenvalues),
        "post_scattering_energy_fraction_range": [
            min(post_scattering_energy_fractions), max(post_scattering_energy_fractions)],
    }
    if audit["max_operator_symmetry_error"] > 1e-7 or audit["min_operator_eigenvalue"] < -1e-5:
        raise RuntimeError(f"operator audit failed: {audit}")
    return audit


def plot_gate(data_dir: Path, manifest: dict, output: Path):
    cases = [c for c in manifest["cases"] if c["resolution"] == min(manifest["resolutions"])
             and c["source_index"] == 0]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        payload = np.load(data_dir/case["file"])
        mask = payload["fluid_mask"].astype(bool)
        image = np.full(mask.shape, np.nan, dtype=np.float32)
        image[mask] = payload["field"][-1]
        scale = np.nanquantile(np.abs(image), .98)
        ax.imshow(image.T, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
        source = payload["source_xy"]
        resolution = mask.shape[0]
        source_pixel = (source+1)*(resolution-1)/2
        ax.scatter(source_pixel[0], source_pixel[1], marker="*", color="black", s=32)
        ax.set_title(case["geometry"]["name"], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Independent wave smoke set: post-scattering pressure at t=2.0, source 0")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/independent_wave_smoke"))
    parser.add_argument("--report-dir", type=Path, default=Path("papers/dataset_gates"))
    parser.add_argument("--resolutions", default="24,32")
    args = parser.parse_args()
    resolutions = tuple(map(int, args.resolutions.split(",")))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_dir/"independent_wave_smoke_summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else None
    manifest = generate_wave_dataset(args.output, resolutions=resolutions)
    plot_gate(args.output, manifest, args.report_dir/"independent_wave_smoke.png")
    checksums = {x["file"]: x["sha256"] for x in manifest["cases"]}
    audit = audit_cases(args.output, manifest)
    compact = {
        "schema_version": manifest["schema_version"],
        "generator": manifest["generator"],
        "equation": manifest["equation"],
        "tensor_semantics": manifest["tensor_semantics"],
        "resolutions": manifest["resolutions"],
        "sources": manifest["sources"],
        "n_geometries": manifest["n_geometries"],
        "n_cases": manifest["n_cases"],
        "field_std_range": [min(x["field_std"] for x in manifest["cases"]),
                            max(x["field_std"] for x in manifest["cases"])],
        "field_abs_max_range": [min(x["field_abs_max"] for x in manifest["cases"]),
                                max(x["field_abs_max"] for x in manifest["cases"])],
        "record_time_range": [manifest["record_times"][0], manifest["record_times"][-1]],
        "audit": audit,
        "repeat_run_checksums_match": bool(previous and previous.get("case_checksums") == checksums),
        "case_checksums": checksums,
    }
    summary_path.write_text(json.dumps(compact, indent=2))
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
