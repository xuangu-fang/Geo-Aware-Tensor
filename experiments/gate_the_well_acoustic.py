#!/usr/bin/env python3
"""Range-read and audit eight trajectories from The Well acoustic maze data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


REPO = "polymathic-ai/acoustic_scattering_maze"
API = "https://huggingface.co/api/datasets"
PINNED_REVISION = "8df383a3223f40f7ce66fe77b4ff4d7006dbc272"
SAMPLE_PLAN = {
    "train": ("data/train/acoustic_scattering_maze_chunk_0.hdf5", range(4)),
    "validation": ("data/valid/acoustic_scattering_maze_chunk_16.hdf5", range(2)),
    "test": ("data/test/acoustic_scattering_maze_chunk_18.hdf5", range(2)),
}
TIME_INDICES = np.asarray([0, 40, 80, 120, 160, 201], dtype=np.int32)


def get_json(url: str):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def source_components(initial_pressure: np.ndarray) -> int:
    threshold = .12*float(np.max(np.abs(initial_pressure)))
    _, count = ndimage.label(np.abs(initial_pressure) > threshold)
    return int(count)


def read_gate_samples(revision: str, output: Path) -> tuple[list[dict], str]:
    samples = []
    arrays = {}
    for split, (shard, trajectory_ids) in SAMPLE_PLAN.items():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{shard}"
        with fsspec.open(url, "rb", block_size=2**20, cache_type="readahead") as remote:
            with h5py.File(remote, "r") as handle:
                times = handle["dimensions/time"][:]
                for trajectory in trajectory_ids:
                    density_full = handle["t0_fields/density"][trajectory]
                    speed_full = handle["t0_fields/speed_of_sound"][trajectory]
                    pressure_full = np.stack([
                        handle["t0_fields/pressure"][trajectory, int(t)]
                        for t in TIME_INDICES])
                    density = density_full[::4, ::4]
                    speed = speed_full[::4, ::4]
                    pressure = pressure_full[:, ::4, ::4]
                    key = f"{split}_{trajectory}"
                    arrays[f"{key}_density"] = density.astype(np.float32)
                    arrays[f"{key}_speed_of_sound"] = speed.astype(np.float32)
                    arrays[f"{key}_pressure"] = pressure.astype(np.float32)
                    arrays[f"{key}_times"] = times[TIME_INDICES].astype(np.float32)
                    samples.append({
                        "id": key,
                        "split": split,
                        "shard": shard,
                        "trajectory_index": trajectory,
                        "geometry_sha256": hashlib.sha256(density.tobytes()).hexdigest(),
                        "wall_fraction": float(np.mean(density > 1e5)),
                        "initial_source_components": source_components(pressure_full[0]),
                        "pressure_std": float(pressure.std()),
                        "finite": bool(np.isfinite(pressure).all()),
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return samples, hashlib.sha256(output.read_bytes()).hexdigest()


def plot_samples(sample_path: Path, samples: list[dict], output: Path):
    with np.load(sample_path) as arrays:
        fig, axes = plt.subplots(2, len(samples), figsize=(16, 4.5), constrained_layout=True)
        for column, sample in enumerate(samples):
            key = sample["id"]
            density = arrays[f"{key}_density"]
            pressure = arrays[f"{key}_pressure"][-1]
            axes[0, column].imshow(density.T, origin="lower", cmap="gray_r")
            scale = np.quantile(np.abs(pressure), .99)
            axes[1, column].imshow(pressure.T, origin="lower", cmap="RdBu_r",
                                   vmin=-scale, vmax=scale)
            axes[0, column].set_title(key, fontsize=8)
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        axes[0, 0].set_ylabel("density / maze")
        axes[1, 0].set_ylabel("pressure at t=4")
        fig.suptitle("The Well acoustic-scattering gate: 8 range-read trajectories")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("data/the_well_acoustic_gate/eight_trajectories.npz"))
    parser.add_argument("--report-dir", type=Path, default=Path("papers/dataset_gates"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = args.report_dir/"the_well_acoustic_gate_summary.json"
    previous = json.loads(report.read_text()) if report.exists() else None

    repo = get_json(f"{API}/{REPO}")
    revision = PINNED_REVISION
    tree = get_json(f"{API}/{REPO}/tree/{revision}?recursive=true&expand=true&limit=100")
    shards = [item for item in tree if item.get("path", "").endswith(".hdf5")]
    samples, sample_checksum = read_gate_samples(revision, args.output)
    if len(samples) != 8 or not all(sample["finite"] for sample in samples):
        raise RuntimeError("eight-trajectory finite-field gate failed")
    geometry_count = len({sample["geometry_sha256"] for sample in samples})
    if geometry_count < 8:
        raise RuntimeError("sampled material geometries are not trajectory-varying")
    plot_samples(args.output, samples, args.report_dir/"the_well_acoustic_gate.png")

    full_bytes = sum(int(item["size"]) for item in shards)
    selected_shards = {plan[0] for plan in SAMPLE_PLAN.values()}
    selected_shard_bytes = sum(int(item["size"]) for item in shards
                               if item["path"] in selected_shards)
    summary = {
        "schema_version": 1,
        "dataset": REPO,
        "revision": revision,
        "repository_head_at_gate": repo["sha"],
        "license": repo.get("cardData", {}).get("license"),
        "dataset_viewer_supported": False,
        "range_read_verified": True,
        "source_format": "HDF5 with contiguous field datasets",
        "source_shape": {"trajectories_per_shard": 100, "time": 202,
                         "space": [256, 256]},
        "n_shards": len(shards),
        "full_repository_bytes": full_bytes,
        "minimum_three_full_shards_bytes": selected_shard_bytes,
        "lfs_shards": {item["path"]: {
            "bytes": item["size"], "sha256": item.get("lfs", {}).get("oid")}
            for item in shards},
        "gate_sample": {
            "n_trajectories": len(samples),
            "time_indices": TIME_INDICES.tolist(),
            "resolution": [64, 64],
            "unique_material_geometries": geometry_count,
            "all_fields_finite": True,
            "source_component_count_range": [
                min(s["initial_source_components"] for s in samples),
                max(s["initial_source_components"] for s in samples)],
            "wall_fraction_range": [min(s["wall_fraction"] for s in samples),
                                    max(s["wall_fraction"] for s in samples)],
            "local_npz_sha256": sample_checksum,
            "repeat_range_read_checksum_verified": bool(
                previous and previous.get("gate_sample", {}).get("local_npz_sha256")
                == sample_checksum),
            "samples": samples,
        },
        "leakage_contract": {
            "inputs": ["density", "speed_of_sound", "pressure at t=0",
                       "coordinates", "query time"],
            "targets": ["pressure at t>0"],
            "forbidden": "No target-time pressure or velocity is used to build geometry or source features."
        },
        "subset_manifest": "experiments/dataset_splits/the_well_acoustic_64_16_32.json",
    }
    report.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "revision": revision,
        "n_shards": len(shards),
        "full_repository_bytes": full_bytes,
        "minimum_three_full_shards_bytes": selected_shard_bytes,
        "unique_material_geometries": geometry_count,
        "source_component_count_range": summary["gate_sample"]["source_component_count_range"],
        "sample_checksum": sample_checksum,
    }, indent=2))


if __name__ == "__main__":
    main()
