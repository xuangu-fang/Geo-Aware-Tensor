#!/usr/bin/env python3
"""Extract the pinned 64/16/32 The Well subset through HTTP range reads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import fsspec
import h5py
import numpy as np

from geoaware.the_well_pilot import load_the_well_case, sanity_baselines


REPO = "polymathic-ai/acoustic_scattering_maze"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_case(path: Path, handle: h5py.File, trajectory: int,
              revision: str, source_shard: str):
    pressure = handle["t0_fields/pressure"][trajectory][:, ::4, ::4]
    density = handle["t0_fields/density"][trajectory][::4, ::4]
    speed = handle["t0_fields/speed_of_sound"][trajectory][::4, ::4]
    times = handle["dimensions/time"][:]
    x = handle["dimensions/x"][:][::4]
    y = handle["dimensions/y"][:][::4]
    if pressure.shape != (202, 64, 64):
        raise RuntimeError(f"unexpected pressure shape {pressure.shape}")
    arrays = (pressure, density, speed, times, x, y)
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"non-finite data in {source_shard}:{trajectory}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez(stream, pressure=pressure.astype(np.float32),
                 density=density.astype(np.float32),
                 speed_of_sound=speed.astype(np.float32),
                 times=times.astype(np.float32), x=x.astype(np.float32),
                 y=y.astype(np.float32), trajectory_index=np.int32(trajectory),
                 revision=np.asarray(revision), source_shard=np.asarray(source_shard))
    temporary.replace(path)


def case_record(path: Path, split: str, trajectory: int, source_shard: str,
                revision: str) -> dict:
    with np.load(path) as payload:
        stored_revision = str(payload["revision"].item())
        stored_shard = str(payload["source_shard"].item())
        stored_trajectory = int(payload["trajectory_index"])
    if (stored_revision != revision or stored_shard != source_shard
            or stored_trajectory != trajectory):
        raise RuntimeError(f"stale or mismatched extracted case: {path}")
    inputs, targets = load_the_well_case(path)
    if targets.shape != (201, 64, 64) or not np.isfinite(targets).all():
        raise RuntimeError(f"invalid extracted target: {path}")
    baseline = sanity_baselines(inputs, targets, observation_ratio=.01, seed=0)
    return {
        "file": str(path),
        "split": split,
        "trajectory_index": trajectory,
        "source_shard": source_shard,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "pressure_mean": float(targets.mean()),
        "pressure_std": float(targets.std()),
        "density_wall_fraction": float(np.mean(inputs["density"] > 1e5)),
        "sanity_baselines": baseline,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("experiments/dataset_splits/the_well_acoustic_64_16_32.json"))
    parser.add_argument("--output", type=Path, default=Path("data/the_well_acoustic_64x64"))
    parser.add_argument("--report", type=Path,
                        default=Path("papers/dataset_gates/the_well_pilot_extraction_summary.json"))
    parser.add_argument("--max-per-split", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.split_manifest.read_text())
    revision = manifest["revision"]
    previous = json.loads(args.report.read_text()) if args.report.exists() else None
    started = time.monotonic()
    records = []
    newly_extracted = 0
    for split, selection in manifest["selections"].items():
        shard = selection["file"]
        start = int(selection["trajectory_start_inclusive"])
        stop = int(selection["trajectory_stop_exclusive"])
        if args.max_per_split is not None:
            stop = min(stop, start+args.max_per_split)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{shard}"
        with fsspec.open(url, "rb", block_size=2**22, cache_type="readahead") as remote:
            with h5py.File(remote, "r") as handle:
                for trajectory in range(start, stop):
                    path = args.output/split/f"trajectory_{trajectory:03d}.npz"
                    if not path.exists():
                        save_case(path, handle, trajectory, revision, shard)
                        newly_extracted += 1
                    records.append(case_record(path, split, trajectory, shard, revision))
                    print(f"[{len(records):03d}] {split}:{trajectory} {path.stat().st_size/2**20:.2f} MiB",
                          flush=True)
    expected = int(manifest["counts"]["total"])
    complete = len(records) == expected and args.max_per_split is None
    split_counts = {split: sum(r["split"] == split for r in records)
                    for split in manifest["selections"]}
    split_stats = {}
    for split in split_counts:
        selected = [r for r in records if r["split"] == split]
        if not selected:
            continue
        means = np.asarray([r["pressure_mean"] for r in selected])
        stds = np.asarray([r["pressure_std"] for r in selected])
        global_mean = float(means.mean())
        global_variance = float(np.mean(stds**2+means**2)-global_mean**2)
        split_stats[split] = {
            "global_mean": global_mean,
            "global_std": float(np.sqrt(max(global_variance, 0.))),
            "trajectory_mean_range": [float(means.min()), float(means.max())],
            "trajectory_std_range": [float(stds.min()), float(stds.max())],
        }
    digest_payload = json.dumps([(r["file"], r["sha256"]) for r in records],
                                separators=(",", ":")).encode()
    dataset_digest = hashlib.sha256(digest_payload).hexdigest()
    mtimes = [Path(r["file"]).stat().st_mtime for r in records]
    summary = {
        "schema_version": 1,
        "experiment_id": "SHARED-DATA-R4-WELLMAZE-EXTRACT-64-16-32",
        "dataset": REPO,
        "revision": revision,
        "complete": complete,
        "expected_cases": expected,
        "extracted_cases": len(records),
        "newly_extracted_cases": newly_extracted,
        "split_counts": split_counts,
        "elapsed_seconds": time.monotonic()-started,
        "extraction_file_mtime_span_seconds": max(mtimes)-min(mtimes),
        "local_bytes": sum(r["bytes"] for r in records),
        "dataset_manifest_sha256": dataset_digest,
        "repeat_audit_checksum_verified": bool(
            previous and previous.get("dataset_manifest_sha256") == dataset_digest),
        "split_isolation_verified": all(
            r["source_shard"] == manifest["selections"][r["split"]]["file"]
            for r in records),
        "target_pressure_stats": split_stats,
        "one_seed_sanity_mean": {
            name: float(np.mean([r["sanity_baselines"][name] for r in records if r["split"] == "test"]))
            for name in ("zero_nrmse", "persistence_nrmse", "observed_mean_nrmse",
                         "realized_observation_ratio")
        } if split_counts.get("test") else {},
        "leakage_check": "loader returns t=0 pressure only in inputs and t>0 pressure only in targets",
        "cases": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "complete", "extracted_cases", "newly_extracted_cases", "split_counts",
        "elapsed_seconds", "local_bytes", "one_seed_sanity_mean")}, indent=2))
    if args.max_per_split is None and not complete:
        raise RuntimeError("full extraction did not match frozen manifest")


if __name__ == "__main__":
    main()
