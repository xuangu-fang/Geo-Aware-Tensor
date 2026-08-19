#!/usr/bin/env python3
"""Audit machine-local links to the shared physics-informed tensor data root."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


SHARED_ROOT = Path("/mnt/data/xuangu-fang")


@dataclass(frozen=True)
class Mapping:
    local: Path
    target: Path
    expected_bytes: int
    expected_files: int


MAPPINGS = (
    Mapping(Path("/home/ubuntu/project/yanjiu/data/active_matter_multi/train"), SHARED_ROOT / "ai-physical-dynamics/datasets/active_matter_multi/train", 4_395_630_592, 5),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/active_matter_multi/test"), SHARED_ROOT / "ai-physical-dynamics/datasets/active_matter_multi/test", 1_384_120_320, 5),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/active_matter_ood_locked/raw"), SHARED_ROOT / "ai-physical-dynamics/datasets/active_matter_ood_locked/raw", 4_345_298_944, 4),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/active_matter_transport_fresh_shift"), SHARED_ROOT / "ai-physical-dynamics/datasets/active_matter_transport_fresh_shift", 1_342_177_280, 4),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/cfdbench/raw"), SHARED_ROOT / "ai-physical-dynamics/datasets/cfdbench/raw", 786_403_674, 1),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/cfdbench/extracted"), SHARED_ROOT / "ai-physical-dynamics/datasets/cfdbench/extracted", 836_587_386, 742),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/kolmogorov_mno/raw"), SHARED_ROOT / "ai-physical-dynamics/datasets/kolmogorov_mno/raw", 6_566_707_456, 2),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/realpde_active_physics_confirmation/raw"), SHARED_ROOT / "ai-physical-dynamics/datasets/realpde_active_physics_confirmation/raw", 4_316_377_840, 8),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/realpde_cylinder_subset/raw"), SHARED_ROOT / "ai-physical-dynamics/datasets/realpde_cylinder_subset/raw", 16_480_728_418, 38),
    Mapping(Path("/home/ubuntu/project/yanjiu/data/realpde_cylinder_subset/prepared"), SHARED_ROOT / "ai-physical-dynamics/datasets/realpde_cylinder_subset/prepared", 456_453_977, 2),
    Mapping(Path("/home/ubuntu/project/measure-what-persists/data/openfwi_curvefault_a"), SHARED_ROOT / "ai-physical-dynamics/datasets/openfwi_curvefault_a", 2_129_400_768, 6),
    Mapping(Path("/home/ubuntu/project/Geo-Aware-Tensor/data"), SHARED_ROOT / "physics-informed-tensor-learning/datasets/Geo-Aware-Tensor/data", 387_470_951, 248),
    Mapping(Path("/home/ubuntu/project/functional-operator-completion/data"), SHARED_ROOT / "physics-informed-tensor-learning/datasets/functional-operator-completion/data", 13_192_704, 64),
)


def tree_stats(root: Path) -> tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for directory, _, filenames in os.walk(root):
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if path.is_file():
                total_bytes += path.stat().st_size
                total_files += 1
    return total_bytes, total_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="recompute recursive byte and file counts")
    args = parser.parse_args()

    failures: list[str] = []
    if not Path("/mnt/data").is_mount():
        failures.append("/mnt/data is not a mount point")
    if not SHARED_ROOT.is_dir():
        failures.append(f"shared root is unavailable: {SHARED_ROOT}")

    for mapping in MAPPINGS:
        state = "OK"
        if not mapping.local.is_symlink():
            state = "NOT_A_SYMLINK"
        elif mapping.local.resolve() != mapping.target.resolve():
            state = "WRONG_TARGET"
        elif not mapping.target.is_dir():
            state = "MISSING_TARGET"
        elif args.deep:
            observed = tree_stats(mapping.target)
            expected = (mapping.expected_bytes, mapping.expected_files)
            if observed != expected:
                state = f"COUNT_MISMATCH expected={expected} observed={observed}"
        print(f"{state:12} {mapping.local} -> {mapping.target}")
        if state != "OK":
            failures.append(f"{mapping.local}: {state}")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"\nPASS: {len(MAPPINGS)} shared-data mappings are healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
