"""Independent wave fields on domains with genuinely irregular outer boundaries.

Unlike the obstacle benchmark, the active computational domain itself changes.
The solver reuses only numerical wave integration utilities; it does not import
any learner or tensor-factorization code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

from .independent_wave_solver import (
    WaveDomain,
    WaveGeometrySpec,
    _laplacian,
    _sparse_payload,
    simulate_damped_wave,
)


@dataclass(frozen=True)
class IrregularBoundarySpec:
    name: str
    kind: str
    parameters: dict[str, float]


def default_irregular_specs() -> list[IrregularBoundarySpec]:
    """Six concave, curved, slanted, or multiply connected domains."""
    return [
        IrregularBoundarySpec("l_shape", "l_shape", {"notch_x": .02, "notch_y": .02}),
        IrregularBoundarySpec("u_notch", "u_notch", {"slot_half": .25, "slot_bottom": -.05}),
        IrregularBoundarySpec("wavy_three_lobe", "wavy", {"base": .82, "a3": .12, "a5": .06}),
        IrregularBoundarySpec("dumbbell", "dumbbell", {"center": .40, "radius": .56, "bridge": .22}),
        IrregularBoundarySpec("slanted_channel", "slanted", {"slope": .20, "half_width": .72}),
        IrregularBoundarySpec("wavy_with_hole", "hole", {"base": .86, "hole_x": .20, "hole_y": .05,
                                                               "hole_rx": .18, "hole_ry": .28}),
    ]


def domain_mask(xx: np.ndarray, yy: np.ndarray, spec: IrregularBoundarySpec) -> np.ndarray:
    """Return the physical domain indicator, including outer and hole boundaries."""
    p = spec.parameters
    box = (np.abs(xx) <= .94) & (np.abs(yy) <= .94)
    if spec.kind == "l_shape":
        return box & ~((xx > p["notch_x"]) & (yy > p["notch_y"]))
    if spec.kind == "u_notch":
        slot = (np.abs(xx) < p["slot_half"]) & (yy > p["slot_bottom"])
        return box & ~slot
    if spec.kind == "wavy":
        radius = np.sqrt(xx**2 + yy**2)
        angle = np.arctan2(yy, xx)
        limit = p["base"] + p["a3"]*np.cos(3*angle) + p["a5"]*np.sin(5*angle)
        return radius <= limit
    if spec.kind == "dumbbell":
        left = (xx+p["center"])**2 + yy**2 <= p["radius"]**2
        right = (xx-p["center"])**2 + yy**2 <= p["radius"]**2
        bridge = (np.abs(xx) <= p["center"]) & (np.abs(yy) <= p["bridge"])
        return left | right | bridge
    if spec.kind == "slanted":
        center = p["slope"]*yy
        channel = np.abs(xx-center) <= p["half_width"]
        ceiling = yy <= .80 + .08*np.sin(3.2*xx)
        floor = yy >= -.88 + .05*np.cos(4.1*xx)
        return channel & ceiling & floor
    if spec.kind == "hole":
        radius = np.sqrt(xx**2 + yy**2)
        angle = np.arctan2(yy, xx)
        outer = radius <= p["base"] + .08*np.cos(3*angle) - .05*np.sin(4*angle)
        hole = ((xx-p["hole_x"])/p["hole_rx"])**2 + ((yy-p["hole_y"])/p["hole_ry"])**2 < 1
        return outer & ~hole
    raise ValueError(f"unknown irregular domain kind: {spec.kind}")


def _connected(mask: np.ndarray) -> bool:
    _, components = ndimage.label(mask)
    return components == 1


def build_irregular_domain(spec: IrregularBoundarySpec, resolution: int) -> WaveDomain:
    if resolution < 16:
        raise ValueError("resolution must be at least 16")
    axis = np.linspace(-1., 1., resolution, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    fluid = domain_mask(xx, yy, spec)
    if not _connected(fluid):
        raise ValueError(f"domain {spec.name} is disconnected at resolution {resolution}")
    grid_indices = np.argwhere(fluid)
    node_of = np.full((resolution, resolution), -1, dtype=np.int64)
    node_of[fluid] = np.arange(len(grid_indices))
    edges: list[tuple[int, int]] = []
    for node, (i, j) in enumerate(grid_indices):
        for di, dj in ((1, 0), (0, 1)):
            ni, nj = i+di, j+dj
            if ni < resolution and nj < resolution and fluid[ni, nj]:
                edges.append((node, int(node_of[ni, nj])))
    edge_array = np.asarray(edges, dtype=np.int64)
    spacing = float(axis[1]-axis[0])
    coords = np.stack([xx[fluid], yy[fluid]], axis=1)
    # Positive distance to any outer or hole boundary. This is geometry
    # metadata, not a target-derived feature.
    boundary_distance = ndimage.distance_transform_edt(fluid)*spacing
    distance_active = boundary_distance[fluid]
    speed = (.80 + .09*np.sin(2.0*coords[:, 0])*np.cos(1.8*coords[:, 1])
             + .06*np.tanh(5*distance_active)).clip(.55, 1.05)
    geometry_operator = _laplacian(len(coords), edge_array, np.ones(len(edge_array)), spacing)
    edge_speed = .5*(speed[edge_array[:, 0]] + speed[edge_array[:, 1]])
    wave_operator = _laplacian(len(coords), edge_array, edge_speed**2, spacing)
    compatible_spec = WaveGeometrySpec(
        spec.name, f"irregular:{spec.kind}", dict(spec.parameters))
    return WaveDomain(
        spec=compatible_spec, resolution=resolution, spacing=spacing,
        coordinates=coords.astype(np.float32), grid_indices=grid_indices.astype(np.int32),
        fluid_mask=fluid, signed_distance=distance_active.astype(np.float32),
        material_speed=speed.astype(np.float32), geometry_operator=geometry_operator,
        wave_operator=wave_operator, undirected_edges=edge_array.astype(np.int32))


def boundary_mask(domain: WaveDomain) -> np.ndarray:
    eroded = ndimage.binary_erosion(domain.fluid_mask)
    return domain.fluid_mask & ~eroded


def generate_irregular_dataset(
    output: Path,
    specs: Iterable[IrregularBoundarySpec] | None = None,
    resolutions: Iterable[int] = (24, 32),
    sources: Iterable[tuple[float, float]] = ((-.58, -.24), (-.58, .24)),
    record_times: np.ndarray | None = None,
) -> dict:
    specs = list(default_irregular_specs() if specs is None else specs)
    resolutions = list(resolutions)
    sources = list(sources)
    times = np.linspace(0., 2., 40) if record_times is None else np.asarray(record_times)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for spec in specs:
        for resolution in resolutions:
            domain = build_irregular_domain(spec, resolution)
            for source_index, source_xy in enumerate(sources):
                fields, simulation = simulate_damped_wave(domain, source_xy, times)
                name = f"{spec.name}_r{resolution}_s{source_index}.npz"
                path = output/name
                payload = {
                    "coordinates": domain.coordinates,
                    "grid_indices": domain.grid_indices,
                    "fluid_mask": domain.fluid_mask.astype(np.uint8),
                    "boundary_mask": boundary_mask(domain).astype(np.uint8),
                    "boundary_distance": domain.signed_distance,
                    "material_speed": domain.material_speed,
                    "undirected_edges": domain.undirected_edges,
                    "record_times": times.astype(np.float32),
                    "field": fields,
                    "source_index": np.asarray(source_index, dtype=np.int32),
                    "source_xy": np.asarray(source_xy, dtype=np.float32),
                    **_sparse_payload("geometry_operator", domain.geometry_operator),
                    **_sparse_payload("wave_operator", domain.wave_operator),
                }
                np.savez_compressed(path, **payload)
                cases.append({
                    "file": name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "geometry": asdict(spec),
                    "resolution": resolution,
                    "n_nodes": len(domain.coordinates),
                    "n_boundary_nodes": int(boundary_mask(domain).sum()),
                    "source_index": source_index,
                    "simulation": simulation,
                    "field_std": float(fields.std()),
                    "field_abs_max": float(np.abs(fields).max()),
                })
    manifest = {
        "schema_version": 1,
        "purpose": "irregular outer-boundary geometry gate",
        "boundary_condition": "reflecting zero-flux on every outer and hole boundary",
        "cases": cases,
    }
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
