"""Geometry-neural-operator tensor models for label-scarce PDE surrogates.

The neural operator sees an ambient-grid description of the *domain*, not the
solution.  Its output is a geometry-conditioned spatial basis.  A CP head then
couples that basis to source and physical-parameter factors.  Consequently the
full geometry is available even when only a tiny subset of output labels is
observed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
import torch
from torch import nn


def ambient_geometry_bundle(mask: np.ndarray) -> np.ndarray:
    """Return occupancy, smooth occupancy, true signed distance, normals, x/y.

    The signed distance is positive inside the physical domain and negative
    outside.  It is defined on the complete ambient grid, including holes.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    h, w = mask.shape
    spacing = 2.0 / max(h - 1, w - 1)
    inside = ndimage.distance_transform_edt(mask) * spacing
    outside = ndimage.distance_transform_edt(~mask) * spacing
    signed = inside - outside
    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.25)
    normal_x, normal_y = np.gradient(signed, spacing, spacing)
    norm = np.sqrt(normal_x**2 + normal_y**2) + 1e-8
    normal_x, normal_y = normal_x / norm, normal_y / norm
    axis_x = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    axis_y = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(axis_x, axis_y, indexing="ij")
    return np.stack([
        mask.astype(np.float32), smooth.astype(np.float32),
        signed.astype(np.float32), normal_x.astype(np.float32),
        normal_y.astype(np.float32), xx, yy,
    ])


class SpectralConv2d(nn.Module):
    """A compact FNO spectral convolution with resolution-safe mode clipping."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, self.modes, self.modes,
                                dtype=torch.cfloat))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        transformed = torch.fft.rfft2(inputs, norm="ortho")
        output = torch.zeros(batch, channels, height, width // 2 + 1,
                             dtype=torch.cfloat, device=inputs.device)
        mx = min(self.modes, height)
        my = min(self.modes, width // 2 + 1)
        output[:, :, :mx, :my] = torch.einsum(
            "bcxy,coxy->boxy", transformed[:, :, :mx, :my],
            self.weight[:, :, :mx, :my])
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class GeometryFNOEncoder(nn.Module):
    """FNO-style encoder mapping domain metadata to spatial basis channels."""

    def __init__(self, output_channels: int, width: int = 24, modes: int = 8,
                 layers: int = 3, masked: bool = True,
                 geometry_inputs: str = "full"):
        super().__init__()
        if geometry_inputs not in {"full", "sdf_only"}:
            raise ValueError("geometry_inputs must be 'full' or 'sdf_only'")
        self.masked = bool(masked)
        self.geometry_inputs = geometry_inputs
        self.lift = nn.Conv2d(7, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1)
                                    for _ in range(layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(4, width)
                                    for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                     nn.Conv2d(width, output_channels, 1))

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.ndim == 3:
            geometry = geometry[None]
        occupancy = geometry[:, :1]
        if self.geometry_inputs == "sdf_only":
            selected = torch.zeros_like(geometry)
            selected[:, 2:3] = geometry[:, 2:3]
            geometry = selected
        hidden = self.lift(geometry)
        if self.masked:
            hidden = hidden * occupancy
        for spectral, local, norm in zip(self.spectral, self.local, self.norms):
            hidden = torch.nn.functional.gelu(norm(spectral(hidden) + local(hidden)))
            if self.masked:
                hidden = hidden * occupancy
        output = self.project(hidden)
        return output * occupancy if self.masked else output


def _factor_mlp(input_dim: int, rank: int, hidden: int = 32) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, rank))


def _query_geometry_table(table: torch.Tensor, case: dict,
                          indices: torch.Tensor) -> torch.Tensor:
    node = indices[:, 2]
    grid = case["active_indices"].to(indices.device)[node]
    return table[0, :, grid[:, 0], grid[:, 1]].T


class GeometryNOFunctionalCP(nn.Module):
    """CP head over source, parameter, and an FNO-produced geometry basis."""

    def __init__(self, rank: int = 20, width: int = 24, modes: int = 8,
                 masked: bool = True, geometry_inputs: str = "full"):
        super().__init__()
        self.encoder = GeometryFNOEncoder(rank, width, modes, masked=masked,
                                          geometry_inputs=geometry_inputs)
        self.source_factor = _factor_mlp(2, rank)
        self.parameter_factor = _factor_mlp(2, rank)
        self.weight = nn.Parameter(torch.ones(rank) / math.sqrt(rank))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, _ = indices.T
        table = self.encoder(case["geometry"].to(indices.device))
        spatial = _query_geometry_table(table, case, indices)
        source_xy = case["source_xy"].to(indices.device)[source]
        value = case["parameters"].to(indices.device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        return (self.source_factor(source_xy)
                * self.parameter_factor(parameter_features)
                * spatial * self.weight).sum(1)


class GeometryNODenseHead(nn.Module):
    """Same geometry encoder with a non-factorized pointwise regression head."""

    def __init__(self, latent: int = 20, width: int = 24, modes: int = 8,
                 masked: bool = True, geometry_inputs: str = "full"):
        super().__init__()
        self.encoder = GeometryFNOEncoder(latent, width, modes, masked=masked,
                                          geometry_inputs=geometry_inputs)
        self.head = nn.Sequential(nn.Linear(latent + 6, 48), nn.GELU(),
                                  nn.Linear(48, 48), nn.GELU(),
                                  nn.Linear(48, 1))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, node = indices.T
        table = self.encoder(case["geometry"].to(indices.device))
        latent = _query_geometry_table(table, case, indices)
        source_xy = case["source_xy"].to(indices.device)[source]
        value = case["parameters"].to(indices.device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        xy = case["coordinates"].to(indices.device)[node]
        return self.head(torch.cat([latent, source_xy, parameter_features, xy], 1)).squeeze(1)


class CoordinateSDFFunctionalCP(nn.Module):
    """No-operator CP baseline using only pointwise coordinates and SDF."""

    def __init__(self, rank: int = 20, hidden: int = 48):
        super().__init__()
        self.source_factor = _factor_mlp(2, rank, hidden)
        self.parameter_factor = _factor_mlp(2, rank, hidden)
        self.space_factor = _factor_mlp(6, rank, hidden)
        self.weight = nn.Parameter(torch.ones(rank) / math.sqrt(rank))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        source, parameter, node = indices.T
        source_xy = case["source_xy"].to(indices.device)[source]
        value = case["parameters"].to(indices.device)[parameter]
        parameter_features = torch.stack([value, torch.log(value)], 1)
        xy = case["coordinates"].to(indices.device)[node]
        sdf = case["active_sdf"].to(indices.device)[node, None]
        distance = torch.linalg.vector_norm(xy - source_xy, dim=1, keepdim=True)
        spatial = torch.cat([xy, sdf, source_xy, distance], 1)
        return (self.source_factor(source_xy)
                * self.parameter_factor(parameter_features)
                * self.space_factor(spatial) * self.weight).sum(1)


class CoordinateSDFPlusGeometryNOCP(nn.Module):
    """Strong local CP mean plus a small geometry-NO CP residual.

    The local mean is exactly ``CoordinateSDFFunctionalCP``.  The residual gate
    starts near zero so adding the operator branch does not destroy the useful
    low-capacity inductive bias at the start of sparse-label optimization.
    """

    def __init__(self, rank: int = 20, hidden: int = 48, width: int = 24,
                 modes: int = 8, masked: bool = True,
                 initial_residual_gate: float = .01):
        super().__init__()
        self.mean = CoordinateSDFFunctionalCP(rank=rank, hidden=hidden)
        self.residual = GeometryNOFunctionalCP(
            rank=rank, width=width, modes=modes, masked=masked,
            geometry_inputs="full")
        self.residual_gate = nn.Parameter(torch.tensor(float(initial_residual_gate)))

    def forward_case(self, case: dict, indices: torch.Tensor) -> torch.Tensor:
        return (self.mean.forward_case(case, indices)
                + self.residual_gate * self.residual.forward_case(case, indices))
