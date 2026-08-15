"""Sign-invariant kernels on irregular domains.

The functions in this module turn a Laplacian eigenbasis into a small set of
Matérn-kernel sections ``k_Omega(x, source)``. A section is invariant to an
eigenvector sign flip and gives a mesh-independent, geometry-aware coordinate
for a query point. This is deliberately a finite-feature POC, not a claim of
full variational Gaussian-process inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def matern_domain_kernel_sections(
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    source_nodes: torch.Tensor,
    *,
    scales: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
    smoothness: float = 1.5,
) -> torch.Tensor:
    """Return normalized ``[source, node, scale]`` domain-kernel sections.

    For graph-Laplacian eigenpairs ``(phi_j, lambda_j)``, channel ``q`` is

    ``sum_j phi_j(x) phi_j(s) (1 + scale_q * lambda_j)^(-smoothness)``.

    This is the finite spectral form of a Matérn-like GP covariance on the
    domain. Normalizing each channel by its RMS makes features comparable
    across meshes and resolutions without changing their topology.
    """
    if basis.ndim != 2 or eigenvalues.ndim != 1:
        raise ValueError("basis must be [node, mode] and eigenvalues [mode]")
    if basis.shape[1] != len(eigenvalues):
        raise ValueError("basis/eigenvalue mode counts do not match")
    if not scales:
        raise ValueError("at least one kernel scale is required")

    phi = basis.float()
    lam = eigenvalues.float().clamp_min(0)
    source_phi = phi[source_nodes.long()]
    scale_tensor = torch.as_tensor(scales, dtype=phi.dtype, device=phi.device)
    filters = (1 + scale_tensor[:, None] * lam[None]).pow(-smoothness)
    sections = torch.einsum("nk,sk,qk->snq", phi, source_phi, filters)
    sections = sections / max(1, basis.shape[1])
    rms = sections.square().mean(dim=(0, 1), keepdim=True).sqrt().clamp_min(1e-6)
    return sections / rms

