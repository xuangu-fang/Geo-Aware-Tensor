"""Direction-4 tests that pin down the current POC semantics."""

import math

import torch
from torch import nn

from geoaware.functional_tucker import FunctionalTucker, sdf_query_features


class _FixedFactor(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs) == len(self.values)
        return self.values


def test_cp_diagonal_core_is_exactly_a_nested_cp_contraction():
    """`cp_diagonal` means CP algebra, not a trained-CP warm start."""
    generator = torch.Generator().manual_seed(47)
    batch, rank = 5, 4
    geometry = torch.randn(batch, rank, generator=generator)
    parameter = torch.randn(batch, rank, generator=generator)
    spatial = torch.randn(batch, rank, generator=generator)

    model = FunctionalTucker(
        spatial_dim=rank, ranks=(rank, rank, rank), hidden=8,
        core_init="cp_diagonal",
    )
    model.geometry_factor = _FixedFactor(geometry)
    model.parameter_factor = _FixedFactor(parameter)
    model.spatial_factor = _FixedFactor(spatial)

    prediction = model.forward_features(
        torch.empty(batch, 7), torch.empty(batch, 2), torch.empty(batch, rank)
    )
    expected = (geometry * parameter * spatial).sum(dim=1) / math.sqrt(rank)
    assert torch.allclose(prediction, expected, atol=1e-6)


def test_boundary_distance_feature_is_positive_interior_metadata():
    """The third query feature is the stored distance, not a recomputed SDF."""
    case = {
        "coords": torch.tensor([[0.0, 0.0], [0.5, -0.25]]),
        "source_xy": torch.tensor([[0.1, 0.2]]),
        "boundary_distance": torch.tensor([0.05, 0.30]),
    }
    indices = torch.tensor([[0, 0, 0], [0, 0, 1]])
    with_distance = sdf_query_features(case, indices, use_sdf=True)
    without_distance = sdf_query_features(case, indices, use_sdf=False)

    assert torch.equal(with_distance[:, 2], case["boundary_distance"])
    assert torch.count_nonzero(without_distance[:, 2]) == 0
    assert torch.all(with_distance[:, 2] >= 0)
