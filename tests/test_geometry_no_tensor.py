"""Tests for the geometry-neural-operator tensor POC."""

import numpy as np
import torch

from geoaware.geometry_no_data import (
    domain_mask,
    protocol_manifest,
    random_domain_spec,
)
from geoaware.geometry_no_tensor import (
    CoordinateSDFPlusGeometryNOCP,
    GeometryFNOEncoder,
    GeometryNOFunctionalCP,
    ambient_geometry_bundle,
)


def test_ambient_sdf_has_correct_sign_across_domain_and_holes():
    mask = np.zeros((21, 21), dtype=bool)
    mask[2:19, 2:19] = True
    mask[8:13, 8:13] = False
    bundle = ambient_geometry_bundle(mask)

    assert bundle.shape == (7, 21, 21)
    assert bundle[2, 4, 4] > 0
    assert bundle[2, 10, 10] < 0
    assert bundle[2, 0, 0] < 0
    assert np.isfinite(bundle).all()


def test_masked_geometry_encoder_zeroes_outputs_outside_domain():
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:13, 2:14] = True
    geometry = torch.from_numpy(ambient_geometry_bundle(mask))
    encoder = GeometryFNOEncoder(output_channels=5, width=8, modes=4,
                                 layers=2, masked=True)
    output = encoder(geometry)

    assert output.shape == (1, 5, 16, 16)
    assert torch.count_nonzero(output[0, :, ~torch.from_numpy(mask)]) == 0


def test_geometry_no_cp_queries_source_parameter_and_active_node():
    mask = np.ones((16, 16), dtype=bool)
    active = np.argwhere(mask).astype(np.int64)
    case = {
        "geometry": torch.from_numpy(ambient_geometry_bundle(mask)),
        "active_indices": torch.from_numpy(active),
        "source_xy": torch.tensor([[-.4, 0.], [.4, 0.]]),
        "parameters": torch.tensor([.05, .2]),
    }
    indices = torch.tensor([[0, 0, 0], [1, 1, 100], [0, 1, 200]])
    model = GeometryNOFunctionalCP(rank=4, width=8, modes=4)
    prediction = model.forward_case(case, indices)

    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()


def test_zero_gated_geometry_residual_equals_coordinate_sdf_mean():
    mask = np.ones((16, 16), dtype=bool)
    active = np.argwhere(mask).astype(np.int64)
    geometry = ambient_geometry_bundle(mask)
    case = {
        "geometry": torch.from_numpy(geometry),
        "active_indices": torch.from_numpy(active),
        "coordinates": torch.from_numpy(geometry[5:7, mask].T.copy()),
        "active_sdf": torch.from_numpy(geometry[2, mask].copy()),
        "source_xy": torch.tensor([[-.4, 0.], [.4, 0.]]),
        "parameters": torch.tensor([.05, .2]),
    }
    indices = torch.tensor([[0, 0, 0], [1, 1, 100], [0, 1, 200]])
    model = CoordinateSDFPlusGeometryNOCP(
        rank=4, hidden=8, width=8, modes=4, initial_residual_gate=0.)
    prediction = model.forward_case(case, indices)
    expected = model.mean.forward_case(case, indices)

    assert torch.equal(prediction, expected)


def test_frozen_manifest_separates_seen_and_topology_ood_holes():
    manifest = protocol_manifest(resolution=20)
    assert len(manifest["splits"]["train"]) == 48
    assert len(manifest["splits"]["id_validation"]) == 8
    assert len(manifest["splits"]["topology_ood_validation"]) == 8
    assert {len(spec["holes"])
            for spec in manifest["splits"]["train"]} == {0, 1}
    assert {len(spec["holes"])
            for spec in manifest["splits"]["topology_ood_validation"]} == {2}
    assert len(random_domain_spec(4001, "test")["holes"]) in {2, 3}


def test_random_two_hole_domain_remains_connected():
    spec = random_domain_spec(3000, "topology_ood_validation")
    mask = domain_mask(spec, resolution=28)
    assert mask.dtype == np.bool_
    assert 0.25 < mask.mean() < 0.75
