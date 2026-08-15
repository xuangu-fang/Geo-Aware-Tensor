import pytest
import torch

from geoaware.neural_tensor import paired_phase_carriers
from geoaware.phase_wave_protocol import absolute_wave_gate


def test_paired_carriers_exactly_span_traveling_cosine_and_sine():
    distance = torch.tensor([[.13], [.72], [1.41]])
    time = torch.tensor([[.05], [.22], [.39]])
    bands = torch.tensor([7., 13.])
    speeds = torch.tensor([.35, 1.0])
    spatial, temporal = paired_phase_carriers(distance, time, bands, speeds)
    products = (spatial * temporal).reshape(len(distance), len(bands),
                                             len(speeds), 4)
    phase = bands[None, :, None] * (
        distance[:, :, None] - time[:, :, None] * speeds[None, None, :])
    reconstructed_cos = products[..., 0] + products[..., 1]
    reconstructed_sin = products[..., 3] - products[..., 2]
    assert torch.allclose(reconstructed_cos, torch.cos(phase), atol=1e-6)
    assert torch.allclose(reconstructed_sin, torch.sin(phase), atol=1e-6)


def test_paired_carriers_reject_ambiguous_shapes():
    with pytest.raises(ValueError):
        paired_phase_carriers(torch.rand(4), torch.rand(4, 1),
                              torch.tensor([1.]), torch.tensor([1.]))


def test_absolute_wave_gate_rejects_the_well_near_null_result():
    result = absolute_wave_gate(.99175, 1.00160)
    assert result.mse_skill == pytest.approx(1-(.99175/1.00160)**2)
    assert not result.passed


def test_absolute_wave_gate_accepts_useful_reconstruction():
    result = absolute_wave_gate(.60, 1.0)
    assert result.mse_skill == pytest.approx(.64)
    assert result.passed
