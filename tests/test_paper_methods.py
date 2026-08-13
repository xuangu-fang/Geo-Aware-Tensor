import numpy as np
import torch

from geoaware.bayes_data import graph_time_features, make_two_room_diffusion
from geoaware.bayes_models import ExactFeatureBayes
from geoaware.neural_geometry import (
    build_obstacle_domain,
    heterogeneous_scattering_field,
)
from geoaware.neural_tensor import (
    GeometryNeuralCP,
    GeometryNeuralTucker,
    PhaseEnvelopeCP,
    PhaseEnvelopeTucker,
    SpeedAlignedPhaseCP,
)
from geoaware.tensor_bayes import OperatorBayesianCP, OperatorBayesianTucker
from geoaware.tensor_data import explicit_mode_bases, operator_cp_tensor, operator_tucker_tensor
from geoaware.the_well_pilot import fixed_random_mask, nrmse_on_mask
from geoaware.independent_wave_solver import (
    WaveGeometrySpec,
    build_wave_domain,
    simulate_damped_wave,
)


def test_anisotropic_exact_posterior_is_finite_under_extreme_sparsity():
    data = make_two_room_diffusion(nx=14, ny=10, n_time=8, n_graph_modes=16)
    features, separate_energy = graph_time_features(
        data, max_features=8 * 16, time_frequencies=4
    )
    assert separate_energy.ndim == 2 and separate_energy.shape[1] == 2
    observed = torch.arange(0, len(data.flat_values), 41)
    model = ExactFeatureBayes(features, separate_energy, selector="loo", device="cpu")
    prediction = model.fit(observed, data.flat_values[observed]).predict()
    assert prediction.mean.shape == data.flat_values.shape
    assert torch.isfinite(prediction.mean).all()
    assert torch.isfinite(prediction.conditional_std).all()
    assert torch.all(prediction.conditional_std > 0)


def test_heterogeneous_graph_target_is_finite_and_not_matched_kernel():
    spec = {"kind": "circle", "cx": 0.0, "cy": 0.0, "r": 0.2, "name": "tiny"}
    domain = build_obstacle_domain(spec, resolution=14, n_eigen=16)
    field = heterogeneous_scattering_field(domain, time=0.3, steps=20)
    assert field.shape == (domain.n_nodes,)
    assert torch.isfinite(field).all()
    assert float(field.std()) > 1e-3
    assert domain.edge_index.shape[0] == 2
    assert domain.edge_index.shape[1] > domain.n_nodes


def test_operator_bayesian_cp_keeps_explicit_mode_factors():
    data = operator_cp_tensor(shape=(7, 8, 9), seed=5)
    basis, eigenvalues = explicit_mode_bases(data, "correct")
    model = OperatorBayesianCP(
        basis, eigenvalues, rank=3, ard=True, device="cpu"
    )
    indices = data.flat_indices()
    factors = model.factor_tables()
    assert [tuple(f.shape) for f in factors] == [(7, 3), (8, 3), (9, 3)]
    design = model.cp_design(indices[:13], factors)
    assert design.shape == (13, 3)
    assert torch.isfinite(model(indices[:13])).all()


def test_operator_bayesian_tucker_keeps_small_explicit_core():
    data = operator_tucker_tensor(shape=(7, 8, 9), seed=6)
    basis, eigenvalues = explicit_mode_bases(data, "correct")
    model = OperatorBayesianTucker(
        basis, eigenvalues, ranks=(2, 3, 3), device="cpu"
    )
    indices = data.flat_indices()
    factors = model.factor_tables()
    assert [tuple(f.shape) for f in factors] == [(7, 2), (8, 3), (9, 3)]
    design = model.tucker_design(indices[:13], factors)
    assert design.shape == (13, 18)
    assert model.core.shape == (2, 3, 3)
    assert torch.isfinite(model(indices[:13])).all()


def test_neural_cp_and_tucker_contract_separate_point_factors():
    n = 11
    geometry = torch.randn(n, 7)
    time = torch.randn(n, 11)
    space = torch.randn(n, 14)
    cp = GeometryNeuralCP(rank=4, hidden=12, use_phase=True)
    tucker = GeometryNeuralTucker(ranks=(2, 3, 4), hidden=12, use_phase=True)
    cp_value = cp.forward_points(geometry, time, space)
    tucker_value = tucker.forward_points(geometry, time, space)
    assert cp_value.shape == (n,)
    assert tucker_value.shape == (n,)
    assert torch.isfinite(cp_value).all()
    assert torch.isfinite(tucker_value).all()


def test_speed_aligned_model_is_an_explicit_cp_contraction():
    n = 9
    geometry = torch.randn(n, 7)
    time = torch.rand(n, 1)
    # x, y, signed boundary distance, and intrinsic source distance
    space = torch.randn(n, 4)
    model = SpeedAlignedPhaseCP(hidden=12)
    spatial_carrier, temporal_carrier = model._carriers(time, space)
    assert spatial_carrier.shape == (n, model.components)
    assert temporal_carrier.shape == (n, model.components)
    value = model.forward_points(geometry, time, space)
    assert value.shape == (n,)
    assert torch.isfinite(value).all()


def test_phase_envelope_uses_separate_distance_and_time_factors():
    n = 9
    geometry = torch.randn(n, 7)
    time = torch.rand(n, 1)
    space = torch.randn(n, 4)
    model = PhaseEnvelopeCP(envelope_rank=3, hidden=12)
    distance_factor, time_factor = model.envelope_factors(time, space)
    assert distance_factor.shape == (n, 3)
    assert time_factor.shape == (n, 3)
    assert model.envelope_weight.shape == (model.components, 3)
    value = model.forward_points(geometry, time, space)
    assert value.shape == (n,)
    assert torch.isfinite(value).all()


def test_phase_envelope_tucker_has_explicit_small_core():
    n = 9
    geometry = torch.randn(n, 7)
    time = torch.rand(n, 1) * .3 + .12
    space = torch.randn(n, 4)
    space[:, 3] = torch.rand(n) * 3.5
    model = PhaseEnvelopeTucker(distance_rank=6, time_rank=4, hidden=12)
    distance_factor, time_factor = model.envelope_factors(time, space)
    assert distance_factor.shape == (n, 6)
    assert time_factor.shape == (n, 4)
    assert model.envelope_core.shape == (model.components, 6, 4)
    assert torch.isfinite(model.forward_points(geometry, time, space)).all()


def test_independent_wave_solver_is_finite_and_operator_is_symmetric_psd():
    spec = WaveGeometrySpec("test_circle", "circle",
                            {"cx": .05, "cy": .0, "radius": .2})
    domain = build_wave_domain(spec, resolution=12)
    difference = domain.wave_operator - domain.wave_operator.T
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.
    assert symmetry_error < 1e-8
    vector = np.linspace(-1., 1., len(domain.coordinates))
    assert float(vector @ (domain.wave_operator @ vector)) >= -1e-7
    fields, metadata = simulate_damped_wave(
        domain, (-.72, -.38), np.linspace(0., .5, 8))
    assert fields.shape == (8, len(domain.coordinates))
    assert np.isfinite(fields).all()
    assert float(fields.std()) > 1e-4
    assert metadata["time_step"] > 0


def test_the_well_mask_is_exact_deterministic_and_metric_held_out():
    shape = (5, 6, 7)
    first = fixed_random_mask(shape, ratio=.05, seed=3)
    second = fixed_random_mask(shape, ratio=.05, seed=3)
    assert np.array_equal(first, second)
    assert int(first.sum()) == round(.05*np.prod(shape))
    target = np.ones(shape, dtype=np.float32)
    prediction = target.copy()
    prediction[~first] = 0.
    assert np.isclose(nrmse_on_mask(target, prediction, ~first), 1.)
