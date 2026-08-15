import math

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
from geoaware.operator_tucker_baselines import NeuralFunctionalCP, NeuralFunctionalTucker
from geoaware.masks import make_observation_split
from geoaware.tensor_data import (explicit_mode_bases, operator_cp_tensor,
                                  operator_nonaligned_tensor, operator_tucker_tensor)
from geoaware.the_well_pilot import block_mean_256_to_64, fixed_random_mask, nrmse_on_mask
from geoaware.well_baselines import WellUNetClassic
from geoaware.independent_wave_solver import (
    WaveGeometrySpec,
    build_wave_domain,
    simulate_damped_wave,
)
from geoaware.domain_kernels import matern_domain_kernel_sections
from geoaware.functional_tucker import (
    DomainKernelFunctionalTucker,
    GeometryConditionedNeuralFunctionalTucker,
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


def test_operator_tucker_spectral_prior_matches_factor_normalization():
    data = operator_tucker_tensor(shape=(7, 8, 9), seed=26)
    basis, eigenvalues = explicit_mode_bases(data, "correct")
    model = OperatorBayesianTucker(
        basis, eigenvalues, ranks=(2, 3, 3), device="cpu"
    )
    indices = data.flat_indices()[:19]
    prediction = model(indices).detach()
    prior = model.factor_prior().detach()
    with torch.no_grad():
        for coefficient in model.coeff:
            coefficient.mul_(3.7)
    assert torch.allclose(model(indices), prediction, atol=1e-6)
    assert torch.allclose(model.factor_prior(), prior, atol=1e-6)


def test_operator_cp_spectral_prior_matches_factor_normalization():
    data = operator_cp_tensor(shape=(7, 8, 9), seed=27)
    basis, eigenvalues = explicit_mode_bases(data, "correct")
    model = OperatorBayesianCP(
        basis, eigenvalues, rank=4, ard=True, device="cpu"
    )
    indices = data.flat_indices()[:19]
    prediction = model(indices).detach()
    prior = model.factor_prior().detach()
    with torch.no_grad():
        for coefficient in model.coeff:
            coefficient.mul_(0.17)
    assert torch.allclose(model(indices), prediction, atol=1e-6)
    assert torch.allclose(model.factor_prior(), prior, atol=1e-6)


def test_operator_tucker_functional_baselines_are_continuous_and_explicit():
    coordinates = torch.rand(17, 3)
    cp = NeuralFunctionalCP((False, False, True), rank=4, hidden=12)
    tucker = NeuralFunctionalTucker(
        (False, False, True), ranks=(2, 3, 4), hidden=12
    )
    assert cp(coordinates).shape == (17,)
    assert tucker(coordinates).shape == (17,)
    assert tucker.core.shape == (2, 3, 4)
    # A periodic factor must agree exactly across the 0/1 seam.
    seam = torch.tensor([[.2, .4, 0.], [.2, .4, 1.]])
    assert torch.allclose(cp(seam), cp(seam.flip(0)), atol=1e-6)
    assert torch.allclose(tucker(seam), tucker(seam.flip(0)), atol=1e-6)


def test_operator_tucker_structured_masks_preserve_the_declared_geometry():
    data = operator_tucker_tensor(shape=(10, 12, 16), seed=16)
    gap = make_observation_split(data, ratio=.05, kind="periodic_gap", seed=3)
    angle = data.flat_indices()[:, 2].float() / data.shape[2]
    excluded_sector = (angle < .125) | (angle >= .875)
    assert not gap.observed[excluded_sector].any()
    assert gap.held_out[excluded_sector].all()
    assert int(gap.observed.sum()) == round(.05 * data.values.numel())

    sensors = make_observation_split(data, ratio=.10, kind="sensor_tracks", seed=4)
    observed = sensors.observed.reshape(data.shape)
    # A selected spatial location contributes its complete time trajectory;
    # unselected locations contribute none of it.
    assert torch.all(observed == observed[:1].expand_as(observed))
    assert int(observed[0].sum()) == round(.10 * math.prod(data.shape[1:]))


def test_nonaligned_operator_benchmark_is_finite_periodic_and_off_basis():
    data = operator_nonaligned_tensor(shape=(9, 11, 24), seed=7)
    assert data.values.shape == (9, 11, 24)
    assert torch.isfinite(data.values).all()
    assert abs(float(data.values.mean())) < 1e-5
    assert abs(float(data.values.std()) - 1) < 1e-5
    # The generator includes periodic harmonic 11 while the learner is locked
    # to seven frequencies, so this is not an exact inverse-crime draw.
    assert data.basis_specs[2].n_frequencies == 7


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


def test_the_well_block_mean_preserves_small_off_stride_source():
    field = np.zeros((256, 256), dtype=np.float32)
    field[55, 57] = 4.
    reduced = block_mean_256_to_64(field)
    assert reduced.shape == (64, 64)
    assert np.isclose(reduced.sum(), .25)
    assert float(reduced.max()) > 0.


def test_well_unet_classic_preserves_spatial_shape():
    model = WellUNetClassic(dim_in=5, dim_out=1, init_features=4)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.randn(2, 5, 32, 32))
    assert prediction.shape == (2, 1, 32, 32)
    assert torch.isfinite(prediction).all()


def test_domain_kernel_sections_are_sign_invariant():
    generator = torch.Generator().manual_seed(9)
    basis, _ = torch.linalg.qr(torch.randn(13, 6, generator=generator))
    eigenvalues = torch.arange(6).float()
    source_nodes = torch.tensor([1, 8])
    first = matern_domain_kernel_sections(basis, eigenvalues, source_nodes)
    signs = torch.tensor([1., -1., 1., -1., -1., 1.])
    second = matern_domain_kernel_sections(basis * signs, eigenvalues, source_nodes)
    assert first.shape == (2, 13, 5)
    assert torch.allclose(first, second, atol=1e-6)


def test_new_functional_tuckers_keep_explicit_small_cores():
    n, n_nodes, n_sources = 10, 17, 3
    indices = torch.stack([
        torch.randint(n_sources, (n,)),
        torch.randint(4, (n,)),
        torch.randint(n_nodes, (n,)),
    ], 1)
    case = {
        "descriptor": torch.randn(7),
        "parameters": torch.linspace(.03, .3, 4),
        "coords": torch.randn(n_nodes, 2),
        "source_xy": torch.randn(n_sources, 2),
        "boundary_distance": torch.rand(n_nodes),
        "domain_kernel_features": torch.randn(n_sources, n_nodes, 5),
    }
    neural = GeometryConditionedNeuralFunctionalTucker(
        ranks=(2, 3, 4), hidden=12)
    kernel = DomainKernelFunctionalTucker(
        kernel_channels=5, ranks=(2, 3, 4), hidden=12)
    assert neural.core.shape == (2, 3, 4)
    assert kernel.core.shape == (2, 3, 4)
    assert torch.isfinite(neural.forward_case(case, indices)).all()
    assert torch.isfinite(kernel.forward_case(case, indices)).all()


def test_neural_functional_tucker_can_start_from_nested_cp_core():
    model = GeometryConditionedNeuralFunctionalTucker(
        ranks=(4, 4, 4), hidden=8, core_init="cp_diagonal")
    diagonal = model.core.detach()[torch.arange(4), torch.arange(4), torch.arange(4)]
    off_diagonal = model.core.detach().clone()
    off_diagonal[torch.arange(4), torch.arange(4), torch.arange(4)] = 0
    assert torch.allclose(diagonal, torch.full((4,), .5))
    assert torch.count_nonzero(off_diagonal) == 0
