import pytest
import torch

from geoaware.domain_kernels import (
    euclidean_rbf_kernel_sections,
    geodesic_rbf_kernel_sections,
    heat_domain_kernel_sections,
    matern_domain_kernel_sections,
)
from geoaware.variational_domain_gp import (
    FiniteFeatureVariationalGP,
    NonnegativeKernelMixture,
    exact_finite_gp_posterior,
    exact_finite_gp_predict,
    tensor_product_gp_features,
)


def test_domain_sections_are_invariant_to_eigenvector_signs():
    generator = torch.Generator().manual_seed(7)
    basis, _ = torch.linalg.qr(torch.randn(9, 5, generator=generator))
    eigenvalues = torch.arange(5, dtype=torch.float32)
    sources = torch.tensor([1, 7])
    signs = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0])

    expected = matern_domain_kernel_sections(basis, eigenvalues, sources)
    actual = matern_domain_kernel_sections(basis * signs, eigenvalues, sources)

    torch.testing.assert_close(actual, expected)


def test_euclidean_sections_peak_at_their_source():
    coordinates = torch.tensor([[0.0, 0.0], [0.3, 0.0], [1.0, 0.0]])
    sections = euclidean_rbf_kernel_sections(
        coordinates, torch.tensor([0, 2]), lengthscales=(0.2, 0.5))

    assert sections.shape == (2, 3, 2)
    assert torch.all(sections[0, 0] > sections[0, 1])
    assert torch.all(sections[1, 2] > sections[1, 1])


def test_heat_sections_are_sign_invariant_and_decay_high_modes_faster():
    basis = torch.eye(4)
    eigenvalues = torch.tensor([0.0, 1.0, 3.0, 7.0])
    sources = torch.tensor([0, 2])
    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    expected = heat_domain_kernel_sections(
        basis, eigenvalues, sources, diffusion_times=(0.1, 1.0))
    actual = heat_domain_kernel_sections(
        basis * signs, eigenvalues, sources, diffusion_times=(0.1, 1.0))
    torch.testing.assert_close(actual, expected)


def test_geodesic_kernel_does_not_shortcut_across_a_wall():
    # Nodes 0 and 3 are ambient-close but the valid path detours through 1,2.
    coordinates = torch.tensor([
        [0.0, 0.0], [0.0, 1.0], [0.1, 1.0], [0.1, 0.0],
    ])
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    source = torch.tensor([0])
    intrinsic = geodesic_rbf_kernel_sections(
        coordinates, source, edges, lengthscales=(0.5,))
    euclidean = euclidean_rbf_kernel_sections(
        coordinates, source, lengthscales=(0.5,))
    assert intrinsic[0, 3, 0] < intrinsic[0, 1, 0]
    assert euclidean[0, 3, 0] > euclidean[0, 1, 0]


@pytest.mark.parametrize("lengthscales", [(), (0.0,), (-1.0,)])
def test_euclidean_sections_reject_invalid_lengthscales(lengthscales):
    with pytest.raises(ValueError):
        euclidean_rbf_kernel_sections(
            torch.zeros(2, 2), torch.tensor([0]), lengthscales=lengthscales)


def test_tensor_product_gp_features_have_matched_finite_kernel_dimension():
    sections = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    parameters = torch.tensor([0.1, 0.3, 1.0])
    indices = torch.tensor([[0, 0, 1], [1, 2, 3]])
    features = tensor_product_gp_features(
        sections, parameters, indices, parameter_centers=5)
    assert features.shape == (2, 15)
    assert torch.isfinite(features).all()


def test_variational_gp_prior_has_zero_kl_and_positive_variance():
    model = FiniteFeatureVariationalGP(4)
    assert float(model.kl_to_prior().detach()) == pytest.approx(0.0, abs=1e-6)
    _, variance = model.predict(torch.eye(4), include_noise=False)
    torch.testing.assert_close(variance, torch.ones(4), atol=2e-4, rtol=2e-4)


def test_exact_finite_gp_observations_reduce_nearby_latent_variance():
    train_features = torch.tensor([[1.0, 0.0], [1.0, 0.1]])
    targets = torch.tensor([1.0, 0.9])
    mean, covariance = exact_finite_gp_posterior(
        train_features, targets, noise_std=0.05)
    query = torch.tensor([[1.0, 0.05], [0.0, 1.0]])
    prediction, variance = exact_finite_gp_predict(query, mean, covariance)
    assert prediction.shape == variance.shape == (2,)
    assert variance[0] < variance[1]


def test_minibatch_elbo_is_finite_and_differentiable():
    model = FiniteFeatureVariationalGP(3)
    features = torch.tensor([[1.0, 0.0, 0.2], [0.0, 1.0, -0.1]])
    targets = torch.tensor([0.5, -0.4])
    loss, diagnostics = model.negative_elbo(
        features, targets, total_count=len(targets))
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["kl"] >= 0
    assert model.variational_mean.grad is not None


def test_variational_gp_elbo_accepts_a_differentiable_neural_mean():
    model = FiniteFeatureVariationalGP(2)
    features = torch.eye(2)
    targets = torch.tensor([0.2, -0.1])
    mean = torch.tensor([0.1, -0.2], requires_grad=True)
    loss, _ = model.negative_elbo(
        features, targets, total_count=2, mean_offset=mean)
    loss.backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def test_nonnegative_kernel_mixture_is_a_simplex_scaled_feature_map():
    mixture = NonnegativeKernelMixture(("heat", "matern"))
    features = {
        "heat": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "matern": torch.tensor([[2.0], [1.0]]),
    }
    combined = mixture(features)
    assert combined.shape == (2, 3)
    torch.testing.assert_close(mixture.weights().sum(), torch.tensor(1.0))
    kernel = combined @ combined.T
    expected = 0.5 * (
        features["heat"] @ features["heat"].T
        + features["matern"] @ features["matern"].T
    )
    torch.testing.assert_close(kernel, expected)
