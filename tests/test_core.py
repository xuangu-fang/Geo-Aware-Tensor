import torch

from geoaware.bases import BasisSpec, evaluate_basis
from geoaware.data import synthetic_boundary, synthetic_wave
from geoaware.masks import make_observation_split
from geoaware.models import build_model


def test_periodic_basis_closes_seam():
    spec = BasisSpec("periodic", 5)
    phi, eig = evaluate_basis(torch.tensor([0.0, 1.0]), spec)
    assert torch.allclose(phi[0], phi[1], atol=1e-5)
    assert torch.all(eig >= 0)


def test_synthetic_datasets_construct():
    assert synthetic_wave((5, 6, 7)).shape == (5, 6, 7)
    assert synthetic_boundary((8, 9)).shape == (8, 9)


def test_masks_are_reproducible_and_low_ratio():
    data = synthetic_wave((8, 10, 12))
    a = make_observation_split(data, 0.05, "periodic_gap", 3)
    b = make_observation_split(data, 0.05, "periodic_gap", 3)
    assert torch.equal(a.observed, b.observed)
    assert abs(a.ratio_actual - 0.05) < 0.002
    assert not torch.any(a.observed & ~a.eligible)


def test_all_models_forward():
    data = synthetic_wave((5, 6, 7))
    coords, indices = data.flat_coordinates()[:11], data.flat_indices()[:11]
    for name in ("cp", "inr", "fourier_inr", "neural_cp", "spectral_cp",
                 "wrong_spectral_cp", "bayesian_spectral_cp", "bayesian_spectral_tensor",
                 "geo_nft"):
        model = build_model(name, data.shape, data.basis_specs, rank=3, hidden=16)
        out = model(coords, indices)
        assert out.shape == (11,)
        assert torch.isfinite(out).all()


def test_exact_bayesian_posterior_is_finite_when_underdetermined():
    data = synthetic_boundary((12, 12))
    model = build_model("bayesian_spectral_tensor", data.shape, data.basis_specs,
                        rank=2, hidden=8)
    coords = data.flat_coordinates()[:8]
    y = data.values.reshape(-1)[:8]
    model.fit_posterior(coords, y)
    assert torch.isfinite(model.posterior_mean).all()
    assert torch.isfinite(model.posterior_cholesky).all()
    assert torch.isfinite(model(coords, sample=True)).all()
