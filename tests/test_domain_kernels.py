import pytest
import torch

from geoaware.domain_kernels import (
    euclidean_rbf_kernel_sections,
    matern_domain_kernel_sections,
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


@pytest.mark.parametrize("lengthscales", [(), (0.0,), (-1.0,)])
def test_euclidean_sections_reject_invalid_lengthscales(lengthscales):
    with pytest.raises(ValueError):
        euclidean_rbf_kernel_sections(
            torch.zeros(2, 2), torch.tensor([0]), lengthscales=lengthscales)
