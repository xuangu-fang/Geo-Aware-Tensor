import numpy as np

from geoaware.irregular_domain_solver import (
    boundary_mask,
    build_irregular_domain,
    default_irregular_specs,
    simulate_screened_elliptic,
)


def test_irregular_domains_are_connected_and_have_nontrivial_boundaries():
    counts = []
    for spec in default_irregular_specs():
        domain = build_irregular_domain(spec, resolution=24)
        counts.append(len(domain.coordinates))
        assert 120 < len(domain.coordinates) < 24*24
        assert boundary_mask(domain).sum() >= 30
        assert domain.geometry_operator.shape == (len(domain.coordinates),)*2
        assert np.all(np.isfinite(domain.signed_distance))
        assert float(domain.signed_distance.min()) > 0
    assert len(set(counts)) >= 5


def test_irregular_domain_operator_is_symmetric_psd():
    domain = build_irregular_domain(default_irregular_specs()[2], resolution=20)
    operator = domain.geometry_operator
    assert (operator-operator.T).nnz == 0
    values = np.linalg.eigvalsh(operator.toarray())
    assert values[0] > -1e-8
    assert values[-1] > 1


def test_screened_elliptic_tensor_is_finite_and_solved_accurately():
    domain = build_irregular_domain(default_irregular_specs()[0], resolution=18)
    field, metadata = simulate_screened_elliptic(
        domain, source_anchors=((-0.5, -0.3), (0.45, -0.35)),
        diffusivities=np.asarray([.04, .16]))
    assert field.shape == (2, 2, len(domain.coordinates))
    assert np.isfinite(field).all()
    assert float(field.std()) > 1e-4
    assert metadata["max_relative_linear_residual"] < 1e-8
