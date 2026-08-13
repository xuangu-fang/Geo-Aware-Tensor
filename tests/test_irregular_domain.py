import numpy as np

from geoaware.irregular_domain_solver import (
    boundary_mask,
    build_irregular_domain,
    default_irregular_specs,
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
