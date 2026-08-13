from geoaware.statistics import paired_seed_summary


def test_paired_seed_summary_uses_seed_pairs():
    result = paired_seed_summary(
        {0: 0.7, 1: 0.8, 2: 0.75}, {0: 1.0, 1: 1.0, 2: 1.0},
        bootstrap_samples=1000,
    )
    assert result["n_seeds"] == 3
    assert result["relative_improvement"] == 0.25
    assert result["paired_difference"] < 0
    # With only three independent seeds, an exact two-sided p-value cannot be
    # below 0.25, regardless of how many nested task rows exist.
    assert result["two_sided_paired_permutation_p"] == 0.25
