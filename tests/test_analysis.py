import numpy as np

from eval.analysis import bootstrap_ci, paired_bootstrap, spearman


def test_bootstrap_interval_contains_mean_and_is_deterministic():
    values = [0.2, 0.5, 1.0, 0.0, 0.8, 0.6, 0.4, 1.0]
    a, b = bootstrap_ci(values), bootstrap_ci(values)
    assert a == b
    assert a['low'] <= a['mean'] <= a['high']
    assert a['n'] == len(values)
    assert bootstrap_ci([]) is None


def test_paired_bootstrap_detects_direction():
    base = [0.0] * 20
    better = [0.5] * 20
    p = paired_bootstrap(base, better)
    assert p['mean_diff'] == 0.5 and p['low'] == 0.5 and p['p_improved'] == 1.0
    assert p['queries_better'] == 20 and p['queries_worse'] == 0
    same = paired_bootstrap(base, base)
    assert same['mean_diff'] == 0.0 and same['p_improved'] == 0.0
    assert paired_bootstrap([1.0], [1.0, 2.0]) is None


def test_spearman_handles_ties_and_constants():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2], [2, 1]) is None
    rho = spearman([1, 2, 2, 3], [1, 2, 3, 4])
    assert 0.9 < rho <= 1.0
