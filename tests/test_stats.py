"""Statistics against reference implementations."""
import numpy as np
from scipy.stats import kruskal
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

from cgmprofiles import agreement as ag
from cgmprofiles import stats

RNG = np.random.default_rng(0)


def test_kruskal_by_hand_matches_scipy_with_ties():
    groups = [np.round(RNG.normal(m, 1, 40), 1) for m in (0, 0.3, 0.6)]
    h, p = stats.kruskal_h(groups)
    ref = kruskal(*groups)
    assert np.isclose(h, ref.statistic) and np.isclose(p, ref.pvalue)


def test_eta_squared_formula_and_clipping():
    assert stats.eta_squared_from_h(10.0, 3, 103) == (10 - 2) / 100
    assert stats.eta_squared_from_h(0.5, 3, 103) == 0.0


def test_holm_is_monotone_and_bounded():
    adjusted = stats.holm([0.01, 0.04, 0.03, 0.5])
    assert adjusted == [0.04, 0.09, 0.09, 0.5]


def test_bootstrap_is_reproducible_with_a_seed():
    a, b = RNG.normal(0, 1, 50), RNG.normal(1, 1, 60)
    first = stats.bootstrap_mean_difference(a, b, np.random.default_rng(5), 500)
    second = stats.bootstrap_mean_difference(a, b, np.random.default_rng(5), 500)
    assert first == second and first[1] < first[0] < first[2]


def test_table_statistics_match_sklearn():
    a, b = RNG.integers(1, 4, 200), RNG.integers(1, 4, 200)
    fast = ag.table_statistics(ag.cross_table(a, b, 3))
    assert np.isclose(fast["ARI"], adjusted_rand_score(a, b))
    assert np.isclose(fast["kappa"], cohen_kappa_score(a, b))
    assert np.isclose(fast["NMI"], normalized_mutual_info_score(a, b, average_method="arithmetic"))


def test_bias_corrected_cramers_v_is_zero_for_independent_and_one_for_identical():
    a = np.repeat([1, 2, 3], 100)
    assert np.isclose(ag.cramers_v_bias_corrected(a, a), 1.0)
    assert ag.cramers_v_bias_corrected(a, RNG.permutation(a)) < 0.1


def test_icc_is_one_for_identical_windows():
    x = RNG.normal(0, 1, 100)
    assert np.isclose(ag.icc_with_ci(x, x, 1)["icc"], 1.0)
