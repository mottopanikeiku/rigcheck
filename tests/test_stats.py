import math

from rigcheck import stats


def test_wilson_matches_reference_values():
    lo, hi = stats.wilson(8, 10)
    assert math.isclose(lo, 0.4902, abs_tol=1e-3)
    assert math.isclose(hi, 0.9433, abs_tol=1e-3)
    assert stats.wilson(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson(20, 20)
    assert hi == 1.0 and math.isclose(lo, 0.8389, abs_tol=1e-3)


def test_quantile_interpolates_like_numpy():
    xs = [1, 2, 3, 4, 10]
    assert stats.quantile(xs, 0.5) == 3
    assert math.isclose(stats.quantile(xs, 0.9), 7.6)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    groups = [[1.0, 1.1], [0.9, 1.2], [1.05, 0.95], [1.3, 1.0]]
    a = stats.bootstrap(groups)
    b = stats.bootstrap(groups)
    assert a == b
    est, lo, hi = a
    assert lo <= est <= hi


def test_bootstrap_with_one_group_has_no_interval():
    est, lo, hi = stats.bootstrap([[1.0, 2.0, 3.0]])
    assert est == 2.0 and math.isnan(lo) and math.isnan(hi)


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert math.isclose(stats.mcnemar_exact(0, 5), 0.0625)
    assert math.isclose(stats.mcnemar_exact(2, 8), 0.109375)


def test_paired_diff():
    pairs = [(True, False)] * 6 + [(True, True)] * 4
    est, lo, hi = stats.paired_diff(pairs)
    assert est == 0.6
    assert lo < 0.6 < hi or lo == hi


def test_ratio_of_medians():
    est, lo, hi = stats.ratio_of_medians([[1.0], [1.0], [1.2]], [[10.0], [9.0], [11.0]])
    assert math.isclose(est, 0.1)
    assert lo <= est <= hi
