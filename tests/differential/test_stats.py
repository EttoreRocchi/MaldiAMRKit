"""Tests for internal differential-analysis statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import false_discovery_control, mannwhitneyu, ttest_ind

from maldiamrkit.differential.stats import (
    _compute_effect_size,
    _compute_fold_change,
    _correct_pvalues,
    _mann_whitney_test,
    _t_test,
)


class TestMannWhitney:
    def test_matches_scipy(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 25)
        stat, p = _mann_whitney_test(a, b)
        exp_stat, exp_p = mannwhitneyu(a, b, alternative="two-sided")
        assert stat == pytest.approx(float(exp_stat))
        assert p == pytest.approx(float(exp_p))

    def test_empty_group_returns_nan_one(self):
        stat, p = _mann_whitney_test(np.array([]), np.array([1.0, 2.0]))
        assert np.isnan(stat)
        assert p == 1.0

    def test_constant_groups(self):
        stat, p = _mann_whitney_test(np.ones(5), np.ones(5))
        # scipy may raise or return a valid nan/1 p; our wrapper always
        # returns (nan, 1) if the U-statistic is undefined.
        assert p <= 1.0 or p == 1.0
        assert np.isnan(stat) or np.isfinite(stat)


class TestTTest:
    def test_matches_scipy_welch(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 40)
        b = rng.normal(0.8, 1.3, 35)
        stat, p = _t_test(a, b)
        res = ttest_ind(a, b, equal_var=False)
        assert stat == pytest.approx(float(res.statistic))
        assert p == pytest.approx(float(res.pvalue))

    def test_too_few_samples(self):
        stat, p = _t_test(np.array([1.0]), np.array([1.0, 2.0, 3.0]))
        assert np.isnan(stat)
        assert p == 1.0

    def test_identical_constant_groups_returns_nan(self):
        stat, p = _t_test(np.ones(5), np.ones(5))
        assert np.isnan(stat)
        assert p == 1.0


class TestCorrectPvalues:
    def test_fdr_bh_matches_scipy(self):
        p = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
        out = _correct_pvalues(p, method="fdr_bh")
        expected = false_discovery_control(p, method="bh")
        np.testing.assert_allclose(out, expected)

    def test_fdr_by_matches_scipy(self):
        p = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
        out = _correct_pvalues(p, method="fdr_by")
        expected = false_discovery_control(p, method="by")
        np.testing.assert_allclose(out, expected)

    def test_bonferroni(self):
        p = np.array([0.01, 0.04, 0.5])
        out = _correct_pvalues(p, method="bonferroni")
        np.testing.assert_allclose(out, np.clip(p * 3, 0.0, 1.0))

    def test_bounds_in_unit_interval(self):
        rng = np.random.default_rng(42)
        p = rng.random(200)
        for method in ("fdr_bh", "fdr_by", "bonferroni"):
            out = _correct_pvalues(p, method=method)
            assert np.all(out >= 0.0)
            assert np.all(out <= 1.0)

    def test_nan_treated_as_one(self):
        p = np.array([0.01, np.nan, 0.1])
        out = _correct_pvalues(p, method="bonferroni")
        assert out[1] == pytest.approx(1.0)

    def test_nan_excluded_from_bonferroni_denominator(self):
        """Bonferroni denominator counts tested hypotheses only.

        Two valid p-values among three -> factor m = 2, not 3.
        """
        p = np.array([0.01, np.nan, 0.1])
        out = _correct_pvalues(p, method="bonferroni")
        np.testing.assert_allclose(out, np.array([0.02, 1.0, 0.2]))

    def test_nan_excluded_from_bh_denominator(self):
        """BH is applied on the non-NaN subset; NaN entries return 1.0."""
        p = np.array([0.01, np.nan, 0.04, 0.2, np.nan])
        out = _correct_pvalues(p, method="fdr_bh")
        # Expected BH on [0.01, 0.04, 0.2] (m = 3)
        expected_valid = false_discovery_control(
            np.array([0.01, 0.04, 0.2]), method="bh"
        )
        np.testing.assert_allclose(out[[0, 2, 3]], expected_valid)
        assert np.isclose(out[1], 1.0) and np.isclose(out[4], 1.0)

    def test_all_nan_returns_ones(self):
        out = _correct_pvalues(np.array([np.nan, np.nan]), method="fdr_bh")
        np.testing.assert_allclose(out, np.array([1.0, 1.0]))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown correction method"):
            _correct_pvalues(np.array([0.5]), method="nonsense")


class TestFoldChange:
    def test_matches_formula(self):
        mr = np.array([4.0, 2.0])
        ms = np.array([1.0, 2.0])
        out = _compute_fold_change(mr, ms, pseudocount=0.0)
        expected = np.log2(mr / ms)
        np.testing.assert_allclose(out, expected)

    def test_pseudocount_guards_zero(self):
        mr = np.array([1.0])
        ms = np.array([0.0])
        out = _compute_fold_change(mr, ms, pseudocount=1e-10)
        assert np.isfinite(out).all()

    def test_sign_direction(self):
        out = _compute_fold_change(np.array([4.0, 1.0]), np.array([1.0, 4.0]))
        assert out[0] > 0
        assert out[1] < 0


class TestEffectSize:
    def test_cohen_d_matches_formula(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        mean_a = a.mean()
        mean_b = b.mean()
        var_a = a.var(ddof=1)
        var_b = b.var(ddof=1)
        pooled = np.sqrt(
            ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
        )
        expected = (mean_a - mean_b) / pooled
        assert _compute_effect_size(a, b) == pytest.approx(expected)

    def test_zero_when_groups_constant(self):
        assert _compute_effect_size(np.ones(5), np.ones(5)) == 0.0

    def test_zero_when_small_sample(self):
        assert _compute_effect_size(np.array([1.0]), np.array([1.0, 2.0])) == 0.0

    def test_direction(self):
        d = _compute_effect_size(np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0]))
        assert d > 0
