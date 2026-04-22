"""Tests for AMR evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.evaluation import (
    amr_classification_report,
    amr_multilabel_report,
    categorical_agreement,
    major_error_rate,
    me_scorer,
    sensitivity_score,
    specificity_score,
    very_major_error_rate,
    vme_me_curve,
    vme_scorer,
)


class TestVeryMajorErrorRate:
    """Tests for VME (resistant called susceptible)."""

    def test_perfect_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        assert very_major_error_rate(y_true, y_pred) == 0.0

    def test_all_resistant_missed(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        assert very_major_error_rate(y_true, y_pred) == 1.0

    def test_half_resistant_missed(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 1, 0, 0]
        assert very_major_error_rate(y_true, y_pred) == 0.5

    def test_no_resistant_samples(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 1]
        assert very_major_error_rate(y_true, y_pred) == 0.0

    def test_custom_resistant_label(self):
        y_true = [2, 2, 0, 0]
        y_pred = [0, 2, 0, 0]
        assert very_major_error_rate(y_true, y_pred, resistant_label=2) == 0.5


class TestMajorErrorRate:
    """Tests for ME (susceptible called resistant)."""

    def test_perfect_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        assert major_error_rate(y_true, y_pred) == 0.0

    def test_all_susceptible_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 1]
        assert major_error_rate(y_true, y_pred) == 1.0

    def test_half_susceptible_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 0]
        assert major_error_rate(y_true, y_pred) == 0.5

    def test_no_susceptible_samples(self):
        y_true = [1, 1, 1]
        y_pred = [1, 0, 1]
        assert major_error_rate(y_true, y_pred) == 0.0


class TestSensitivityScore:
    """Tests for sensitivity (TP / (TP + FN))."""

    def test_perfect(self):
        assert sensitivity_score([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_zero(self):
        assert sensitivity_score([1, 1, 0, 0], [0, 0, 0, 0]) == 0.0

    def test_complement_of_vme(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 1, 0, 0]
        vme = very_major_error_rate(y_true, y_pred)
        sens = sensitivity_score(y_true, y_pred)
        assert pytest.approx(vme + sens) == 1.0


class TestSpecificityScore:
    """Tests for specificity (TN / (TN + FP))."""

    def test_perfect(self):
        assert specificity_score([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_zero(self):
        assert specificity_score([1, 1, 0, 0], [1, 1, 1, 1]) == 0.0

    def test_complement_of_me(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 0]
        me = major_error_rate(y_true, y_pred)
        spec = specificity_score(y_true, y_pred)
        assert pytest.approx(me + spec) == 1.0


class TestCategoricalAgreement:
    """Tests for categorical agreement (accuracy)."""

    def test_perfect(self):
        assert categorical_agreement([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_all_wrong(self):
        assert categorical_agreement([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0

    def test_half_correct(self):
        assert categorical_agreement([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5

    def test_empty(self):
        assert categorical_agreement([], []) == 0.0


class TestVmeMeCurve:
    """Tests for VME/ME curve computation."""

    def test_returns_correct_shapes(self):
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.6, 0.4, 0.1]
        vme_rates, me_rates, thresholds = vme_me_curve(y_true, y_score)
        assert len(vme_rates) == len(thresholds)
        assert len(me_rates) == len(thresholds)

    def test_monotonicity(self):
        """VME should generally increase with threshold; ME should decrease."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.7, 0.5, 0.4, 0.2, 0.1])
        vme_rates, me_rates, thresholds = vme_me_curve(y_true, y_score)
        # At low thresholds: everything predicted resistant -> VME=0, ME high
        # At high thresholds: everything predicted susceptible -> VME high, ME=0
        assert vme_rates[0] <= vme_rates[-1] or me_rates[0] >= me_rates[-1]

    def test_extreme_thresholds(self):
        y_true = [1, 0]
        y_score = [0.8, 0.2]
        vme_rates, me_rates, thresholds = vme_me_curve(y_true, y_score)
        # At lowest threshold (0.2): both predicted resistant -> VME=0
        assert vme_rates[0] == 0.0

    def test_matches_per_threshold_reference(self):
        """Vectorised implementation must match a per-threshold loop
        calling the canonical VME/ME functions element-wise."""
        rng = np.random.default_rng(0)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        y_score = rng.random(n)

        vme, me, thresholds = vme_me_curve(y_true, y_score)

        exp_vme = np.empty_like(vme)
        exp_me = np.empty_like(me)
        for i, t in enumerate(thresholds):
            y_pred = (y_score >= t).astype(int)
            exp_vme[i] = very_major_error_rate(y_true, y_pred)
            exp_me[i] = major_error_rate(y_true, y_pred)
        np.testing.assert_allclose(vme, exp_vme)
        np.testing.assert_allclose(me, exp_me)


class TestAmrClassificationReport:
    """Tests for the full classification report."""

    def test_report_keys(self):
        report = amr_classification_report([1, 1, 0, 0], [1, 0, 0, 1])
        expected_keys = {
            "vme",
            "me",
            "sensitivity",
            "specificity",
            "categorical_agreement",
            "n_resistant",
            "n_susceptible",
            "n_total",
        }
        assert set(report.keys()) == expected_keys

    def test_report_values(self):
        report = amr_classification_report([1, 1, 0, 0], [1, 1, 0, 0])
        assert report["vme"] == 0.0
        assert report["me"] == 0.0
        assert report["sensitivity"] == 1.0
        assert report["specificity"] == 1.0
        assert report["categorical_agreement"] == 1.0
        assert report["n_resistant"] == 2
        assert report["n_susceptible"] == 2
        assert report["n_total"] == 4

    def test_report_counts(self):
        report = amr_classification_report([1, 1, 1, 0], [1, 0, 0, 0])
        assert report["n_resistant"] == 3
        assert report["n_susceptible"] == 1
        assert report["n_total"] == 4


class TestSklearnScorers:
    """Tests for pre-built sklearn scorers."""

    def test_vme_scorer_sign(self):
        """VME scorer should return negative values (greater_is_better=False)."""
        from sklearn.dummy import DummyClassifier

        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 1, 0, 0])
        clf = DummyClassifier(strategy="most_frequent").fit(X, y)
        score = vme_scorer(clf, X, y)
        assert score <= 0

    def test_me_scorer_sign(self):
        """ME scorer should return negative values (greater_is_better=False)."""
        from sklearn.dummy import DummyClassifier

        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 1, 0, 0])
        clf = DummyClassifier(strategy="most_frequent").fit(X, y)
        score = me_scorer(clf, X, y)
        assert score <= 0


class TestSingleClassEdgeCase:
    """Tests for single-class edge cases."""

    def test_single_class_all_resistant(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 0])
        report = amr_classification_report(y_true, y_pred)
        assert "vme" in report
        assert report["n_resistant"] == 3
        assert report["n_susceptible"] == 0

    def test_single_class_all_susceptible(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 1, 0])
        report = amr_classification_report(y_true, y_pred)
        assert "me" in report
        assert report["n_susceptible"] == 3
        assert report["n_resistant"] == 0

    def test_vme_all_resistant(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 0, 1])
        assert very_major_error_rate(y_true, y_pred) == pytest.approx(1 / 3)

    def test_me_all_susceptible(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 0])
        assert major_error_rate(y_true, y_pred) == pytest.approx(1 / 3)


class TestVmeMeCurveCustomLabels:
    """Tests for vme_me_curve with non-default labels."""

    def test_vme_me_curve_non_default_labels(self):
        y_true = np.array([2, 2, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.3, 0.1])
        vme_rates, me_rates, thresholds = vme_me_curve(
            y_true, y_scores, resistant_label=2
        )
        assert len(thresholds) > 0
        assert len(vme_rates) == len(thresholds)
        assert len(me_rates) == len(thresholds)

    def test_vme_me_curve_string_labels(self):
        y_true = np.array(["R", "R", "S", "S", "S"])
        y_scores = np.array([0.9, 0.8, 0.2, 0.3, 0.1])
        vme_rates, me_rates, thresholds = vme_me_curve(
            y_true, y_scores, resistant_label="R"
        )
        assert len(thresholds) > 0
        # At lowest threshold, all predicted resistant -> VME should be 0
        assert vme_rates[0] == 0.0


class TestComputeMacroAverage:
    """Tests for _compute_macro_average helper."""

    def test_empty_reports_returns_zeros(self):
        """Empty reports dict should produce zero-valued metrics."""
        from maldiamrkit.evaluation.metrics import _compute_macro_average

        macro = _compute_macro_average({})
        assert macro["vme"] == 0.0
        assert macro["me"] == 0.0
        assert macro["n_total"] == 0

    def test_single_drug_equals_drug(self):
        """Macro average of a single drug should equal that drug's metrics."""
        from maldiamrkit.evaluation.metrics import _compute_macro_average

        drug_report = {
            "vme": 0.25,
            "me": 0.1,
            "sensitivity": 0.75,
            "specificity": 0.9,
            "categorical_agreement": 0.85,
            "n_resistant": 4,
            "n_susceptible": 6,
            "n_total": 10,
        }
        macro = _compute_macro_average({"Drug1": drug_report})
        assert macro["vme"] == pytest.approx(0.25)
        assert macro["n_total"] == 10


class TestAmrMultilabelReport:
    def test_basic(self):
        y_true = pd.DataFrame({"Drug1": [1, 1, 0, 0], "Drug2": [0, 0, 1, 1]})
        y_pred = pd.DataFrame({"Drug1": [1, 0, 0, 0], "Drug2": [0, 1, 1, 1]})
        report = amr_multilabel_report(y_true, y_pred)
        assert "Drug1" in report
        assert "Drug2" in report
        assert "macro_avg" in report
        assert report["Drug1"]["vme"] == 0.5
        assert report["Drug2"]["me"] == pytest.approx(0.5)

    def test_as_dataframe(self):
        y_true = pd.DataFrame({"A": [1, 1, 0], "B": [0, 1, 0]})
        y_pred = pd.DataFrame({"A": [1, 1, 0], "B": [0, 1, 0]})
        df = amr_multilabel_report(y_true, y_pred, as_dataframe=True)
        assert isinstance(df, pd.DataFrame)
        assert "macro_avg" in df.index
        assert "A" in df.index
        assert "vme" in df.columns

    def test_with_nan(self):
        y_true = pd.DataFrame({"D1": [1, 0, np.nan], "D2": [0, 1, 1]})
        y_pred = pd.DataFrame({"D1": [1, 0, 1], "D2": [0, 1, 0]})
        report = amr_multilabel_report(y_true, y_pred)
        assert report["D1"]["n_total"] == 2
        assert report["D2"]["n_total"] == 3

    def test_single_drug(self):
        y_true = pd.DataFrame({"Drug": [1, 1, 0, 0]})
        y_pred = pd.DataFrame({"Drug": [1, 0, 0, 0]})
        report = amr_multilabel_report(y_true, y_pred)
        assert "Drug" in report
        assert "macro_avg" in report
        assert report["Drug"]["vme"] == report["macro_avg"]["vme"]

    def test_no_common_columns_raises(self):
        y_true = pd.DataFrame({"A": [1, 0]})
        y_pred = pd.DataFrame({"B": [1, 0]})
        with pytest.raises(ValueError, match="No common columns"):
            amr_multilabel_report(y_true, y_pred)
