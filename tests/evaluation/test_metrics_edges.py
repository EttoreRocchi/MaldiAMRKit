"""Edge case tests for AMR metrics - single-class and multi-class inputs."""

from __future__ import annotations

import pytest

from maldiamrkit.evaluation.metrics import (
    _get_confusion_values,
    major_error_rate,
    sensitivity_score,
    specificity_score,
    very_major_error_rate,
)


class TestSingleClassConfusion:
    """Tests for _get_confusion_values with single-class inputs."""

    def test_all_resistant_true_and_pred(self):
        tp, tn, fp, fn = _get_confusion_values([1, 1, 1], [1, 1, 1])
        assert tp == 3
        assert fn == 0
        assert tn == 0
        assert fp == 0

    def test_all_resistant_true_some_pred_susceptible(self):
        tp, tn, fp, fn = _get_confusion_values([1, 1, 1], [1, 0, 1])
        assert tp == 2
        assert fn == 1

    def test_all_susceptible_true_and_pred(self):
        tp, tn, fp, fn = _get_confusion_values([0, 0, 0], [0, 0, 0])
        assert tn == 3
        assert fp == 0
        assert tp == 0
        assert fn == 0

    def test_all_susceptible_true_some_pred_resistant(self):
        tp, tn, fp, fn = _get_confusion_values([0, 0, 0], [0, 1, 0])
        assert tn == 2
        assert fp == 1

    def test_resistant_label_not_in_labels(self):
        tp, tn, fp, fn = _get_confusion_values([0, 0, 0], [0, 0, 0], resistant_label=1)
        assert tn == 3
        assert tp == 0

    def test_multi_label_raises(self):
        with pytest.raises(ValueError, match="binary labels"):
            _get_confusion_values([0, 1, 2], [0, 1, 2])


class TestSingleClassMetrics:
    """Tests for metric functions with single-class edge cases."""

    def test_vme_all_resistant(self):
        assert very_major_error_rate([1, 1], [1, 1]) == 0.0

    def test_me_all_susceptible(self):
        assert major_error_rate([0, 0], [0, 0]) == 0.0

    def test_sensitivity_no_resistant(self):
        assert sensitivity_score([0, 0], [0, 0]) == 0.0

    def test_specificity_no_susceptible(self):
        assert specificity_score([1, 1], [1, 1]) == 0.0
