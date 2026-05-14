"""Tests for mic_regression_report."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

from maldiamrkit.evaluation import mic_regression_report
from maldiamrkit.susceptibility import BreakpointTable


@pytest.fixture
def bp() -> BreakpointTable:
    path = resources.files("maldiamrkit") / "data" / "breakpoints" / "example.yaml"
    return BreakpointTable.from_yaml(path)


class TestMicRegressionReportBasics:
    def test_perfect_predictions(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        rep = mic_regression_report(y, y)
        assert rep["n"] == 4
        assert rep["rmse_log2"] == 0.0
        assert rep["mae_log2"] == 0.0
        assert rep["bias_log2"] == 0.0
        assert rep["essential_agreement"] == 1.0

    def test_essential_agreement_within_one_dilution(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 1.0, 2.0])
        rep = mic_regression_report(y_true, y_pred)
        assert rep["essential_agreement"] == pytest.approx(2 / 3)

    def test_bias(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        rep = mic_regression_report(y_true, y_pred)
        assert rep["bias_log2"] == 1.0
        assert rep["mae_log2"] == 1.0
        assert rep["rmse_log2"] == 1.0

    def test_nan_inputs_excluded(self):
        y_true = np.array([0.0, np.nan, 1.0])
        y_pred = np.array([0.0, 5.0, 1.0])
        rep = mic_regression_report(y_true, y_pred)
        assert rep["n"] == 2
        assert rep["rmse_log2"] == 0.0

    def test_empty_after_nan_filter(self):
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([0.0, 0.0])
        rep = mic_regression_report(y_true, y_pred)
        assert rep["n"] == 0
        assert np.isnan(rep["rmse_log2"])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            mic_regression_report([1, 2, 3], [1, 2])


class TestMicRegressionReportCategorical:
    def test_with_breakpoints(self, bp):
        y_true = np.array([0.0, 1.0, 4.0])
        y_pred = np.array([0.0, 1.0, 4.0])
        rep = mic_regression_report(
            y_true,
            y_pred,
            breakpoints=bp,
            species="Klebsiella pneumoniae",
            drug="Ceftriaxone",
        )
        assert rep["categorical_agreement"] == 1.0
        assert rep["n_categorical"] == 3

    def test_very_major_error(self, bp):
        y_true = np.array([4.0, 4.0])
        y_pred = np.array([-2.0, -2.0])
        rep = mic_regression_report(
            y_true,
            y_pred,
            breakpoints=bp,
            species="Klebsiella pneumoniae",
            drug="Ceftriaxone",
        )
        assert rep["very_major_error_rate"] == 1.0

    def test_major_error(self, bp):
        y_true = np.array([-2.0, -2.0])
        y_pred = np.array([4.0, 4.0])
        rep = mic_regression_report(
            y_true,
            y_pred,
            breakpoints=bp,
            species="Klebsiella pneumoniae",
            drug="Ceftriaxone",
        )
        assert rep["major_error_rate"] == 1.0

    def test_requires_species_and_drug(self, bp):
        with pytest.raises(ValueError, match="species and drug"):
            mic_regression_report([0.0], [0.0], breakpoints=bp)
