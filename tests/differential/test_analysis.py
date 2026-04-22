"""Tests for DifferentialAnalysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.differential import DifferentialAnalysis

_EXPECTED_COLUMNS = [
    "mz_bin",
    "mean_r",
    "mean_s",
    "fold_change",
    "p_value",
    "adjusted_p_value",
    "effect_size",
]


class TestConstruction:
    def test_accepts_ndarray_labels(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y.to_numpy())
        assert len(analysis.y) == len(X)

    def test_requires_dataframe_X(self):
        with pytest.raises(TypeError, match="DataFrame"):
            DifferentialAnalysis(np.zeros((4, 2)), np.array([0, 1, 0, 1]))

    def test_length_mismatch_raises(self):
        X = pd.DataFrame(np.zeros((3, 2)))
        with pytest.raises(ValueError, match="does not match"):
            DifferentialAnalysis(X, np.array([0, 1]))

    def test_non_binary_labels_raise(self):
        X = pd.DataFrame(np.zeros((4, 2)))
        with pytest.raises(ValueError, match="binary labels"):
            DifferentialAnalysis(X, np.array([0, 1, 2, 1]))

    def test_missing_labels_are_dropped(self):
        X = pd.DataFrame(np.zeros((4, 2)), index=["a", "b", "c", "d"])
        y = pd.Series([0, 1, np.nan, 1], index=["a", "b", "c", "d"])
        analysis = DifferentialAnalysis(X, y)
        assert list(analysis.y.index) == ["a", "b", "d"]
        assert analysis.X.shape == (3, 2)


class TestRun:
    def test_results_schema(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        assert list(analysis.results.columns) == _EXPECTED_COLUMNS
        assert len(analysis.results) == X.shape[1]

    def test_results_raises_before_run(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y)
        with pytest.raises(RuntimeError, match="not been run"):
            _ = analysis.results

    def test_run_returns_self(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y)
        assert analysis.run() is analysis

    def test_recovers_injected_markers(self, differential_dataset):
        X, y, marker_names = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        top = analysis.top_peaks(n=len(marker_names))
        assert set(marker_names).issubset(set(top["mz_bin"].astype(str)))

    @pytest.mark.parametrize("test", ["mann_whitney", "t_test"])
    @pytest.mark.parametrize("correction", ["fdr_bh", "fdr_by", "bonferroni"])
    def test_all_test_correction_combos(self, differential_dataset, test, correction):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run(test=test, correction=correction)
        assert analysis.results["adjusted_p_value"].between(0.0, 1.0).all()
        assert analysis.results["p_value"].between(0.0, 1.0).all()

    def test_single_class_raises(self, differential_dataset_single_class):
        X, y = differential_dataset_single_class
        with pytest.raises(ValueError, match="at least one sample in each class"):
            DifferentialAnalysis(X, y).run()

    def test_tiny_groups_do_not_crash(self, differential_dataset_tiny):
        X, y = differential_dataset_tiny
        analysis = DifferentialAnalysis(X, y).run(test="t_test")
        assert analysis.results.shape[0] == X.shape[1]

    def test_fold_change_sign_matches_injection(self, differential_dataset):
        X, y, marker_names = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        res = analysis.results.set_index("mz_bin")
        for name in marker_names:
            assert res.loc[name, "fold_change"] > 0


class TestRunFilters:
    """mz_ranges + peak_detector parameters to run()."""

    def test_mz_ranges_single_tuple(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run(mz_ranges=(2000, 2030))
        mz = pd.to_numeric(analysis.results["mz_bin"]).to_numpy()
        assert ((mz >= 2000) & (mz <= 2030)).all()

    def test_mz_ranges_list_of_tuples(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run(
            mz_ranges=[(2000, 2020), (2080, 2110)]
        )
        mz = pd.to_numeric(analysis.results["mz_bin"]).to_numpy()
        in_any = ((mz >= 2000) & (mz <= 2020)) | ((mz >= 2080) & (mz <= 2110))
        assert in_any.all()

    def test_mz_ranges_reduces_testing_burden(self, differential_dataset):
        X, y, _ = differential_dataset
        full = DifferentialAnalysis(X, y).run()
        narrow = DifferentialAnalysis(X, y).run(mz_ranges=(2000, 2030))
        assert len(narrow.results) < len(full.results)
        # Narrower testing -> smaller multiplier -> smaller or equal adj p-values
        # for bins present in both runs.
        shared = set(narrow.results["mz_bin"]) & set(full.results["mz_bin"])
        for bin_id in list(shared)[:5]:
            p_full = full.results.set_index("mz_bin").loc[bin_id, "adjusted_p_value"]
            p_narrow = narrow.results.set_index("mz_bin").loc[
                bin_id, "adjusted_p_value"
            ]
            assert p_narrow <= p_full + 1e-12

    def test_mz_ranges_swapped_endpoints_are_normalized(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run(mz_ranges=(2030, 2000))
        mz = pd.to_numeric(analysis.results["mz_bin"]).to_numpy()
        assert ((mz >= 2000) & (mz <= 2030)).all()

    def test_mz_ranges_empty_result_raises(self, differential_dataset):
        X, y, _ = differential_dataset
        with pytest.raises(ValueError, match="No bins remain"):
            DifferentialAnalysis(X, y).run(mz_ranges=(9.9e9, 1e10))

    def test_mz_ranges_invalid_tuple_raises(self, differential_dataset):
        X, y, _ = differential_dataset
        with pytest.raises(ValueError, match="low, high"):
            DifferentialAnalysis(X, y).run(mz_ranges=[(1, 2, 3)])

    def test_peak_detector_filter(self, differential_dataset):
        from maldiamrkit.detection import MaldiPeakDetector

        X, y, marker_names = differential_dataset
        detector = MaldiPeakDetector(method="local", binary=True, prominence=0.1)
        analysis = DifferentialAnalysis(X, y).run(peak_detector=detector)
        assert len(analysis.results) <= X.shape[1]
        # Injected markers are boosted intensities - they should survive as peaks
        kept = set(analysis.results["mz_bin"].astype(str))
        assert any(name in kept for name in marker_names)

    def test_peak_detector_reduces_testing_burden(self, differential_dataset):
        from maldiamrkit.detection import MaldiPeakDetector

        X, y, _ = differential_dataset
        detector = MaldiPeakDetector(method="local", binary=True, prominence=0.1)
        full = DifferentialAnalysis(X, y).run()
        filtered = DifferentialAnalysis(X, y).run(peak_detector=detector)
        assert len(filtered.results) < len(full.results)

    def test_filters_combine_by_intersection(self, differential_dataset):
        from maldiamrkit.detection import MaldiPeakDetector

        X, y, _ = differential_dataset
        detector = MaldiPeakDetector(method="local", binary=True, prominence=0.05)
        combined = DifferentialAnalysis(X, y).run(
            mz_ranges=(2000, 2030), peak_detector=detector
        )
        mz = pd.to_numeric(combined.results["mz_bin"]).to_numpy()
        assert ((mz >= 2000) & (mz <= 2030)).all()

    def test_no_filters_matches_default_run(self, differential_dataset):
        X, y, _ = differential_dataset
        a = DifferentialAnalysis(X, y).run()
        b = DifferentialAnalysis(X, y).run(mz_ranges=None, peak_detector=None)
        pd.testing.assert_frame_equal(a.results, b.results)


class TestTopAndSignificant:
    def test_top_peaks_count(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        assert len(analysis.top_peaks(n=7)) == 7

    def test_top_peaks_sorted_ascending(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        top = analysis.top_peaks(n=10)["adjusted_p_value"].to_numpy()
        assert np.all(np.diff(top) >= 0)

    def test_top_peaks_capped_at_size(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        huge = analysis.top_peaks(n=10_000)
        assert len(huge) == X.shape[1]

    def test_significant_peaks_filter(self, differential_dataset):
        X, y, marker_names = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        sig = analysis.significant_peaks(fc_threshold=1.0, p_threshold=0.05)
        assert set(marker_names).issubset(set(sig["mz_bin"].astype(str)))
        assert (sig["adjusted_p_value"] <= 0.05).all()
        assert (sig["fold_change"].abs() >= 1.0).all()

    def test_significant_peaks_empty_when_strict(self, differential_dataset):
        X, y, _ = differential_dataset
        analysis = DifferentialAnalysis(X, y).run()
        sig = analysis.significant_peaks(fc_threshold=100.0, p_threshold=1e-50)
        assert len(sig) == 0


class TestCompareDrugs:
    def test_boolean_matrix_shape(self, differential_dataset):
        X, y, marker_names = differential_dataset
        a = DifferentialAnalysis(X, y).run()
        b = DifferentialAnalysis(X, y).run()
        out = DifferentialAnalysis.compare_drugs({"drugA": a, "drugB": b})
        assert out.shape[1] == 2
        assert all(dt == np.dtype(bool) for dt in out.dtypes)
        assert set(out.columns) == {"drugA", "drugB"}
        assert set(marker_names).issubset(set(out.index.astype(str)))
        assert out["drugA"].equals(out["drugB"])

    def test_union_of_significant_bins(self, differential_dataset):
        X, y, _ = differential_dataset
        a = DifferentialAnalysis(X, y).run()
        b = DifferentialAnalysis(X, y).run()
        out = DifferentialAnalysis.compare_drugs(
            {"drugA": a, "drugB": b}, fc_threshold=0.5, p_threshold=0.1
        )
        sig_a = set(a.significant_peaks(fc_threshold=0.5, p_threshold=0.1)["mz_bin"])
        sig_b = set(b.significant_peaks(fc_threshold=0.5, p_threshold=0.1)["mz_bin"])
        assert set(out.index) == sig_a.union(sig_b)

    def test_empty_analyses_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            DifferentialAnalysis.compare_drugs({})
