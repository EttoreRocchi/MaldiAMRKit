"""Unit tests for MaldiPeakDetector class."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from maldiamrkit.detection import MaldiPeakDetector


class TestMaldiPeakDetectorInit:
    """Tests for MaldiPeakDetector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        detector = MaldiPeakDetector()

        assert detector.method == "local"
        assert detector.binary is True
        assert detector.n_jobs == 1

    def test_custom_init(self):
        """Test custom initialization."""
        detector = MaldiPeakDetector(
            method="ph",
            binary=False,
            persistence_threshold=1e-5,
            n_jobs=4,
        )

        assert detector.method == "ph"
        assert detector.binary is False
        assert detector.persistence_threshold == 1e-5
        assert detector.n_jobs == 4

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            MaldiPeakDetector(method="invalid")


class TestMaldiPeakDetectorTransform:
    """Tests for MaldiPeakDetector transform."""

    def test_local_maxima_finds_peaks(self, binned_dataset: pd.DataFrame):
        """Test that local maxima method finds peaks."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        result = detector.fit_transform(binned_dataset)

        assert result.shape == binned_dataset.shape
        # Should detect some peaks (non-zero values)
        assert (result.sum(axis=1) > 0).all()

    def test_binary_mode(self, binned_dataset: pd.DataFrame):
        """Test binary mode returns only 0s and 1s."""
        detector = MaldiPeakDetector(binary=True, prominence=1e-4)
        result = detector.fit_transform(binned_dataset)

        unique_values = np.unique(result.values)
        assert set(unique_values).issubset({0, 1})

    def test_intensity_mode(self, binned_dataset: pd.DataFrame):
        """Test intensity mode preserves peak values."""
        detector = MaldiPeakDetector(binary=False, prominence=1e-4)
        result = detector.fit_transform(binned_dataset)

        # Non-zero values should match original intensities
        for i in range(len(result)):
            peak_mask = result.iloc[i] > 0
            if peak_mask.any():
                np.testing.assert_array_almost_equal(
                    result.iloc[i][peak_mask].values,
                    binned_dataset.iloc[i][peak_mask].values,
                )

    @pytest.mark.slow
    def test_ph_method(self, binned_dataset: pd.DataFrame):
        """Test persistent homology method."""
        small_dataset = binned_dataset.iloc[:5]
        detector = MaldiPeakDetector(method="ph", persistence_threshold=1e-5)
        result = detector.fit_transform(small_dataset)

        assert result.shape == small_dataset.shape

    def test_series_input(self, binned_dataset: pd.DataFrame):
        """Test that Series input works."""
        detector = MaldiPeakDetector(prominence=1e-4)
        single_spectrum = binned_dataset.iloc[0]

        result = detector.fit_transform(single_spectrum)

        assert isinstance(result, pd.Series)
        assert len(result) == len(single_spectrum)


class TestMaldiPeakDetectorParallelization:
    """Tests for parallelization."""

    def test_parallel_produces_same_results(self, binned_dataset: pd.DataFrame):
        """Test that parallel processing produces same results as sequential."""
        detector_seq = MaldiPeakDetector(method="local", prominence=1e-4, n_jobs=1)
        detector_par = MaldiPeakDetector(method="local", prominence=1e-4, n_jobs=2)

        result_seq = detector_seq.fit_transform(binned_dataset)
        result_par = detector_par.fit_transform(binned_dataset)

        pd.testing.assert_frame_equal(result_seq, result_par)

    @pytest.mark.slow
    def test_parallel_ph_method(self, binned_dataset: pd.DataFrame):
        """Test parallel processing with PH method."""
        # Use smaller dataset for faster test
        small_dataset = binned_dataset.iloc[:5]

        detector_seq = MaldiPeakDetector(
            method="ph", persistence_threshold=1e-5, n_jobs=1
        )
        detector_par = MaldiPeakDetector(
            method="ph", persistence_threshold=1e-5, n_jobs=2
        )

        result_seq = detector_seq.fit_transform(small_dataset)
        result_par = detector_par.fit_transform(small_dataset)

        pd.testing.assert_frame_equal(result_seq, result_par)


class TestMaldiPeakDetectorStatistics:
    """Tests for get_peak_statistics."""

    def test_get_peak_statistics(self, binned_dataset: pd.DataFrame):
        """Test get_peak_statistics method."""
        detector = MaldiPeakDetector(prominence=1e-4)
        detector.fit(binned_dataset)
        stats = detector.get_peak_statistics(binned_dataset)

        assert "n_peaks" in stats.columns
        assert "mean_intensity" in stats.columns
        assert "max_intensity" in stats.columns
        assert len(stats) == len(binned_dataset)


class TestMaldiPeakDetectorSklearn:
    """Tests for sklearn compatibility."""

    def test_sklearn_clone(self, binned_dataset: pd.DataFrame):
        """Test that detector can be cloned."""
        from sklearn.base import clone

        detector = MaldiPeakDetector(method="ph", persistence_threshold=1e-5, n_jobs=2)
        cloned = clone(detector)

        assert cloned.method == "ph"
        assert cloned.persistence_threshold == 1e-5
        assert cloned.n_jobs == 2

    def test_sklearn_pipeline(self, binned_dataset: pd.DataFrame):
        """Test detector in sklearn pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline(
            [
                ("peaks", MaldiPeakDetector(binary=False, prominence=1e-4)),
                ("scaler", StandardScaler()),
            ]
        )

        result = pipe.fit_transform(binned_dataset)

        assert result.shape == binned_dataset.shape


class TestPeakDetectorTransformDirect:
    """Tests for transform method called directly."""

    def test_transform_after_fit(self, binned_dataset: pd.DataFrame):
        """Test transform called separately from fit."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        result = detector.transform(binned_dataset)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == binned_dataset.shape

    def test_transform_series_input(self, binned_dataset: pd.DataFrame):
        """Test transform with Series input."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        # Convert to Series
        series = binned_dataset.iloc[0]
        result = detector.transform(series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)


class TestPeakDetectorPlot:
    """Tests for plot_peaks method."""

    def test_plot_peaks_basic(self, binned_dataset: pd.DataFrame):
        """Test basic peak plot."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        fig, ax = detector.plot_peaks(binned_dataset, indices=[0])
        assert fig is not None
        plt.close(fig)

    def test_plot_peaks_multiple_indices(self, binned_dataset: pd.DataFrame):
        """Test peak plot with multiple indices."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        fig, axes = detector.plot_peaks(binned_dataset, indices=[0, 1])
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_peaks_invalid_index(self, binned_dataset: pd.DataFrame):
        """Test plot_peaks with invalid index raises ValueError."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        with pytest.raises(ValueError, match="out of bounds"):
            detector.plot_peaks(binned_dataset, indices=[999])

    def test_plot_peaks_default_index(self, binned_dataset: pd.DataFrame):
        """Test plot_peaks with default index (None)."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        fig, ax = detector.plot_peaks(binned_dataset)  # indices=None
        assert fig is not None
        plt.close(fig)


class TestPeakStatisticsEdgeCases:
    """Tests for get_peak_statistics edge cases."""

    def test_statistics_high_prominence(self, binned_dataset: pd.DataFrame):
        """Test statistics with high prominence (few peaks detected)."""
        detector = MaldiPeakDetector(method="local", prominence=0.5)
        detector.fit(binned_dataset)
        stats = detector.get_peak_statistics(binned_dataset)
        assert isinstance(stats, pd.DataFrame)
        assert "n_peaks" in stats.columns

    def test_statistics_series_input(self, binned_dataset: pd.DataFrame):
        """Test statistics with Series input."""
        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        stats = detector.get_peak_statistics(binned_dataset.iloc[0])
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 1
