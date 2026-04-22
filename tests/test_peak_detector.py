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
        with pytest.raises(ValueError, match="is not a valid"):
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
    """Tests for the plot_peaks function."""

    def test_plot_peaks_basic(self, binned_dataset: pd.DataFrame):
        """Test basic peak plot."""
        from maldiamrkit.visualization import plot_peaks

        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        fig, ax = plot_peaks(detector, binned_dataset, indices=[0])
        assert fig is not None
        plt.close(fig)

    def test_plot_peaks_multiple_indices(self, binned_dataset: pd.DataFrame):
        """Test peak plot with multiple indices."""
        from maldiamrkit.visualization import plot_peaks

        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        fig, axes = plot_peaks(detector, binned_dataset, indices=[0, 1])
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_peaks_invalid_index(self, binned_dataset: pd.DataFrame):
        """Test plot_peaks with invalid index raises ValueError."""
        from maldiamrkit.visualization import plot_peaks

        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        with pytest.raises(ValueError, match="out of bounds"):
            plot_peaks(detector, binned_dataset, indices=[999])

    def test_plot_peaks_default_index(self, binned_dataset: pd.DataFrame):
        """Test plot_peaks with default index (None)."""
        from maldiamrkit.visualization import plot_peaks

        detector = MaldiPeakDetector(method="local", prominence=1e-4)
        detector.fit(binned_dataset)
        fig, ax = plot_peaks(detector, binned_dataset)  # indices=None
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

    def test_statistics_no_peaks_found(self):
        """Test statistics when no peaks are found."""
        detector = MaldiPeakDetector(method="local", prominence=999)
        flat = pd.DataFrame(np.zeros((1, 100)), columns=[str(i) for i in range(100)])
        detector.fit(flat)
        stats = detector.get_peak_statistics(flat)
        assert stats.iloc[0]["n_peaks"] == 0
        assert stats.iloc[0]["mean_intensity"] == 0.0
        assert stats.iloc[0]["max_intensity"] == 0.0


class TestPeakDetectorPHEdges:
    """Edge case tests for persistent homology peak detection."""

    @pytest.mark.slow
    def test_constant_signal_returns_empty(self):
        """Test _detect_peaks_ph with constant signal returns no peaks."""
        detector = MaldiPeakDetector(method="ph")
        constant = pd.DataFrame(np.ones((1, 100)), columns=[str(i) for i in range(100)])
        detector.fit(constant)
        result = detector.transform(constant)
        assert result.iloc[0].sum() == 0

    def test_fit_empty_dataframe(self):
        detector = MaldiPeakDetector()
        with pytest.raises(ValueError, match="empty"):
            detector.fit(pd.DataFrame())

    @pytest.mark.slow
    def test_ph_fallback_candidates(self):
        rng = np.random.default_rng(99)
        signal = rng.uniform(0.0, 0.001, 200)
        signal[50] = 0.5
        signal[51] = 0.49
        signal[100] = 0.8
        signal[150] = 0.3
        X = pd.DataFrame([signal], columns=[str(i) for i in range(200)])
        detector = MaldiPeakDetector(method="ph", persistence_threshold=1e-4)
        detector.fit(X)
        result = detector.transform(X)
        assert result.iloc[0].sum() > 0

    @pytest.mark.slow
    def test_ph_statistics(self):
        rng = np.random.default_rng(42)
        signal = rng.uniform(0.0, 0.01, 100)
        signal[30] = 1.0
        signal[70] = 0.5
        X = pd.DataFrame([signal], columns=[str(i) for i in range(100)])
        detector = MaldiPeakDetector(method="ph", persistence_threshold=1e-3)
        detector.fit(X)
        stats = detector.get_peak_statistics(X)
        assert isinstance(stats, pd.DataFrame)
        assert "n_peaks" in stats.columns
        assert stats.iloc[0]["n_peaks"] > 0

    @pytest.mark.slow
    def test_ph_indices_match_true_maxima(self):
        """PH peak indices land exactly on the injected maxima (no
        value-proximity heuristic)."""
        signal = np.zeros(200)
        signal[40] = 1.0
        signal[90] = 0.8
        signal[160] = 1.2
        X = pd.DataFrame([signal], columns=[str(i) for i in range(200)])
        detector = MaldiPeakDetector(
            method="ph", persistence_threshold=0.5, binary=True
        )
        detector.fit(X)
        result = detector.transform(X)
        detected = np.flatnonzero(result.iloc[0].to_numpy())
        assert set(detected.tolist()) == {40, 90, 160}

    @pytest.mark.slow
    def test_ph_plateau_no_ambiguity(self):
        """On a plateau of equal values, the heuristic would pick an
        ambiguous index; the direct cell-index recovery must pick a
        single well-defined local maximum per persistent component."""
        signal = np.zeros(100)
        signal[30:35] = 0.9  # five-sample plateau
        signal[70] = 1.0
        X = pd.DataFrame([signal], columns=[str(i) for i in range(100)])
        detector = MaldiPeakDetector(
            method="ph", persistence_threshold=0.5, binary=True
        )
        detector.fit(X)
        result = detector.transform(X)
        detected = np.flatnonzero(result.iloc[0].to_numpy())
        # The 0.9 plateau gives one persistent 0D component;
        # the 1.0 point gives another. Expect exactly two peaks.
        assert len(detected) == 2
        assert 70 in detected.tolist()
        # The plateau peak must land somewhere inside the plateau.
        plateau_peak = [i for i in detected.tolist() if i != 70]
        assert 30 <= plateau_peak[0] <= 34


class TestPeakDetectorEdgeCases:
    """Additional edge case tests for peak detector."""

    def test_unknown_method_raises_on_transform(self, binned_dataset):
        """Verify unknown method raises ValueError during transform."""
        detector = MaldiPeakDetector(method="local")
        detector.method = "invalid"  # bypass __init__ validation
        with pytest.raises(ValueError, match="Unknown method"):
            detector.transform(binned_dataset.iloc[:1])

    def test_unknown_method_raises_on_statistics(self, binned_dataset):
        """Verify unknown method raises ValueError in get_peak_statistics."""
        detector = MaldiPeakDetector(method="local")
        detector.method = "invalid"
        with pytest.raises(ValueError, match="Unknown method"):
            detector.get_peak_statistics(binned_dataset.iloc[:1])
