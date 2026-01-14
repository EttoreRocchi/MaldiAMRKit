"""Unit tests for Warping class."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from maldiamrkit import Warping


class TestWarpingInit:
    """Tests for Warping initialization."""

    def test_default_init(self):
        """Test default initialization."""
        warper = Warping()

        assert warper.method == "shift"
        assert warper.reference == "median"
        assert warper.n_jobs == 1

    def test_custom_init(self):
        """Test custom initialization."""
        warper = Warping(
            method="dtw",
            reference=0,
            n_jobs=4,
            dtw_radius=20,
        )

        assert warper.method == "dtw"
        assert warper.reference == 0
        assert warper.n_jobs == 4
        assert warper.dtw_radius == 20


class TestWarpingFit:
    """Tests for Warping fit."""

    def test_fit_with_median(self, binned_dataset: pd.DataFrame):
        """Test fitting with median reference."""
        warper = Warping(reference="median")
        warper.fit(binned_dataset)

        assert hasattr(warper, 'ref_spec_')
        assert len(warper.ref_spec_) == binned_dataset.shape[1]

    def test_fit_with_index(self, binned_dataset: pd.DataFrame):
        """Test fitting with index reference."""
        warper = Warping(reference=5)
        warper.fit(binned_dataset)

        assert hasattr(warper, 'ref_spec_')
        np.testing.assert_array_equal(
            warper.ref_spec_,
            binned_dataset.iloc[5].to_numpy()
        )

    def test_fit_invalid_index_raises(self, binned_dataset: pd.DataFrame):
        """Test that invalid reference index raises ValueError."""
        warper = Warping(reference=1000)

        with pytest.raises(ValueError, match="out of bounds"):
            warper.fit(binned_dataset)

    def test_fit_invalid_method_raises(self, binned_dataset: pd.DataFrame):
        """Test that invalid method raises ValueError."""
        warper = Warping(method="invalid")

        with pytest.raises(ValueError, match="Unknown warping method"):
            warper.fit(binned_dataset)


class TestWarpingTransform:
    """Tests for Warping transform."""

    @pytest.mark.parametrize("method", ["shift", "linear", "piecewise"])
    def test_all_methods_run(
        self, binned_dataset: pd.DataFrame, method: str
    ):
        """Test that all methods run without error."""
        warper = Warping(method=method)
        warper.fit(binned_dataset)
        result = warper.transform(binned_dataset)

        assert result.shape == binned_dataset.shape
        assert result.index.equals(binned_dataset.index)

    @pytest.mark.slow
    def test_dtw_method_run(self, binned_dataset: pd.DataFrame):
        """Test that DTW method runs without error."""
        small_dataset = binned_dataset.iloc[:5]
        warper = Warping(method="dtw")
        warper.fit(small_dataset)
        result = warper.transform(small_dataset)

        assert result.shape == small_dataset.shape
        assert result.index.equals(small_dataset.index)

    def test_transform_before_fit_raises(self, binned_dataset: pd.DataFrame):
        """Test that transform before fit raises RuntimeError."""
        warper = Warping()

        with pytest.raises(RuntimeError, match="must be fitted"):
            warper.transform(binned_dataset)

    def test_transform_preserves_shape(self, binned_dataset: pd.DataFrame):
        """Test that transform preserves shape."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        result = warper.transform(binned_dataset)

        assert result.shape == binned_dataset.shape
        assert result.columns.equals(binned_dataset.columns)


class TestWarpingParallelization:
    """Tests for parallelization."""

    @pytest.mark.parametrize("method", ["shift", "linear", "piecewise"])
    def test_parallel_produces_same_results(
        self, binned_dataset: pd.DataFrame, method: str
    ):
        """Test that parallel processing produces same results as sequential."""
        warper_seq = Warping(method=method, n_jobs=1)
        warper_par = Warping(method=method, n_jobs=2)

        warper_seq.fit(binned_dataset)
        warper_par.fit(binned_dataset)

        result_seq = warper_seq.transform(binned_dataset)
        result_par = warper_par.transform(binned_dataset)

        pd.testing.assert_frame_equal(result_seq, result_par)

    @pytest.mark.slow
    def test_parallel_dtw(self, binned_dataset: pd.DataFrame):
        """Test parallel processing with DTW method."""
        # Use smaller dataset for faster test
        small_dataset = binned_dataset.iloc[:5]

        warper_seq = Warping(method="dtw", n_jobs=1)
        warper_par = Warping(method="dtw", n_jobs=2)

        warper_seq.fit(small_dataset)
        warper_par.fit(small_dataset)

        result_seq = warper_seq.transform(small_dataset)
        result_par = warper_par.transform(small_dataset)

        pd.testing.assert_frame_equal(result_seq, result_par)


class TestWarpingQuality:
    """Tests for alignment quality."""

    def test_get_alignment_quality(self, binned_dataset_with_shift: pd.DataFrame):
        """Test get_alignment_quality method."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset_with_shift)

        X_aligned = warper.transform(binned_dataset_with_shift)
        quality = warper.get_alignment_quality(
            binned_dataset_with_shift, X_aligned
        )

        assert "correlation_before" in quality.columns
        assert "correlation_after" in quality.columns
        assert "improvement" in quality.columns
        assert "rmse_before" in quality.columns
        assert "rmse_after" in quality.columns
        assert len(quality) == len(binned_dataset_with_shift)

    def test_get_alignment_quality_linear(self, binned_dataset: pd.DataFrame):
        """Test quality with linear method."""
        warper = Warping(method="linear")
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        quality = warper.get_alignment_quality(binned_dataset, X_aligned)
        assert "improvement" in quality.columns

    def test_get_alignment_quality_piecewise(self, binned_dataset: pd.DataFrame):
        """Test quality with piecewise method."""
        warper = Warping(method="piecewise", n_segments=3)
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        quality = warper.get_alignment_quality(binned_dataset, X_aligned)
        assert len(quality) == len(binned_dataset)

    def test_get_alignment_quality_without_aligned(self, binned_dataset: pd.DataFrame):
        """Test quality when X_aligned is None (auto-transform)."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        quality = warper.get_alignment_quality(binned_dataset)
        assert quality is not None
        assert len(quality) == len(binned_dataset)


class TestWarpingPlot:
    """Tests for plot_alignment method."""

    def test_plot_alignment_basic(self, binned_dataset: pd.DataFrame):
        """Test basic alignment plot."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        fig, axes = warper.plot_alignment(binned_dataset, X_aligned, indices=[0, 1])
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_plot_alignment_single_index(self, binned_dataset: pd.DataFrame):
        """Test plot with single index."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        X_aligned = warper.transform(binned_dataset)
        fig, axes = warper.plot_alignment(binned_dataset, X_aligned, indices=[0])
        assert fig is not None
        plt.close(fig)

    def test_plot_alignment_without_aligned(self, binned_dataset: pd.DataFrame):
        """Test plot when X_aligned is None (auto-transform)."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        fig, axes = warper.plot_alignment(binned_dataset, indices=[0])
        assert fig is not None
        plt.close(fig)

    def test_plot_alignment_invalid_index(self, binned_dataset: pd.DataFrame):
        """Test plot with invalid index raises ValueError."""
        warper = Warping(method="shift")
        warper.fit(binned_dataset)
        with pytest.raises(ValueError, match="out of bounds"):
            warper.plot_alignment(binned_dataset, indices=[999])


class TestWarpingSklearn:
    """Tests for sklearn compatibility."""

    def test_sklearn_clone(self, binned_dataset: pd.DataFrame):
        """Test that warper can be cloned."""
        from sklearn.base import clone

        warper = Warping(method="piecewise", n_segments=3, n_jobs=2)
        cloned = clone(warper)

        assert cloned.method == "piecewise"
        assert cloned.n_segments == 3
        assert cloned.n_jobs == 2

    def test_sklearn_pipeline(self, binned_dataset: pd.DataFrame):
        """Test warper in sklearn pipeline."""
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("warp", Warping(method="shift")),
        ])

        result = pipe.fit_transform(binned_dataset)

        assert result.shape == binned_dataset.shape
