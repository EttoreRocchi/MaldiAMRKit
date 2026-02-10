"""Unit tests for binning functions."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.preprocessing import bin_spectrum, preprocess


class TestBinSpectrum:
    """Tests for bin_spectrum function."""

    @pytest.fixture
    def preprocessed_spectrum(self, synthetic_spectrum: pd.DataFrame) -> pd.DataFrame:
        """Preprocessed synthetic spectrum."""
        return preprocess(synthetic_spectrum)

    def test_bin_spectrum_uniform(self, preprocessed_spectrum: pd.DataFrame):
        """Test uniform binning."""
        result, metadata = bin_spectrum(
            preprocessed_spectrum, bin_width=3, method="uniform"
        )

        assert isinstance(result, pd.DataFrame)
        assert "mass" in result.columns
        assert "intensity" in result.columns

        # Check expected number of bins
        expected_bins = (20000 - 2000) // 3
        assert len(result) == expected_bins

    def test_bin_spectrum_logarithmic(self, preprocessed_spectrum: pd.DataFrame):
        """Test logarithmic binning."""
        result, metadata = bin_spectrum(
            preprocessed_spectrum, bin_width=3, method="logarithmic"
        )

        assert len(result) > 0
        # Logarithmic binning should have fewer bins than uniform (bins grow)
        uniform_result, _ = bin_spectrum(
            preprocessed_spectrum, bin_width=3, method="uniform"
        )
        # Log bins grow, so we get fewer bins with same starting width
        assert len(result) < len(uniform_result)

    def test_bin_spectrum_adaptive(self, preprocessed_spectrum: pd.DataFrame):
        """Test adaptive binning."""
        result, metadata = bin_spectrum(
            preprocessed_spectrum,
            method="adaptive",
            adaptive_min_width=1.0,
            adaptive_max_width=10.0,
        )

        assert len(result) > 0

    def test_bin_spectrum_custom(self, preprocessed_spectrum: pd.DataFrame):
        """Test custom binning."""
        edges = [2000, 5000, 10000, 15000, 20000]
        result, metadata = bin_spectrum(
            preprocessed_spectrum, method="custom", custom_edges=edges
        )

        assert len(result) == len(edges) - 1

    def test_bin_spectrum_preserves_intensity(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that binning preserves total intensity."""
        original_sum = preprocessed_spectrum["intensity"].sum()

        result, _ = bin_spectrum(preprocessed_spectrum, bin_width=3)
        binned_sum = result["intensity"].sum()

        # Allow small numerical tolerance
        assert np.isclose(original_sum, binned_sum, rtol=0.01)

    def test_bin_spectrum_invalid_method_raises(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            bin_spectrum(preprocessed_spectrum, method="invalid")

    def test_bin_spectrum_custom_without_edges_raises(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that custom method without edges raises ValueError."""
        with pytest.raises(ValueError, match="custom_edges"):
            bin_spectrum(preprocessed_spectrum, method="custom")

    def test_bin_spectrum_invalid_mz_range(self, preprocessed_spectrum: pd.DataFrame):
        """Test that mz_min >= mz_max raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            bin_spectrum(preprocessed_spectrum, mz_min=20000, mz_max=2000)

    def test_adaptive_custom_prominence(self, preprocessed_spectrum: pd.DataFrame):
        """Test that explicit prominence changes bin count."""
        result_default, _ = bin_spectrum(
            preprocessed_spectrum,
            method="adaptive",
            adaptive_min_width=1.0,
            adaptive_max_width=10.0,
        )
        # Very high prominence → no peaks detected → uniform fallback
        result_high, _ = bin_spectrum(
            preprocessed_spectrum,
            method="adaptive",
            adaptive_min_width=1.0,
            adaptive_max_width=10.0,
            adaptive_peak_prominence=1e10,
        )
        # With no peaks, falls back to uniform with max_width=10
        assert len(result_default) != len(result_high)

    def test_adaptive_custom_bandwidth(self, preprocessed_spectrum: pd.DataFrame):
        """Test that explicit bandwidth changes bin distribution."""
        _, meta_narrow = bin_spectrum(
            preprocessed_spectrum,
            method="adaptive",
            adaptive_min_width=1.0,
            adaptive_max_width=10.0,
            adaptive_kde_bandwidth=100.0,
        )
        _, meta_wide = bin_spectrum(
            preprocessed_spectrum,
            method="adaptive",
            adaptive_min_width=1.0,
            adaptive_max_width=10.0,
            adaptive_kde_bandwidth=5000.0,
        )
        # Different bandwidths should produce different bin distributions
        assert not np.array_equal(
            meta_narrow["bin_width"].values, meta_wide["bin_width"].values
        )

    def test_adaptive_silverman_default(self, preprocessed_spectrum: pd.DataFrame):
        """Test that default (None) parameters run without error."""
        result, metadata = bin_spectrum(
            preprocessed_spectrum,
            method="adaptive",
            adaptive_min_width=1.0,
            adaptive_max_width=10.0,
            adaptive_peak_prominence=None,
            adaptive_kde_bandwidth=None,
        )
        assert len(result) > 0
        assert len(metadata) > 0


class TestBinMetadata:
    """Tests for bin metadata."""

    @pytest.fixture
    def preprocessed_spectrum(self, synthetic_spectrum: pd.DataFrame) -> pd.DataFrame:
        """Preprocessed synthetic spectrum."""
        return preprocess(synthetic_spectrum)

    def test_bin_metadata_columns(self, preprocessed_spectrum: pd.DataFrame):
        """Test that bin metadata has expected columns."""
        _, metadata = bin_spectrum(preprocessed_spectrum, bin_width=3)

        assert "bin_index" in metadata.columns
        assert "bin_start" in metadata.columns
        assert "bin_end" in metadata.columns
        assert "bin_width" in metadata.columns

    def test_bin_metadata_uniform_width(self, preprocessed_spectrum: pd.DataFrame):
        """Test that uniform binning has consistent bin widths."""
        _, metadata = bin_spectrum(preprocessed_spectrum, bin_width=3, method="uniform")

        # All bins should have width 3
        assert np.allclose(metadata["bin_width"], 3.0)

    def test_bin_metadata_logarithmic_increasing_width(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that logarithmic binning has increasing bin widths."""
        _, metadata = bin_spectrum(
            preprocessed_spectrum, bin_width=3, method="logarithmic"
        )

        widths = metadata["bin_width"].values
        # Widths should generally increase (allowing some tolerance)
        assert widths[-1] > widths[0]


class TestBinningReproducibility:
    """Tests for reproducibility of binning."""

    @pytest.fixture
    def preprocessed_spectrum(self, synthetic_spectrum: pd.DataFrame) -> pd.DataFrame:
        """Preprocessed synthetic spectrum."""
        return preprocess(synthetic_spectrum)

    def test_same_input_same_output(self, preprocessed_spectrum: pd.DataFrame):
        """Test that same input produces same output."""
        result1, meta1 = bin_spectrum(preprocessed_spectrum.copy(), bin_width=3)
        result2, meta2 = bin_spectrum(preprocessed_spectrum.copy(), bin_width=3)

        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(meta1, meta2)

    @pytest.mark.parametrize("method", ["uniform", "logarithmic", "adaptive"])
    def test_reproducible_across_methods(
        self, preprocessed_spectrum: pd.DataFrame, method: str
    ):
        """Test that all methods produce reproducible results."""
        kwargs = {}
        if method == "adaptive":
            kwargs = {"adaptive_min_width": 1.0, "adaptive_max_width": 10.0}

        result1, _ = bin_spectrum(
            preprocessed_spectrum.copy(), bin_width=3, method=method, **kwargs
        )
        result2, _ = bin_spectrum(
            preprocessed_spectrum.copy(), bin_width=3, method=method, **kwargs
        )

        pd.testing.assert_frame_equal(result1, result2)
