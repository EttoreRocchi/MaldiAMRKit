"""Unit tests for binning functions."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.core.config import PreprocessingSettings
from maldiamrkit.preprocessing.binning import (
    bin_spectrum,
)
from maldiamrkit.preprocessing.pipeline import preprocess


class TestBinSpectrum:
    """Tests for bin_spectrum function."""

    @pytest.fixture
    def preprocessed_spectrum(self, synthetic_spectrum: pd.DataFrame) -> pd.DataFrame:
        """Preprocessed synthetic spectrum."""
        return preprocess(synthetic_spectrum)

    def test_bin_spectrum_uniform(self, preprocessed_spectrum: pd.DataFrame):
        """Test uniform binning."""
        cfg = PreprocessingSettings()
        result, metadata = bin_spectrum(
            preprocessed_spectrum, cfg, bin_width=3, method="uniform"
        )

        assert isinstance(result, pd.DataFrame)
        assert "mass" in result.columns
        assert "intensity" in result.columns

        # Check expected number of bins
        expected_bins = (cfg.trim_to - cfg.trim_from) // 3
        assert len(result) == expected_bins

    def test_bin_spectrum_logarithmic(self, preprocessed_spectrum: pd.DataFrame):
        """Test logarithmic binning."""
        cfg = PreprocessingSettings()
        result, metadata = bin_spectrum(
            preprocessed_spectrum, cfg, bin_width=3, method="logarithmic"
        )

        assert len(result) > 0
        # Logarithmic binning should have fewer bins than uniform (bins grow)
        uniform_result, _ = bin_spectrum(
            preprocessed_spectrum, cfg, bin_width=3, method="uniform"
        )
        # Log bins grow, so we get fewer bins with same starting width
        assert len(result) < len(uniform_result)

    def test_bin_spectrum_adaptive(self, preprocessed_spectrum: pd.DataFrame):
        """Test adaptive binning."""
        cfg = PreprocessingSettings()
        result, metadata = bin_spectrum(
            preprocessed_spectrum,
            cfg,
            method="adaptive",
            adaptive_min_width=1.0,
            adaptive_max_width=10.0,
        )

        assert len(result) > 0

    def test_bin_spectrum_custom(self, preprocessed_spectrum: pd.DataFrame):
        """Test custom binning."""
        cfg = PreprocessingSettings()
        edges = [2000, 5000, 10000, 15000, 20000]
        result, metadata = bin_spectrum(
            preprocessed_spectrum, cfg, method="custom", custom_edges=edges
        )

        assert len(result) == len(edges) - 1

    def test_bin_spectrum_preserves_intensity(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that binning preserves total intensity."""
        cfg = PreprocessingSettings()
        original_sum = preprocessed_spectrum["intensity"].sum()

        result, _ = bin_spectrum(preprocessed_spectrum, cfg, bin_width=3)
        binned_sum = result["intensity"].sum()

        # Allow small numerical tolerance
        assert np.isclose(original_sum, binned_sum, rtol=0.01)

    def test_bin_spectrum_invalid_method_raises(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that invalid method raises ValueError."""
        cfg = PreprocessingSettings()

        with pytest.raises(ValueError, match="Invalid method"):
            bin_spectrum(preprocessed_spectrum, cfg, method="invalid")

    def test_bin_spectrum_custom_without_edges_raises(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that custom method without edges raises ValueError."""
        cfg = PreprocessingSettings()

        with pytest.raises(ValueError, match="custom_edges"):
            bin_spectrum(preprocessed_spectrum, cfg, method="custom")


class TestBinMetadata:
    """Tests for bin metadata."""

    @pytest.fixture
    def preprocessed_spectrum(self, synthetic_spectrum: pd.DataFrame) -> pd.DataFrame:
        """Preprocessed synthetic spectrum."""
        return preprocess(synthetic_spectrum)

    def test_bin_metadata_columns(self, preprocessed_spectrum: pd.DataFrame):
        """Test that bin metadata has expected columns."""
        cfg = PreprocessingSettings()
        _, metadata = bin_spectrum(preprocessed_spectrum, cfg, bin_width=3)

        assert "bin_index" in metadata.columns
        assert "bin_start" in metadata.columns
        assert "bin_end" in metadata.columns
        assert "bin_width" in metadata.columns

    def test_bin_metadata_uniform_width(self, preprocessed_spectrum: pd.DataFrame):
        """Test that uniform binning has consistent bin widths."""
        cfg = PreprocessingSettings()
        _, metadata = bin_spectrum(
            preprocessed_spectrum, cfg, bin_width=3, method="uniform"
        )

        # All bins should have width 3
        assert np.allclose(metadata["bin_width"], 3.0)

    def test_bin_metadata_logarithmic_increasing_width(
        self, preprocessed_spectrum: pd.DataFrame
    ):
        """Test that logarithmic binning has increasing bin widths."""
        cfg = PreprocessingSettings()
        _, metadata = bin_spectrum(
            preprocessed_spectrum, cfg, bin_width=3, method="logarithmic"
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
        cfg = PreprocessingSettings()

        result1, meta1 = bin_spectrum(preprocessed_spectrum.copy(), cfg, bin_width=3)
        result2, meta2 = bin_spectrum(preprocessed_spectrum.copy(), cfg, bin_width=3)

        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(meta1, meta2)

    @pytest.mark.parametrize("method", ["uniform", "logarithmic", "adaptive"])
    def test_reproducible_across_methods(
        self, preprocessed_spectrum: pd.DataFrame, method: str
    ):
        """Test that all methods produce reproducible results."""
        cfg = PreprocessingSettings()

        kwargs = {}
        if method == "adaptive":
            kwargs = {"adaptive_min_width": 1.0, "adaptive_max_width": 10.0}

        result1, _ = bin_spectrum(
            preprocessed_spectrum.copy(), cfg, bin_width=3, method=method, **kwargs
        )
        result2, _ = bin_spectrum(
            preprocessed_spectrum.copy(), cfg, bin_width=3, method=method, **kwargs
        )

        pd.testing.assert_frame_equal(result1, result2)
