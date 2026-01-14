"""Unit tests for utils module."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from maldiamrkit.utils.plotting import plot_spectra_comparison, plot_spectrum
from maldiamrkit.utils.validation import validate_mz_range, validate_spectrum_input


class TestValidation:
    """Tests for validation functions."""

    def test_validate_spectrum_input_dataframe(self):
        """Test with valid DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        result = validate_spectrum_input(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_validate_spectrum_input_series(self):
        """Test with Series input (converts to DataFrame)."""
        s = pd.Series([0.1, 0.2, 0.3], index=[1, 2, 3], name="intensity")
        result = validate_spectrum_input(s)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 3)

    def test_validate_spectrum_input_invalid_type(self):
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError):
            validate_spectrum_input([1, 2, 3])

    def test_validate_spectrum_input_empty(self):
        """Test empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            validate_spectrum_input(pd.DataFrame())

    def test_validate_mz_range_valid(self):
        """Test valid m/z range returns True."""
        mz_values = np.array([2000, 5000, 10000, 15000, 20000])
        result = validate_mz_range(mz_values, min_mz=1000, max_mz=25000)
        assert result is True

    def test_validate_mz_range_valid_no_bounds(self):
        """Test m/z validation with no bounds specified."""
        mz_values = np.array([100, 200, 300])
        result = validate_mz_range(mz_values)
        assert result is True

    def test_validate_mz_range_below_min(self):
        """Test values below min_mz raises ValueError."""
        mz_values = np.array([500, 1000, 2000])
        with pytest.raises(ValueError, match="below minimum"):
            validate_mz_range(mz_values, min_mz=1000)

    def test_validate_mz_range_above_max(self):
        """Test values above max_mz raises ValueError."""
        mz_values = np.array([15000, 20000, 25000])
        with pytest.raises(ValueError, match="above maximum"):
            validate_mz_range(mz_values, max_mz=20000)


class TestPlotting:
    """Tests for plotting functions."""

    @pytest.fixture
    def sample_spectrum(self):
        """Create sample spectrum DataFrame with 'mass' column."""
        return pd.DataFrame({
            "mass": np.linspace(2000, 20000, 100),
            "intensity": np.random.rand(100)
        })

    def test_plot_spectrum_basic(self, sample_spectrum):
        """Test basic spectrum plot."""
        ax = plot_spectrum(sample_spectrum)
        assert ax is not None
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_spectrum_with_ax(self, sample_spectrum):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        returned_ax = plot_spectrum(sample_spectrum, ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_plot_spectrum_with_title(self, sample_spectrum):
        """Test plot with title."""
        ax = plot_spectrum(sample_spectrum, title="Test Spectrum")
        assert ax.get_title() == "Test Spectrum"
        plt.close("all")

    def test_plot_spectrum_with_color(self, sample_spectrum):
        """Test plot with custom color."""
        ax = plot_spectrum(sample_spectrum, color="red")
        assert ax is not None
        plt.close("all")

    def test_plot_spectra_comparison(self, sample_spectrum):
        """Test comparison plot with multiple spectra."""
        spectra = [sample_spectrum.copy(), sample_spectrum.copy()]
        spectra[1]["intensity"] = spectra[1]["intensity"] * 0.8
        ax = plot_spectra_comparison(spectra, labels=["A", "B"])
        assert ax is not None
        plt.close("all")

    def test_plot_spectra_comparison_default_labels(self, sample_spectrum):
        """Test comparison with default labels."""
        spectra = [sample_spectrum.copy(), sample_spectrum.copy()]
        ax = plot_spectra_comparison(spectra)
        assert ax is not None
        plt.close("all")

    def test_plot_spectra_comparison_with_colors(self, sample_spectrum):
        """Test comparison with custom colors."""
        spectra = [sample_spectrum.copy(), sample_spectrum.copy()]
        ax = plot_spectra_comparison(spectra, colors=["red", "blue"])
        assert ax is not None
        plt.close("all")

    def test_plot_spectra_comparison_with_ax(self, sample_spectrum):
        """Test comparison plot on existing axes."""
        fig, ax = plt.subplots()
        spectra = [sample_spectrum.copy(), sample_spectrum.copy()]
        returned_ax = plot_spectra_comparison(spectra, ax=ax)
        assert returned_ax is ax
        plt.close("all")
