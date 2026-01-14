"""Unit tests for preprocessing pipeline."""
import numpy as np
import pandas as pd

from maldiamrkit.core.config import PreprocessingSettings
from maldiamrkit.preprocessing.pipeline import preprocess


class TestPreprocess:
    """Tests for the preprocess function."""

    def test_preprocess_returns_dataframe(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocess returns a DataFrame."""
        result = preprocess(synthetic_spectrum)

        assert isinstance(result, pd.DataFrame)
        assert "mass" in result.columns
        assert "intensity" in result.columns

    def test_preprocess_normalizes_to_one(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessed intensities sum to 1."""
        result = preprocess(synthetic_spectrum)

        assert np.isclose(result["intensity"].sum(), 1.0, atol=1e-6)

    def test_preprocess_removes_negatives(self, synthetic_spectrum: pd.DataFrame):
        """Test that negative intensities are removed."""
        df = synthetic_spectrum.copy()
        df.loc[0:100, "intensity"] = -100

        result = preprocess(df)

        assert (result["intensity"] >= 0).all()

    def test_preprocess_trims_range(self, synthetic_spectrum: pd.DataFrame):
        """Test that spectrum is trimmed to configured range."""
        cfg = PreprocessingSettings(trim_from=3000, trim_to=15000)
        result = preprocess(synthetic_spectrum, cfg)

        assert result["mass"].min() >= 3000
        assert result["mass"].max() <= 15000

    def test_preprocess_default_trim_range(self, synthetic_spectrum: pd.DataFrame):
        """Test default trim range (2000-20000)."""
        result = preprocess(synthetic_spectrum)

        assert result["mass"].min() >= 2000
        assert result["mass"].max() <= 20000

    def test_preprocess_preserves_peak_positions(self, synthetic_spectrum: pd.DataFrame):
        """Test that peak positions are preserved after preprocessing."""
        # The synthetic spectrum has peaks at 3000, 5000, 7500, 10000, 12500, 15000
        result = preprocess(synthetic_spectrum)

        # Find peak positions (local maxima)
        intensity = result["intensity"].values
        peak_mask = (
            (intensity[1:-1] > intensity[:-2]) &
            (intensity[1:-1] > intensity[2:])
        )
        peak_indices = np.where(peak_mask)[0] + 1
        peak_mz = result["mass"].iloc[peak_indices].values

        # Check that major peaks are near expected positions
        expected_peaks = [3000, 5000, 7500, 10000, 12500, 15000]
        for expected in expected_peaks:
            # Find closest peak
            distances = np.abs(peak_mz - expected)
            closest_distance = distances.min()
            # Peak should be within 100 Da of expected position
            assert closest_distance < 100, f"Peak at {expected} not found"

    def test_preprocess_custom_savgol_params(self, synthetic_spectrum: pd.DataFrame):
        """Test preprocessing with custom Savitzky-Golay parameters."""
        cfg = PreprocessingSettings(savgol_window=30, savgol_poly=3)
        result = preprocess(synthetic_spectrum, cfg)

        assert (result["intensity"] >= 0).all()
        assert np.isclose(result["intensity"].sum(), 1.0, atol=1e-6)

    def test_preprocess_handles_empty_after_trim(self):
        """Test that preprocessing handles edge case of empty result."""
        # Create spectrum outside the default trim range
        df = pd.DataFrame({
            "mass": np.linspace(500, 1000, 1000),
            "intensity": np.random.uniform(0, 100, 1000),
        })

        result = preprocess(df)

        # Result should be empty or handle gracefully
        assert len(result) == 0 or result["intensity"].sum() == 0

    def test_preprocess_does_not_modify_input(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessing does not modify the input DataFrame."""
        original = synthetic_spectrum.copy()
        _ = preprocess(synthetic_spectrum)

        pd.testing.assert_frame_equal(synthetic_spectrum, original)


class TestPreprocessingSettings:
    """Tests for PreprocessingSettings configuration."""

    def test_default_settings(self):
        """Test default preprocessing settings."""
        cfg = PreprocessingSettings()

        assert cfg.trim_from == 2000
        assert cfg.trim_to == 20000
        assert cfg.savgol_window == 20
        assert cfg.savgol_poly == 2
        assert cfg.baseline_half_window == 40

    def test_custom_settings(self):
        """Test custom preprocessing settings."""
        cfg = PreprocessingSettings(
            trim_from=3000,
            trim_to=15000,
            savgol_window=30,
            savgol_poly=3,
            baseline_half_window=50,
        )

        assert cfg.trim_from == 3000
        assert cfg.trim_to == 15000
        assert cfg.savgol_window == 30
        assert cfg.savgol_poly == 3
        assert cfg.baseline_half_window == 50

    def test_as_dict(self):
        """Test as_dict method."""
        cfg = PreprocessingSettings(trim_from=3000)
        d = cfg.as_dict()

        assert isinstance(d, dict)
        assert d["trim_from"] == 3000


class TestPreprocessReproducibility:
    """Tests for reproducibility of preprocessing."""

    def test_same_input_same_output(self, synthetic_spectrum: pd.DataFrame):
        """Test that same input produces same output."""
        result1 = preprocess(synthetic_spectrum.copy())
        result2 = preprocess(synthetic_spectrum.copy())

        pd.testing.assert_frame_equal(result1, result2)

    def test_deterministic_with_cfg(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessing is deterministic with same config."""
        cfg = PreprocessingSettings(trim_from=3000, trim_to=15000)

        result1 = preprocess(synthetic_spectrum.copy(), cfg)
        result2 = preprocess(synthetic_spectrum.copy(), cfg)

        pd.testing.assert_frame_equal(result1, result2)
