"""Unit tests for preprocessing pipeline."""

import numpy as np
import pandas as pd

from maldiamrkit.preprocessing import (
    MzTrimmer,
    PreprocessingPipeline,
    SavitzkyGolaySmooth,
    preprocess,
)


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
        pipe = PreprocessingPipeline.default()
        # Replace the MzTrimmer step
        pipe.steps = [
            (n, s) if not isinstance(s, MzTrimmer) else (n, MzTrimmer(3000, 15000))
            for n, s in pipe.steps
        ]
        result = preprocess(synthetic_spectrum, pipe)

        assert result["mass"].min() >= 3000
        assert result["mass"].max() <= 15000

    def test_preprocess_default_trim_range(self, synthetic_spectrum: pd.DataFrame):
        """Test default trim range (2000-20000)."""
        result = preprocess(synthetic_spectrum)

        assert result["mass"].min() >= 2000
        assert result["mass"].max() <= 20000

    def test_preprocess_preserves_peak_positions(
        self, synthetic_spectrum: pd.DataFrame
    ):
        """Test that peak positions are preserved after preprocessing."""
        # The synthetic spectrum has peaks at 3000, 5000, 7500, 10000, 12500, 15000
        result = preprocess(synthetic_spectrum)

        # Find peak positions (local maxima)
        intensity = result["intensity"].values
        peak_mask = (intensity[1:-1] > intensity[:-2]) & (
            intensity[1:-1] > intensity[2:]
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
        pipe = PreprocessingPipeline.default()
        pipe.steps = [
            (n, s)
            if not isinstance(s, SavitzkyGolaySmooth)
            else (n, SavitzkyGolaySmooth(window_length=30, polyorder=3))
            for n, s in pipe.steps
        ]
        result = preprocess(synthetic_spectrum, pipe)

        assert (result["intensity"] >= 0).all()
        assert np.isclose(result["intensity"].sum(), 1.0, atol=1e-6)

    def test_preprocess_handles_empty_after_trim(self):
        """Test that preprocessing handles edge case of empty result."""
        # Create spectrum outside the default trim range
        df = pd.DataFrame(
            {
                "mass": np.linspace(500, 1000, 1000),
                "intensity": np.random.uniform(0, 100, 1000),
            }
        )

        result = preprocess(df)

        # Result should be empty or handle gracefully
        assert len(result) == 0 or result["intensity"].sum() == 0

    def test_preprocess_does_not_modify_input(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessing does not modify the input DataFrame."""
        original = synthetic_spectrum.copy()
        _ = preprocess(synthetic_spectrum)

        pd.testing.assert_frame_equal(synthetic_spectrum, original)


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline."""

    def test_default_pipeline_has_6_steps(self):
        """Test that default pipeline has 6 steps."""
        pipe = PreprocessingPipeline.default()
        assert len(pipe) == 6

    def test_default_step_names(self):
        """Test default pipeline step names."""
        pipe = PreprocessingPipeline.default()
        assert pipe.step_names == [
            "clip",
            "sqrt",
            "smooth",
            "baseline",
            "trim",
            "normalize",
        ]

    def test_mz_range(self):
        """Test mz_range property."""
        pipe = PreprocessingPipeline.default()
        assert pipe.mz_range == (2000, 20000)

    def test_custom_mz_range(self):
        """Test custom mz_range."""
        pipe = PreprocessingPipeline(
            [
                ("trim", MzTrimmer(mz_min=3000, mz_max=15000)),
            ]
        )
        assert pipe.mz_range == (3000, 15000)

    def test_get_step(self):
        """Test get_step method."""
        pipe = PreprocessingPipeline.default()
        trim = pipe.get_step("trim")
        assert isinstance(trim, MzTrimmer)

    def test_get_step_missing_raises(self):
        """Test that get_step raises for missing step."""
        pipe = PreprocessingPipeline.default()
        with pytest.raises(KeyError, match="nonexistent"):
            pipe.get_step("nonexistent")

    def test_repr(self):
        """Test repr is informative."""
        pipe = PreprocessingPipeline.default()
        r = repr(pipe)
        assert "PreprocessingPipeline" in r
        assert "clip" in r

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization round-trip."""
        pipe = PreprocessingPipeline.default()
        d = pipe.to_dict()
        pipe2 = PreprocessingPipeline.from_dict(d)

        assert pipe.step_names == pipe2.step_names
        assert pipe.mz_range == pipe2.mz_range


class TestPreprocessReproducibility:
    """Tests for reproducibility of preprocessing."""

    def test_same_input_same_output(self, synthetic_spectrum: pd.DataFrame):
        """Test that same input produces same output."""
        result1 = preprocess(synthetic_spectrum.copy())
        result2 = preprocess(synthetic_spectrum.copy())

        pd.testing.assert_frame_equal(result1, result2)

    def test_deterministic_with_pipeline(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessing is deterministic with same pipeline."""
        pipe = PreprocessingPipeline.default()
        # Custom MzTrimmer
        pipe.steps = [
            (n, s) if not isinstance(s, MzTrimmer) else (n, MzTrimmer(3000, 15000))
            for n, s in pipe.steps
        ]

        result1 = preprocess(synthetic_spectrum.copy(), pipe)
        result2 = preprocess(synthetic_spectrum.copy(), pipe)

        pd.testing.assert_frame_equal(result1, result2)


# Need pytest for raises
import pytest  # noqa: E402
