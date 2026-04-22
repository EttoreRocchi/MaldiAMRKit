"""Tests for spectral distance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.similarity.metrics import (
    METRIC_REGISTRY,
    _extract_mz_intensity,
    spectral_distance,
)


class TestExtractMzIntensity:
    """Input normalization helper."""

    def test_maldi_spectrum(self, spectrum_pair):
        spec_a, _ = spectrum_pair
        mz, intensity = _extract_mz_intensity(spec_a)
        assert mz is not None
        assert len(mz) == len(intensity)
        assert mz.dtype == np.float64 or np.issubdtype(mz.dtype, np.floating)

    def test_dataframe_mass_intensity(self):
        df = pd.DataFrame(
            {
                "mass": [1.0, 2.0, 3.0],
                "intensity": [10.0, 20.0, 30.0],
            }
        )
        mz, intensity = _extract_mz_intensity(df)
        assert mz is not None
        np.testing.assert_array_equal(mz, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(intensity, [10.0, 20.0, 30.0])

    def test_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        mz, intensity = _extract_mz_intensity(arr)
        assert mz is None
        np.testing.assert_array_equal(intensity, arr)


_RAW_METRICS = ["wasserstein", "dtw"]
_BINNED_METRICS = ["cosine", "spectral_contrast_angle", "pearson"]


class TestRawMetricProperties:
    """Identity, symmetry, non-negativity for raw-input metrics."""

    @pytest.mark.parametrize("metric", _RAW_METRICS)
    def test_identity(self, spectrum_pair, metric):
        spec_a, _ = spectrum_pair
        d = spectral_distance(spec_a, spec_a, metric=metric)
        assert d == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("metric", _RAW_METRICS)
    def test_symmetry(self, spectrum_pair, metric):
        spec_a, spec_b = spectrum_pair
        d_ab = spectral_distance(spec_a, spec_b, metric=metric)
        d_ba = spectral_distance(spec_b, spec_a, metric=metric)
        assert d_ab == pytest.approx(d_ba, rel=1e-6)

    @pytest.mark.parametrize("metric", _RAW_METRICS)
    def test_non_negative(self, spectrum_pair, metric):
        spec_a, spec_b = spectrum_pair
        assert spectral_distance(spec_a, spec_b, metric=metric) >= 0.0


class TestBinnedMetricProperties:
    """Identity, symmetry, non-negativity for binned-input metrics."""

    @pytest.mark.parametrize("metric", _BINNED_METRICS)
    def test_identity(self, binned_pair, metric):
        a, _ = binned_pair
        d = spectral_distance(a, a, metric=metric)
        assert d == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("metric", _BINNED_METRICS)
    def test_symmetry(self, binned_pair, metric):
        a, b = binned_pair
        d_ab = spectral_distance(a, b, metric=metric)
        d_ba = spectral_distance(b, a, metric=metric)
        assert d_ab == pytest.approx(d_ba, rel=1e-6)

    @pytest.mark.parametrize("metric", _BINNED_METRICS)
    def test_non_negative(self, binned_pair, metric):
        a, b = binned_pair
        assert spectral_distance(a, b, metric=metric) >= 0.0


class TestSpecificMetrics:
    """Deterministic checks on known inputs."""

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert spectral_distance(a, b, metric="cosine") == pytest.approx(1.0, abs=1e-6)

    def test_cosine_identical(self):
        a = np.array([3.0, 4.0])
        assert spectral_distance(a, a, metric="cosine") == pytest.approx(0.0, abs=1e-6)

    def test_cosine_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert spectral_distance(a, b, metric="cosine") == 1.0

    def test_pearson_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0])
        assert spectral_distance(a, a, metric="pearson") == pytest.approx(0.0, abs=1e-6)

    def test_spectral_contrast_angle_range(self, binned_pair):
        a, b = binned_pair
        d = spectral_distance(a, b, metric="spectral_contrast_angle")
        assert 0.0 <= d <= 1.0

    def test_wasserstein_requires_raw(self):
        a = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="requires raw spectra"):
            spectral_distance(a, a, metric="wasserstein")

    def test_dtw_requires_raw(self):
        a = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="requires raw spectra"):
            spectral_distance(a, a, metric="dtw")

    def test_wasserstein_handles_negative_intensities(self):
        """Negative intensities are clipped to zero so scipy's
        non-negative-weight precondition is respected."""
        import pandas as pd

        mz = np.linspace(2000.0, 3000.0, 50)
        int_a = np.abs(np.sin(mz / 200.0))
        int_b = int_a.copy()
        int_b[5] = -0.3  # inject a negative
        df_a = pd.DataFrame({"mass": mz, "intensity": int_a})
        df_b = pd.DataFrame({"mass": mz, "intensity": int_b})
        d = spectral_distance(df_a, df_b, metric="wasserstein")
        assert np.isfinite(d)
        assert d >= 0.0


class TestMetricRegistry:
    """METRIC_REGISTRY completeness and dispatcher validation."""

    def test_expected_keys(self):
        expected = {
            "wasserstein",
            "dtw",
            "cosine",
            "spectral_contrast_angle",
            "pearson",
        }
        assert set(METRIC_REGISTRY) == expected

    def test_invalid_metric_raises(self):
        a = np.array([1.0])
        with pytest.raises(ValueError, match="is not a valid"):
            spectral_distance(a, a, metric="nonexistent")
