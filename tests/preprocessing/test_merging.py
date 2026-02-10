"""Unit tests for spectral replicate merging."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit import MaldiSpectrum
from maldiamrkit.preprocessing import (
    detect_outlier_replicates,
    merge_replicates,
)


def _make_spectrum(mz, intensity):
    return MaldiSpectrum(pd.DataFrame({"mass": mz, "intensity": intensity}))


@pytest.fixture
def common_mz():
    return np.linspace(2000, 20000, 500)


@pytest.fixture
def identical_replicates(common_mz):
    """Three identical spectra."""
    intensity = np.sin(common_mz / 1000) ** 2 * 100
    return [_make_spectrum(common_mz, intensity) for _ in range(3)]


@pytest.fixture
def replicates_with_outlier(common_mz):
    """Three similar spectra plus one very different."""
    rng = np.random.default_rng(42)
    base = np.sin(common_mz / 1000) ** 2 * 100
    specs = [
        _make_spectrum(common_mz, base + rng.normal(0, 1, len(common_mz)))
        for _ in range(3)
    ]
    # Outlier: completely different shape
    outlier = _make_spectrum(common_mz, rng.uniform(0, 500, len(common_mz)))
    specs.append(outlier)
    return specs


class TestMergeReplicates:
    """Tests for merge_replicates function."""

    def test_merge_mean_identical(self, identical_replicates):
        """Mean of identical spectra should equal original."""
        merged = merge_replicates(identical_replicates, method="mean")

        assert "mass" in merged.columns
        assert "intensity" in merged.columns
        np.testing.assert_allclose(
            merged["intensity"].values,
            identical_replicates[0].raw["intensity"].values,
        )

    def test_merge_median_robust_to_outlier(self, replicates_with_outlier):
        """Median should be robust to a single outlier replicate."""
        merged_median = merge_replicates(replicates_with_outlier, method="median")
        merged_mean = merge_replicates(replicates_with_outlier, method="mean")

        # Median should be closer to the 3 good replicates than mean is
        good_avg = merge_replicates(replicates_with_outlier[:3], method="mean")
        diff_median = np.abs(
            merged_median["intensity"].values - good_avg["intensity"].values
        ).mean()
        diff_mean = np.abs(
            merged_mean["intensity"].values - good_avg["intensity"].values
        ).mean()
        assert diff_median < diff_mean

    def test_merge_weighted(self, common_mz):
        """Higher weight should pull result toward that replicate."""
        low = _make_spectrum(common_mz, np.zeros(len(common_mz)))
        high = _make_spectrum(common_mz, np.ones(len(common_mz)) * 100)

        merged = merge_replicates([low, high], method="mean", weights=[1.0, 9.0])
        # Should be closer to high (weight 9) than low (weight 1)
        assert merged["intensity"].mean() > 50

    def test_merge_different_grids(self):
        """Spectra with different m/z grids should be interpolated."""
        s1 = _make_spectrum(np.array([2000, 3000, 4000]), np.array([10, 20, 30]))
        s2 = _make_spectrum(np.array([2500, 3500, 4500]), np.array([15, 25, 35]))

        merged = merge_replicates([s1, s2], method="mean")
        # Should have the union of m/z values
        assert len(merged) == 6

    def test_merge_single_spectrum(self, common_mz):
        """Single spectrum should be returned as-is."""
        spec = _make_spectrum(common_mz, np.ones(len(common_mz)) * 42)
        merged = merge_replicates([spec])

        pd.testing.assert_frame_equal(merged, spec.raw)

    def test_merge_empty_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            merge_replicates([])

    def test_merge_invalid_method_raises(self, identical_replicates):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            merge_replicates(identical_replicates, method="invalid")

    def test_merge_weights_length_mismatch_raises(self, identical_replicates):
        """Weights length mismatch should raise ValueError."""
        with pytest.raises(ValueError, match="weights length"):
            merge_replicates(identical_replicates, method="mean", weights=[1.0, 2.0])


class TestDetectOutlierReplicates:
    """Tests for detect_outlier_replicates function."""

    def test_detect_outlier(self, replicates_with_outlier):
        """Should flag the outlier replicate."""
        keep = detect_outlier_replicates(replicates_with_outlier)

        assert isinstance(keep, np.ndarray)
        assert keep.dtype == bool
        assert len(keep) == 4
        # First 3 should be kept, last should be flagged
        assert all(keep[:3])
        assert not keep[3]

    def test_detect_no_outlier(self, identical_replicates):
        """All identical spectra should be kept."""
        keep = detect_outlier_replicates(identical_replicates)
        assert all(keep)

    def test_too_few_replicates_raises(self, common_mz):
        """Fewer than 3 replicates should raise ValueError."""
        specs = [_make_spectrum(common_mz, np.ones(len(common_mz)))] * 2
        with pytest.raises(ValueError, match="at least 3"):
            detect_outlier_replicates(specs)
