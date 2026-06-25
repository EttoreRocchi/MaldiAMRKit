"""Tests for fit-free peak-set alignment (align_peaks)."""

import numpy as np
import pytest

from maldiamrkit.alignment import align_peaks
from maldiamrkit.detection import PeakSet


class TestAlignPeaks:
    def test_shift_recovers_offset(self):
        ref = np.array([3000.0, 5000.0, 7500.0, 10000.0])
        sample = PeakSet(ref - 5.0, np.ones(4))
        aligned = align_peaks(sample, ref, method="shift", max_shift_da=50.0)

        before = np.mean(np.abs(sample.mz - ref))
        after = np.mean(np.abs(aligned.mz - ref))
        assert after < before
        np.testing.assert_allclose(aligned.mz, ref, atol=1e-6)
        # intensities unchanged
        np.testing.assert_array_equal(aligned.intensity, sample.intensity)

    def test_linear_method(self):
        ref = np.array([3000.0, 5000.0, 7500.0, 10000.0])
        sample = PeakSet(ref - 4.0, np.arange(4.0))
        aligned = align_peaks(sample, ref, method="linear")
        assert aligned.n_peaks == 4
        assert np.mean(np.abs(aligned.mz - ref)) < np.mean(np.abs(sample.mz - ref))

    def test_dtw_rejected(self):
        sample = PeakSet([3000.0, 5000.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="dtw"):
            align_peaks(sample, [3000.0, 5000.0], method="dtw")

    def test_empty_peakset_returns_copy(self):
        empty = PeakSet([], [])
        out = align_peaks(empty, [3000.0, 5000.0], method="shift")
        assert out.n_peaks == 0

    def test_empty_reference_returns_copy(self):
        sample = PeakSet([3000.0, 5000.0], [1.0, 2.0])
        out = align_peaks(sample, [], method="shift")
        np.testing.assert_array_equal(out.mz, sample.mz)

    def test_intensities_preserved_after_warp(self):
        ref = np.array([3000.0, 5000.0, 7500.0])
        sample = PeakSet([2990.0, 4990.0, 7490.0], [0.5, 0.3, 0.2])
        aligned = align_peaks(sample, ref, method="shift")
        # m/z-sorted order preserved, intensities follow their peaks
        np.testing.assert_array_equal(aligned.intensity, [0.5, 0.3, 0.2])
