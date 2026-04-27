"""Tests for alignment strategy classes and helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from maldiamrkit.alignment.strategies import (
    ALIGNMENT_REGISTRY,
    DTWStrategy,
    LinearStrategy,
    LOWESSStrategy,
    PiecewiseStrategy,
    PolynomialStrategy,
    ShiftStrategy,
    _lowess_fit,
    _match_peak_pairs,
    _match_peaks_to_ref,
    _nearest_ref_indices,
    monotonic_interp,
)


class TestMonotonicInterp:
    """Tests for the monotonic_interp helper."""

    def test_monotonic_input_no_warning(self):
        mz = np.arange(10, dtype=float)
        positions = mz + 0.5
        row = np.ones(10)
        result = monotonic_interp(mz, positions, row)
        assert result.shape == mz.shape

    def test_non_monotonic_input_warns_and_recovers(self):
        mz = np.arange(10, dtype=float)
        positions = mz.copy()
        positions[3] = positions[5]
        row = np.arange(10, dtype=float)
        with pytest.warns(UserWarning, match="non-monotonic"):
            result = monotonic_interp(mz, positions, row)
        assert result.shape == mz.shape
        assert np.all(np.isfinite(result))

    def test_monotonic_preserves_values(self):
        mz = np.array([0.0, 1.0, 2.0, 3.0])
        positions = mz.copy()
        row = np.array([1.0, 2.0, 3.0, 4.0])
        result = monotonic_interp(mz, positions, row)
        np.testing.assert_allclose(result, row)


class TestHelperFunctions:
    """Tests for peak matching helpers."""

    def test_nearest_ref_indices(self):
        peaks = np.array([100.0, 200.0, 300.0])
        ref_peaks = np.array([95.0, 205.0, 310.0])
        idx = _nearest_ref_indices(peaks, ref_peaks)
        assert idx[0] == 0
        assert idx[1] == 1
        assert idx[2] == 2

    def test_match_peaks_to_ref(self):
        peaks = np.array([100.0, 200.0])
        ref_peaks = np.array([102.0, 198.0])
        shifts = _match_peaks_to_ref(peaks, ref_peaks)
        np.testing.assert_allclose(shifts, [2.0, -2.0])

    def test_match_peak_pairs(self):
        peaks = np.array([100.0, 200.0])
        ref_peaks = np.array([105.0, 195.0])
        sample, ref = _match_peak_pairs(peaks, ref_peaks)
        np.testing.assert_array_equal(sample, peaks)
        np.testing.assert_array_equal(ref, [105.0, 195.0])


class TestShiftStrategy:
    """Tests for ShiftStrategy."""

    def setup_method(self):
        self.strategy = ShiftStrategy(max_shift=50)

    def test_align_binned_empty_peaks_returns_original(self):
        row = np.array([1.0, 2.0, 3.0])
        result = self.strategy.align_binned(
            row, np.array([]), np.array([10]), np.arange(3, dtype=float)
        )
        np.testing.assert_array_equal(result, row)

    def test_align_binned_empty_ref_peaks_returns_original(self):
        row = np.array([1.0, 2.0, 3.0])
        result = self.strategy.align_binned(
            row, np.array([1]), np.array([]), np.arange(3, dtype=float)
        )
        np.testing.assert_array_equal(result, row)

    def test_align_binned_positive_shift(self):
        row = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
        peaks = np.array([1])
        ref_peaks = np.array([3])
        result = self.strategy.align_binned(
            row, peaks, ref_peaks, np.arange(5, dtype=float)
        )
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_align_binned_negative_shift(self):
        row = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
        peaks = np.array([3])
        ref_peaks = np.array([1])
        result = self.strategy.align_binned(
            row, peaks, ref_peaks, np.arange(5, dtype=float)
        )
        assert result.shape == row.shape

    def test_align_binned_zero_shift(self):
        row = np.array([0.0, 5.0, 0.0])
        peaks = np.array([1])
        ref_peaks = np.array([1])
        result = self.strategy.align_binned(
            row, peaks, ref_peaks, np.arange(3, dtype=float)
        )
        np.testing.assert_array_equal(result, row)

    def test_align_raw_empty_peaks_returns_original(self):
        mz = np.array([100.0, 200.0])
        intensity = np.array([1.0, 2.0])
        result_mz, result_int = self.strategy.align_raw(
            mz, intensity, np.array([]), np.array([100.0]), mz, intensity
        )
        np.testing.assert_array_equal(result_mz, mz)

    def test_align_raw_applies_shift(self):
        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([1.0, 2.0, 3.0])
        peaks_mz = np.array([100.0])
        ref_peaks_mz = np.array([105.0])
        result_mz, result_int = self.strategy.align_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, mz, intensity
        )
        np.testing.assert_allclose(result_mz, mz + 5.0)


class TestLinearStrategy:
    """Tests for LinearStrategy."""

    def setup_method(self):
        self.strategy = LinearStrategy(max_shift=50)

    def test_align_binned_few_peaks_falls_back_to_shift(self):
        row = np.array([0.0, 5.0, 0.0])
        peaks = np.array([1])
        ref_peaks = np.array([1])
        result = self.strategy.align_binned(
            row, peaks, ref_peaks, np.arange(3, dtype=float)
        )
        assert result.shape == row.shape

    def test_align_binned_enough_peaks(self):
        mz_axis = np.arange(100, dtype=float)
        row = np.zeros(100)
        row[20] = 1.0
        row[80] = 1.0
        peaks = np.array([20, 80])
        ref_peaks = np.array([22, 82])
        result = self.strategy.align_binned(row, peaks, ref_peaks, mz_axis)
        assert result.shape == row.shape

    def test_align_raw_few_peaks_falls_back(self):
        mz = np.array([100.0, 200.0])
        intensity = np.array([1.0, 2.0])
        result_mz, result_int = self.strategy.align_raw(
            mz, intensity, np.array([100.0]), np.array([105.0]), mz, intensity
        )
        assert len(result_mz) == len(mz)

    def test_align_raw_enough_peaks(self):
        mz = np.linspace(100, 300, 200)
        intensity = np.random.default_rng(42).random(200)
        peaks_mz = np.array([120.0, 250.0])
        ref_peaks_mz = np.array([122.0, 252.0])
        result_mz, result_int = self.strategy.align_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, mz, intensity
        )
        assert len(result_mz) == len(mz)


class TestPiecewiseStrategy:
    """Tests for PiecewiseStrategy."""

    def setup_method(self):
        self.strategy = PiecewiseStrategy(n_segments=3, smooth_sigma=2.0, max_shift=50)

    def test_align_binned_empty_peaks_returns_original(self):
        row = np.ones(50)
        result = self.strategy.align_binned(
            row, np.array([]), np.array([10]), np.arange(50, dtype=float)
        )
        np.testing.assert_array_equal(result, row)

    def test_align_binned_with_peaks(self):
        mz_axis = np.arange(100, dtype=float)
        row = np.zeros(100)
        row[[10, 30, 50, 70, 90]] = 1.0
        peaks = np.array([10, 30, 50, 70, 90])
        ref_peaks = np.array([12, 32, 52, 72, 92])
        result = self.strategy.align_binned(row, peaks, ref_peaks, mz_axis)
        assert result.shape == row.shape

    def test_align_raw_empty_peaks_returns_original(self):
        mz = np.linspace(100, 300, 100)
        intensity = np.ones(100)
        result_mz, result_int = self.strategy.align_raw(
            mz, intensity, np.array([]), np.array([150.0]), mz, intensity
        )
        np.testing.assert_array_equal(result_mz, mz)

    def test_align_raw_with_smoothing(self):
        mz = np.linspace(100, 300, 200)
        intensity = np.random.default_rng(42).random(200)
        peaks_mz = np.array([120.0, 160.0, 200.0, 240.0, 280.0])
        ref_peaks_mz = np.array([122.0, 162.0, 202.0, 242.0, 282.0])
        result_mz, result_int = self.strategy.align_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, mz, intensity
        )
        assert len(result_mz) == len(mz)

    def test_align_raw_no_smoothing(self):
        strategy = PiecewiseStrategy(n_segments=2, smooth_sigma=0.0, max_shift=50)
        mz = np.linspace(100, 300, 200)
        intensity = np.ones(200)
        peaks_mz = np.array([150.0, 250.0])
        ref_peaks_mz = np.array([152.0, 252.0])
        result_mz, _ = strategy.align_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, mz, intensity
        )
        assert len(result_mz) == len(mz)


class TestDTWStrategy:
    """Tests for DTWStrategy."""

    def test_align_binned_raises(self):
        strategy = DTWStrategy(dtw_radius=5)
        with pytest.raises(NotImplementedError):
            strategy.align_binned(
                np.ones(10), np.array([1]), np.array([2]), np.arange(10, dtype=float)
            )

    @pytest.mark.slow
    def test_align_raw(self):
        strategy = DTWStrategy(dtw_radius=5)
        mz = np.linspace(0, 100, 50)
        intensity = np.sin(mz / 10)
        ref_intensity = np.sin((mz - 2) / 10)
        result_mz, result_int = strategy.align_raw(
            mz, intensity, np.array([]), np.array([]), mz, ref_intensity
        )
        np.testing.assert_array_equal(result_mz, mz)
        assert result_int.shape == intensity.shape


class TestAlignmentRegistry:
    """Tests for ALIGNMENT_REGISTRY."""

    def test_all_methods_registered(self):
        assert set(ALIGNMENT_REGISTRY.keys()) == {
            "shift",
            "linear",
            "piecewise",
            "dtw",
            "quadratic",
            "cubic",
            "lowess",
        }

    def test_registry_values_are_strategy_classes(self):
        for cls in ALIGNMENT_REGISTRY.values():
            assert issubclass(cls, ShiftStrategy.__mro__[1])

    def test_polynomial_registered_for_quadratic_and_cubic(self):
        assert ALIGNMENT_REGISTRY["quadratic"] is PolynomialStrategy
        assert ALIGNMENT_REGISTRY["cubic"] is PolynomialStrategy


class TestPolynomialStrategy:
    """Tests for PolynomialStrategy."""

    @pytest.mark.parametrize("degree", [2, 3])
    def test_recovers_polynomial_miscalibration_binned(self, degree: int):
        mz_axis = np.arange(500, dtype=float)
        ref_peaks = np.array([50, 150, 300, 450], dtype=float)
        # Known polynomial miscalibration: sample_mz = ref_mz - small correction
        true_coeffs = np.array([1e-5, 0.0, 1.0, 0.0])[-(degree + 1) :]
        peaks = np.polyval(true_coeffs, ref_peaks)
        row = np.zeros(500)
        for p in peaks.astype(int):
            if 0 <= p < 500:
                row[p] = 1.0

        strategy = PolynomialStrategy(max_shift=50, degree=degree)
        out = strategy.align_binned(row, peaks, ref_peaks, mz_axis)
        assert out.shape == row.shape
        # Most mass should now sit near the reference peaks
        peak_mass = sum(out[int(p) - 2 : int(p) + 3].sum() for p in ref_peaks)
        assert peak_mass > 0.5 * row.sum()

    def test_degree_zero_raises(self):
        with pytest.raises(ValueError, match="degree"):
            PolynomialStrategy(max_shift=50, degree=0)

    @pytest.mark.parametrize("degree", [2, 3])
    def test_falls_back_when_too_few_peaks(self, degree: int):
        mz_axis = np.arange(100, dtype=float)
        row = np.zeros(100)
        row[10] = 1.0
        peaks = np.array([10])
        ref_peaks = np.array([12])
        strategy = PolynomialStrategy(max_shift=50, degree=degree)
        out = strategy.align_binned(row, peaks, ref_peaks, mz_axis)
        # Behaves like ShiftStrategy: peak moves from index 10 to index 12
        assert np.argmax(out) == 12

    def test_align_raw_quadratic(self):
        mz = np.linspace(2000.0, 20000.0, 500)
        intensity = np.zeros(500)
        peaks_mz = np.array([2500.0, 7500.0, 12500.0, 17500.0])
        ref_peaks_mz = peaks_mz + 1e-4 * peaks_mz**2 - 0.5 * peaks_mz
        strategy = PolynomialStrategy(max_shift=50.0, degree=2)
        new_mz, new_int = strategy.align_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, mz, intensity
        )
        assert new_mz.shape == mz.shape
        np.testing.assert_array_equal(new_int, intensity)


class TestLowessFit:
    """Tests for the in-repo LOWESS helper."""

    def test_fits_linear_signal_exactly(self):
        x = np.linspace(0.0, 10.0, 50)
        y = 2.0 * x + 1.0
        fitted = _lowess_fit(x, y, frac=0.5, it=0)
        np.testing.assert_allclose(fitted, y, atol=1e-6)

    def test_smooths_noise(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 10.0, 200)
        y = np.sin(x) + rng.normal(0, 0.2, 200)
        fitted = _lowess_fit(x, y, frac=0.3, it=2)
        assert np.std(fitted - np.sin(x)) < np.std(y - np.sin(x))

    def test_empty_input(self):
        fitted = _lowess_fit(np.empty(0), np.empty(0), frac=0.3, it=1)
        assert fitted.size == 0


class TestLOWESSStrategy:
    """Tests for LOWESSStrategy."""

    @pytest.mark.filterwarnings("ignore:Warping produced non-monotonic")
    def test_recovers_nonlinear_drift_binned(self):
        mz_axis = np.arange(500, dtype=float)
        ref_peaks = np.array([50, 150, 250, 350, 450], dtype=float)
        # Non-linear drift: sinusoidal offset
        drift = 3 * np.sin(ref_peaks / 60.0)
        peaks = ref_peaks + drift
        row = np.zeros(500)
        for p in peaks.astype(int):
            if 0 <= p < 500:
                row[p] = 1.0

        strategy = LOWESSStrategy(max_shift=50, frac=0.6, it=1)
        out = strategy.align_binned(row, peaks, ref_peaks, mz_axis)
        assert out.shape == row.shape
        peak_mass = sum(out[int(p) - 2 : int(p) + 3].sum() for p in ref_peaks)
        assert peak_mass > 0.5 * row.sum()

    def test_falls_back_when_too_few_peaks(self):
        mz_axis = np.arange(100, dtype=float)
        row = np.zeros(100)
        row[10] = 1.0
        strategy = LOWESSStrategy(max_shift=50, frac=0.5, it=0)
        out = strategy.align_binned(row, np.array([10]), np.array([12]), mz_axis)
        assert np.argmax(out) == 12

    def test_align_raw_passes_through_intensity(self):
        mz = np.linspace(2000.0, 20000.0, 500)
        intensity = np.arange(500.0)
        peaks_mz = np.array([3000.0, 5000.0, 10000.0, 15000.0, 18000.0])
        ref_peaks_mz = peaks_mz + 2.0 * np.sin(peaks_mz / 5000.0)
        strategy = LOWESSStrategy(max_shift=50.0, frac=0.5, it=1)
        new_mz, new_int = strategy.align_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, mz, intensity
        )
        assert new_mz.shape == mz.shape
        np.testing.assert_array_equal(new_int, intensity)

    def test_invalid_frac_raises(self):
        with pytest.raises(ValueError, match="frac"):
            LOWESSStrategy(max_shift=50, frac=0.0, it=1)
        with pytest.raises(ValueError, match="frac"):
            LOWESSStrategy(max_shift=50, frac=1.5, it=1)

    def test_invalid_it_raises(self):
        with pytest.raises(ValueError, match="it"):
            LOWESSStrategy(max_shift=50, frac=0.3, it=-1)
