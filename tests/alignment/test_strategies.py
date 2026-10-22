"""Tests for alignment strategy classes and helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from maldiamrkit.alignment.strategies import (
    _ALIGNMENT_REGISTRY,
    DTWStrategy,
    LinearStrategy,
    LOWESSStrategy,
    PiecewiseStrategy,
    PolynomialStrategy,
    ShiftStrategy,
    _conservative_resample,
    _gated_lag,
    _lowess_fit,
    _most_prominent,
    _mutual_pairs,
    _nearest_ref_indices,
    _pearson_by_lag,
    _shift_row,
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


class TestShiftHelpers:
    """Integer shift, correlation by lag and the gated lag."""

    def test_shift_row_zero_fills(self):
        row = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(_shift_row(row, 1), [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_shift_row(row, -2), [3.0, 4.0, 0.0, 0.0])
        np.testing.assert_array_equal(_shift_row(row, 0), row)

    def test_pearson_by_lag_matches_explicit_shifts(self):
        rng = np.random.default_rng(0)
        row, ref = rng.random(200), rng.random(200)
        lags, corr = _pearson_by_lag(row, ref, 7)
        expected = [_pearson(_shift_row(row, int(k)), ref) for k in lags]
        np.testing.assert_allclose(corr, expected, atol=1e-12)

    def test_gated_lag_recovers_a_clear_shift(self):
        x = np.linspace(0, 1, 300)
        ref = np.exp(-(((x - 0.5) / 0.02) ** 2))
        assert _gated_lag(np.roll(ref, -4), ref, 10, 0.05) == 4

    def test_gated_lag_ignores_small_gains(self):
        rng = np.random.default_rng(1)
        ref = rng.random(500)
        row = ref + 0.5 * rng.random(500)
        assert _gated_lag(row, ref, 5, 0.05) == 0

    def test_gated_lag_respects_the_bound(self):
        x = np.linspace(0, 1, 300)
        ref = np.exp(-(((x - 0.5) / 0.02) ** 2))
        assert abs(_gated_lag(np.roll(ref, -20), ref, 3, 0.0)) <= 3

    def test_gated_lag_flat_row_is_not_shifted(self):
        assert _gated_lag(np.ones(50), np.arange(50.0), 5, 0.0) == 0


class TestPeakMatching:
    """Nearest and mutually nearest peak matching."""

    def test_nearest_ref_indices(self):
        peaks = np.array([100.0, 200.0, 300.0])
        ref_peaks = np.array([95.0, 205.0, 310.0])
        np.testing.assert_array_equal(_nearest_ref_indices(peaks, ref_peaks), [0, 1, 2])

    def test_mutual_pairs_drop_far_and_shared_matches(self):
        peaks = np.array([100.0, 300.0, 330.0, 900.0])
        ref_peaks = np.array([102.0, 310.0])
        sample, ref = _mutual_pairs(peaks, ref_peaks, 0.0, 15.0)
        np.testing.assert_array_equal(sample, [100.0, 300.0])
        np.testing.assert_array_equal(ref, [102.0, 310.0])

    def test_mutual_pairs_use_the_offset(self):
        peaks = np.array([100.0, 200.0])
        ref_peaks = np.array([120.0, 220.0])
        assert len(_mutual_pairs(peaks, ref_peaks, 0.0, 5.0)[0]) == 0
        sample, ref = _mutual_pairs(peaks, ref_peaks, 20.0, 5.0)
        np.testing.assert_array_equal(ref - sample, [20.0, 20.0])

    def test_peaks_beyond_the_last_reference_peak_are_not_matched(self):
        peaks = np.array([100.0, 4600.0, 5200.0, 5900.0])
        ref_peaks = np.array([101.0, 4500.0])
        sample, _ = _mutual_pairs(peaks, ref_peaks, 0.0, 3.0)
        np.testing.assert_array_equal(sample, [100.0])

    def test_most_prominent_keeps_the_tallest_peaks_sorted(self):
        x = np.zeros(100)
        x[[10, 30, 50, 70, 90]] = [1.0, 5.0, 2.0, 4.0, 3.0]
        peaks = np.array([10, 30, 50, 70, 90])
        np.testing.assert_array_equal(_most_prominent(x, peaks, 3), [30, 70, 90])

    def test_most_prominent_keeps_everything_when_n_is_large_or_none(self):
        x = np.zeros(20)
        x[[5, 15]] = 1.0
        peaks = np.array([5, 15])
        np.testing.assert_array_equal(_most_prominent(x, peaks, None), peaks)
        np.testing.assert_array_equal(_most_prominent(x, peaks, 2), peaks)


class TestConservativeResample:
    """Intensity-preserving resampling of a warped binned row."""

    def test_identity(self):
        row = np.random.default_rng(0).random(50)
        axis = np.arange(50.0)
        np.testing.assert_allclose(_conservative_resample(axis, axis, row), row)

    def test_integer_shift_equals_shift_row(self):
        row = np.random.default_rng(1).random(50)
        axis = np.arange(50.0)
        np.testing.assert_allclose(
            _conservative_resample(axis, axis + 3, row), _shift_row(row, 3), atol=1e-12
        )

    def test_preserves_total_intensity_inside_the_axis(self):
        row = np.zeros(100)
        row[20:80] = np.random.default_rng(2).random(60)
        axis = np.arange(100.0)
        out = _conservative_resample(axis, 1.03 * axis - 1.0, row)
        assert out.sum() == pytest.approx(row.sum())
        assert np.all(out >= 0)


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

    def test_segment_shift_is_a_trimmed_mean_with_sub_bin_precision(self):
        strategy = PiecewiseStrategy(n_segments=1, smooth_sigma=0.0, max_shift=3)
        sample = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        offsets = np.array([1.0, 1.0, 2.0, 2.0, 2.0, -2.0])
        _, shifts = strategy._compute_segment_shifts(sample, sample + offsets)
        assert shifts[0] == pytest.approx(1.6)

    def test_segment_shift_without_offsets_near_the_median(self):
        # Two pairs pulling apart: none lies near their median of 0.
        strategy = PiecewiseStrategy(n_segments=1, smooth_sigma=0.0, max_shift=3)
        sample = np.array([10.0, 20.0])
        _, shifts = strategy._compute_segment_shifts(sample, sample + [-3.0, 3.0])
        assert shifts == [0.0]

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
    """Tests for _ALIGNMENT_REGISTRY."""

    def test_all_methods_registered(self):
        assert set(_ALIGNMENT_REGISTRY.keys()) == {
            "shift",
            "linear",
            "piecewise",
            "dtw",
            "quadratic",
            "cubic",
            "lowess",
        }

    def test_registry_values_are_strategy_classes(self):
        for cls in _ALIGNMENT_REGISTRY.values():
            assert issubclass(cls, ShiftStrategy.__mro__[1])

    def test_polynomial_registered_for_quadratic_and_cubic(self):
        assert _ALIGNMENT_REGISTRY["quadratic"] is PolynomialStrategy
        assert _ALIGNMENT_REGISTRY["cubic"] is PolynomialStrategy


class TestPolynomialStrategy:
    """Tests for PolynomialStrategy."""

    @pytest.mark.parametrize("degree", [2, 3])
    def test_recovers_polynomial_miscalibration_binned(self, degree: int):
        mz_axis = np.arange(500, dtype=float)
        ref_peaks = np.array([50, 150, 300, 450], dtype=float)
        leading = {2: 1e-5, 3: 2e-7}[degree]
        true_coeffs = np.array([leading, 0.0, 1.0, 0.0])[-(degree + 1) :]
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

    def test_robustness_does_not_empty_a_window(self):
        x = np.arange(0.0, 2000.0, 10.0)
        y = x + (np.arange(len(x)) < 110)
        fitted = _lowess_fit(x, y, frac=0.3, it=3)
        assert np.max(np.abs(fitted - x)) <= 1.0 + 1e-9


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


class TestGates:
    """The min_gain gate on the shift and DTW, the support rule on the warps."""

    @staticmethod
    def _spectrum(centres, n=300):
        x = np.arange(n, dtype=float)
        return sum(np.exp(-0.5 * ((x - c) / 1.5) ** 2) for c in centres)

    def test_peak_shift_is_rejected_when_it_does_not_help(self):
        reference = self._spectrum([100, 200])
        row = reference.copy()
        # A single spurious pair suggests a shift that only lowers correlation.
        strategy = ShiftStrategy(max_shift=5, estimator="peaks", min_gain=0.05)
        shift = strategy.estimate_binned(
            row, np.array([103]), np.array([100]), reference
        )
        assert shift == 0
        assert strategy.estimate_binned(row, np.array([103]), np.array([100])) == -3

    def test_unsupported_warp_falls_back_to_the_shift(self, monkeypatch):
        reference = self._spectrum([60, 150, 240])
        row = reference.copy()
        peaks = np.array([60, 150, 240])
        monkeypatch.setattr(
            LinearStrategy, "_positions", lambda self, s, r, axis, raw: 1.05 * axis
        )
        strategy = LinearStrategy(max_shift=3, estimator="xcorr", min_support=0.6)
        out = strategy.align_binned(
            row, peaks, peaks, np.arange(300.0), reference=reference
        )
        np.testing.assert_array_equal(out, row)
        unchecked = LinearStrategy(max_shift=3, estimator="xcorr")
        out = unchecked.align_binned(
            row, peaks, peaks, np.arange(300.0), reference=reference
        )
        assert not np.array_equal(out, row)

    def test_supported_warp_is_used(self):
        centres = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
        sample_centres = (centres - 1.0) / 1.01
        reference = self._spectrum(centres)
        row = self._spectrum(sample_centres)
        strategy = LinearStrategy(max_shift=3, estimator="xcorr", min_support=0.6)
        out = strategy.align_binned(
            row,
            np.round(sample_centres),
            centres,
            np.arange(300.0),
            reference=reference,
        )
        np.testing.assert_allclose(find_peaks(out)[0], centres, atol=1)

    def test_raw_warp_needs_support(self):
        mz = np.arange(2000.0, 3000.0, 0.5)
        intensity = np.ones_like(mz)
        peaks_mz = np.array([2100.0, 2400.0, 2700.0, 2900.0])
        ref_peaks_mz = peaks_mz + np.array([0.0, 6.0, -6.0, 0.0])
        strategy = LinearStrategy(max_shift=9.0, min_support=0.9)
        new_mz, _ = strategy.align_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, mz, intensity
        )
        np.testing.assert_array_equal(new_mz, mz)

    def test_dtw_result_rejected_when_it_does_not_help(self, monkeypatch):
        reference = self._spectrum([100, 200])
        row = self._spectrum([101, 199])
        strategy = DTWStrategy(dtw_radius=5, min_gain=0.05)
        monkeypatch.setattr(
            DTWStrategy, "_dtw_align", lambda self, q, r: np.zeros_like(r)
        )
        np.testing.assert_array_equal(strategy.align_to_reference(row, reference), row)

    def test_xcorr_raw_shift_recovers_daltons(self):
        mz = np.arange(2000.0, 3000.0, 0.5)
        intensity = sum(
            np.exp(-0.5 * ((mz - c) / 2.0) ** 2) for c in (2200, 2500, 2800)
        )
        strategy = ShiftStrategy(max_shift=10.0, estimator="xcorr", min_gain=0.05)
        shift = strategy.estimate_raw(mz + 6.0, intensity, None, None, mz, intensity)
        assert shift == pytest.approx(-6.0)

    def test_xcorr_needs_a_reference(self):
        strategy = ShiftStrategy(max_shift=3, estimator="xcorr")
        with pytest.raises(ValueError, match="reference"):
            strategy.estimate_binned(np.ones(10), None, None)

    def test_invalid_estimator(self):
        with pytest.raises(ValueError, match="estimator"):
            ShiftStrategy(max_shift=3, estimator="nope")
