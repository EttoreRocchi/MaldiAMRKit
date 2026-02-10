"""Numerical stability tests for preprocessing transformers.

Tests all transformers with problematic inputs: inf, -inf, NaN, all-zero,
very large values, and extreme conditions.
"""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.preprocessing import (
    ClipNegatives,
    LogTransform,
    MedianNormalizer,
    MzMultiTrimmer,
    MzTrimmer,
    PQNNormalizer,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    SqrtTransform,
    TICNormalizer,
)


def _make_df(intensity_values):
    """Create a spectrum DataFrame with given intensity values."""
    n = len(intensity_values)
    mz = np.linspace(2000, 20000, n)
    return pd.DataFrame({"mass": mz, "intensity": intensity_values})


class TestClipNegatives:
    """Numerical stability tests for ClipNegatives."""

    def test_clips_negative_to_zero(self):
        df = _make_df([100, -50, 200, -10, 0])
        result = ClipNegatives()(df)
        assert (result["intensity"] >= 0).all()
        assert result["intensity"].iloc[1] == 0
        assert result["intensity"].iloc[0] == 100

    def test_nan_passthrough(self):
        df = _make_df([100, np.nan, 200])
        result = ClipNegatives()(df)
        assert np.isnan(result["intensity"].iloc[1])


class TestSqrtTransform:
    """Numerical stability tests for SqrtTransform."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_negative_produces_nan(self):
        df = _make_df([100, -50, 200])
        result = SqrtTransform()(df)
        assert np.isnan(result["intensity"].iloc[1])

    def test_inf_produces_inf(self):
        df = _make_df([100, np.inf, 200])
        result = SqrtTransform()(df)
        assert np.isinf(result["intensity"].iloc[1])

    def test_nan_produces_nan(self):
        df = _make_df([100, np.nan, 200])
        result = SqrtTransform()(df)
        assert np.isnan(result["intensity"].iloc[1])


class TestLogTransform:
    """Numerical stability tests for LogTransform."""

    def test_negative_finite(self):
        # log1p(-0.5) = log(0.5) which is finite
        df = _make_df([100, -0.5, 200])
        result = LogTransform()(df)
        assert np.isfinite(result["intensity"].iloc[1])

    def test_large_values(self):
        df = _make_df([1e15, 1e12, 1e8])
        result = LogTransform()(df)
        assert result["intensity"].isna().sum() == 0
        assert np.all(np.isfinite(result["intensity"].values))

    def test_nan_passthrough(self):
        df = _make_df([100, np.nan, 200])
        result = LogTransform()(df)
        assert np.isnan(result["intensity"].iloc[1])


class TestSavitzkyGolaySmooth:
    """Numerical stability tests for SavitzkyGolaySmooth."""

    def test_nan_propagation(self):
        values = np.ones(100)
        values[50] = np.nan
        df = _make_df(values)
        result = SavitzkyGolaySmooth(window_length=21, polyorder=2)(df)
        # NaN should propagate to neighboring points within the window
        assert np.isnan(result["intensity"].iloc[50])

    def test_inf_propagation(self):
        values = np.ones(100)
        values[50] = np.inf
        df = _make_df(values)
        result = SavitzkyGolaySmooth(window_length=21, polyorder=2)(df)
        # inf should propagate to neighboring points
        assert not np.isfinite(result["intensity"].iloc[50])

    def test_all_zero(self):
        df = _make_df(np.zeros(100))
        result = SavitzkyGolaySmooth(window_length=21, polyorder=2)(df)
        assert np.allclose(result["intensity"].values, 0, atol=1e-15)


class TestSNIPBaseline:
    """Numerical stability tests for SNIPBaseline."""

    def test_all_zero(self):
        df = _make_df(np.zeros(500))
        result = SNIPBaseline(half_window=40)(df)
        assert (result["intensity"] >= 0).all()
        assert np.allclose(result["intensity"].values, 0, atol=1e-10)

    def test_constant_input(self):
        df = _make_df(np.full(500, 100.0))
        result = SNIPBaseline(half_window=40)(df)
        # Baseline should approximate the constant; result should be near zero
        assert (result["intensity"] >= 0).all()
        assert result["intensity"].max() < 10.0  # most of signal removed


class TestTICNormalizer:
    """Numerical stability tests for TICNormalizer."""

    def test_all_zero_unchanged(self):
        df = _make_df(np.zeros(100))
        result = TICNormalizer()(df)
        assert (result["intensity"] == 0).all()

    def test_inf_in_input(self):
        df = _make_df([100, np.inf, 200])
        result = TICNormalizer()(df)
        # Total is inf, so 100/inf=0, inf/inf=nan, 200/inf=0
        # The guard `total > 0` is True for inf, so division happens
        assert len(result) == 3  # no crash

    def test_normal_sums_to_one(self):
        df = _make_df([100, 200, 300])
        result = TICNormalizer()(df)
        assert np.isclose(result["intensity"].sum(), 1.0)


class TestMedianNormalizer:
    """Numerical stability tests for MedianNormalizer."""

    def test_all_zero_unchanged(self):
        df = _make_df(np.zeros(100))
        result = MedianNormalizer()(df)
        assert (result["intensity"] == 0).all()

    def test_constant_input(self):
        df = _make_df(np.full(100, 42.0))
        result = MedianNormalizer()(df)
        assert np.allclose(result["intensity"].values, 1.0)


class TestPQNNormalizer:
    """Numerical stability tests for PQNNormalizer."""

    def test_all_zero_unchanged(self):
        df = _make_df(np.zeros(100))
        result = PQNNormalizer()(df)
        assert (result["intensity"] == 0).all()

    def test_with_reference(self):
        ref = np.ones(100)
        df = _make_df(np.full(100, 50.0))
        result = PQNNormalizer(reference=ref)(df)
        assert len(result) == 100
        assert result["intensity"].sum() > 0


class TestSavitzkyGolaySmoothValidation:
    """Input validation tests for SavitzkyGolaySmooth."""

    def test_window_exceeds_data(self):
        df = _make_df(np.ones(5))
        with pytest.raises(ValueError, match="exceeds data length"):
            SavitzkyGolaySmooth(window_length=20, polyorder=2)(df)

    def test_polyorder_exceeds_window(self):
        df = _make_df(np.ones(100))
        with pytest.raises(ValueError, match="must be greater than"):
            SavitzkyGolaySmooth(window_length=3, polyorder=5)(df)


class TestMzTrimmerValidation:
    """Input validation tests for MzTrimmer."""

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="must be less than"):
            MzTrimmer(mz_min=20000, mz_max=2000)

    def test_equal_range(self):
        with pytest.raises(ValueError, match="must be less than"):
            MzTrimmer(mz_min=5000, mz_max=5000)


class TestMzTrimmer:
    """Numerical stability tests for MzTrimmer."""

    def test_empty_after_trim(self):
        df = _make_df(np.ones(100))  # mass range: 2000-20000
        result = MzTrimmer(mz_min=30000, mz_max=40000)(df)
        assert len(result) == 0

    def test_full_range_noop(self):
        df = _make_df(np.ones(100))
        result = MzTrimmer(mz_min=0, mz_max=50000)(df)
        assert len(result) == len(df)


class TestMzMultiTrimmer:
    """Tests for MzMultiTrimmer."""

    def test_no_matching_ranges(self):
        df = _make_df(np.ones(100))  # mass range: 2000-20000
        result = MzMultiTrimmer(mz_ranges=[(30000, 40000)])(df)
        assert len(result) == 0
