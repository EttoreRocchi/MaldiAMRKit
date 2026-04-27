"""Numerical stability tests for preprocessing transformers.

Tests all transformers with problematic inputs: inf, -inf, NaN, all-zero,
very large values, and extreme conditions.
"""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.preprocessing import (
    ClipNegatives,
    ConvexHullBaseline,
    LogTransform,
    MedianBaseline,
    MedianNormalizer,
    MovingAverageSmooth,
    MzMultiTrimmer,
    MzTrimmer,
    PQNNormalizer,
    PreprocessingPipeline,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    SqrtTransform,
    TICNormalizer,
    TopHatBaseline,
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
        with pytest.warns(UserWarning, match="TIC total is zero"):
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
        with pytest.warns(UserWarning, match="Median intensity is zero"):
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
        df = _make_df(np.ones(100))
        result = MzMultiTrimmer(mz_ranges=[(30000, 40000)])(df)
        assert len(result) == 0

    def test_multiple_ranges(self):
        df = _make_df(np.ones(100))
        result = MzMultiTrimmer(mz_ranges=[(2000, 5000), (10000, 15000)])(df)
        assert len(result) > 0
        assert all((2000 <= m <= 5000) or (10000 <= m <= 15000) for m in result["mass"])

    def test_repr(self):
        t = MzMultiTrimmer(mz_ranges=[(2000, 5000)])
        assert "MzMultiTrimmer" in repr(t)


class TestLogTransformRepr:
    """Test LogTransform repr."""

    def test_repr(self):
        from maldiamrkit.preprocessing.transformers import LogTransform

        assert repr(LogTransform()) == "LogTransform()"


class TestMedianNormalizerEdges:
    """Tests for MedianNormalizer edge cases."""

    def test_all_zero_median(self):
        from maldiamrkit.preprocessing.transformers import MedianNormalizer

        df = _make_df(np.zeros(50))
        with pytest.warns(UserWarning, match="Median intensity is zero"):
            result = MedianNormalizer()(df)
        assert result["intensity"].sum() == 0.0


class TestPQNNormalizerEdges:
    """Tests for PQNNormalizer edge cases."""

    def test_reference_length_mismatch_warns(self):
        from maldiamrkit.preprocessing.transformers import PQNNormalizer

        ref = np.ones(200)
        pqn = PQNNormalizer(reference=ref)
        df = _make_df(np.ones(50))
        with pytest.raises(ValueError, match="reference length"):
            pqn(df)

    def test_no_positive_reference(self):
        from maldiamrkit.preprocessing.transformers import PQNNormalizer

        ref = np.zeros(50)
        pqn = PQNNormalizer(reference=ref)
        df = _make_df(np.ones(50))
        result = pqn(df)
        assert len(result) == 50

    def test_repr(self):
        from maldiamrkit.preprocessing.transformers import PQNNormalizer

        assert "PQNNormalizer" in repr(PQNNormalizer())


class TestMedianNormalizerSerialization:
    """Tests for MedianNormalizer to_dict and __repr__ (L236, L239)."""

    def test_to_dict(self):
        """Verify to_dict returns correct dictionary."""
        from maldiamrkit.preprocessing.transformers import MedianNormalizer

        result = MedianNormalizer().to_dict()
        assert result == {"name": "MedianNormalizer"}

    def test_repr(self):
        """Verify __repr__ returns expected string."""
        from maldiamrkit.preprocessing.transformers import MedianNormalizer

        assert repr(MedianNormalizer()) == "MedianNormalizer()"


class TestPQNNormalizerExtras:
    """Additional PQNNormalizer tests for uncovered branches."""

    def test_list_reference_converted(self):
        """Verify list reference is converted to ndarray."""
        from maldiamrkit.preprocessing.transformers import PQNNormalizer

        pqn = PQNNormalizer(reference=[1.0, 2.0, 3.0])
        assert isinstance(pqn.reference, np.ndarray)

    def test_to_dict_with_reference(self):
        """Verify to_dict includes reference when set."""
        from maldiamrkit.preprocessing.transformers import PQNNormalizer

        pqn = PQNNormalizer(reference=[1.0, 2.0])
        d = pqn.to_dict()
        assert "reference" in d
        assert d["reference"] == [1.0, 2.0]


class TestMzMultiTrimmerExtras:
    """Additional MzMultiTrimmer tests for uncovered branches."""

    def test_empty_ranges_raises(self):
        """Verify empty mz_ranges raises ValueError."""
        from maldiamrkit.preprocessing.transformers import MzMultiTrimmer

        with pytest.raises(ValueError, match="must not be empty"):
            MzMultiTrimmer(mz_ranges=[])

    def test_to_dict(self):
        """Verify to_dict returns correct dictionary."""
        from maldiamrkit.preprocessing.transformers import MzMultiTrimmer

        trimmer = MzMultiTrimmer(mz_ranges=[(2000, 5000), (8000, 12000)])
        d = trimmer.to_dict()
        assert d["name"] == "MzMultiTrimmer"
        assert d["mz_ranges"] == [(2000, 5000), (8000, 12000)]


def _make_df_with_baseline(
    n: int = 500,
    peak_positions: tuple[int, ...] = (100, 250, 400),
    peak_amplitude: float = 10.0,
    baseline_slope: float = 2.0,
    baseline_intercept: float = 5.0,
):
    """Create a synthetic spectrum with a known linear baseline and peaks."""
    mz = np.linspace(2000, 20000, n)
    baseline = baseline_intercept + baseline_slope * np.linspace(0, 1, n)
    intensity = baseline.copy()
    for pos in peak_positions:
        intensity[pos] += peak_amplitude
    return pd.DataFrame({"mass": mz, "intensity": intensity})


class TestTopHatBaseline:
    """Tests for TopHatBaseline."""

    def test_removes_flat_baseline(self):
        df = _make_df(np.full(500, 42.0))
        result = TopHatBaseline(half_window=20)(df)
        assert np.allclose(result["intensity"].values, 0.0, atol=1e-10)

    def test_preserves_narrow_peak(self):
        df = _make_df_with_baseline(
            n=500,
            peak_positions=(250,),
            peak_amplitude=20.0,
            baseline_slope=0.0,
            baseline_intercept=5.0,
        )
        result = TopHatBaseline(half_window=20)(df)
        assert (result["intensity"] >= 0).all()
        assert result["intensity"].iloc[250] > 15.0
        mask = np.ones(500, dtype=bool)
        mask[245:256] = False
        assert result["intensity"][mask].max() < 1e-6

    def test_clips_negative_residuals(self):
        df = _make_df(np.linspace(0, 100, 500))
        result = TopHatBaseline(half_window=20)(df)
        assert (result["intensity"] >= 0).all()

    def test_invalid_half_window_raises(self):
        df = _make_df(np.ones(500))
        with pytest.raises(ValueError, match="positive integer"):
            TopHatBaseline(half_window=0)(df)

    def test_window_too_large_raises(self):
        df = _make_df(np.ones(10))
        with pytest.raises(ValueError, match="exceeds data length"):
            TopHatBaseline(half_window=100)(df)

    def test_to_dict_repr(self):
        t = TopHatBaseline(half_window=75)
        assert t.to_dict() == {"name": "TopHatBaseline", "half_window": 75}
        assert repr(t) == "TopHatBaseline(half_window=75)"


class TestConvexHullBaseline:
    """Tests for ConvexHullBaseline."""

    def test_removes_linear_baseline(self):
        df = _make_df_with_baseline(
            n=500,
            peak_positions=(250,),
            peak_amplitude=50.0,
            baseline_slope=10.0,
            baseline_intercept=5.0,
        )
        result = ConvexHullBaseline()(df)
        assert (result["intensity"] >= 0).all()
        assert result["intensity"].iloc[250] > 40.0
        mask = np.ones(500, dtype=bool)
        mask[245:256] = False
        assert result["intensity"][mask].max() < 1.0

    def test_flat_input_returns_zero(self):
        df = _make_df(np.full(500, 7.0))
        result = ConvexHullBaseline()(df)
        assert np.allclose(result["intensity"].values, 0.0, atol=1e-10)

    def test_short_input_no_crash(self):
        df = _make_df([1.0, 2.0])
        result = ConvexHullBaseline()(df)
        assert len(result) == 2
        assert (result["intensity"] >= 0).all()

    def test_to_dict_repr(self):
        t = ConvexHullBaseline()
        assert t.to_dict() == {"name": "ConvexHullBaseline"}
        assert repr(t) == "ConvexHullBaseline()"


class TestMedianBaseline:
    """Tests for MedianBaseline."""

    def test_removes_slow_baseline(self):
        df = _make_df_with_baseline(
            n=500,
            peak_positions=(250,),
            peak_amplitude=30.0,
            baseline_slope=5.0,
            baseline_intercept=2.0,
        )
        result = MedianBaseline(half_window=30, iterations=1)(df)
        assert (result["intensity"] >= 0).all()
        assert result["intensity"].iloc[250] > 20.0

    def test_iterations_does_not_invert(self):
        df = _make_df(np.full(500, 3.0))
        r1 = MedianBaseline(half_window=30, iterations=1)(df)
        r3 = MedianBaseline(half_window=30, iterations=3)(df)
        assert np.allclose(r1["intensity"].values, 0.0, atol=1e-10)
        assert np.allclose(r3["intensity"].values, 0.0, atol=1e-10)

    def test_invalid_half_window_raises(self):
        df = _make_df(np.ones(500))
        with pytest.raises(ValueError, match="half_window"):
            MedianBaseline(half_window=0)(df)

    def test_invalid_iterations_raises(self):
        df = _make_df(np.ones(500))
        with pytest.raises(ValueError, match="iterations"):
            MedianBaseline(half_window=30, iterations=0)(df)

    def test_window_too_large_raises(self):
        df = _make_df(np.ones(10))
        with pytest.raises(ValueError, match="exceeds data length"):
            MedianBaseline(half_window=100)(df)

    def test_to_dict_repr(self):
        t = MedianBaseline(half_window=50, iterations=2)
        assert t.to_dict() == {
            "name": "MedianBaseline",
            "half_window": 50,
            "iterations": 2,
        }
        assert repr(t) == "MedianBaseline(half_window=50, iterations=2)"


class TestMovingAverageSmooth:
    """Tests for MovingAverageSmooth."""

    def test_smooths_gaussian_noise(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=10.0, scale=1.0, size=500)
        df = _make_df(values)
        result = MovingAverageSmooth(window_length=7)(df)
        assert result["intensity"].std() < df["intensity"].std()
        assert np.isclose(result["intensity"].mean(), df["intensity"].mean(), atol=0.1)

    def test_constant_input_unchanged(self):
        df = _make_df(np.full(500, 42.0))
        result = MovingAverageSmooth(window_length=5)(df)
        assert np.allclose(result["intensity"].values, 42.0, atol=1e-10)

    def test_even_window_raises(self):
        df = _make_df(np.ones(500))
        with pytest.raises(ValueError, match="odd integer"):
            MovingAverageSmooth(window_length=6)(df)

    def test_window_too_small_raises(self):
        df = _make_df(np.ones(500))
        with pytest.raises(ValueError, match="odd integer"):
            MovingAverageSmooth(window_length=1)(df)

    def test_window_exceeds_data_raises(self):
        df = _make_df(np.ones(7))
        with pytest.raises(ValueError, match="exceeds data length"):
            MovingAverageSmooth(window_length=21)(df)

    def test_to_dict_repr(self):
        t = MovingAverageSmooth(window_length=11)
        assert t.to_dict() == {"name": "MovingAverageSmooth", "window_length": 11}
        assert repr(t) == "MovingAverageSmooth(window_length=11)"


class TestNewTransformersPipelineSerialization:
    """Round-trip serialization via PreprocessingPipeline for new transformers."""

    def _build_pipeline(self) -> PreprocessingPipeline:
        return PreprocessingPipeline(
            [
                ("smooth", MovingAverageSmooth(window_length=7)),
                ("tophat", TopHatBaseline(half_window=50)),
                ("hull", ConvexHullBaseline()),
                ("median_base", MedianBaseline(half_window=40, iterations=2)),
            ]
        )

    def test_dict_round_trip_preserves_params(self):
        pipe = self._build_pipeline()
        pipe2 = PreprocessingPipeline.from_dict(pipe.to_dict())
        assert pipe.step_names == pipe2.step_names
        for (_, a), (_, b) in zip(pipe.steps, pipe2.steps, strict=True):
            assert a.to_dict() == b.to_dict()

    def test_dict_round_trip_produces_identical_output(self):
        pipe = self._build_pipeline()
        pipe2 = PreprocessingPipeline.from_dict(pipe.to_dict())
        df = _make_df_with_baseline(
            n=500,
            peak_positions=(120, 260, 400),
            peak_amplitude=25.0,
            baseline_slope=8.0,
            baseline_intercept=3.0,
        )
        out1 = pipe(df.copy())
        out2 = pipe2(df.copy())
        pd.testing.assert_frame_equal(out1, out2)
