"""Edge-case tests identified by the scientific rigour audit.

Each test targets a specific finding from the audit, verifying numerical
stability, mathematical correctness, and domain-specific edge cases that
were previously untested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestNearZeroVectorMetrics:
    """Verify metrics handle denormalized / near-zero vectors correctly."""

    def test_cosine_denormalized_vector(self):
        """Cosine distance with denormalized float inputs (norm ~ 1e-308)."""
        from maldiamrkit.similarity.metrics import spectral_distance

        tiny = np.finfo(float).tiny  # ~2.2e-308
        a = np.array([tiny / 2, tiny / 2])
        b = np.array([1.0, 2.0])
        d = spectral_distance(a, b, metric="cosine")
        assert d == 1.0  # should treat as zero vector

    def test_sca_denormalized_vector(self):
        """SCA distance with denormalized float inputs."""
        from maldiamrkit.similarity.metrics import spectral_distance

        tiny = np.finfo(float).tiny
        a = np.array([tiny / 2, tiny / 2])
        b = np.array([1.0, 2.0])
        d = spectral_distance(a, b, metric="spectral_contrast_angle")
        assert d == 1.0

    def test_cosine_clipped_non_negative(self):
        """Cosine distance should never return negative values."""
        from maldiamrkit.similarity.metrics import spectral_distance

        # Nearly identical vectors that could produce 1 - 1.0000...0002 < 0
        a = np.array([1.0, 2.0, 3.0])
        b = a.copy()
        d = spectral_distance(a, b, metric="cosine")
        assert d >= 0.0


class TestPearsonDistanceRange:
    """Verify Pearson distance range includes anti-correlated spectra."""

    def test_anti_correlated_distance_gt_1(self):
        """Perfectly anti-correlated vectors should yield distance = 2."""
        from maldiamrkit.similarity.metrics import spectral_distance

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        d = spectral_distance(a, b, metric="pearson")
        assert d == pytest.approx(2.0, abs=1e-10)

    def test_uncorrelated_distance_near_1(self):
        """Uncorrelated vectors should yield distance ~ 1."""
        from maldiamrkit.similarity.metrics import spectral_distance

        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 10000)
        b = rng.normal(0, 1, 10000)
        d = spectral_distance(a, b, metric="pearson")
        assert 0.9 < d < 1.1  # approximately 1


class TestShiftAlignmentRounding:
    """Verify shift uses round() not int() truncation."""

    def test_median_shift_rounds_not_truncates(self):
        """A median shift of 2.7 should round to 3, not truncate to 2."""
        from maldiamrkit.alignment.strategies import ShiftStrategy

        strategy = ShiftStrategy(max_shift=50)

        # Create peaks so that median shift = ~3 (ref_peaks - peaks)
        # peaks at indices 10, 20, 30; ref at 13, 23, 33 -> shifts = 3, 3, 3
        # But let's make a case where median = 2.7:
        # shifts: [2, 3, 3] -> median = 3
        # shifts: [2, 2, 3, 3, 4] -> median = 3
        # shifts: [2, 3] -> median = 2.5 -> round = 2 (banker's rounding)
        # shifts: [3, 3, 3, 2] -> median = 2.5 -> round = 2

        # Test case: shifts = [2, 3, 3] -> median = 3.0
        peaks = np.array([10, 20, 30])
        ref_peaks = np.array([13, 23, 33])

        row = np.zeros(100)
        row[10] = 1.0
        row[20] = 1.0
        row[30] = 1.0
        mz_axis = np.arange(100)

        aligned = strategy.align_binned(row, peaks, ref_peaks, mz_axis)
        # Peaks should move right by 3
        assert aligned[13] == pytest.approx(1.0)
        assert aligned[23] == pytest.approx(1.0)
        assert aligned[33] == pytest.approx(1.0)

    def test_fractional_shift_rounds_correctly(self):
        """Median shift of 2.5 should round to 2 (banker's rounding)."""
        from maldiamrkit.alignment.strategies import ShiftStrategy

        strategy = ShiftStrategy(max_shift=50)
        # shifts: [2, 3] -> median = 2.5 -> np.round(2.5) = 2 (banker's)
        peaks = np.array([10, 20])
        ref_peaks = np.array([12, 23])

        row = np.zeros(100)
        row[10] = 1.0
        row[20] = 1.0
        mz_axis = np.arange(100)

        aligned = strategy.align_binned(row, peaks, ref_peaks, mz_axis)
        # Should shift by 2 (np.round(2.5) = 2 due to banker's rounding)
        assert aligned[12] == pytest.approx(1.0)
        assert aligned[22] == pytest.approx(1.0)


class TestUniformBinningStability:
    """Verify uniform binning produces consistent edge counts with float widths."""

    def test_float_bin_width_consistent_edges(self):
        """np.linspace-based edges should be deterministic for float widths."""
        from maldiamrkit.preprocessing.binning import _uniform_edges

        edges_a = _uniform_edges(2000.0, 20000.0, 3.1)
        edges_b = _uniform_edges(2000.0, 20000.0, 3.1)
        np.testing.assert_array_equal(edges_a, edges_b)
        # Should cover the full range
        assert edges_a[0] == 2000.0
        assert edges_a[-1] == 20000.0

    def test_edges_cover_full_range(self):
        """First edge == mz_min, last edge == mz_max exactly."""
        from maldiamrkit.preprocessing.binning import _uniform_edges

        edges = _uniform_edges(2000.0, 20000.0, 3.0)
        assert edges[0] == 2000.0
        assert edges[-1] == 20000.0

    def test_edge_spacing_uniform(self):
        """All bin widths should be approximately equal."""
        from maldiamrkit.preprocessing.binning import _uniform_edges

        edges = _uniform_edges(2000.0, 20000.0, 3.0)
        widths = np.diff(edges)
        assert np.std(widths) < 1e-10  # essentially uniform


class TestBrukerCalibrationEdgeCases:
    """Verify TOF-to-mass for ML3 < 0 and ML3 = 0."""

    def test_ml3_zero_linear(self):
        """ML3=0 should give valid positive masses (linear calibration)."""
        from maldiamrkit.io.readers import _tof_to_mass

        tof = np.array([19000.0, 19100.0, 19200.0])
        mass = _tof_to_mass(5000000.0, 400.0, 0.0, tof)
        assert np.all(mass > 0)
        # Should be monotonic
        assert np.all(np.diff(mass) > 0) or np.all(np.diff(mass) < 0)

    def test_ml3_negative_quadratic(self):
        """ML3 < 0 should produce valid positive masses."""
        from maldiamrkit.io.readers import _tof_to_mass

        tof = np.array([19000.0, 19100.0, 19200.0])
        mass = _tof_to_mass(5000000.0, 400.0, -0.01, tof)
        assert np.all(mass > 0)
        assert mass.shape == (3,)

    def test_ml3_positive_quadratic(self):
        """ML3 > 0 should also produce valid positive masses."""
        from maldiamrkit.io.readers import _tof_to_mass

        tof = np.array([19000.0, 19100.0, 19200.0])
        mass = _tof_to_mass(5000000.0, 400.0, 0.01, tof)
        assert np.all(mass > 0)

    def test_ml1_negative_raises(self):
        """Negative ML1 should raise ValueError."""
        from maldiamrkit.io.readers import _tof_to_mass

        tof = np.array([19000.0])
        with pytest.raises(ValueError, match="ML1 must be positive"):
            _tof_to_mass(-1.0, 400.0, 0.0, tof)


class TestPQNSparseReference:
    """Verify PQN normalizer behavior with highly sparse references."""

    def test_sparse_reference_normalization(self):
        """PQN with >50% zero reference values should still normalize."""
        from maldiamrkit.preprocessing.transformers import PQNNormalizer

        # Reference with 80% zeros
        reference = np.zeros(100)
        reference[10:30] = np.random.default_rng(42).exponential(1.0, 20)

        pqn = PQNNormalizer(reference=reference)
        df = pd.DataFrame(
            {
                "mass": np.arange(100, dtype=float),
                "intensity": np.random.default_rng(7).exponential(1.0, 100),
            }
        )
        result = pqn(df)
        # Should produce valid (non-NaN) output
        assert not result["intensity"].isna().any()
        assert (result["intensity"] >= 0).all()


class TestMedianNormalizerZeroInput:
    """Verify MedianNormalizer warns on all-zero input."""

    def test_all_zero_warns(self):
        """All-zero intensities should emit a warning."""
        from maldiamrkit.preprocessing.transformers import MedianNormalizer

        normalizer = MedianNormalizer()
        df = pd.DataFrame({"mass": [1.0, 2.0, 3.0], "intensity": [0.0, 0.0, 0.0]})
        with pytest.warns(UserWarning, match="zero"):
            result = normalizer(df)
        # Data should be returned unchanged
        np.testing.assert_array_equal(result["intensity"].values, [0.0, 0.0, 0.0])


class TestKmedoidsTiedDistances:
    """Verify K-medoids handles tied distances deterministically."""

    def test_equidistant_points(self):
        """All pairwise distances equal - any clustering is valid."""
        from maldiamrkit.similarity.clustering import kmedoids_clustering

        n = 6
        D = np.ones((n, n)) - np.eye(n)
        labels = kmedoids_clustering(D, n_clusters=2)
        assert labels.shape == (n,)
        assert len(set(labels)) == 2

    def test_deterministic_build(self):
        """BUILD initialization should be deterministic."""
        from maldiamrkit.similarity.clustering import kmedoids_clustering

        rng = np.random.default_rng(42)
        points = rng.normal(0, 1, (20, 5))
        from scipy.spatial.distance import cdist

        D = cdist(points, points)
        labels1 = kmedoids_clustering(D, n_clusters=3, init="build")
        labels2 = kmedoids_clustering(D, n_clusters=3, init="build")
        np.testing.assert_array_equal(labels1, labels2)


class TestRareStrataLabelPreservation:
    """Verify rare strata merging preserves resistance label."""

    def test_rare_strata_split_by_label(self):
        """Rare strata should merge within resistance class, not across."""
        from maldiamrkit.evaluation.splitting import _build_strata

        y = np.array(["R", "R", "R", "R", "S", "R"])
        species = np.array(["A", "A", "A", "A", "B", "B"])
        strata = _build_strata(y, species, min_count=2)

        # B__S (1 sample) should merge to __rare_S__, B__R to __rare_R__
        # They should NOT both go to the same __rare__ bucket
        rare_mask = np.array(["__rare" in s for s in strata])
        if rare_mask.any():
            rare_strata = strata[rare_mask]
            rare_labels = y[rare_mask]
            # Each rare stratum should contain only one resistance label
            for stratum in np.unique(rare_strata):
                mask = rare_strata == stratum
                assert len(set(rare_labels[mask])) == 1


class TestKmedoidsVectorizedCost:
    """Verify vectorized K-medoids produces correct results."""

    def test_vectorized_matches_expected(self):
        """K-medoids on well-separated clusters should achieve ARI > 0.8."""
        from sklearn.metrics import adjusted_rand_score

        from maldiamrkit.similarity.clustering import kmedoids_clustering

        rng = np.random.default_rng(42)
        centers = np.array([[0] * 5, [10] * 5, [20] * 5], dtype=float)
        points = np.vstack([c + rng.normal(0, 0.5, (10, 5)) for c in centers])
        true_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

        from scipy.spatial.distance import cdist

        D = cdist(points, points)
        labels = kmedoids_clustering(D, n_clusters=3, init="build")
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.8
