"""Tests for pairwise distance matrix computation."""

from __future__ import annotations

import numpy as np
import pytest

from maldiamrkit.similarity.pairwise import pairwise_distances


class TestPairwiseDistances:
    """Shape, symmetry, diagonal, and parallelism checks."""

    def test_shape(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="cosine")
        n = len(small_binned_df)
        assert D.shape == (n, n)

    def test_symmetry(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="cosine")
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_zero_diagonal(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="cosine")
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_njobs_equivalence(self, small_binned_df):
        D1 = pairwise_distances(small_binned_df, metric="cosine", n_jobs=1)
        D2 = pairwise_distances(small_binned_df, metric="cosine", n_jobs=2)
        np.testing.assert_allclose(D1, D2, atol=1e-12)

    def test_fast_path_matches_general(self, small_binned_df):
        """Binned fast path and general path produce the same result."""
        D_fast = pairwise_distances(small_binned_df, metric="cosine")
        # Force general path by passing row arrays as a list.
        rows = [small_binned_df.iloc[i].values for i in range(len(small_binned_df))]
        D_general = pairwise_distances(rows, metric="cosine", n_jobs=1)
        np.testing.assert_allclose(D_fast, D_general, atol=1e-10)

    def test_invalid_metric(self, small_binned_df):
        with pytest.raises(ValueError, match="is not a valid"):
            pairwise_distances(small_binned_df, metric="bad")

    def test_pearson_metric(self, small_binned_df):
        D = pairwise_distances(small_binned_df, metric="pearson")
        assert D.shape == (len(small_binned_df), len(small_binned_df))
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
