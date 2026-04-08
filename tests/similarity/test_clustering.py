"""Tests for clustering algorithms and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from maldiamrkit.similarity.clustering import (
    cluster_metadata_concordance,
    cluster_spectra,
    hdbscan_clustering,
    hierarchical_clustering,
    kmedoids_clustering,
    silhouette_scores,
)


class TestHierarchicalClustering:
    """Linkage matrix shape and basic correctness."""

    def test_linkage_shape(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        Z = hierarchical_clustering(D)
        n = D.shape[0]
        assert Z.shape == (n - 1, 4)

    def test_known_cluster_recovery(self, distance_matrix_3clusters):
        D, true_labels = distance_matrix_3clusters
        Z = hierarchical_clustering(D, method="average")
        from scipy.cluster.hierarchy import fcluster

        labels = fcluster(Z, t=3, criterion="maxclust")
        # Labels may differ in numbering; check via ARI.
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.8


class TestHDBSCAN:
    """Basic HDBSCAN checks."""

    def test_returns_correct_length(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels = hdbscan_clustering(D, min_samples=2)
        assert len(labels) == D.shape[0]

    def test_at_least_one_cluster(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels = hdbscan_clustering(D, min_samples=2)
        assert len(set(labels) - {-1}) >= 1


class TestKmedoids:
    """PAM implementation checks."""

    def test_known_cluster_recovery(self, distance_matrix_3clusters):
        D, true_labels = distance_matrix_3clusters
        labels = kmedoids_clustering(D, n_clusters=3, init="build")
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.8

    def test_build_deterministic(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels1 = kmedoids_clustering(D, n_clusters=3, init="build")
        labels2 = kmedoids_clustering(D, n_clusters=3, init="build")
        np.testing.assert_array_equal(labels1, labels2)

    def test_random_reproducible(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels1 = kmedoids_clustering(D, n_clusters=3, init="random", random_state=0)
        labels2 = kmedoids_clustering(D, n_clusters=3, init="random", random_state=0)
        np.testing.assert_array_equal(labels1, labels2)

    def test_invalid_init(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        with pytest.raises(ValueError, match="is not a valid"):
            kmedoids_clustering(D, n_clusters=3, init="bad")


class TestSilhouetteScores:
    """Silhouette score range and behaviour."""

    def test_range(self, distance_matrix_3clusters):
        D, true_labels = distance_matrix_3clusters
        score = silhouette_scores(D, true_labels)
        assert -1.0 <= score <= 1.0

    def test_well_separated_positive(self, distance_matrix_3clusters):
        D, true_labels = distance_matrix_3clusters
        score = silhouette_scores(D, true_labels)
        assert score > 0.0


class TestClusterMetadataConcordance:
    """Concordance dict keys and perfect-label behaviour."""

    def test_keys(self, distance_matrix_3clusters):
        _, true_labels = distance_matrix_3clusters
        result = cluster_metadata_concordance(true_labels, pd.Series(true_labels))
        assert set(result) == {"adjusted_rand_index", "normalized_mutual_info"}

    def test_perfect_labels(self, distance_matrix_3clusters):
        _, true_labels = distance_matrix_3clusters
        result = cluster_metadata_concordance(true_labels, pd.Series(true_labels))
        assert result["adjusted_rand_index"] == pytest.approx(1.0)
        assert result["normalized_mutual_info"] == pytest.approx(1.0)


class TestClusterSpectra:
    """cluster_spectra dispatcher validation."""

    def test_hierarchical_n_clusters(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels = cluster_spectra(D, method="hierarchical", n_clusters=3)
        assert len(labels) == D.shape[0]
        assert len(set(labels)) == 3

    def test_hierarchical_threshold(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels = cluster_spectra(D, method="hierarchical", threshold=15.0)
        assert len(labels) == D.shape[0]

    def test_hierarchical_both_raises(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        with pytest.raises(ValueError, match="Exactly one"):
            cluster_spectra(D, method="hierarchical", n_clusters=3, threshold=1.0)

    def test_hierarchical_neither_raises(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        with pytest.raises(ValueError, match="Exactly one"):
            cluster_spectra(D, method="hierarchical")

    def test_kmedoids_requires_n_clusters(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        with pytest.raises(ValueError, match="n_clusters"):
            cluster_spectra(D, method="kmedoids")

    def test_kmedoids_works(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels = cluster_spectra(D, method="kmedoids", n_clusters=3)
        assert len(labels) == D.shape[0]

    def test_hdbscan_works(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        labels = cluster_spectra(D, method="hdbscan", min_samples=2)
        assert len(labels) == D.shape[0]

    def test_invalid_method(self, distance_matrix_3clusters):
        D, _ = distance_matrix_3clusters
        with pytest.raises(ValueError, match="is not a valid"):
            cluster_spectra(D, method="bad")
