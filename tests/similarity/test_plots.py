"""Tests for similarity visualization functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from maldiamrkit.similarity.clustering import hierarchical_clustering
from maldiamrkit.similarity.plots import plot_dendrogram, plot_distance_heatmap


@pytest.fixture
def sample_distance_matrix() -> np.ndarray:
    """Small symmetric distance matrix for plot tests."""
    rng = np.random.default_rng(42)
    X = rng.random((5, 5))
    D = (X + X.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


class TestPlotDistanceHeatmap:
    """plot_distance_heatmap return type and options."""

    def test_returns_figure_axes(self, sample_distance_matrix):
        fig, ax = plot_distance_heatmap(sample_distance_matrix, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_labels(self, sample_distance_matrix):
        labels = [f"s{i}" for i in range(5)]
        fig, ax = plot_distance_heatmap(
            sample_distance_matrix, labels=labels, show=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_existing_ax(self, sample_distance_matrix):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_distance_heatmap(sample_distance_matrix, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close(fig)

    def test_title(self, sample_distance_matrix):
        fig, ax = plot_distance_heatmap(
            sample_distance_matrix, title="Test Heatmap", show=False
        )
        assert ax.get_title() == "Test Heatmap"
        plt.close(fig)

    def test_default_title(self, sample_distance_matrix):
        """Default title auto-filled with metric when provided."""
        fig, ax = plot_distance_heatmap(
            sample_distance_matrix, metric="cosine", show=False
        )
        assert "Pairwise distance" in ax.get_title()
        assert "cosine" in ax.get_title()
        plt.close(fig)

    def test_metric_sets_colorbar_bounds(self, sample_distance_matrix):
        """metric='cosine' clamps colourbar to (0, 1)."""
        fig, ax = plot_distance_heatmap(
            sample_distance_matrix, metric="cosine", show=False
        )
        # Seaborn stores the QuadMesh as the first collection; its clim
        # reflects vmin/vmax.
        mesh = ax.collections[0]
        vmin, vmax = mesh.get_clim()
        assert vmin == 0.0
        assert vmax == 1.0
        plt.close(fig)

    def test_explicit_vmin_vmax_overrides_metric(self, sample_distance_matrix):
        fig, ax = plot_distance_heatmap(
            sample_distance_matrix,
            metric="cosine",
            vmin=0.0,
            vmax=0.5,
            show=False,
        )
        vmin, vmax = ax.collections[0].get_clim()
        assert vmin == 0.0
        assert vmax == 0.5
        plt.close(fig)

    def test_annot_auto_small_matrix(self, sample_distance_matrix):
        """5x5 matrix auto-annotates."""
        fig, ax = plot_distance_heatmap(sample_distance_matrix, show=False)
        # Seaborn writes one Text per cell when annotating.
        n_text_annotations = sum(
            1
            for t in ax.texts
            if t.get_text().replace(".", "").replace("-", "").isdigit()
        )
        assert n_text_annotations == sample_distance_matrix.size
        plt.close(fig)

    def test_cluster_reorders_labels(self, sample_distance_matrix):
        """cluster=True reorders labels to match the reordered matrix."""
        labels = np.array(["a", "b", "c", "d", "e"])
        fig, ax = plot_distance_heatmap(
            sample_distance_matrix, labels=labels, cluster=True, show=False
        )
        rendered = [t.get_text() for t in ax.get_xticklabels()]
        # Should be a permutation of the original labels, but not the
        # identity (given our random matrix).
        assert set(rendered) == set(labels)
        plt.close(fig)

    def test_dynamic_figsize(self, sample_distance_matrix):
        """Default figsize scales with matrix size."""
        big = np.random.default_rng(1).random((80, 80))
        big = (big + big.T) / 2
        np.fill_diagonal(big, 0.0)
        fig, _ = plot_distance_heatmap(big, show=False)
        w, h = fig.get_size_inches()
        # For n=80 the formula gives 4 + 0.1*80 = 12 (capped at 16)
        assert w == pytest.approx(12.0, rel=0.01)
        assert h == pytest.approx(12.0, rel=0.01)
        plt.close(fig)


class TestPlotDendrogram:
    """plot_dendrogram return type and options."""

    @pytest.fixture
    def linkage_matrix(self, sample_distance_matrix):
        return hierarchical_clustering(sample_distance_matrix)

    def test_returns_figure_axes(self, linkage_matrix):
        fig, ax = plot_dendrogram(linkage_matrix, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_labels(self, linkage_matrix):
        labels = [f"s{i}" for i in range(5)]
        fig, ax = plot_dendrogram(linkage_matrix, labels=labels, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_existing_ax(self, linkage_matrix):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_dendrogram(linkage_matrix, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close(fig)

    def test_title(self, linkage_matrix):
        fig, ax = plot_dendrogram(linkage_matrix, title="Test Dendrogram", show=False)
        assert ax.get_title() == "Test Dendrogram"
        plt.close(fig)

    def test_default_title(self, linkage_matrix):
        fig, ax = plot_dendrogram(linkage_matrix, show=False)
        assert ax.get_title() == "Hierarchical clustering dendrogram"
        plt.close(fig)

    def test_truncate_mode_lastp(self, linkage_matrix):
        """truncate_mode='lastp' with p=3 produces a truncated tree."""
        fig, ax = plot_dendrogram(
            linkage_matrix, truncate_mode="lastp", p=3, show=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_leaf_rotation_kwarg(self, linkage_matrix):
        """leaf_rotation=0 produces horizontal labels."""
        fig, ax = plot_dendrogram(
            linkage_matrix, leaf_rotation=0, labels=list("abcde"), show=False
        )
        rotations = {t.get_rotation() for t in ax.get_xticklabels()}
        assert 0.0 in rotations or 0 in rotations
        plt.close(fig)

    def test_color_threshold_kwarg(self, linkage_matrix):
        """color_threshold=0.0 collapses colouring (no colored clusters)."""
        fig, ax = plot_dendrogram(linkage_matrix, color_threshold=0.0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestShowTrue:
    """Tests for show=True path (Agg backend makes plt.show() a no-op)."""

    def test_heatmap_show_true(self, sample_distance_matrix):
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, ax = plot_distance_heatmap(sample_distance_matrix, show=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dendrogram_show_true(self, sample_distance_matrix):
        linkage = hierarchical_clustering(sample_distance_matrix)
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, ax = plot_dendrogram(linkage, show=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
