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
