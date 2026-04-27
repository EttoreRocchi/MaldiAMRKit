"""Tests for exploratory dimensionality-reduction plots."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from maldiamrkit.visualization.exploratory_plots import (  # noqa: E402
    _reduce_dimensions,
    _scatter_embedding,
    plot_pca,
    plot_tsne,
    plot_umap,
)


@pytest.fixture
def small_X():
    """Small synthetic feature matrix (30 samples, 100 features)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 100))
    columns = [str(2000 + i * 3) for i in range(100)]
    index = [f"s{i}" for i in range(30)]
    return pd.DataFrame(data, columns=columns, index=index)


@pytest.fixture
def labels():
    """Categorical labels for 30 samples (3 groups)."""
    return pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 10, name="Species")


@pytest.fixture
def binary_labels():
    """Binary labels for 30 samples."""
    return np.array([0] * 15 + [1] * 15)


class TestReduceDimensions:
    def test_pca_shape(self, small_X):
        emb, reducer = _reduce_dimensions(small_X, "pca", n_components=2)
        assert emb.shape == (30, 2)
        # Default standardize=True -> Pipeline(StandardScaler, PCA)
        assert hasattr(reducer, "named_steps")
        assert hasattr(reducer.named_steps["pca"], "explained_variance_ratio_")

    def test_pca_no_standardize(self, small_X):
        emb, reducer = _reduce_dimensions(
            small_X, "pca", n_components=2, standardize=False
        )
        assert emb.shape == (30, 2)
        assert hasattr(reducer, "explained_variance_ratio_")

    def test_tsne_shape(self, small_X):
        emb, _ = _reduce_dimensions(small_X, "tsne", n_components=2, perplexity=5)
        assert emb.shape == (30, 2)

    def test_invalid_method(self, small_X):
        with pytest.raises(ValueError, match="Unknown method"):
            _reduce_dimensions(small_X, "invalid")

    def test_umap_import_error(self, small_X):
        try:
            import umap  # noqa: F401

            pytest.skip("umap-learn is installed; cannot test ImportError")
        except ImportError:
            with pytest.raises(ImportError, match="maldiamrkit\\[batch\\]"):
                _reduce_dimensions(small_X, "umap")


class TestScatterEmbedding:
    def test_no_color(self):
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(emb, show=False)
        assert isinstance(fig, Figure)

    def test_with_color(self, labels):
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(emb, color_by=labels, show=False)
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        assert set(legend_labels) == {"A", "B", "C"}

    def test_existing_ax(self):
        import matplotlib.pyplot as plt

        _, ax_ext = plt.subplots()
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(emb, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close("all")

    def test_custom_palette(self, labels):
        emb = np.random.default_rng(0).standard_normal((30, 2))
        palette = {"A": "red", "B": "blue", "C": "green"}
        fig, ax = _scatter_embedding(emb, color_by=labels, palette=palette, show=False)
        assert isinstance(fig, Figure)

    def test_rs_labels_rendered_as_susceptible_resistant(self):
        """R/S labels auto-map to the readable 'Susceptible'/'Resistant' strings."""
        rs = np.array(["R", "S", "R", "S"] * 8)[:30]
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(emb, color_by=rs, show=False)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Susceptible" in t for t in legend_texts)
        assert any("Resistant" in t for t in legend_texts)

    def test_rs_labels_order_s_then_r(self):
        """Groups render in S -> I -> R order, not alphabetical."""
        rs = np.array(["R"] * 10 + ["S"] * 20)
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(emb, color_by=rs, show=False)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Susceptible" in legend_texts[0]
        assert "Resistant" in legend_texts[1]

    def test_binary_labels_mapped(self):
        """0/1 labels auto-map to Susceptible/Resistant."""
        bin_labels = np.array([0, 1, 0, 1] * 8)[:30]
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(emb, color_by=bin_labels, show=False)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Susceptible" in t for t in legend_texts)
        assert any("Resistant" in t for t in legend_texts)

    def test_label_map_override(self):
        """User label_map overrides the defaults."""
        bin_labels = np.array([0, 1, 0, 1] * 8)[:30]
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(
            emb,
            color_by=bin_labels,
            label_map={0: "Yes", 1: "No"},
            show=False,
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert set(legend_texts) == {"Yes", "No"}

    def test_marker_size_auto_scales_with_n(self):
        """s=None shrinks marker for large n."""
        from maldiamrkit.visualization.exploratory_plots import _auto_marker_size

        assert _auto_marker_size(10) > _auto_marker_size(1000)

    def test_legend_loc_outside(self, labels):
        """legend_loc='outside' places the legend outside the axes box."""
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig, ax = _scatter_embedding(
            emb,
            color_by=labels,
            legend_loc="outside",
            show=False,
        )
        # Outside legends have bbox_to_anchor > 1 on x.
        legend = ax.get_legend()
        assert legend is not None
        bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        assert bbox.xmin > 1.0

    def test_grid_enabled_by_default(self, labels):
        """Grid is drawn by default; disabled via grid=False."""
        emb = np.random.default_rng(0).standard_normal((30, 2))
        fig1, ax1 = _scatter_embedding(emb, color_by=labels, show=False)
        fig2, ax2 = _scatter_embedding(emb, color_by=labels, grid=False, show=False)
        gridlines_on = any(line.get_visible() for line in ax1.get_xgridlines())
        gridlines_off = any(line.get_visible() for line in ax2.get_xgridlines())
        assert gridlines_on
        assert not gridlines_off


class TestPlotPCA:
    def test_returns_fig_ax(self, small_X, labels):
        fig, ax = plot_pca(small_X, color_by=labels, show=False)
        assert isinstance(fig, Figure)

    def test_variance_in_labels(self, small_X):
        fig, ax = plot_pca(small_X, show=False)
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        assert "PC1" in xlabel and "%" in xlabel
        assert "PC2" in ylabel and "%" in ylabel

    def test_title_default(self, small_X):
        fig, ax = plot_pca(small_X, show=False)
        assert ax.get_title() == "PCA"

    def test_custom_title(self, small_X):
        fig, ax = plot_pca(small_X, title="My PCA", show=False)
        assert ax.get_title() == "My PCA"

    def test_existing_ax(self, small_X):
        import matplotlib.pyplot as plt

        _, ax_ext = plt.subplots()
        fig, ax = plot_pca(small_X, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close("all")

    def test_with_binary_labels(self, small_X, binary_labels):
        fig, ax = plot_pca(small_X, color_by=binary_labels, show=False)
        assert isinstance(fig, Figure)


class TestPlotTSNE:
    def test_returns_fig_ax(self, small_X, labels):
        fig, ax = plot_tsne(small_X, color_by=labels, perplexity=5, show=False)
        assert isinstance(fig, Figure)

    def test_axis_labels(self, small_X):
        fig, ax = plot_tsne(small_X, perplexity=5, show=False)
        assert "t-SNE 1" in ax.get_xlabel()
        assert "t-SNE 2" in ax.get_ylabel()

    def test_title_default(self, small_X):
        fig, ax = plot_tsne(small_X, perplexity=5, show=False)
        assert ax.get_title() == "t-SNE"


@pytest.mark.filterwarnings("ignore:n_jobs value .* overridden")
class TestPlotUMAP:
    @pytest.fixture(autouse=True)
    def _require_umap(self):
        pytest.importorskip("umap")

    def test_returns_fig_ax(self, small_X, labels):
        fig, ax = plot_umap(small_X, color_by=labels, n_neighbors=5, show=False)
        assert isinstance(fig, Figure)

    def test_axis_labels(self, small_X):
        fig, ax = plot_umap(small_X, n_neighbors=5, show=False)
        assert "UMAP 1" in ax.get_xlabel()
        assert "UMAP 2" in ax.get_ylabel()

    def test_title_default(self, small_X):
        fig, ax = plot_umap(small_X, n_neighbors=5, show=False)
        assert ax.get_title() == "UMAP"


class TestUmapImportError:
    """Tests for UMAP import error handling."""

    def test_umap_import_error(self, small_X):
        """Verify ImportError when umap-learn is not installed."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"umap": None}):
            with pytest.raises(ImportError, match="umap-learn"):
                _reduce_dimensions(small_X, method="umap")


class TestScatterEmbeddingShow:
    """Tests for show parameter in _scatter_embedding."""

    def test_show_false_returns_without_calling_show(self, small_X):
        """Verify show=False returns fig and ax without plt.show."""
        fig, ax = plot_pca(small_X, show=False)
        assert fig is not None
