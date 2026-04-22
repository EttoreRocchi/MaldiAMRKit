"""Tests for drift-monitoring visualization functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from maldiamrkit.drift import (
    plot_effect_size_drift,
    plot_pca_drift,
    plot_peak_stability,
    plot_reference_drift,
)


@pytest.fixture
def ref_df() -> pd.DataFrame:
    starts = pd.date_range("2025-01-01", periods=6, freq="30D")
    return pd.DataFrame(
        {
            "window_start": starts,
            "window_end": starts + pd.Timedelta(days=30),
            "n_spectra": [10, 12, 9, 11, 10, 8],
            "distance_to_reference": [0.05, 0.08, 0.12, 0.20, 0.25, 0.30],
        }
    )


@pytest.fixture
def pca_df() -> pd.DataFrame:
    starts = pd.date_range("2025-01-01", periods=6, freq="30D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "window_start": starts,
            "window_end": starts + pd.Timedelta(days=30),
            "centroid_pc1": np.linspace(0, 2.0, 6) + rng.normal(0, 0.05, 6),
            "centroid_pc2": np.linspace(0, 1.0, 6) + rng.normal(0, 0.05, 6),
            "dispersion": np.linspace(0.5, 1.5, 6),
            "n_spectra": [10, 12, 9, 11, 10, 8],
        }
    )


@pytest.fixture
def stability_df() -> pd.DataFrame:
    starts = pd.date_range("2025-01-01", periods=6, freq="30D")
    return pd.DataFrame(
        {
            "window_start": starts,
            "stability_score": [0.9, 0.85, 0.7, 0.5, 0.3, 0.2],
            "n_spectra": [10, 12, 9, 11, 10, 8],
        }
    )


@pytest.fixture
def effect_df() -> pd.DataFrame:
    starts = pd.date_range("2025-01-01", periods=6, freq="30D")
    return pd.DataFrame(
        {
            "window_start": starts,
            "2000.0": [1.2, 1.1, 0.9, 0.5, 0.1, -0.2],
            "2030.0": [0.8, 0.9, 0.7, 0.6, 0.4, 0.3],
        }
    )


class TestReferenceDrift:
    def test_returns_figure_axes(self, ref_df):
        fig, ax = plot_reference_drift(ref_df, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_existing_ax(self, ref_df):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_reference_drift(ref_df, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close(fig)

    def test_missing_columns_raise(self):
        bad = pd.DataFrame({"window_start": [], "other": []})
        with pytest.raises(ValueError, match="missing"):
            plot_reference_drift(bad, show=False)


class TestPcaDrift:
    def test_returns_figure_axes(self, pca_df):
        fig, ax = plot_pca_drift(pca_df, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_draws_arrows(self, pca_df):
        from matplotlib.patches import FancyArrowPatch

        fig, ax = plot_pca_drift(pca_df, show=False)
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == len(pca_df) - 1
        plt.close(fig)

    def test_single_row_ok(self, pca_df):
        fig, ax = plot_pca_drift(pca_df.head(1), show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_columns_raise(self):
        bad = pd.DataFrame({"window_start": []})
        with pytest.raises(ValueError, match="missing"):
            plot_pca_drift(bad, show=False)


class TestPeakStability:
    def test_returns_figure_axes(self, stability_df):
        fig, ax = plot_peak_stability(stability_df, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_y_range_clamped_like(self, stability_df):
        fig, ax = plot_peak_stability(stability_df, show=False)
        lo, hi = ax.get_ylim()
        assert lo <= 0.0 and hi >= 1.0
        plt.close(fig)

    def test_missing_columns_raise(self):
        bad = pd.DataFrame({"window_start": []})
        with pytest.raises(ValueError, match="missing"):
            plot_peak_stability(bad, show=False)


class TestEffectSizeDrift:
    def test_returns_figure_axes(self, effect_df):
        fig, ax = plot_effect_size_drift(effect_df, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_peak_subset(self, effect_df):
        fig, ax = plot_effect_size_drift(effect_df, peaks=["2000.0"], show=False)
        labels = [
            ln.get_label() for ln in ax.get_lines() if ln.get_label() != "_child0"
        ]
        assert "2000.0" in labels
        assert "2030.0" not in labels
        plt.close(fig)

    def test_unknown_peak_raises(self, effect_df):
        with pytest.raises(ValueError, match="not found"):
            plot_effect_size_drift(effect_df, peaks=["bogus"], show=False)

    def test_missing_window_start(self):
        bad = pd.DataFrame({"some_peak": [0.1, 0.2]})
        with pytest.raises(ValueError, match="window_start"):
            plot_effect_size_drift(bad, show=False)


class TestShowTrue:
    def test_reference_show_true(self, ref_df):
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, _ = plot_reference_drift(ref_df, show=True)
        plt.close(fig)

    def test_pca_show_true(self, pca_df):
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, _ = plot_pca_drift(pca_df, show=True)
        plt.close(fig)

    def test_stability_show_true(self, stability_df):
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, _ = plot_peak_stability(stability_df, show=True)
        plt.close(fig)

    def test_effect_show_true(self, effect_df):
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, _ = plot_effect_size_drift(effect_df, show=True)
        plt.close(fig)
