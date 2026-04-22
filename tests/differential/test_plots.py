"""Tests for differential-analysis visualization functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from maldiamrkit.differential import (
    DifferentialAnalysis,
    plot_drug_comparison,
    plot_manhattan,
    plot_volcano,
)


@pytest.fixture
def fitted_results(differential_dataset) -> pd.DataFrame:
    X, y, _ = differential_dataset
    return DifferentialAnalysis(X, y).run().results


class TestVolcano:
    def test_returns_figure_axes(self, fitted_results):
        fig, ax = plot_volcano(fitted_results, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_draws_threshold_lines(self, fitted_results):
        fig, ax = plot_volcano(
            fitted_results, fc_threshold=1.5, p_threshold=0.01, show=False
        )
        line_ys = [ln.get_ydata()[0] for ln in ax.get_lines()]
        line_xs = [ln.get_xdata()[0] for ln in ax.get_lines()]
        assert any(abs(y - (-np.log10(0.01))) < 1e-9 for y in line_ys)
        assert any(abs(x - 1.5) < 1e-9 for x in line_xs)
        assert any(abs(x + 1.5) < 1e-9 for x in line_xs)
        plt.close(fig)

    def test_with_existing_ax(self, fitted_results):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_volcano(fitted_results, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close(fig)

    def test_title(self, fitted_results):
        fig, ax = plot_volcano(fitted_results, title="Volcano", show=False)
        assert ax.get_title() == "Volcano"
        plt.close(fig)

    def test_missing_columns_raise(self):
        bad = pd.DataFrame({"mz_bin": [1, 2], "p_value": [0.1, 0.2]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_volcano(bad, show=False)


class TestManhattan:
    def test_returns_figure_axes(self, fitted_results):
        fig, ax = plot_manhattan(fitted_results, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_draws_threshold_line(self, fitted_results):
        fig, ax = plot_manhattan(fitted_results, p_threshold=0.01, show=False)
        line_ys = [ln.get_ydata()[0] for ln in ax.get_lines()]
        assert any(abs(y - (-np.log10(0.01))) < 1e-9 for y in line_ys)
        plt.close(fig)

    def test_with_existing_ax(self, fitted_results):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_manhattan(fitted_results, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close(fig)

    def test_missing_columns_raise(self):
        bad = pd.DataFrame({"mz_bin": [1, 2], "p_value": [0.1, 0.2]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_manhattan(bad, show=False)


class TestDrugComparison:
    @pytest.fixture
    def comparison_df(self, differential_dataset) -> pd.DataFrame:
        X, y, _ = differential_dataset
        a = DifferentialAnalysis(X, y).run()
        b = DifferentialAnalysis(X, y).run()
        return DifferentialAnalysis.compare_drugs({"drugA": a, "drugB": b})

    def test_heatmap_returns_figure_axes(self, comparison_df):
        fig, ax = plot_drug_comparison(comparison_df, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_heatmap_with_existing_ax(self, comparison_df):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_drug_comparison(comparison_df, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close(fig)

    def test_upset_returns_figure_axes(self, comparison_df):
        fig, ax = plot_drug_comparison(comparison_df, kind="upset", show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert len(fig.axes) >= 2  # bar chart + dot matrix
        plt.close(fig)

    def test_upset_ignores_ax_with_warning(self, comparison_df):
        fig_ext, ax_ext = plt.subplots()
        with pytest.warns(UserWarning, match="composite figure"):
            fig, ax = plot_drug_comparison(
                comparison_df, kind="upset", ax=ax_ext, show=False
            )
        assert ax is not ax_ext
        plt.close(fig)
        plt.close(fig_ext)

    def test_upset_handles_empty_matrix(self):
        empty = pd.DataFrame(columns=["drugA", "drugB"])
        fig, ax = plot_drug_comparison(empty, kind="upset", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_kind_raises(self, comparison_df):
        with pytest.raises(ValueError, match="is not a valid"):
            plot_drug_comparison(comparison_df, kind="bogus", show=False)


class TestShowTrue:
    def test_volcano_show_true(self, fitted_results):
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, _ = plot_volcano(fitted_results, show=True)
        plt.close(fig)

    def test_manhattan_show_true(self, fitted_results):
        with pytest.warns(UserWarning, match="non-interactive"):
            fig, _ = plot_manhattan(fitted_results, show=True)
        plt.close(fig)
