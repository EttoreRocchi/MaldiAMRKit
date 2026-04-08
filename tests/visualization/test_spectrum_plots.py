"""Tests for spectrum and pseudogel plotting functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from maldiamrkit.visualization.spectrum_plots import (
    _apply_region_filter,
    _set_pseudogel_xaxis,
    plot_pseudogel,
    plot_spectrum,
)


@pytest.fixture
def binned_maldi_spectrum():
    """Return a MaldiSpectrum that has been preprocessed and binned."""
    from maldiamrkit import MaldiSpectrum
    from tests.conftest import _generate_synthetic_spectrum

    spec = MaldiSpectrum(_generate_synthetic_spectrum(random_state=42), pipeline=None)
    spec.preprocess()
    spec.bin(bin_width=3)
    return spec


@pytest.fixture
def pseudogel_dataset():
    """Minimal MaldiSet-like object for pseudogel tests.

    Uses a mock that provides the attributes plot_pseudogel reads:
    X, antibiotics, get_y_single().
    """
    rng = np.random.default_rng(42)
    n_samples = 10
    n_bins = 50
    columns = [str(2000 + i * 3) for i in range(n_bins)]
    index = [f"s{i}" for i in range(n_samples)]

    X = pd.DataFrame(
        rng.exponential(0.01, (n_samples, n_bins)), columns=columns, index=index
    )
    y = pd.Series(["R"] * 5 + ["S"] * 5, index=index, name="Ceftriaxone")

    ds = MagicMock()
    ds.X = X
    ds.antibiotics = ["Ceftriaxone"]
    ds.get_y_single.return_value = y
    return ds


@pytest.fixture
def single_group_dataset():
    """Dataset where all samples belong to the same class."""
    rng = np.random.default_rng(99)
    n, p = 6, 30
    columns = [str(2000 + i * 3) for i in range(p)]
    index = [f"s{i}" for i in range(n)]

    X = pd.DataFrame(rng.exponential(0.01, (n, p)), columns=columns, index=index)
    y = pd.Series(["R"] * n, index=index, name="Drug")

    ds = MagicMock()
    ds.X = X
    ds.antibiotics = ["Drug"]
    ds.get_y_single.return_value = y
    return ds


@pytest.fixture
def no_antibiotic_dataset():
    """Dataset with no antibiotics defined."""
    ds = MagicMock()
    ds.antibiotics = None
    return ds


@pytest.fixture
def feature_matrix():
    """Simple feature matrix for _apply_region_filter tests."""
    rng = np.random.default_rng(42)
    columns = [str(float(2000 + i * 3)) for i in range(100)]
    index = [f"s{i}" for i in range(5)]
    return pd.DataFrame(rng.random((5, 100)), columns=columns, index=index)


class TestPlotSpectrum:
    """Tests for the plot_spectrum function."""

    def test_returns_axes_type(self, binned_maldi_spectrum):
        """Verify plot_spectrum returns a matplotlib Axes."""
        ax = plot_spectrum(binned_maldi_spectrum, binned=True)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_binned_true_uses_barplot(self, binned_maldi_spectrum):
        """Verify binned=True path executes without error."""
        ax = plot_spectrum(binned_maldi_spectrum, binned=True)
        assert isinstance(ax, Axes)
        assert "(binned)" in ax.get_title()
        plt.close("all")

    def test_binned_false_uses_lineplot(self, binned_maldi_spectrum):
        """Verify binned=False path uses line plot."""
        ax = plot_spectrum(binned_maldi_spectrum, binned=False)
        assert isinstance(ax, Axes)
        assert "(binned)" not in ax.get_title()
        plt.close("all")

    def test_custom_ax_reused(self, binned_maldi_spectrum):
        """Verify that a user-provided Axes is reused."""
        fig, ax_ext = plt.subplots()
        ax = plot_spectrum(binned_maldi_spectrum, binned=False, ax=ax_ext)
        assert ax is ax_ext
        plt.close("all")

    def test_title_contains_spectrum_id(self, binned_maldi_spectrum):
        """Verify that the plot title contains the spectrum ID."""
        ax = plot_spectrum(binned_maldi_spectrum, binned=True)
        assert binned_maldi_spectrum.id in ax.get_title()
        plt.close("all")

    def test_kwargs_forwarded_no_error(self, binned_maldi_spectrum):
        """Verify extra kwargs are forwarded to the plotting function."""
        ax = plot_spectrum(binned_maldi_spectrum, binned=False, color="red")
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_ylim_starts_at_zero(self, binned_maldi_spectrum):
        """Verify y-axis lower limit is 0."""
        ax = plot_spectrum(binned_maldi_spectrum, binned=False)
        assert ax.get_ylim()[0] == 0
        plt.close("all")


class TestPlotPseudogel:
    """Tests for the plot_pseudogel function."""

    def test_returns_fig_and_axes_array(self, pseudogel_dataset):
        """Verify return types are (Figure, ndarray)."""
        fig, axes = plot_pseudogel(pseudogel_dataset, show=False)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        plt.close("all")

    def test_default_antibiotic_picks_first(self, pseudogel_dataset):
        """Verify antibiotic=None defaults to the first antibiotic."""
        fig, axes = plot_pseudogel(pseudogel_dataset, show=False)
        # Should not raise; antibiotics[0] is "Ceftriaxone"
        assert fig is not None
        plt.close("all")

    def test_no_antibiotic_raises_ValueError(self, no_antibiotic_dataset):
        """Verify ValueError when no antibiotic is defined."""
        with pytest.raises(ValueError, match="not defined"):
            plot_pseudogel(no_antibiotic_dataset, show=False)

    def test_single_group_wraps_axes(self, single_group_dataset):
        """Verify single-group plot normalizes axes to ndarray."""
        fig, axes = plot_pseudogel(single_group_dataset, show=False)
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 1
        plt.close("all")

    def test_two_groups_two_panels(self, pseudogel_dataset):
        """Verify two-group dataset produces two subplot panels."""
        fig, axes = plot_pseudogel(pseudogel_dataset, show=False)
        assert len(axes) == 2
        plt.close("all")

    def test_log_scale_false(self, pseudogel_dataset):
        """Verify log_scale=False does not raise."""
        fig, axes = plot_pseudogel(pseudogel_dataset, log_scale=False, show=False)
        assert fig is not None
        plt.close("all")

    def test_sort_by_intensity_false(self, pseudogel_dataset):
        """Verify sort_by_intensity=False does not raise."""
        fig, axes = plot_pseudogel(
            pseudogel_dataset, sort_by_intensity=False, show=False
        )
        assert fig is not None
        plt.close("all")

    def test_show_false_no_error(self, pseudogel_dataset):
        """Verify show=False returns without calling plt.show()."""
        fig, axes = plot_pseudogel(pseudogel_dataset, show=False)
        assert fig is not None
        plt.close("all")

    def test_custom_title_suptitle(self, pseudogel_dataset):
        """Verify title kwarg sets figure suptitle."""
        fig, axes = plot_pseudogel(pseudogel_dataset, title="My Gel", show=False)
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "My Gel"
        plt.close("all")

    def test_custom_figsize(self, pseudogel_dataset):
        """Verify custom figsize is applied."""
        fig, axes = plot_pseudogel(pseudogel_dataset, figsize=(12, 8), show=False)
        w, h = fig.get_size_inches()
        assert pytest.approx(w, abs=0.5) == 12
        assert pytest.approx(h, abs=0.5) == 8
        plt.close("all")


class TestApplyRegionFilter:
    """Tests for the _apply_region_filter helper."""

    def test_none_returns_unchanged(self, feature_matrix):
        """Verify regions=None returns the original DataFrame."""
        result = _apply_region_filter(feature_matrix, None)
        pd.testing.assert_frame_equal(result, feature_matrix)

    def test_single_tuple_treated_as_one_region(self, feature_matrix):
        """Verify a single (min, max) tuple is treated as one region."""
        result = _apply_region_filter(feature_matrix, (2000.0, 2030.0))
        assert result.shape[1] > 0
        assert result.shape[1] < feature_matrix.shape[1]

    def test_list_of_one_region(self, feature_matrix):
        """Verify list with one region tuple works."""
        result = _apply_region_filter(feature_matrix, [(2000.0, 2030.0)])
        assert result.shape[1] > 0

    def test_multiple_regions_with_blank_columns(self, feature_matrix):
        """Verify blank separator columns are inserted between regions."""
        result = _apply_region_filter(
            feature_matrix, [(2000.0, 2030.0), (2200.0, 2230.0)]
        )
        blank_cols = [c for c in result.columns if str(c).startswith("_blank_")]
        assert len(blank_cols) == 1

    def test_min_gt_max_raises_ValueError(self, feature_matrix):
        """Verify ValueError when min_mz > max_mz."""
        with pytest.raises(ValueError, match="Invalid region"):
            _apply_region_filter(feature_matrix, (5000.0, 2000.0))

    def test_no_mz_in_region_raises_ValueError(self, feature_matrix):
        """Verify ValueError when region contains no m/z values."""
        with pytest.raises(ValueError, match="No m/z values"):
            _apply_region_filter(feature_matrix, (99000.0, 99999.0))


class TestSetPseudogelXaxis:
    """Tests for the _set_pseudogel_xaxis helper."""

    def test_xticks_count_capped_at_10(self):
        """Verify n_ticks is at most 10."""
        fig, ax = plt.subplots()
        axes = np.array([ax])
        X = pd.DataFrame(
            np.zeros((2, 50)),
            columns=[str(2000 + i * 3) for i in range(50)],
        )
        _set_pseudogel_xaxis(axes, X)
        xticks = axes[-1].get_xticks()
        assert len(xticks) <= 10
        plt.close("all")

    def test_blank_columns_get_empty_labels(self):
        """Verify _blank_ columns produce empty tick labels."""
        fig, ax = plt.subplots()
        axes = np.array([ax])
        cols = (
            [str(2000 + i * 3) for i in range(10)]
            + ["_blank_1"]
            + [str(2200 + i * 3) for i in range(10)]
        )
        X = pd.DataFrame(np.zeros((2, len(cols))), columns=cols)
        _set_pseudogel_xaxis(axes, X)
        labels = [t.get_text() for t in axes[-1].get_xticklabels()]
        # Any label corresponding to a _blank_ column should be ""
        for lbl in labels:
            if "_blank_" in lbl:
                pytest.fail("_blank_ column should have been replaced with ''")
        plt.close("all")
