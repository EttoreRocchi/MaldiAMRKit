"""Tests for alignment visualization (plot_alignment)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.alignment.warping import Warping
from maldiamrkit.visualization.alignment_plots import plot_alignment


@pytest.fixture
def small_binned_shifted():
    """Small binned dataset with systematic shifts for alignment testing."""
    rng = np.random.default_rng(42)
    n_samples, n_bins = 5, 100
    columns = [str(2000 + i * 3) for i in range(n_bins)]
    base = rng.exponential(0.01, n_bins)
    base[[20, 50, 80]] += 1.0

    data = np.zeros((n_samples, n_bins))
    for i in range(n_samples):
        shift = (i % 3) - 1  # -1, 0, 1
        data[i] = np.roll(base, shift) + rng.normal(0, 0.001, n_bins)
    data = np.maximum(data, 0)
    return pd.DataFrame(
        data, columns=columns, index=[f"s{i}" for i in range(n_samples)]
    )


@pytest.fixture
def fitted_warper(small_binned_shifted):
    """Warping transformer fitted on the small shifted dataset."""
    warper = Warping(method="shift", reference="median")
    warper.fit(small_binned_shifted)
    return warper


class TestPlotAlignment:
    """Tests for the plot_alignment function."""

    def test_unfitted_raises(self, small_binned_shifted):
        """Verify RuntimeError when warper is not fitted."""
        warper = Warping()
        with pytest.raises(RuntimeError, match="fitted"):
            plot_alignment(warper, small_binned_shifted)

    def test_auto_transform(self, fitted_warper, small_binned_shifted):
        """Verify X_aligned=None triggers transform."""
        fig, axes = plot_alignment(fitted_warper, small_binned_shifted)
        assert fig is not None
        assert axes.shape == (1, 2)

    def test_int_index_normalized(self, fitted_warper, small_binned_shifted):
        """Verify int index is normalized to list."""
        fig, axes = plot_alignment(fitted_warper, small_binned_shifted, indices=1)
        assert axes.shape == (1, 2)

    def test_show_peaks_false(self, fitted_warper, small_binned_shifted):
        """Verify show_peaks=False skips peak computation."""
        fig, axes = plot_alignment(
            fitted_warper, small_binned_shifted, show_peaks=False
        )
        assert fig is not None

    def test_with_xlim(self, fitted_warper, small_binned_shifted):
        """Verify xlim parameter is applied (L200, L244)."""
        fig, axes = plot_alignment(
            fitted_warper, small_binned_shifted, xlim=(2050.0, 2200.0)
        )
        # Check both before and after panels
        assert axes[0, 0].get_xlim()[0] == pytest.approx(2050.0)
        assert axes[0, 1].get_xlim()[0] == pytest.approx(2050.0)

    def test_out_of_bounds_raises(self, fitted_warper, small_binned_shifted):
        """Verify ValueError for out-of-bounds indices."""
        with pytest.raises(ValueError, match="out of bounds"):
            plot_alignment(fitted_warper, small_binned_shifted, indices=99)

    def test_multiple_indices(self, fitted_warper, small_binned_shifted):
        """Verify multiple indices produce correct subplot grid."""
        fig, axes = plot_alignment(fitted_warper, small_binned_shifted, indices=[0, 1])
        assert axes.shape == (2, 2)
