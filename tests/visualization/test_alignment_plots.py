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
        fig, axes = plot_alignment(fitted_warper, small_binned_shifted, show=False)
        assert fig is not None
        assert axes.shape == (1, 2)

    def test_int_index_normalized(self, fitted_warper, small_binned_shifted):
        """Verify int index is normalized to list."""
        fig, axes = plot_alignment(
            fitted_warper, small_binned_shifted, indices=1, show=False
        )
        assert axes.shape == (1, 2)

    def test_show_peaks_false(self, fitted_warper, small_binned_shifted):
        """Verify show_peaks=False skips peak computation."""
        fig, axes = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            show_peaks=False,
            show=False,
        )
        assert fig is not None

    def test_with_xlim(self, fitted_warper, small_binned_shifted):
        """Verify xlim parameter is applied (L200, L244)."""
        fig, axes = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            xlim=(2050.0, 2200.0),
            show=False,
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
        fig, axes = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            indices=[0, 1],
            show=False,
        )
        assert axes.shape == (2, 2)

    def test_suptitle_default_names_method(self, fitted_warper, small_binned_shifted):
        """Figure suptitle mentions the warping method."""
        fig, _axes = plot_alignment(fitted_warper, small_binned_shifted, show=False)
        assert fitted_warper.method in fig._suptitle.get_text()

    def test_suptitle_override(self, fitted_warper, small_binned_shifted):
        """title= overrides the default suptitle."""
        fig, _axes = plot_alignment(
            fitted_warper, small_binned_shifted, title="Custom Title", show=False
        )
        assert fig._suptitle.get_text() == "Custom Title"

    def test_sharex_sharey(self, fitted_warper, small_binned_shifted):
        """Multi-row panels share both x and y axes."""
        fig, axes = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            indices=[0, 1],
            show=False,
        )
        # Before (column 0) and After (column 1) panels share both axes.
        assert axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[0, 1])
        assert axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 0])

    def test_figsize_auto_scales_with_n_spectra(
        self, fitted_warper, small_binned_shifted
    ):
        """Default figsize grows with n_spectra."""
        fig1, _ = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            indices=[0],
            show=False,
        )
        fig3, _ = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            indices=[0, 1, 2],
            show=False,
        )
        # Height should grow; width stays the same.
        assert fig3.get_size_inches()[1] > fig1.get_size_inches()[1]

    def test_sample_peaks_opt_in(self, fitted_warper, small_binned_shifted):
        """show_sample_peaks=False draws only reference axvlines; True adds sample ones."""
        fig_off, axes_off = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            show_sample_peaks=False,
            show=False,
        )
        fig_on, axes_on = plot_alignment(
            fitted_warper,
            small_binned_shifted,
            show_sample_peaks=True,
            show=False,
        )

        # Both should have reference peak axvlines; `on` should have
        # strictly more dashed vertical lines.
        def _n_dashed(ax):
            return len([ln for ln in ax.lines if ln.get_linestyle() == "--"])

        assert _n_dashed(axes_on[0, 0]) >= _n_dashed(axes_off[0, 0])

    def test_dtw_method_gets_peak_markers(self, small_binned_shifted):
        """DTW-aligned spectra also receive sample-peak markers when enabled."""
        from maldiamrkit.alignment.warping import Warping

        dtw = Warping(method="dtw", reference="median")
        dtw.fit(small_binned_shifted)
        fig, axes = plot_alignment(
            dtw,
            small_binned_shifted,
            show_sample_peaks=True,
            show=False,
        )
        # With sample peaks enabled, the after panel should have at least
        # as many dashed vlines as the before panel (per-sample + ref peaks).
        dashed_after = [ln for ln in axes[0, 1].lines if ln.get_linestyle() == "--"]
        assert len(dashed_after) > 0
