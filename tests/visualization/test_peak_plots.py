"""Tests for peak detection visualization (plot_peaks)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.detection.peak_detector import MaldiPeakDetector
from maldiamrkit.visualization.peak_plots import plot_peaks


@pytest.fixture
def small_binned():
    """Small synthetic binned dataset (5 samples x 100 bins)."""
    rng = np.random.default_rng(42)
    data = rng.exponential(0.01, (5, 100))
    # Add peaks at known positions
    for i in range(5):
        data[i, [20, 50, 80]] += 1.0
    columns = [str(2000 + i * 3) for i in range(100)]
    return pd.DataFrame(data, columns=columns, index=[f"s{i}" for i in range(5)])


@pytest.fixture
def detector():
    """Fitted peak detector."""
    return MaldiPeakDetector(method="local", binary=True)


class TestPlotPeaks:
    """Tests for the plot_peaks function."""

    def test_default_indices_plots_first(self, detector, small_binned):
        """Verify indices=None defaults to plotting first spectrum."""
        fig, ax = plot_peaks(detector, small_binned, show=False)
        assert fig is not None

    def test_series_input(self, detector, small_binned):
        """Verify pd.Series input is handled."""
        row = small_binned.iloc[0]
        fig, ax = plot_peaks(detector, row, show=False)
        assert fig is not None

    def test_int_index_normalized(self, detector, small_binned):
        """Verify int index is normalized to list."""
        fig, ax = plot_peaks(detector, small_binned, indices=2, show=False)
        assert fig is not None

    def test_multiple_indices(self, detector, small_binned):
        """Verify multiple indices produce multiple subplots."""
        fig, axes = plot_peaks(detector, small_binned, indices=[0, 2], show=False)
        assert len(axes) == 2

    def test_out_of_bounds_raises(self, detector, small_binned):
        """Verify out-of-bounds index raises ValueError."""
        with pytest.raises(ValueError, match="out of bounds"):
            plot_peaks(detector, small_binned, indices=10, show=False)

    def test_with_xlim(self, detector, small_binned):
        """Verify xlim parameter is applied."""
        fig, ax = plot_peaks(detector, small_binned, xlim=(2050.0, 2200.0), show=False)
        xlim = ax.get_xlim()
        assert xlim[0] == pytest.approx(2050.0)
        assert xlim[1] == pytest.approx(2200.0)
