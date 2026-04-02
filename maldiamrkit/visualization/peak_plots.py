"""Peak detection visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..detection.peak_detector import MaldiPeakDetector


def plot_peaks(
    detector: MaldiPeakDetector,
    X: pd.DataFrame,
    indices: int | list[int] | None = None,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14, 6),
    alpha: float = 0.7,
):
    """Plot detected peaks overlaid on original spectra.

    Parameters
    ----------
    detector : MaldiPeakDetector
        Fitted peak detector.
    X : pd.DataFrame or pd.Series
        Input spectra with shape (n_samples, n_bins).
    indices : int or list of int, optional
        Indices of spectra to plot. If None, plots the first spectrum.
    xlim : tuple of (float, float), optional
        X-axis limits for zooming into specific m/z range.
    figsize : tuple of (float, float), default=(14, 6)
        Figure size in inches.
    alpha : float, default=0.7
        Transparency for spectrum lines.

    Raises
    ------
    ValueError
        If any index in ``indices`` is out of bounds for the data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    axes : Axes or array of Axes
        The subplot axes.
    """
    import matplotlib.pyplot as plt

    X, indices, mz_axis = _normalize_peak_inputs(X, indices)

    # Detect peaks for selected spectra using public API
    peaks_df = detector.transform(X.iloc[indices])

    n_spectra = len(indices)
    fig, axes = plt.subplots(n_spectra, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for plot_idx, spectrum_idx in enumerate(indices):
        row = X.iloc[spectrum_idx].values
        peak_mask = peaks_df.iloc[plot_idx].to_numpy() != 0
        peaks = np.where(peak_mask)[0]
        _draw_peak_panel(
            axes[plot_idx],
            mz_axis,
            row,
            peaks,
            detector.method,
            spectrum_idx,
            xlim,
            alpha,
        )

    plt.tight_layout()

    if n_spectra == 1:
        return fig, axes[0]
    return fig, axes


def _normalize_peak_inputs(X, indices):
    """Normalize inputs for peak plotting: handle Series, indices, mz_axis."""
    if isinstance(X, pd.Series):
        X = X.to_frame().T

    if indices is None:
        indices = [0]
    elif isinstance(indices, int):
        indices = [indices]

    for idx in indices:
        if idx < 0 or idx >= len(X):
            raise ValueError(
                f"Index {idx} out of bounds for data with {len(X)} samples"
            )

    mz_axis = X.columns.to_numpy()
    if not np.issubdtype(mz_axis.dtype, np.number):
        mz_axis = np.arange(len(mz_axis))

    return X, indices, mz_axis


def _draw_peak_panel(ax, mz_axis, row, peaks, method, spectrum_idx, xlim, alpha):
    """Draw a single peak detection panel on the given axes."""
    ax.plot(mz_axis, row, color="black", linewidth=1, alpha=alpha, label="Spectrum")

    if len(peaks) > 0:
        ax.scatter(
            mz_axis[peaks],
            row[peaks],
            color="red",
            s=50,
            zorder=5,
            label=f"Peaks (n={len(peaks)})",
            marker="o",
        )
        for peak in peaks:
            ax.axvline(
                mz_axis[peak],
                color="red",
                linestyle="--",
                alpha=0.3,
                linewidth=0.8,
            )

    ax.set_xlabel("m/z" if np.issubdtype(mz_axis.dtype, np.number) else "Index")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Peak Detection (method={method}, idx={spectrum_idx})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if xlim:
        ax.set_xlim(xlim)
