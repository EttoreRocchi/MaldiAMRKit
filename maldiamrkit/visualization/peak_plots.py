"""Peak detection visualization functions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..detection.peak_detector import MaldiPeakDetector


def _show_with_warning(show: bool) -> None:
    """Match the ``show=True`` pattern used in sibling plot modules."""
    import matplotlib
    import matplotlib.pyplot as plt

    if show:
        if not matplotlib.is_interactive():
            warnings.warn(
                "matplotlib is using a non-interactive backend; "
                "plt.show() may not display the figure",
                UserWarning,
                stacklevel=3,
            )
        plt.show()


def plot_peaks(
    detector: MaldiPeakDetector,
    X: pd.DataFrame,
    indices: int | list[int] | str | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.7,
    show_axvlines: bool = False,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Any]:
    """Plot detected peaks overlaid on original spectra.

    Parameters
    ----------
    detector : MaldiPeakDetector
        Fitted peak detector.
    X : pd.DataFrame or pd.Series
        Input spectra with shape (n_samples, n_bins).
    indices : int, list of int, "all", or None, default=None
        Indices of spectra to plot.  ``None`` plots the first spectrum
        (unchanged from prior behaviour); ``"all"`` plots every
        spectrum in ``X``.
    xlim : tuple of (float, float), optional
        X-axis limits for zooming into a specific m/z range.
    figsize : tuple of (float, float), optional
        Figure size in inches.  When ``None``, defaults to
        ``(14, 3 * n_spectra)`` so stacking many spectra gives a
        proportionally tall figure.  Ignored when ``ax`` is provided.
    alpha : float, default=0.7
        Transparency for spectrum lines.
    show_axvlines : bool, default=False
        If True, draw a dashed vertical line at each detected peak.
        Default off because the scatter markers alone already mark
        peak positions, and the vertical lines become visually noisy
        with dense peak sets.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on.  Only honoured when exactly one
        spectrum is plotted; multi-panel calls always create a new
        Figure.
    show : bool, default=True
        Call ``plt.show()`` at the end.

    Raises
    ------
    ValueError
        If any index in ``indices`` is out of bounds for the data, or
        if an ``ax`` is provided together with multiple spectra.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure holding the peak panels.
    axes : Axes or ndarray of Axes
        A single ``Axes`` when ``indices`` resolves to one spectrum,
        otherwise an ndarray of ``Axes`` (one per spectrum, stacked
        vertically with shared x-axis).
    """
    import matplotlib.pyplot as plt

    X, indices, mz_axis = _normalize_peak_inputs(X, indices)
    n_spectra = len(indices)

    # Detect peaks for selected spectra using the public API
    peaks_df = detector.transform(X.iloc[indices])

    if ax is not None and n_spectra > 1:
        raise ValueError(
            "plot_peaks(ax=...) is only supported when a single spectrum is "
            f"plotted (got {n_spectra})."
        )

    if ax is not None:
        fig = ax.get_figure()
        axes = np.array([ax])
    else:
        if figsize is None:
            figsize = (14, max(3.0, 3.0 * n_spectra))
        fig, axes = plt.subplots(
            n_spectra, 1, figsize=figsize, squeeze=False, sharex=True
        )
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
            show_axvlines,
        )

    if ax is None:
        fig.tight_layout()

    _show_with_warning(show)

    if n_spectra == 1:
        return fig, axes[0]
    return fig, axes


def _normalize_peak_inputs(X, indices):
    """Normalize inputs for peak plotting: handle Series, indices, mz_axis."""
    if isinstance(X, pd.Series):
        X = X.to_frame().T

    if indices is None:
        indices = [0]
    elif isinstance(indices, str):
        if indices.lower() == "all":
            indices = list(range(len(X)))
        else:
            raise ValueError(f"String `indices` must be 'all', got {indices!r}.")
    elif isinstance(indices, int):
        indices = [indices]
    else:
        indices = list(indices)

    for idx in indices:
        if idx < 0 or idx >= len(X):
            raise ValueError(
                f"Index {idx} out of bounds for data with {len(X)} samples"
            )

    mz_axis = X.columns.to_numpy()
    if not np.issubdtype(mz_axis.dtype, np.number):
        mz_axis = np.arange(len(mz_axis))

    return X, indices, mz_axis


def _draw_peak_panel(
    ax,
    mz_axis,
    row,
    peaks,
    method,
    spectrum_idx,
    xlim,
    alpha,
    show_axvlines,
):
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
        if show_axvlines:
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
