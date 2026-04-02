"""Alignment visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..alignment.warping import Warping


def plot_alignment(
    warper: Warping,
    X_original: pd.DataFrame,
    X_aligned: pd.DataFrame | None = None,
    indices: int | list[int] | None = None,
    show_peaks: bool = True,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14, 6),
    alpha: float = 0.7,
):
    """Plot comparison of original vs aligned spectra against reference.

    Parameters
    ----------
    warper : Warping
        Fitted warping transformer.
    X_original : pd.DataFrame
        Original (unaligned) spectra.
    X_aligned : pd.DataFrame, optional
        Aligned spectra. If None, will compute by calling transform().
    indices : int or list of int, optional
        Indices of spectra to plot. If None, plots the first spectrum.
    show_peaks : bool, default=True
        Whether to show detected peaks as vertical lines.
    xlim : tuple of (float, float), optional
        X-axis limits for zooming into specific m/z range.
    figsize : tuple of (float, float), default=(14, 6)
        Figure size in inches.
    alpha : float, default=0.7
        Transparency for spectrum lines.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    axes : array of matplotlib.axes.Axes
        The subplot axes.

    Raises
    ------
    RuntimeError
        If the transformer has not been fitted.
    ValueError
        If any index is out of bounds for the data.
    """
    import matplotlib.pyplot as plt

    indices, mz_axis, X_aligned = _validate_alignment_inputs(
        warper, X_original, X_aligned, indices
    )

    ref_peaks, sample_peaks_dict, aligned_peaks_dict = _compute_peak_positions(
        warper, X_original, X_aligned, indices, mz_axis, show_peaks
    )

    n_spectra = len(indices)
    fig, axes = plt.subplots(n_spectra, 2, figsize=figsize, squeeze=False)

    for plot_idx, spectrum_idx in enumerate(indices):
        original = X_original.iloc[spectrum_idx].to_numpy()
        aligned = X_aligned.iloc[spectrum_idx].to_numpy()

        _plot_before_panel(
            axes[plot_idx, 0],
            mz_axis,
            warper.ref_spec_,
            original,
            spectrum_idx,
            warper.method,
            ref_peaks,
            sample_peaks_dict.get(spectrum_idx),
            xlim,
            alpha,
        )
        _plot_after_panel(
            axes[plot_idx, 1],
            mz_axis,
            warper.ref_spec_,
            aligned,
            spectrum_idx,
            warper.method,
            ref_peaks,
            aligned_peaks_dict.get(spectrum_idx),
            xlim,
            alpha,
        )

    plt.tight_layout()
    return fig, axes


def _validate_alignment_inputs(warper, X_original, X_aligned, indices):
    """Validate inputs and normalize indices for alignment plotting."""
    if not hasattr(warper, "ref_spec_"):
        raise RuntimeError("Warping must be fitted before plotting")

    if X_aligned is None:
        X_aligned = warper.transform(X_original)

    if indices is None:
        indices = [0]
    elif isinstance(indices, int):
        indices = [indices]

    for idx in indices:
        if idx < 0 or idx >= len(X_original):
            raise ValueError(
                f"Index {idx} out of bounds for data with {len(X_original)} samples"
            )

    mz_axis = X_original.columns.to_numpy()
    if not np.issubdtype(mz_axis.dtype, np.number):
        mz_axis = np.arange(len(mz_axis))

    return indices, mz_axis, X_aligned


def _compute_peak_positions(
    warper,
    X_original,
    X_aligned,
    indices,
    mz_axis,
    show_peaks,
):
    """Compute peak positions for reference and selected spectra."""
    ref_peaks = None
    sample_peaks_dict: dict = {}
    aligned_peaks_dict: dict = {}

    if not show_peaks:
        return ref_peaks, sample_peaks_dict, aligned_peaks_dict

    ref_peaks_df = warper.peak_detector.transform(
        pd.DataFrame(warper.ref_spec_[np.newaxis, :], columns=X_original.columns)
    )
    ref_peaks = mz_axis[ref_peaks_df.iloc[0].to_numpy().nonzero()[0]]

    if warper.method != "dtw":
        sample_peaks_df = warper.peak_detector.transform(X_original.iloc[indices])
        aligned_peaks_df = warper.peak_detector.transform(X_aligned.iloc[indices])

        for i, idx in enumerate(indices):
            sample_peaks_dict[idx] = mz_axis[
                sample_peaks_df.iloc[i].to_numpy().nonzero()[0]
            ]
            aligned_peaks_dict[idx] = mz_axis[
                aligned_peaks_df.iloc[i].to_numpy().nonzero()[0]
            ]

    return ref_peaks, sample_peaks_dict, aligned_peaks_dict


def _plot_before_panel(
    ax,
    mz_axis,
    ref_spec,
    original,
    spectrum_idx,
    method,
    ref_peaks,
    sample_peaks,
    xlim,
    alpha,
):
    """Plot the 'before alignment' panel."""
    ax.plot(
        mz_axis,
        ref_spec,
        label="Reference",
        color="black",
        linewidth=1.5,
        alpha=alpha,
    )
    ax.plot(
        mz_axis,
        original,
        label=f"Original (idx={spectrum_idx})",
        color="red",
        linewidth=1,
        alpha=alpha,
    )

    if ref_peaks is not None:
        for peak in ref_peaks:
            ax.axvline(peak, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
        if sample_peaks is not None:
            for peak in sample_peaks:
                ax.axvline(peak, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_ylabel("Intensity")
    ax.set_title(f"Before Alignment ({method} method)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)


def _plot_after_panel(
    ax,
    mz_axis,
    ref_spec,
    aligned,
    spectrum_idx,
    method,
    ref_peaks,
    aligned_peaks,
    xlim,
    alpha,
):
    """Plot the 'after alignment' panel."""
    ax.plot(
        mz_axis,
        ref_spec,
        label="Reference",
        color="black",
        linewidth=1.5,
        alpha=alpha,
    )
    ax.plot(
        mz_axis,
        aligned,
        label=f"Aligned (idx={spectrum_idx})",
        color="blue",
        linewidth=1,
        alpha=alpha,
    )

    if ref_peaks is not None:
        for peak in ref_peaks:
            ax.axvline(peak, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
        if aligned_peaks is not None:
            for peak in aligned_peaks:
                ax.axvline(peak, color="blue", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_title(f"After Alignment ({method} method)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
