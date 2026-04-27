"""Alignment visualization functions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..alignment.warping import Warping


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


def plot_alignment(
    warper: Warping,
    X_original: pd.DataFrame,
    X_aligned: pd.DataFrame | None = None,
    indices: int | list[int] | None = None,
    *,
    show_peaks: bool = True,
    show_sample_peaks: bool = False,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.7,
    color_reference: str = "black",
    color_original: str = "red",
    color_aligned: str = "blue",
    title: str | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
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
        Whether to draw reference peak positions (vertical dashed lines).
        These are the calibration markers used to judge alignment
        quality and are on by default.
    show_sample_peaks : bool, default=False
        If True, additionally draw per-sample (and per-aligned) peak
        positions as dashed vertical lines.  Off by default because
        dense peak sets clutter the panel.
    xlim : tuple of (float, float), optional
        X-axis limits for zooming into specific m/z range.
    figsize : tuple of (float, float), optional
        Figure size in inches.  When ``None``, defaults to
        ``(14, 3 * n_spectra)``.
    alpha : float, default=0.7
        Transparency for spectrum lines.
    color_reference : str, default="black"
        Line colour for the reference spectrum.
    color_original : str, default="red"
        Line colour for the original (before-alignment) spectrum.
    color_aligned : str, default="blue"
        Line colour for the aligned (after-alignment) spectrum.
    title : str, optional
        Figure-level title (suptitle). Defaults to
        ``f"Warping ({warper.method})"``.
    show : bool, default=True
        Call ``plt.show()`` at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    axes : ndarray of matplotlib.axes.Axes
        2-D array of shape ``(n_spectra, 2)``: column 0 = before, 1 = after.

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

    peaks_ctx = _compute_peak_positions(
        warper,
        X_original,
        X_aligned,
        indices,
        mz_axis,
        show_peaks,
        show_sample_peaks,
    )

    n_spectra = len(indices)
    if figsize is None:
        figsize = (14, max(3.0, 3.0 * n_spectra))

    fig, axes = plt.subplots(
        n_spectra,
        2,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for plot_idx, spectrum_idx in enumerate(indices):
        original = X_original.iloc[spectrum_idx].to_numpy()
        aligned = X_aligned.iloc[spectrum_idx].to_numpy()
        is_bottom = plot_idx == n_spectra - 1

        _plot_alignment_panel(
            axes[plot_idx, 0],
            mz_axis,
            warper.ref_spec_,
            original,
            spectrum_idx,
            peaks_ctx["ref"],
            peaks_ctx["sample"].get(spectrum_idx),
            xlim,
            alpha,
            label="Original",
            sample_color=color_original,
            ref_color=color_reference,
            column_title="Before" if plot_idx == 0 else None,
            ylabel=f"idx={spectrum_idx}",
            is_bottom=is_bottom,
            show_sample_peaks=show_sample_peaks,
        )
        _plot_alignment_panel(
            axes[plot_idx, 1],
            mz_axis,
            warper.ref_spec_,
            aligned,
            spectrum_idx,
            peaks_ctx["ref"],
            peaks_ctx["aligned"].get(spectrum_idx),
            xlim,
            alpha,
            label="Aligned",
            sample_color=color_aligned,
            ref_color=color_reference,
            column_title="After" if plot_idx == 0 else None,
            ylabel=None,
            is_bottom=is_bottom,
            show_sample_peaks=show_sample_peaks,
        )

    fig.suptitle(title or f"Warping ({warper.method})")
    fig.tight_layout()

    _show_with_warning(show)

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
    show_sample_peaks,
):
    """Compute peak positions for reference and (optionally) selected spectra.

    Returns a dict ``{"ref": np.ndarray | None, "sample": dict[int, np.ndarray],
    "aligned": dict[int, np.ndarray]}``.  DTW outputs are detected the same
    way as any other aligned spectra so users see peak markers regardless
    of warping method.
    """
    ctx: dict = {"ref": None, "sample": {}, "aligned": {}}
    if not show_peaks:
        return ctx

    ref_peaks_df = warper.peak_detector.transform(
        pd.DataFrame(warper.ref_spec_[np.newaxis, :], columns=X_original.columns)
    )
    ctx["ref"] = mz_axis[ref_peaks_df.iloc[0].to_numpy().nonzero()[0]]

    if show_sample_peaks:
        sample_peaks_df = warper.peak_detector.transform(X_original.iloc[indices])
        aligned_peaks_df = warper.peak_detector.transform(X_aligned.iloc[indices])
        for i, idx in enumerate(indices):
            ctx["sample"][idx] = mz_axis[
                sample_peaks_df.iloc[i].to_numpy().nonzero()[0]
            ]
            ctx["aligned"][idx] = mz_axis[
                aligned_peaks_df.iloc[i].to_numpy().nonzero()[0]
            ]

    return ctx


def _plot_alignment_panel(
    ax,
    mz_axis,
    ref_spec,
    sample_spec,
    spectrum_idx,
    ref_peaks,
    sample_peaks,
    xlim,
    alpha,
    *,
    label,
    sample_color,
    ref_color,
    column_title,
    ylabel,
    is_bottom,
    show_sample_peaks,
):
    """Draw one (before or after) panel of the alignment plot."""
    ax.plot(
        mz_axis,
        ref_spec,
        label="Reference",
        color=ref_color,
        linewidth=1.5,
        alpha=alpha,
    )
    ax.plot(
        mz_axis,
        sample_spec,
        label=f"{label} (idx={spectrum_idx})",
        color=sample_color,
        linewidth=1,
        alpha=alpha,
    )

    if ref_peaks is not None:
        for peak in ref_peaks:
            ax.axvline(peak, color=ref_color, linestyle="--", alpha=0.3, linewidth=0.8)
        if show_sample_peaks and sample_peaks is not None:
            for peak in sample_peaks:
                ax.axvline(
                    peak,
                    color=sample_color,
                    linestyle="--",
                    alpha=0.3,
                    linewidth=0.8,
                )

    if column_title is not None:
        ax.set_title(column_title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if is_bottom:
        ax.set_xlabel("m/z (Da)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
