"""Spectral alignment and warping transformers for binned spectra."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin
from tslearn.metrics import dtw_path

from ..detection.peak_detector import MaldiPeakDetector


class Warping(BaseEstimator, TransformerMixin):
    """
    Align MALDI-TOF spectra to a reference using different strategies.

    Supports multiple alignment methods for correcting mass calibration drift
    in binned spectra.

    Parameters
    ----------
    peak_detector : MaldiPeakDetector, optional
        Peak detector used to find peaks in spectra. If None, a default
        detector is created with binary=True and prominence=1e-5.
    reference : str or int, default="median"
        How to choose the reference spectrum:
        - "median" : median spectrum across all samples
        - int : use that row index as reference
    method : str, default="shift"
        Alignment method:
        - "shift" : global median shift
        - "linear" : least-squares linear transform
        - "piecewise" : local median shifts across segments
        - "dtw" : dynamic time warping
    n_segments : int, default=5
        Number of segments for piecewise warping.
    max_shift : int, default=50
        Max allowed shift in bins (for shift/linear modes).
    dtw_radius : int, default=10
        Radius constraint for DTW to limit warping path search space.
    smooth_sigma : float, default=2.0
        Gaussian smoothing parameter for piecewise segment shifts.
    min_reference_peaks : int, default=5
        Minimum number of peaks expected in reference for quality check.
    n_jobs : int, default=1
        Number of parallel jobs for transform. Use -1 for all available
        cores, 1 for sequential processing. Parallelization is particularly
        beneficial for the "dtw" method which is CPU-intensive.

    Attributes
    ----------
    ref_spec_ : np.ndarray
        The fitted reference spectrum (stored after fit()).

    Examples
    --------
    >>> from maldiamrkit.alignment import Warping
    >>> warper = Warping(method="shift")
    >>> warper.fit(X_train)
    >>> X_aligned = warper.transform(X_test)
    """

    def __init__(
        self,
        peak_detector: MaldiPeakDetector | None = None,
        reference: str | int = "median",
        method: str = "shift",
        n_segments: int = 5,
        max_shift: int = 50,
        dtw_radius: int = 10,
        smooth_sigma: float = 2.0,
        min_reference_peaks: int = 5,
        n_jobs: int = 1,
    ) -> None:
        self.peak_detector = peak_detector or MaldiPeakDetector(
            binary=True, prominence=1e-5
        )
        self.reference = reference
        self.method = method
        self.n_segments = n_segments
        self.max_shift = max_shift
        self.dtw_radius = dtw_radius
        self.smooth_sigma = smooth_sigma
        self.min_reference_peaks = min_reference_peaks
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by selecting or computing the reference spectrum.

        Parameters
        ----------
        X : pd.DataFrame
            Input spectra with shape (n_samples, n_bins).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : Warping
            Fitted transformer.

        Raises
        ------
        ValueError
            If the input DataFrame is empty, the reference index is out of
            bounds, the reference specifier is unsupported, the warping
            method is unknown, or parameters are invalid.
        """
        if X.empty:
            raise ValueError("Input DataFrame X is empty")

        if self.reference == "median":
            self.ref_spec_ = X.median(axis=0).to_numpy()
        elif isinstance(self.reference, int):
            if self.reference < 0 or self.reference >= len(X):
                raise ValueError(
                    f"Reference index {self.reference} is out of bounds "
                    f"for X with {len(X)} samples"
                )
            self.ref_spec_ = X.iloc[self.reference].to_numpy()
        else:
            raise ValueError(
                f"Unsupported reference specifier: {self.reference}. "
                f"Must be 'median' or int."
            )

        # Validate parameters
        if self.method not in ["shift", "linear", "piecewise", "dtw"]:
            raise ValueError(
                f"Unknown warping method: {self.method}. "
                f"Must be one of: shift, linear, piecewise, dtw"
            )
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {self.n_segments}")
        if self.max_shift < 0:
            raise ValueError(f"max_shift must be >= 0, got {self.max_shift}")

        # Validate reference quality
        self._validate_reference_quality(X)

        return self

    def _validate_reference_quality(self, X: pd.DataFrame):
        """Validate that the reference spectrum has sufficient quality."""
        ref_peaks_df = self.peak_detector.transform(
            pd.DataFrame(self.ref_spec_[np.newaxis, :], columns=X.columns)
        )
        n_peaks = ref_peaks_df.iloc[0].to_numpy().nonzero()[0].size

        if n_peaks < self.min_reference_peaks:
            warnings.warn(
                f"Reference spectrum has only {n_peaks} peaks detected. "
                f"Expected at least {self.min_reference_peaks}. "
                f"This may result in poor alignment quality. "
                f"Consider adjusting peak detection parameters or "
                f"choosing a different reference.",
                UserWarning,
                stacklevel=2,
            )

    def _shift_only(self, row, peaks, ref_peaks):
        """Apply global median shift alignment with zero-padding."""
        if len(peaks) == 0 or len(ref_peaks) == 0:
            return row

        shifts = []
        for p in peaks:
            nearest = ref_peaks[np.argmin(np.abs(ref_peaks - p))]
            shifts.append(nearest - p)

        shift = int(np.median(shifts)) if shifts else 0
        shift = np.clip(shift, -self.max_shift, self.max_shift)

        # Apply shift with zero-padding
        if shift > 0:
            # Shift right: pad left with zeros
            aligned = np.zeros_like(row)
            aligned[shift:] = row[:-shift]
        elif shift < 0:
            # Shift left: pad right with zeros
            aligned = np.zeros_like(row)
            aligned[:shift] = row[-shift:]
        else:
            aligned = row.copy()

        return aligned

    def _linear_fit(self, row, peaks, ref_peaks, mz_axis):
        """Apply linear transformation alignment using least squares fit."""
        if len(peaks) < 2 or len(ref_peaks) < 2:
            # Not enough peaks for linear fit, fall back to shift
            return self._shift_only(row, peaks, ref_peaks)

        # Match each peak to nearest reference peak
        pairs = [(p, ref_peaks[np.argmin(np.abs(ref_peaks - p))]) for p in peaks]
        sample = np.array([p[0] for p in pairs])
        ref = np.array([p[1] for p in pairs])

        # Fit linear transformation: ref = a * sample + b
        A = np.vstack([sample, np.ones_like(sample)]).T
        a, b = np.linalg.lstsq(A, ref, rcond=None)[0]

        # Apply inverse transformation to align
        new_positions = a * mz_axis + b

        # Ensure monotonicity for interpolation
        return self._monotonic_interp(mz_axis, new_positions, row)

    def _piecewise(self, row, peaks, ref_peaks, mz_axis):
        """Apply piecewise linear alignment with smoothed local shifts."""
        if len(peaks) == 0 or len(ref_peaks) == 0:
            return row

        # Match each peak to nearest reference peak
        pairs = [(p, ref_peaks[np.argmin(np.abs(ref_peaks - p))]) for p in peaks]
        sample = np.array([p[0] for p in pairs])
        ref = np.array([p[1] for p in pairs])

        # Divide peak positions into segments
        quantiles = np.linspace(0, 1, self.n_segments + 1)
        boundaries = np.quantile(sample, quantiles)

        seg_x, seg_shift = [], []
        for q in range(self.n_segments):
            # Include upper boundary for last segment
            if q == self.n_segments - 1:
                mask = (sample >= boundaries[q]) & (sample <= boundaries[q + 1])
            else:
                mask = (sample >= boundaries[q]) & (sample < boundaries[q + 1])

            if mask.sum() > 0:
                seg_x.append(np.median(sample[mask]))
                seg_shift.append(np.median(ref[mask] - sample[mask]))

        if len(seg_x) == 0:
            return row

        # Interpolate shifts across the spectrum
        shift_interp = np.interp(
            mz_axis, seg_x, seg_shift, left=seg_shift[0], right=seg_shift[-1]
        )

        # Apply Gaussian smoothing to reduce abrupt transitions
        if self.smooth_sigma > 0:
            shift_interp = gaussian_filter1d(
                shift_interp, sigma=self.smooth_sigma, mode="nearest"
            )

        new_positions = mz_axis + shift_interp

        # Ensure monotonicity for interpolation
        return self._monotonic_interp(mz_axis, new_positions, row)

    def _monotonic_interp(self, mz_axis, new_positions, row):
        """
        Perform interpolation with monotonicity enforcement.

        Sorts new_positions and corresponding intensities to ensure
        monotonic mapping before interpolation.
        """
        # Check if already monotonic
        if np.all(np.diff(new_positions) > 0):
            return np.interp(mz_axis, new_positions, row, left=0.0, right=0.0)

        warnings.warn(
            "Warping produced non-monotonic m/z mapping for a sample. "
            "This may indicate poor alignment quality. "
            "Consider adjusting alignment parameters (e.g., reduce max_shift_da "
            "or increase n_segments).",
            UserWarning,
            stacklevel=3,
        )

        # Sort to enforce monotonicity
        sort_idx = np.argsort(new_positions)
        new_positions_sorted = new_positions[sort_idx]
        row_sorted = row[sort_idx]

        # Remove duplicates by averaging
        unique_pos, inverse = np.unique(new_positions_sorted, return_inverse=True)
        unique_intensities = np.zeros(len(unique_pos))

        for i, pos_idx in enumerate(inverse):
            unique_intensities[pos_idx] += row_sorted[i]

        # Average duplicates
        counts = np.bincount(inverse)
        unique_intensities = unique_intensities / counts

        return np.interp(mz_axis, unique_pos, unique_intensities, left=0.0, right=0.0)

    def _dtw(self, row):
        """
        Align intensity vector using Dynamic Time Warping.

        This maps the query spectrum to the reference using the optimal
        warping path found by DTW. Uses averaging for multiple mappings
        to the same reference index.
        """
        # Compute DTW alignment path with radius constraint
        # Use squared Euclidean distance for better intensity matching
        path, distance = dtw_path(
            row,
            self.ref_spec_,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=self.dtw_radius,
        )

        # Create aligned spectrum by following the warping path
        aligned_sum = np.zeros_like(self.ref_spec_)
        aligned_count = np.zeros_like(self.ref_spec_)

        for i, j in path:
            # i is index in query (row), j is index in reference
            if 0 <= j < len(aligned_sum):
                aligned_sum[j] += row[i]
                aligned_count[j] += 1

        # Average where multiple query points map to same reference index
        aligned = np.zeros_like(self.ref_spec_)
        mask = aligned_count > 0
        aligned[mask] = aligned_sum[mask] / aligned_count[mask]

        return aligned

    def _align_single_row(
        self,
        row: np.ndarray,
        peaks: np.ndarray | None,
        ref_peaks: np.ndarray,
        mz_axis: np.ndarray,
    ) -> np.ndarray:
        """Align a single row (helper for parallelization)."""
        if self.method == "dtw":
            return self._dtw(row)
        elif self.method == "shift":
            return self._shift_only(row, peaks, ref_peaks)
        elif self.method == "linear":
            return self._linear_fit(row, peaks, ref_peaks, mz_axis)
        elif self.method == "piecewise":
            return self._piecewise(row, peaks, ref_peaks, mz_axis)
        else:
            raise ValueError(f"Unknown warping method {self.method}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform spectra by aligning them to the reference.

        Parameters
        ----------
        X : pd.DataFrame
            Input spectra with shape (n_samples, n_bins).

        Returns
        -------
        X_aligned : pd.DataFrame
            Aligned spectra with same shape as input.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted.
        ValueError
            If the number of features in X does not match the reference
            spectrum length.
        """
        if not hasattr(self, "ref_spec_"):
            raise RuntimeError("Warping must be fitted before transform")

        if X.shape[1] != len(self.ref_spec_):
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match "
                f"reference spectrum length ({len(self.ref_spec_)})"
            )

        mz_axis = np.arange(X.shape[1])

        # Detect peaks in reference (do once)
        ref_peaks_df = self.peak_detector.transform(
            pd.DataFrame(self.ref_spec_[np.newaxis, :], columns=X.columns)
        )
        ref_peaks = ref_peaks_df.iloc[0].to_numpy().nonzero()[0]

        # Batch peak detection for efficiency (for non-DTW methods)
        peaks_list = None
        if self.method != "dtw":
            peaks_df = self.peak_detector.transform(X)
            peaks_list = [
                peaks_df.iloc[i].to_numpy().nonzero()[0] for i in range(len(X))
            ]

        # Use parallel processing with joblib
        aligned_rows = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._align_single_row)(
                X.iloc[i].to_numpy(),
                peaks_list[i] if peaks_list is not None else None,
                ref_peaks,
                mz_axis,
            )
            for i in range(len(X))
        )

        return pd.DataFrame(aligned_rows, index=X.index, columns=X.columns)

    def get_alignment_quality(
        self, X_original: pd.DataFrame, X_aligned: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Compute alignment quality metrics.

        Parameters
        ----------
        X_original : pd.DataFrame
            Original (unaligned) spectra.
        X_aligned : pd.DataFrame, optional
            Aligned spectra. If None, will compute by calling transform().

        Returns
        -------
        pd.DataFrame
            Quality metrics with columns:
            - correlation_before: Pearson correlation with reference (before)
            - correlation_after: Pearson correlation with reference (after)
            - improvement: correlation_after - correlation_before
            - rmse_before: RMSE with reference (before)
            - rmse_after: RMSE with reference (after)

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted.
        """
        if not hasattr(self, "ref_spec_"):
            raise RuntimeError("Warping must be fitted before computing quality")

        if X_aligned is None:
            X_aligned = self.transform(X_original)

        metrics = []
        for i in range(len(X_original)):
            original = X_original.iloc[i].to_numpy()
            aligned = X_aligned.iloc[i].to_numpy()

            # Correlation with reference (NaN when a signal has zero variance)
            corr_before = np.corrcoef(original, self.ref_spec_)[0, 1]
            corr_after = np.corrcoef(aligned, self.ref_spec_)[0, 1]

            if np.isnan(corr_before) or np.isnan(corr_after):
                warnings.warn(
                    f"Correlation undefined for sample {X_original.index[i]} "
                    f"(constant signal); defaulting to 0.0",
                    UserWarning,
                    stacklevel=2,
                )
                corr_before = 0.0 if np.isnan(corr_before) else corr_before
                corr_after = 0.0 if np.isnan(corr_after) else corr_after

            # RMSE with reference
            rmse_before = np.sqrt(np.mean((original - self.ref_spec_) ** 2))
            rmse_after = np.sqrt(np.mean((aligned - self.ref_spec_) ** 2))

            metrics.append(
                {
                    "correlation_before": corr_before,
                    "correlation_after": corr_after,
                    "improvement": corr_after - corr_before,
                    "rmse_before": rmse_before,
                    "rmse_after": rmse_after,
                }
            )

        return pd.DataFrame(metrics, index=X_original.index)

    def plot_alignment(
        self,
        X_original: pd.DataFrame,
        X_aligned: pd.DataFrame | None = None,
        indices: int | list[int] | None = None,
        show_peaks: bool = True,
        xlim: tuple[float, float] | None = None,
        figsize: tuple[float, float] = (14, 6),
        alpha: float = 0.7,
    ):
        """
        Plot comparison of original vs aligned spectra against reference.

        Parameters
        ----------
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
        if not hasattr(self, "ref_spec_"):
            raise RuntimeError("Warping must be fitted before plotting")

        # Compute aligned spectra if not provided
        if X_aligned is None:
            X_aligned = self.transform(X_original)

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

        ref_peaks, sample_peaks_dict, aligned_peaks_dict = None, {}, {}
        if show_peaks:
            ref_peaks_df = self.peak_detector.transform(
                pd.DataFrame(self.ref_spec_[np.newaxis, :], columns=X_original.columns)
            )
            ref_peaks = mz_axis[ref_peaks_df.iloc[0].to_numpy().nonzero()[0]]

            if self.method != "dtw":
                sample_peaks_df = self.peak_detector.transform(X_original.iloc[indices])
                aligned_peaks_df = self.peak_detector.transform(X_aligned.iloc[indices])

                for i, idx in enumerate(indices):
                    sample_peaks_dict[idx] = mz_axis[
                        sample_peaks_df.iloc[i].to_numpy().nonzero()[0]
                    ]
                    aligned_peaks_dict[idx] = mz_axis[
                        aligned_peaks_df.iloc[i].to_numpy().nonzero()[0]
                    ]

        # Create figure with subplots
        n_spectra = len(indices)
        fig, axes = plt.subplots(n_spectra, 2, figsize=figsize, squeeze=False)

        for plot_idx, spectrum_idx in enumerate(indices):
            ax_before = axes[plot_idx, 0]
            ax_after = axes[plot_idx, 1]

            # Get spectra
            original = X_original.iloc[spectrum_idx].to_numpy()
            aligned = X_aligned.iloc[spectrum_idx].to_numpy()

            # Plot before alignment
            ax_before.plot(
                mz_axis,
                self.ref_spec_,
                label="Reference",
                color="black",
                linewidth=1.5,
                alpha=alpha,
            )
            ax_before.plot(
                mz_axis,
                original,
                label=f"Original (idx={spectrum_idx})",
                color="red",
                linewidth=1,
                alpha=alpha,
            )

            if show_peaks and ref_peaks is not None:
                for peak in ref_peaks:
                    ax_before.axvline(
                        peak, color="black", linestyle="--", alpha=0.3, linewidth=0.8
                    )
                if spectrum_idx in sample_peaks_dict:
                    for peak in sample_peaks_dict[spectrum_idx]:
                        ax_before.axvline(
                            peak, color="red", linestyle="--", alpha=0.3, linewidth=0.8
                        )

            ax_before.set_ylabel("Intensity")
            ax_before.set_title(f"Before Alignment ({self.method} method)")
            ax_before.legend(loc="upper right")
            ax_before.grid(True, alpha=0.3)
            if xlim:
                ax_before.set_xlim(xlim)

            # Plot after alignment
            ax_after.plot(
                mz_axis,
                self.ref_spec_,
                label="Reference",
                color="black",
                linewidth=1.5,
                alpha=alpha,
            )
            ax_after.plot(
                mz_axis,
                aligned,
                label=f"Aligned (idx={spectrum_idx})",
                color="blue",
                linewidth=1,
                alpha=alpha,
            )

            if show_peaks and ref_peaks is not None:
                for peak in ref_peaks:
                    ax_after.axvline(
                        peak, color="black", linestyle="--", alpha=0.3, linewidth=0.8
                    )
                if spectrum_idx in aligned_peaks_dict:
                    for peak in aligned_peaks_dict[spectrum_idx]:
                        ax_after.axvline(
                            peak, color="blue", linestyle="--", alpha=0.3, linewidth=0.8
                        )

            ax_after.set_title(f"After Alignment ({self.method} method)")
            ax_after.legend(loc="upper right")
            ax_after.grid(True, alpha=0.3)
            if xlim:
                ax_after.set_xlim(xlim)

        plt.tight_layout()
        return fig, axes
