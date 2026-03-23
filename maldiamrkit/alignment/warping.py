"""Spectral alignment and warping transformers for binned spectra."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from ..detection.peak_detector import MaldiPeakDetector
from .strategies import ALIGNMENT_REGISTRY, DTWStrategy


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
        if self.method not in ALIGNMENT_REGISTRY:
            raise ValueError(
                f"Unknown warping method: {self.method}. "
                f"Must be one of: {', '.join(ALIGNMENT_REGISTRY)}"
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

    def _get_strategy(self):
        """Build strategy instance from current parameters."""
        if self.method not in ALIGNMENT_REGISTRY:
            raise ValueError(
                f"Unknown warping method {self.method}. "
                f"Must be one of: {', '.join(ALIGNMENT_REGISTRY)}"
            )
        cls = ALIGNMENT_REGISTRY[self.method]
        if self.method == "shift":
            return cls(max_shift=self.max_shift)
        elif self.method == "linear":
            return cls(max_shift=self.max_shift)
        elif self.method == "piecewise":
            return cls(
                n_segments=self.n_segments,
                smooth_sigma=self.smooth_sigma,
                max_shift=self.max_shift,
            )
        elif self.method == "dtw":
            return cls(dtw_radius=self.dtw_radius)
        return cls()

    def _align_single_row(
        self,
        row: np.ndarray,
        peaks: np.ndarray | None,
        ref_peaks: np.ndarray,
        mz_axis: np.ndarray,
    ) -> np.ndarray:
        """Align a single row (helper for parallelization)."""
        strategy = self._get_strategy()
        if isinstance(strategy, DTWStrategy):
            return strategy._dtw_align(row, self.ref_spec_)
        return strategy.align_binned(row, peaks, ref_peaks, mz_axis)

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
