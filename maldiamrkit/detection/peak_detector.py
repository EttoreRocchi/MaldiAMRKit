"""Peak detection algorithms for MALDI-TOF spectra."""

from __future__ import annotations

import gudhi
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, TransformerMixin


class MaldiPeakDetector(BaseEstimator, TransformerMixin):
    """
    Peak detector for MALDI-TOF spectra with local maxima and topological methods.

    The transformer maintains the original feature dimension; all non-peak
    positions are set to 0. Peaks can be returned as binary flags or with
    their original intensities.

    Parameters
    ----------
    method : {"local", "ph"}, default="local"
        Detection method to use:
        - "local" : Local maxima detection using scipy.signal.find_peaks
        - "ph" : Persistent homology based detection using gudhi
    binary : bool, default=True
        If True, peaks are marked with 1; otherwise, original intensity is kept.
    persistence_threshold : float, default=1e-6
        Minimum persistence (death - birth) required for a peak when using
        method="ph". For normalized spectra (sum=1), typical values are 1e-6
        to 1e-4. Higher values detect fewer, more prominent peaks.
    n_jobs : int, default=1
        Number of parallel jobs for peak detection. Use -1 for all available
        cores, 1 for sequential processing. Parallelization is particularly
        beneficial for the "ph" method which is CPU-intensive.
    **kwargs :
        Additional keyword arguments passed to scipy.signal.find_peaks
        when method="local". Common parameters:
        - prominence : float, minimum prominence of peaks (recommended: 1e-5 to 1e-2)
        - height : float, minimum height of peaks
        - distance : int, minimum distance between peaks in bins
        - width : float, minimum width of peaks

    Notes
    -----
    For MALDI-TOF spectra normalized to sum=1:
    - prominence=1e-5 to 1e-3 typically works well for local maxima
    - persistence_threshold=1e-6 to 1e-4 for persistent homology

    Raises
    ------
    ValueError
        If ``method`` is not one of 'local' or 'ph'.

    Examples
    --------
    >>> # Local maxima detection with prominence filter
    >>> detector = MaldiPeakDetector(method="local", prominence=0.01)
    >>> peaks = detector.fit_transform(spectra_df)

    >>> # Persistent homology based detection
    >>> detector = MaldiPeakDetector(method="ph", persistence_threshold=1e-6)
    >>> peaks = detector.fit_transform(spectra_df)
    """

    def __init__(
        self,
        method: str = "local",
        binary: bool = True,
        persistence_threshold: float = 1e-6,
        n_jobs: int = 1,
        prominence: float | None = None,
        height: float | None = None,
        distance: int | None = None,
        width: float | None = None,
        **kwargs,
    ) -> None:
        self.method = method
        self.binary = binary
        self.persistence_threshold = persistence_threshold
        self.n_jobs = n_jobs
        self.prominence = prominence
        self.height = height
        self.distance = distance
        self.width = width
        # Build kwargs from explicit params + extra kwargs
        self.kwargs = dict(kwargs)
        for param in ("prominence", "height", "distance", "width"):
            val = getattr(self, param)
            if val is not None:
                self.kwargs.setdefault(param, val)

        if self.method not in ["local", "ph"]:
            raise ValueError(
                f"Unknown method '{self.method}'. Must be one of: 'local', 'ph'"
            )

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the peak detector (no learning performed).

        Parameters
        ----------
        X : pd.DataFrame
            Input spectra with shape (n_samples, n_bins).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : MaldiPeakDetector
            Fitted transformer.

        Raises
        ------
        ValueError
            If the input DataFrame is empty.
        """
        if X.empty:
            raise ValueError("Input DataFrame X is empty")

        return self

    def _detect_peaks_local(self, row: np.ndarray) -> np.ndarray:
        """
        Detect peaks using local maxima detection.

        Uses scipy.signal.find_peaks with configurable parameters.

        Parameters
        ----------
        row : np.ndarray
            1D spectrum intensity array.

        Returns
        -------
        peaks : np.ndarray
            Array of peak indices.
        """
        peaks, _ = find_peaks(row, **self.kwargs)
        return peaks

    def _detect_peaks_ph(self, row: np.ndarray) -> np.ndarray:
        """
        Detect peaks using persistent homology (0D persistence).

        Computes the 0D persistence diagram of the signal treated as a
        1D cubical complex. Peaks correspond to local maxima with sufficient
        persistence (death - birth) above the threshold.

        Parameters
        ----------
        row : np.ndarray
            1D spectrum intensity array.

        Returns
        -------
        peaks : np.ndarray
            Array of peak indices corresponding to persistent maxima.

        Notes
        -----
        The algorithm:
        1. Negates the signal for superlevel set analysis
        2. Computes 0D persistence using cubical complexes
        3. Filters features by persistence threshold
        4. Maps birth values back to peak indices using local maxima constraint
        """
        if np.allclose(row, row[0]):
            return np.array([], dtype=int)

        # Negate signal for superlevel analysis (peaks become valleys)
        signal = -row
        signal = signal - signal.min()

        cc = gudhi.CubicalComplex(top_dimensional_cells=signal[np.newaxis, :])
        persistence_diagram = cc.persistence()

        # Find all local maxima in original signal for accurate index mapping
        local_maxima, _ = find_peaks(row)
        local_maxima_set = set(local_maxima)

        peak_indices = []
        signal_max = np.max(signal)

        for dim, (birth, death) in persistence_diagram:
            if dim != 0:
                continue

            persistence = (
                (death - birth) if not np.isinf(death) else (signal_max - birth)
            )

            if persistence >= self.persistence_threshold:
                # Find candidates close to birth intensity
                tol = persistence * 0.1 + 1e-10
                candidates = np.where(np.abs(signal - birth) < tol)[0]

                # Prefer candidates that are local maxima in original signal
                maxima_candidates = [c for c in candidates if c in local_maxima_set]

                if maxima_candidates:
                    # Use the candidate closest to birth value among local maxima
                    idx = maxima_candidates[
                        np.argmin(np.abs(signal[maxima_candidates] - birth))
                    ]
                elif len(candidates) > 0:
                    # Fallback: use closest candidate
                    idx = candidates[np.argmin(np.abs(signal[candidates] - birth))]
                else:
                    # Last resort: global argmin
                    idx = np.argmin(np.abs(signal - birth))

                peak_indices.append(idx)

        return np.array(sorted(set(peak_indices)), dtype=int)

    def _process_single_row(self, row: np.ndarray) -> np.ndarray:
        """Process a single row and return masked array (helper for parallelization)."""
        if self.method == "local":
            peaks = self._detect_peaks_local(row)
        elif self.method == "ph":
            peaks = self._detect_peaks_ph(row)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        masked = np.zeros_like(row, dtype=row.dtype)

        if self.binary:
            masked[peaks] = 1
        else:
            masked[peaks] = row[peaks]

        return masked

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect peaks in each spectrum and mask non-peak positions.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Input spectra with shape (n_samples, n_bins).

        Returns
        -------
        X_peaks : pd.DataFrame or pd.Series
            Transformed spectra where non-peak positions are set to 0.
            Peak positions contain 1 (if binary=True) or original intensity.
        """
        input_is_series = isinstance(X, pd.Series)
        if input_is_series:
            X = X.to_frame().T

        # Use parallel processing with joblib
        # "processes" backend is better for CPU-bound tasks like PH
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._process_single_row)(X.iloc[i].values)
            for i in range(X.shape[0])
        )

        # Reconstruct DataFrame from results
        X_out = pd.DataFrame(np.vstack(results), index=X.index, columns=X.columns)

        if input_is_series:
            return X_out.iloc[0]

        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Input spectra with shape (n_samples, n_bins).
        y : array-like, optional
            Target values (ignored).
        **fit_params :
            Additional fit parameters (unused).

        Returns
        -------
        X_peaks : pd.DataFrame or pd.Series
            Transformed spectra with detected peaks.
        """
        if isinstance(X, pd.Series):
            X_fit = X.to_frame().T
        else:
            X_fit = X

        self.fit(X_fit, y)
        return self.transform(X)

    def get_peak_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics about detected peaks for each spectrum.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Input spectra with shape (n_samples, n_bins).

        Returns
        -------
        stats : pd.DataFrame
            DataFrame with columns:
            - n_peaks: number of peaks detected
            - mean_intensity: mean intensity of detected peaks
            - max_intensity: maximum intensity of detected peaks
        """
        input_is_series = isinstance(X, pd.Series)
        if input_is_series:
            X = X.to_frame().T

        stats = []

        for i in range(X.shape[0]):
            row = X.iloc[i].values

            if self.method == "local":
                peaks = self._detect_peaks_local(row)
            elif self.method == "ph":
                peaks = self._detect_peaks_ph(row)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            n_peaks = len(peaks)
            if n_peaks > 0:
                peak_intensities = row[peaks]
                mean_intensity = np.mean(peak_intensities)
                max_intensity = np.max(peak_intensities)
            else:
                mean_intensity = 0.0
                max_intensity = 0.0

            stats.append(
                {
                    "n_peaks": n_peaks,
                    "mean_intensity": mean_intensity,
                    "max_intensity": max_intensity,
                }
            )

        return pd.DataFrame(stats, index=X.index)
