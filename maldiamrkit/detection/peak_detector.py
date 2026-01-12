"""Peak detection algorithms for MALDI-TOF spectra."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks
import gudhi


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
        **kwargs
    ) -> MaldiPeakDetector:
        self.method = method
        self.binary = binary
        self.persistence_threshold = persistence_threshold
        self.kwargs = kwargs

        if self.method not in ["local", "ph"]:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Must be one of: 'local', 'ph'"
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

        X_out = X.copy()

        for i in range(X_out.shape[0]):
            row = X_out.iloc[i].values

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

            X_out.iloc[i] = masked

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

            stats.append({
                'n_peaks': n_peaks,
                'mean_intensity': mean_intensity,
                'max_intensity': max_intensity
            })

        return pd.DataFrame(stats, index=X.index)

    def plot_peaks(
        self,
        X: pd.DataFrame,
        indices: int | list[int] | None = None,
        xlim: tuple[float, float] | None = None,
        figsize: tuple[float, float] = (14, 6),
        alpha: float = 0.7
    ):
        """
        Plot detected peaks overlaid on original spectra.

        Parameters
        ----------
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

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.
        axes : Axes or array of Axes
            The subplot axes.
        """
        input_is_series = isinstance(X, pd.Series)
        if input_is_series:
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

        n_spectra = len(indices)
        fig, axes = plt.subplots(n_spectra, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for plot_idx, spectrum_idx in enumerate(indices):
            ax = axes[plot_idx]

            row = X.iloc[spectrum_idx].values

            if self.method == "local":
                peaks = self._detect_peaks_local(row)
            elif self.method == "ph":
                peaks = self._detect_peaks_ph(row)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            ax.plot(
                mz_axis, row,
                color='black', linewidth=1, alpha=alpha, label='Spectrum'
            )

            if len(peaks) > 0:
                ax.scatter(
                    mz_axis[peaks], row[peaks],
                    color='red', s=50, zorder=5,
                    label=f'Peaks (n={len(peaks)})', marker='o'
                )

                for peak in peaks:
                    ax.axvline(
                        mz_axis[peak], color='red', linestyle='--',
                        alpha=0.3, linewidth=0.8
                    )

            ax.set_xlabel(
                'm/z' if np.issubdtype(mz_axis.dtype, np.number) else 'Index'
            )
            ax.set_ylabel('Intensity')
            ax.set_title(f'Peak Detection (method={self.method}, idx={spectrum_idx})')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            if xlim:
                ax.set_xlim(xlim)

        plt.tight_layout()

        if n_spectra == 1:
            return fig, axes[0]
        return fig, axes
