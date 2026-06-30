"""Peak detection algorithms for MALDI-TOF spectra."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import gudhi
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import find_peaks, peak_prominences
from sklearn.base import BaseEstimator, TransformerMixin

from .peaklist import (
    _RANK_BY,
    PeakList,
    PeakSet,
    _array_hash,
    _config_hash,
    _PeakCache,
)


class PeakMethod(str, Enum):
    """Supported peak detection methods.

    Attributes
    ----------
    local : str
        Local maxima detection via ``scipy.signal.find_peaks``.
    ph : str
        Persistent homology based peak detection.
    """

    local = "local"
    ph = "ph"


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
    prominence : float or None, default=None
        Minimum prominence of peaks (recommended: 1e-5 to 1e-2).
        Passed to :func:`scipy.signal.find_peaks` when ``method="local"``.
    height : float or None, default=None
        Minimum height of peaks.
        Passed to :func:`scipy.signal.find_peaks` when ``method="local"``.
    distance : int or None, default=None
        Minimum distance between peaks in bins.
        Passed to :func:`scipy.signal.find_peaks` when ``method="local"``.
    width : float or None, default=None
        Minimum width of peaks.
        Passed to :func:`scipy.signal.find_peaks` when ``method="local"``.
    **kwargs :
        Additional keyword arguments passed to
        :func:`scipy.signal.find_peaks` when ``method="local"``.

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
        method: str | PeakMethod = PeakMethod.local,
        binary: bool = True,
        persistence_threshold: float = 1e-6,
        n_jobs: int = 1,
        prominence: float | None = None,
        height: float | None = None,
        distance: int | None = None,
        width: float | None = None,
        **kwargs,
    ) -> None:
        self.method = PeakMethod(method)
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
        1. Negates the signal so that ``row``'s maxima become sub-level
           minima (0D component births).
        2. Builds a 1D cubical complex and computes 0D persistence.
        3. Recovers the exact birth-cell index for each pair.
        4. Filters pairs by ``persistence >= persistence_threshold``.
        """
        if np.allclose(row, row[0]):
            return np.array([], dtype=int)

        # Negate signal for sub-level-set filtration on the negated signal
        # (so that peaks of ``row`` become births of 0D features).
        signal = -row
        signal = signal - signal.min()

        cc = gudhi.CubicalComplex(top_dimensional_cells=signal[np.newaxis, :])
        cc.persistence()
        regular_pairs, essential_pairs = cc.cofaces_of_persistence_pairs()

        regular = regular_pairs[0] if len(regular_pairs) else np.empty((0, 2), int)
        essential = essential_pairs[0] if len(essential_pairs) else np.empty(0, int)
        signal_max = float(np.max(signal))

        peak_indices: list[int] = []

        if regular.size:
            births = signal[regular[:, 0]]
            deaths = signal[regular[:, 1]]
            persistences = deaths - births
            keep = persistences >= self.persistence_threshold
            peak_indices.extend(int(i) for i in regular[keep, 0].tolist())

        if essential.size:
            essential_births = signal[essential]
            persistences = signal_max - essential_births
            keep = persistences >= self.persistence_threshold
            peak_indices.extend(int(i) for i in essential[keep].tolist())

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

    def _detect_peaks_ph_scored(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect persistent-homology peaks and their persistence values.

        Like ``_detect_peaks_ph`` but also returns the persistence
        (death - birth) of each kept peak, index-ascending.

        Parameters
        ----------
        row : np.ndarray
            1D spectrum intensity array.

        Returns
        -------
        indices : np.ndarray
            Peak indices, ascending.
        persistences : np.ndarray
            Persistence of each peak, aligned with ``indices``.
        """
        row = np.asarray(row, dtype=float)
        if np.allclose(row, row[0]):
            return np.array([], dtype=int), np.array([], dtype=float)

        signal = -row
        signal = signal - signal.min()

        cc = gudhi.CubicalComplex(top_dimensional_cells=signal[np.newaxis, :])
        cc.persistence()
        regular_pairs, essential_pairs = cc.cofaces_of_persistence_pairs()

        regular = regular_pairs[0] if len(regular_pairs) else np.empty((0, 2), int)
        essential = essential_pairs[0] if len(essential_pairs) else np.empty(0, int)
        signal_max = float(np.max(signal))

        idx_to_pers: dict[int, float] = {}
        if regular.size:
            births = signal[regular[:, 0]]
            deaths = signal[regular[:, 1]]
            persistences = deaths - births
            keep = persistences >= self.persistence_threshold
            for i, pers in zip(
                regular[keep, 0].tolist(), persistences[keep].tolist(), strict=True
            ):
                idx_to_pers[int(i)] = max(idx_to_pers.get(int(i), 0.0), float(pers))
        if essential.size:
            essential_births = signal[essential]
            persistences = signal_max - essential_births
            keep = persistences >= self.persistence_threshold
            for i, pers in zip(
                essential[keep].tolist(), persistences[keep].tolist(), strict=True
            ):
                idx_to_pers[int(i)] = max(idx_to_pers.get(int(i), 0.0), float(pers))

        if not idx_to_pers:
            return np.array([], dtype=int), np.array([], dtype=float)
        indices = np.array(sorted(idx_to_pers), dtype=int)
        pers = np.array([idx_to_pers[int(i)] for i in indices], dtype=float)
        return indices, pers

    def _detect_scored(
        self, row: np.ndarray, rank_by: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(peak_indices, ranking_scores)`` for one spectrum row.

        Peak indices are ascending. ``rank_by="persistence"`` requires
        ``method="ph"``; ``"intensity"`` and ``"prominence"`` work for either
        detection method.
        """
        row = np.asarray(row, dtype=float)
        if self.method == "local":
            indices = self._detect_peaks_local(row)
            persistences: np.ndarray | None = None
        elif self.method == "ph":
            indices, persistences = self._detect_peaks_ph_scored(row)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return indices, np.empty(0, dtype=float)

        if rank_by == "intensity":
            scores = row[indices]
        elif rank_by == "prominence":
            scores = peak_prominences(row, indices)[0]
        elif rank_by == "persistence":
            if persistences is None:
                raise ValueError("rank_by='persistence' requires method='ph'.")
            scores = persistences
        else:
            raise ValueError(
                f"Unknown rank_by={rank_by!r}; expected one of {_RANK_BY}."
            )
        return indices, np.asarray(scores, dtype=float)

    def detect_peakset(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        top_k: int | None = 200,
        rank_by: str = "intensity",
    ) -> PeakSet:
        """Detect peaks in one spectrum and return a :class:`PeakSet`.

        Stateless and per-spectrum: the result depends only on this spectrum,
        so it is identical whether computed over a whole dataset or inside a
        single fold.

        Parameters
        ----------
        mz : array-like
            m/z axis of the spectrum, shape ``(n_points,)``.
        intensity : array-like
            Intensity values aligned with ``mz``.
        top_k : int or None, default=200
            Keep at most this many peaks, ranked by ``rank_by``. ``None`` keeps
            all detected peaks.
        rank_by : {"intensity", "persistence", "prominence"}, default="intensity"
            Ranking used to select the ``top_k`` peaks.

        Returns
        -------
        PeakSet
            The detected peaks (always carrying their true intensities,
            regardless of the detector's ``binary`` setting).
        """
        mz = np.asarray(mz, dtype=float).ravel()
        intensity = np.asarray(intensity, dtype=float).ravel()
        if mz.shape != intensity.shape:
            raise ValueError(
                "mz and intensity must have the same shape; "
                f"got {mz.shape} and {intensity.shape}."
            )
        indices, scores = self._detect_scored(intensity, rank_by)
        if indices.size == 0:
            return PeakSet(np.empty(0), np.empty(0))
        if top_k is not None and indices.size > top_k:
            keep = np.argsort(scores, kind="stable")[::-1][: int(top_k)]
            indices = indices[keep]
            scores = scores[keep]
        peak_score = None if rank_by == "intensity" else scores
        return PeakSet(mz[indices], intensity[indices], peak_score)

    def _resolve_mz_index(
        self, X: Any, mz: np.ndarray | None
    ) -> tuple[np.ndarray, pd.Index]:
        """Recover the m/z axis (from ``mz`` or numeric columns) and the index."""
        if isinstance(X, pd.DataFrame):
            index = X.index
            columns: pd.Index | None = X.columns
        else:
            index = pd.RangeIndex(np.asarray(X).shape[0])
            columns = None

        if mz is not None:
            return np.asarray(mz, dtype=float).ravel(), index
        if columns is not None:
            try:
                return columns.to_numpy(dtype=float), index
            except (ValueError, TypeError):
                raise ValueError(
                    "Could not recover an m/z axis from the DataFrame columns "
                    f"({list(columns[:3])}...). Pass mz= explicitly (e.g. the bin "
                    "centres from maldiamrkit.preprocessing.get_bin_metadata)."
                ) from None
        raise ValueError(
            "X has no column labels to recover m/z from; pass mz= explicitly."
        )

    def transform_peaklist(
        self,
        X: Any,
        top_k: int | None = 200,
        rank_by: str = "intensity",
        mz: np.ndarray | None = None,
        cache_dir: str | Path | None = None,
    ) -> PeakList:
        """Extract a compact :class:`PeakList` from binned spectra.

        Per-spectrum and stateless. The m/z axis is taken from ``mz`` if given,
        otherwise recovered from the (numeric) DataFrame column labels.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or ndarray
            Binned spectra of shape ``(n_samples, n_bins)``.
        top_k : int or None, default=200
            Maximum number of peaks kept per spectrum.
        rank_by : {"intensity", "persistence", "prominence"}, default="intensity"
            Ranking used to select the ``top_k`` peaks.
        mz : array-like, optional
            Explicit m/z axis of length ``n_bins``. Overrides column labels.
        cache_dir : str, optional
            If given, cache the resulting :class:`PeakList` keyed by a content +
            config hash and reuse it on subsequent calls.

        Returns
        -------
        PeakList
            One :class:`PeakSet` per spectrum.
        """
        if rank_by not in _RANK_BY:
            raise ValueError(
                f"Unknown rank_by={rank_by!r}; expected one of {_RANK_BY}."
            )
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        mz_axis, index = self._resolve_mz_index(X, mz)
        values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != mz_axis.shape[0]:
            raise ValueError(
                f"m/z axis length {mz_axis.shape[0]} does not match the "
                f"{values.shape[1]} feature columns of X."
            )

        config = {
            "source": "transform_peaklist",
            "top_k": None if top_k is None else int(top_k),
            "rank_by": rank_by,
            "detector": self.get_params(),
            "detector_kwargs": dict(self.kwargs),
            "mz_axis": _array_hash(mz_axis),
        }
        config_hash = _config_hash(config)
        content_hash = _array_hash(values)

        cache = _PeakCache(cache_dir)
        cached = cache.get(content_hash, config_hash)
        if cached is not None:
            return cached

        rows = [values[i] for i in range(values.shape[0])]
        if self.n_jobs == 1:
            peaks = [
                self.detect_peakset(mz_axis, row, top_k=top_k, rank_by=rank_by)
                for row in rows
            ]
        else:
            peaks = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(self.detect_peakset)(mz_axis, row, top_k, rank_by)
                for row in rows
            )

        peaklist = PeakList(
            peaks,
            index=index,
            meta={
                "method": self.method.value,
                "top_k": None if top_k is None else int(top_k),
                "rank_by": rank_by,
                "mz_range": [float(mz_axis.min()), float(mz_axis.max())],
                "warped": False,
                "content_hash": content_hash,
                "config_hash": config_hash,
                "source": "transform_peaklist",
            },
        )
        cache.put(content_hash, config_hash, peaklist)
        return peaklist
