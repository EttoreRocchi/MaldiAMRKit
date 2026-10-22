"""Spectral alignment and warping transformers for binned spectra."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin

from ..detection.peak_detector import MaldiPeakDetector
from .strategies import (
    _ALIGNMENT_REGISTRY,
    _NON_MONOTONIC,
    _SHIFT_ESTIMATORS,
    AlignmentMethod,
    DTWStrategy,
    ShiftStrategy,
    _most_prominent,
    _shift_row,
)

_MAX_CHUNK_ROWS = 256


def _renormalize(rows: np.ndarray) -> None:
    """Divide each row by its sum in place, leaving rows that sum to 0."""
    totals = rows.sum(axis=1, keepdims=True)
    np.divide(rows, totals, out=rows, where=totals != 0)


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
        - "shift" : global (rigid) shift, see ``shift_estimator``
        - "linear" : least-squares linear transform
        - "piecewise" : local shifts interpolated between segments
        - "dtw" : dynamic time warping
        - "quadratic" : quadratic polynomial fit on matched peak pairs
        - "cubic" : cubic polynomial fit on matched peak pairs
        - "lowess" : LOWESS (Cleveland 1979) non-linear warping
    n_segments : int, default=5
        Number of segments for piecewise warping.
    max_shift : int, default=3
        Largest shift considered, in bins. It bounds the global shift (the
        total of both steps with ``prealign``) and the distance between
        matched peak pairs. The default suits the
        package's 3 Da bins (about 9 Da); a value close to 10 Da divided
        by the bin width is a reasonable choice for other bin widths.
    dtw_radius : int, default=10
        Radius constraint for DTW to limit warping path search space.
    smooth_sigma : float, default=2.0
        Gaussian smoothing parameter for piecewise segment shifts.
    lowess_frac : float, default=0.3
        LOWESS smoothing bandwidth (fraction of matched peaks used for
        each local fit). Applies when ``method="lowess"``.
    lowess_it : int, default=3
        Number of LOWESS robustness iterations. Applies when
        ``method="lowess"``.
    min_reference_peaks : int, default=5
        Minimum number of peaks expected in reference for quality check.
    n_jobs : int, default=1
        Number of parallel jobs for transform. Use -1 for all available
        cores, 1 for sequential processing. Parallelization is particularly
        beneficial for the "dtw" method which is CPU-intensive.
    shift_estimator : {"xcorr", "peaks"}, default="xcorr"
        How the global shift is estimated. ``"xcorr"`` takes the lag (within
        ``max_shift``) that maximises the Pearson correlation with the
        reference spectrum, and needs no peak detection. ``"peaks"`` takes
        the median offset between mutually nearest sample and reference
        peaks within ``max_shift``. Either shift is applied only if it
        raises the correlation with the reference by ``min_gain``.
    min_gain : float, default=0.05
        Minimum relative gain in correlation with the reference (as a
        fraction of the correlation it improves on) required to apply a
        shift and to keep the "dtw" result instead of the unwarped
        spectrum. With ``prealign``, each step is checked against its own
        reference.
    min_support : float, default=0.6
        Share of the matched peak pairs that a non-rigid warp must bring
        within ``max_shift / 3`` bins of their reference partner (1 bin by
        default) to be used instead of the global shift. A warp that leaves
        many pairs unexplained signals a sample that does not share its
        peaks with the reference, for which the warp is unreliable.
    n_peaks : int or None, default=200
        Number of most prominent detected peaks, in each spectrum and in
        the reference, used to match peaks (by the "peaks" shift estimator
        and the non-rigid methods). ``None`` uses every detected peak.
    prealign : bool, default=True
        Align the spectra with a global shift first and fit the alignment
        to the reference of the shifted spectra; :meth:`transform` applies
        the same two steps, and the global shift of both steps together
        stays within ``max_shift``. This keeps a median reference sharp
        when the spectra come from different calibrations (for instance
        several instruments).
    renormalize : bool, default=False
        Divide each aligned spectrum by its total intensity, so that it
        sums to 1 (spectra with no intensity are left unchanged).

    Attributes
    ----------
    ref_spec_ : np.ndarray
        The fitted reference spectrum (stored after fit()). With
        ``prealign``, the reference of the shifted spectra.
    prealign_ref_ : np.ndarray or None
        Reference of the global-shift step applied first (with
        ``prealign=True``), otherwise None.

    Notes
    -----
    The non-rigid methods ("linear", "piecewise", "quadratic", "cubic",
    "lowess") fit their warp on the ``n_peaks`` most prominent peaks,
    matched around the global shift (after the shift step of
    ``prealign``). A warp is used only if its m/z mapping is strictly
    increasing (otherwise the global shift is used and the sample is
    counted in a warning) and only if it is supported by at least
    ``min_support`` of the matched pairs; otherwise the global shift is
    used. Their output is resampled so that the total intensity is
    preserved, apart from the part moved past either end of the spectrum;
    "shift" preserves it in the same way. "dtw" does not preserve the total
    intensity, because it averages the sample points matched to each
    reference point. ``renormalize=True`` makes every aligned spectrum sum
    to 1 whatever the method.

    A median reference stands for spectra that share most of their peaks,
    such as spectra of one species. For a collection of several species,
    fitting one transformer per species aligns each spectrum to peaks it
    actually has.

    Only the reference spectrum is learned in :meth:`fit`; :meth:`transform`
    aligns each spectrum on its own, so its result does not depend on the
    other spectra transformed with it.

    The output has the same floating dtype as ``X``.

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
        method: str | AlignmentMethod = AlignmentMethod.shift,
        n_segments: int = 5,
        max_shift: int = 3,
        dtw_radius: int = 10,
        smooth_sigma: float = 2.0,
        lowess_frac: float = 0.3,
        lowess_it: int = 3,
        min_reference_peaks: int = 5,
        n_jobs: int = 1,
        shift_estimator: str = "xcorr",
        min_gain: float = 0.05,
        min_support: float = 0.6,
        n_peaks: int | None = 200,
        prealign: bool = True,
        renormalize: bool = False,
    ) -> None:
        self.peak_detector = peak_detector or MaldiPeakDetector(
            binary=True, prominence=1e-5
        )
        self.reference = reference
        self.method = AlignmentMethod(method)
        self.n_segments = n_segments
        self.max_shift = max_shift
        self.dtw_radius = dtw_radius
        self.smooth_sigma = smooth_sigma
        self.lowess_frac = lowess_frac
        self.lowess_it = lowess_it
        self.min_reference_peaks = min_reference_peaks
        self.n_jobs = n_jobs
        self.shift_estimator = shift_estimator
        self.min_gain = min_gain
        self.min_support = min_support
        self.n_peaks = n_peaks
        self.prealign = prealign
        self.renormalize = renormalize

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

        # Validate parameters
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {self.n_segments}")
        if self.max_shift < 0:
            raise ValueError(f"max_shift must be >= 0, got {self.max_shift}")
        if not (0.0 < self.lowess_frac <= 1.0):
            raise ValueError(f"lowess_frac must be in (0, 1], got {self.lowess_frac}")
        if self.lowess_it < 0:
            raise ValueError(f"lowess_it must be >= 0, got {self.lowess_it}")
        if self.shift_estimator not in _SHIFT_ESTIMATORS:
            raise ValueError(
                f"shift_estimator must be one of {_SHIFT_ESTIMATORS}, "
                f"got {self.shift_estimator!r}"
            )
        if self.min_gain < 0:
            raise ValueError(f"min_gain must be >= 0, got {self.min_gain}")
        if not (0.0 <= self.min_support <= 1.0):
            raise ValueError(f"min_support must be in [0, 1], got {self.min_support}")
        if self.n_peaks is not None and self.n_peaks < 1:
            raise ValueError(f"n_peaks must be >= 1 or None, got {self.n_peaks}")

        self.prealign_ref_ = None
        if self._prealigns():
            self.prealign_ref_ = self._reference_of(X)
            prealign_peaks = self._prealign_reference_peaks(X.columns)
            shifted = np.empty(X.shape, dtype=self._out_dtype(X))
            for start, stop, block in self._run_chunks(
                X, self._prealign_chunk, prealign_peaks
            ):
                shifted[start:stop] = block
            X = pd.DataFrame(shifted, index=X.index, columns=X.columns)
        self.ref_spec_ = self._reference_of(X)

        # Validate reference quality
        self._validate_reference_quality(X)

        return self

    def _reference_of(self, X: pd.DataFrame) -> np.ndarray:
        """Return the reference spectrum of ``X`` chosen by ``reference``."""
        if self.reference == "median":
            return X.median(axis=0).to_numpy()
        if isinstance(self.reference, int):
            if self.reference < 0 or self.reference >= len(X):
                raise ValueError(
                    f"Reference index {self.reference} is out of bounds "
                    f"for X with {len(X)} samples"
                )
            return X.iloc[self.reference].to_numpy()
        raise ValueError(
            f"Unsupported reference specifier: {self.reference}. "
            f"Must be 'median' or int."
        )

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
        cls = _ALIGNMENT_REGISTRY[self.method]
        shared = dict(
            max_shift=self.max_shift,
            estimator=self.shift_estimator,
            min_gain=self.min_gain,
        )
        warp = dict(shared, min_support=self.min_support)
        if self.method == "shift":
            return cls(**shared)
        elif self.method == "linear":
            return cls(**warp)
        elif self.method == "piecewise":
            return cls(
                n_segments=self.n_segments, smooth_sigma=self.smooth_sigma, **warp
            )
        elif self.method == "dtw":
            return cls(dtw_radius=self.dtw_radius, min_gain=self.min_gain)
        elif self.method == "quadratic":
            return cls(degree=2, **warp)
        elif self.method == "cubic":
            return cls(degree=3, **warp)
        elif self.method == "lowess":
            return cls(frac=self.lowess_frac, it=self.lowess_it, **warp)
        return cls()

    def _select_peaks(self, row: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Keep the ``n_peaks`` most prominent of the peaks detected in ``row``."""
        return _most_prominent(row, peaks, self.n_peaks)

    def _prealigns(self) -> bool:
        """Whether a global-shift step runs before the alignment."""
        return bool(self.prealign)

    @staticmethod
    def _out_dtype(X: pd.DataFrame) -> np.dtype:
        """Floating dtype of ``X``, or float64 when ``X`` is not floating."""
        dtype = X.iloc[:0].to_numpy().dtype
        return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)

    def _reference_peaks(self, spectrum: np.ndarray, columns: pd.Index) -> np.ndarray:
        """Return the ``n_peaks`` most prominent peaks detected in ``spectrum``."""
        detected = self.peak_detector.transform(
            pd.DataFrame(spectrum[np.newaxis, :], columns=columns)
        )
        return self._select_peaks(spectrum, detected.iloc[0].to_numpy().nonzero()[0])

    def _prealign_reference_peaks(self, columns: pd.Index) -> np.ndarray:
        """Peaks of the prealign reference (used by the "peaks" estimator)."""
        if self.shift_estimator != "peaks":
            return np.empty(0, dtype=int)
        return self._reference_peaks(self.prealign_ref_, columns)

    def _prealign_rows(
        self, chunk: pd.DataFrame, prealign_peaks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shift every row of ``chunk`` to the prealign reference.

        Returns the shifted rows and the shift (in bins) of each row.
        """
        rows = chunk.to_numpy()
        peaks = None
        if self.shift_estimator == "peaks":
            peaks = self.peak_detector.transform(chunk).to_numpy()
        strategy = ShiftStrategy(self.max_shift, self.shift_estimator, self.min_gain)
        shifted = np.empty_like(rows)
        lags = np.zeros(len(rows), dtype=int)
        for i, row in enumerate(rows):
            row_peaks = None
            if peaks is not None:
                row_peaks = self._select_peaks(row, peaks[i].nonzero()[0])
            lags[i] = strategy.estimate_binned(
                row, row_peaks, prealign_peaks, self.prealign_ref_
            )
            shifted[i] = _shift_row(row, lags[i])
        return shifted, lags

    def _prealign_chunk(
        self, chunk: pd.DataFrame, prealign_peaks: np.ndarray
    ) -> np.ndarray:
        """Shift every row of ``chunk`` to the prealign reference (in a worker)."""
        return self._prealign_rows(chunk, prealign_peaks)[0]

    def _needs_peaks(self) -> bool:
        """Whether the method uses detected peaks (DTW and the xcorr shift do not)."""
        if self.method == "dtw":
            return False
        return not (self.method == "shift" and self.shift_estimator == "xcorr")

    def _align_single_row(
        self,
        row: np.ndarray,
        peaks: np.ndarray | None,
        ref_peaks: np.ndarray,
        mz_axis: np.ndarray,
        applied_shift: int = 0,
    ) -> tuple[np.ndarray, bool]:
        """Align a single row (helper for parallelization).

        ``applied_shift`` is the shift (in bins) the prealign step already
        gave the row; the global shift left to this step is bounded so that
        both together stay within ``max_shift``.

        Returns ``(aligned_row, was_non_monotonic)``. The flag records that a
        warp was rejected for not being strictly increasing (the strategy
        then used the global shift), so :meth:`transform` can emit a single
        aggregated warning instead of one per sample.
        """
        strategy = self._get_strategy()
        max_shift = int(self.max_shift)
        bounds = (-max_shift - applied_shift, max_shift - applied_shift)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            if isinstance(strategy, DTWStrategy):
                aligned = strategy.align_to_reference(row, self.ref_spec_)
            else:
                aligned = strategy.align_binned(
                    row,
                    peaks,
                    ref_peaks,
                    mz_axis,
                    reference=self.ref_spec_,
                    shift_bounds=bounds,
                )
        non_monotonic = any(_NON_MONOTONIC in str(w.message) for w in caught)
        return aligned, non_monotonic

    def _align_chunk(
        self,
        chunk: pd.DataFrame,
        ref_peaks: np.ndarray,
        prealign_peaks: np.ndarray,
        mz_axis: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Prealign, detect peaks in and align every row of ``chunk`` (in a worker).

        Returns ``(aligned, n_non_monotonic)``: the aligned rows stacked
        into one array and the number of rows whose warp was non-monotonic.
        """
        lags = np.zeros(len(chunk), dtype=int)
        if self.prealign_ref_ is not None:
            shifted, lags = self._prealign_rows(chunk, prealign_peaks)
            chunk = pd.DataFrame(shifted, index=chunk.index, columns=chunk.columns)
        peaks = None
        if self._needs_peaks():
            peaks = self.peak_detector.transform(chunk).to_numpy()
        aligned_rows = []
        n_non_monotonic = 0
        for i, row in enumerate(chunk.to_numpy()):
            aligned, non_monotonic = self._align_single_row(
                row,
                self._select_peaks(row, peaks[i].nonzero()[0])
                if peaks is not None
                else None,
                ref_peaks,
                mz_axis,
                int(lags[i]),
            )
            aligned_rows.append(aligned)
            n_non_monotonic += non_monotonic
        return np.vstack(aligned_rows), n_non_monotonic

    def _run_chunks(self, X: pd.DataFrame, func, *args):
        """Run ``func(chunk, *args)`` on row chunks of ``X`` in parallel.

        Yields ``(start, stop, result)`` for every chunk ``X[start:stop]``.
        Chunks are copied before dispatch and processed in waves of one
        chunk per worker, so transient memory stays bounded.
        """
        n_rows = len(X)
        n_workers = effective_n_jobs(self.n_jobs)
        chunk_rows = min(_MAX_CHUNK_ROWS, max(1, math.ceil(n_rows / (4 * n_workers))))
        bounds = [
            (start, min(start + chunk_rows, n_rows))
            for start in range(0, n_rows, chunk_rows)
        ]
        with Parallel(
            n_jobs=min(n_workers, max(len(bounds), 1)),
            prefer="processes",
            batch_size=1,
        ) as parallel:
            for wave_start in range(0, len(bounds), n_workers):
                wave = bounds[wave_start : wave_start + n_workers]
                results = parallel(
                    delayed(func)(X.iloc[start:stop].copy(), *args)
                    for start, stop in wave
                )
                for (start, stop), result in zip(wave, results, strict=True):
                    yield start, stop, result
                del results

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

        Notes
        -----
        Spectra are processed in chunks of at most 256 rows, each chunk
        being one joblib task (prealign shift, peak detection and
        alignment), so memory
        use beyond the input and the output stays bounded however many
        spectra are transformed.
        """
        if not hasattr(self, "ref_spec_"):
            raise RuntimeError("Warping must be fitted before transform")

        if X.shape[1] != len(self.ref_spec_):
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match "
                f"reference spectrum length ({len(self.ref_spec_)})"
            )

        mz_axis = np.arange(X.shape[1])
        ref_peaks = np.empty(0, dtype=int)
        if self._needs_peaks():
            ref_peaks = self._reference_peaks(self.ref_spec_, X.columns)
        prealign_peaks = np.empty(0, dtype=int)
        if self.prealign_ref_ is not None:
            prealign_peaks = self._prealign_reference_peaks(X.columns)

        aligned = np.empty(X.shape, dtype=self._out_dtype(X))
        n_non_monotonic = 0
        for start, stop, (block, n_bad) in self._run_chunks(
            X, self._align_chunk, ref_peaks, prealign_peaks, mz_axis
        ):
            aligned[start:stop] = block
            n_non_monotonic += n_bad

        if n_non_monotonic:
            warnings.warn(
                f"Warping produced non-monotonic m/z mappings for "
                f"{n_non_monotonic}/{len(X)} sample(s); the global shift was "
                "used for them instead. Consider a rigid method or fewer "
                "segments.",
                UserWarning,
                stacklevel=2,
            )

        if self.renormalize:
            _renormalize(aligned)

        return pd.DataFrame(aligned, index=X.index, columns=X.columns, copy=False)

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
