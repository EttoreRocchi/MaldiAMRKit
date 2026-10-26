"""Shared alignment strategy classes for spectral warping.

Each strategy implements one alignment algorithm that can operate on either
binned (index-based) or raw (m/z-based) coordinate systems.
"""

from __future__ import annotations

import numbers
import warnings
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_prominences
from scipy.stats import median_abs_deviation


class AlignmentMethod(str, Enum):
    """Supported alignment/warping methods.

    Attributes
    ----------
    shift : str
        Rigid global shift alignment.
    linear : str
        Linear (affine) recalibration.
    piecewise : str
        Piecewise-linear recalibration.
    dtw : str
        Dynamic time warping alignment.
    quadratic : str
        Quadratic polynomial recalibration.
    cubic : str
        Cubic polynomial recalibration.
    lowess : str
        Non-linear LOWESS (Cleveland 1979) recalibration.
    """

    shift = "shift"
    linear = "linear"
    piecewise = "piecewise"
    dtw = "dtw"
    quadratic = "quadratic"
    cubic = "cubic"
    lowess = "lowess"


class AlignmentStrategy(ABC):
    """Base class for alignment strategies."""

    @abstractmethod
    def align_binned(
        self,
        row: np.ndarray,
        peaks: np.ndarray,
        ref_peaks: np.ndarray,
        mz_axis: np.ndarray,
    ) -> np.ndarray:
        """Align a binned spectrum row to the reference.

        Parameters
        ----------
        row : np.ndarray
            Intensity values of the spectrum to align.
        peaks : np.ndarray
            Detected peak indices in ``row``.
        ref_peaks : np.ndarray
            Detected peak indices in the reference spectrum.
        mz_axis : np.ndarray
            Array of bin positions (e.g. ``np.arange(len(row))``).

        Returns
        -------
        np.ndarray
            Aligned intensity array with the same length as ``row``.
        """

    @abstractmethod
    def align_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray,
        ref_peaks_mz: np.ndarray,
        ref_mz: np.ndarray,
        ref_intensity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align a raw spectrum to the reference.

        Parameters
        ----------
        mz : np.ndarray
            m/z values of the spectrum to align.
        intensity : np.ndarray
            Intensity values of the spectrum to align.
        peaks_mz : np.ndarray
            Detected peak m/z positions in the sample spectrum.
        ref_peaks_mz : np.ndarray
            Detected peak m/z positions in the reference spectrum.
        ref_mz : np.ndarray
            m/z values of the reference spectrum.
        ref_intensity : np.ndarray
            Intensity values of the reference spectrum.

        Returns
        -------
        aligned_mz : np.ndarray
            Aligned m/z values.
        aligned_intensity : np.ndarray
            Aligned intensity values.
        """


_SHIFT_ESTIMATORS = ("peaks", "xcorr")

_NON_MONOTONIC = "non-monotonic m/z mapping"


def _warn_non_monotonic() -> None:
    """Signal that a warp was rejected for not being strictly increasing."""
    warnings.warn(
        f"Warping produced a {_NON_MONOTONIC} for a sample; the global shift "
        "was used for it instead.",
        UserWarning,
        stacklevel=4,
    )


def _shift_row(row: np.ndarray, shift: int) -> np.ndarray:
    """Shift a binned row by ``shift`` bins, filling the vacated bins with 0."""
    out = np.zeros_like(row)
    if shift > 0:
        out[shift:] = row[:-shift]
    elif shift < 0:
        out[:shift] = row[-shift:]
    else:
        out[:] = row
    return out


def _pearson_by_lag(
    row: np.ndarray, reference: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation with ``reference`` of ``row`` shifted by each lag.

    The shifted row is zero-filled exactly as :func:`_shift_row` produces it,
    so each value is the correlation the aligned output would have. Lags run
    from ``-max_lag`` to ``max_lag``; undefined correlations (flat inputs)
    are ``-inf``.
    """
    row = np.asarray(row, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n = len(row)
    max_lag = max(0, min(int(max_lag), n - 1))
    lags = np.arange(-max_lag, max_lag + 1)
    dot = np.correlate(np.pad(reference, max_lag), row, mode="valid")
    cum = np.concatenate(([0.0], np.cumsum(row)))
    cum_sq = np.concatenate(([0.0], np.cumsum(row * row)))
    start = np.where(lags >= 0, 0, -lags)
    stop = np.where(lags >= 0, n - lags, n)
    s1 = cum[stop] - cum[start]
    s2 = cum_sq[stop] - cum_sq[start]
    r1 = reference.sum()
    r2 = np.dot(reference, reference)
    denom = np.sqrt(np.maximum(n * s2 - s1 * s1, 0.0) * max(n * r2 - r1 * r1, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, (n * dot - s1 * r1) / denom, -np.inf)
    return lags, corr


def _gated_lag(
    row: np.ndarray,
    reference: np.ndarray,
    max_lag: int,
    min_gain: float,
    bounds: tuple[int, int] | None = None,
) -> int:
    """Lag that best correlates ``row`` with ``reference``, if it pays off.

    Returns the lag in ``[-max_lag, max_lag]`` (or in ``bounds``, which
    must contain 0) with the highest Pearson correlation (the smallest in
    absolute value among ties), or 0 when that correlation does not exceed
    the unshifted one by at least ``min_gain`` times its magnitude, or when
    the unshifted correlation is undefined.
    """
    lo, hi = (-max_lag, max_lag) if bounds is None else bounds
    lags, corr = _pearson_by_lag(row, reference, max(-lo, hi))
    inside = (lags >= lo) & (lags <= hi)
    lags, corr = lags[inside], corr[inside]
    unshifted = corr[lags == 0][0]
    best = corr.max()
    if not (np.isfinite(unshifted) and np.isfinite(best)):
        return 0
    candidates = lags[corr == best]
    lag = int(candidates[np.argmin(np.abs(candidates))])
    if lag == 0 or best < unshifted + min_gain * abs(unshifted):
        return 0
    return lag


def _nearest_ref_indices(peaks: np.ndarray, ref_peaks: np.ndarray) -> np.ndarray:
    """For each peak, find the index of the nearest reference peak (O(P log R))."""
    idx = np.searchsorted(ref_peaks, peaks)
    idx = np.clip(idx, 0, len(ref_peaks) - 1)
    left = np.clip(idx - 1, 0, len(ref_peaks) - 1)
    use_left = np.abs(ref_peaks[left] - peaks) < np.abs(ref_peaks[idx] - peaks)
    idx[use_left] = left[use_left]
    return idx


def _most_prominent(
    intensity: np.ndarray, peaks: np.ndarray, n: int | None
) -> np.ndarray:
    """Keep the ``n`` most prominent of ``peaks`` (indices into ``intensity``).

    The kept peaks are returned sorted; all of ``peaks`` is kept when ``n``
    is None or not smaller than their number.
    """
    peaks = np.asarray(peaks, dtype=int)
    if n is None or len(peaks) <= n:
        return peaks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prominences = peak_prominences(intensity, peaks)[0]
    keep = np.argsort(-prominences, kind="stable")[:n]
    return np.sort(peaks[keep])


def _mutual_pairs(
    peaks: np.ndarray, ref_peaks: np.ndarray, offset: float, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mutually nearest (sample, reference) peak pairs around ``offset``.

    The sample peaks are first moved by ``offset``. A pair is kept when each
    peak is the other's nearest one and the two lie within ``tolerance`` of
    each other. Returns the original sample positions and their reference
    partners (both sorted, as the inputs are).
    """
    peaks = np.asarray(peaks, dtype=float)
    ref_peaks = np.asarray(ref_peaks, dtype=float)
    if len(peaks) == 0 or len(ref_peaks) == 0:
        return np.empty(0), np.empty(0)
    moved = peaks + offset
    to_ref = _nearest_ref_indices(moved, ref_peaks)
    to_sample = _nearest_ref_indices(ref_peaks, moved)
    keep = (to_sample[to_ref] == np.arange(len(peaks))) & (
        np.abs(ref_peaks[to_ref] - moved) <= tolerance
    )
    return peaks[keep], ref_peaks[to_ref[keep]]


def _uniform_grid(mz: np.ndarray) -> tuple[np.ndarray, float]:
    """Uniform grid spanning ``mz`` at its median spacing."""
    step = float(np.median(np.diff(mz)))
    return np.arange(mz[0], mz[-1] + step / 2, step), step


def _strictly_increasing(positions: np.ndarray) -> bool:
    """Whether every warped position lies after the previous one."""
    return bool(np.all(np.diff(positions) > 0))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two vectors, NaN when undefined."""
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _beats(candidate, baseline, reference, min_gain: float) -> bool:
    """Whether ``candidate`` correlates with ``reference`` clearly better.

    "Clearly" means by at least ``min_gain`` times the magnitude of the
    baseline correlation.
    """
    c_new = _correlation(candidate, reference)
    c_old = _correlation(baseline, reference)
    if np.isnan(c_new):
        return False
    if np.isnan(c_old):
        return True
    return c_new >= c_old + min_gain * abs(c_old)


def _conservative_resample(
    mz_axis: np.ndarray, positions: np.ndarray, row: np.ndarray
) -> np.ndarray:
    """Resample a warped binned row onto ``mz_axis``, preserving intensity.

    Bin ``i`` of ``row`` (centred on ``mz_axis[i]``) is moved to
    ``positions[i]`` and its intensity is shared among the target bins it
    overlaps, so the total intensity is preserved apart from the part moved
    past either end of the axis. ``positions`` must be strictly increasing.
    """
    mz_axis = np.asarray(mz_axis, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if len(mz_axis) < 2:
        return np.asarray(row, dtype=np.float64).copy()

    def edges(centres: np.ndarray) -> np.ndarray:
        inner = (centres[:-1] + centres[1:]) / 2
        first = centres[0] - (centres[1] - centres[0]) / 2
        last = centres[-1] + (centres[-1] - centres[-2]) / 2
        return np.concatenate(([first], inner, [last]))

    cumulative = np.concatenate(([0.0], np.cumsum(row, dtype=np.float64)))
    return np.diff(np.interp(edges(mz_axis), edges(positions), cumulative))


class ShiftStrategy(AlignmentStrategy):
    """Global (rigid) shift alignment.

    Parameters
    ----------
    max_shift : float
        Largest shift considered, in bins for binned spectra and in Daltons
        for raw ones.
    estimator : {"peaks", "xcorr"}, default="peaks"
        How the shift is estimated. ``"peaks"`` takes the median offset
        between mutually nearest sample and reference peaks (pairs farther
        apart than ``max_shift`` are ignored). ``"xcorr"`` takes the lag that
        maximises the Pearson correlation with the reference spectrum.
    min_gain : float or None, default=None
        Relative correlation gain (a fraction of the unshifted correlation
        with the reference spectrum) required to apply a shift, when the
        reference spectrum is available. ``None`` disables the check for
        ``"peaks"`` and lets ``"xcorr"`` apply any improvement.
    """

    def __init__(
        self,
        max_shift: float,
        estimator: str = "peaks",
        min_gain: float | None = None,
    ) -> None:
        if estimator not in _SHIFT_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {_SHIFT_ESTIMATORS}, got {estimator!r}"
            )
        self.max_shift = max_shift
        self.estimator = estimator
        self.min_gain = min_gain

    def _median_peak_offset(
        self,
        peaks: np.ndarray,
        ref_peaks: np.ndarray,
        bounds: tuple[float, float] | None = None,
    ) -> float:
        sample, ref = _mutual_pairs(peaks, ref_peaks, 0.0, self.max_shift)
        if len(sample) == 0:
            return 0.0
        lo, hi = (-self.max_shift, self.max_shift) if bounds is None else bounds
        return float(np.clip(np.median(ref - sample), lo, hi))

    def estimate_binned(
        self,
        row: np.ndarray,
        peaks: np.ndarray,
        ref_peaks: np.ndarray,
        reference: np.ndarray | None = None,
        bounds: tuple[int, int] | None = None,
    ) -> int:
        """Estimate the shift (in bins) of a binned spectrum.

        ``reference`` (the reference spectrum) is required by ``"xcorr"``.
        ``bounds`` (containing 0) narrows the allowed shifts, by default
        ``[-max_shift, max_shift]``.
        """
        if self.estimator == "xcorr":
            if reference is None:
                raise ValueError("estimator='xcorr' needs the reference spectrum")
            return _gated_lag(
                row, reference, int(self.max_shift), self.min_gain or 0.0, bounds
            )
        shift = int(np.round(self._median_peak_offset(peaks, ref_peaks, bounds)))
        if (
            shift != 0
            and reference is not None
            and self.min_gain is not None
            and not _beats(_shift_row(row, shift), row, reference, self.min_gain)
        ):
            return 0
        return shift

    def align_binned(
        self,
        row: np.ndarray,
        peaks: np.ndarray,
        ref_peaks: np.ndarray,
        mz_axis: np.ndarray,
        *,
        reference: np.ndarray | None = None,
        shift_bounds: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Apply the estimated global shift to a binned spectrum."""
        shift = self.estimate_binned(row, peaks, ref_peaks, reference, shift_bounds)
        return _shift_row(row, shift)

    def estimate_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray,
        ref_peaks_mz: np.ndarray,
        ref_mz: np.ndarray,
        ref_intensity: np.ndarray,
        bounds: tuple[float, float] | None = None,
    ) -> float:
        """Estimate the shift (in Daltons) of a raw spectrum.

        With ``"xcorr"``, both spectra are resampled onto a uniform grid at
        the reference spacing and the lag is searched in whole grid steps.
        ``bounds`` (containing 0) narrows the allowed shifts, by default
        ``[-max_shift, max_shift]``.
        """
        if self.estimator == "xcorr":
            grid, step = _uniform_grid(ref_mz)
            sample = np.interp(grid, mz, intensity, left=0.0, right=0.0)
            reference = np.interp(grid, ref_mz, ref_intensity)
            max_lag = int(self.max_shift // step)
            lag_bounds = None
            if bounds is not None:
                lag_bounds = (
                    min(0, int(np.ceil(bounds[0] / step - 1e-9))),
                    max(0, int(np.floor(bounds[1] / step + 1e-9))),
                )
            lag = _gated_lag(
                sample, reference, max_lag, self.min_gain or 0.0, lag_bounds
            )
            return lag * step
        shift = self._median_peak_offset(peaks_mz, ref_peaks_mz, bounds)
        if shift != 0 and self.min_gain is not None:
            grid, _ = _uniform_grid(ref_mz)
            shifted = np.interp(grid, mz + shift, intensity, left=0.0, right=0.0)
            unshifted = np.interp(grid, mz, intensity, left=0.0, right=0.0)
            reference = np.interp(grid, ref_mz, ref_intensity)
            if not _beats(shifted, unshifted, reference, self.min_gain):
                return 0.0
        return shift

    def align_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray,
        ref_peaks_mz: np.ndarray,
        ref_mz: np.ndarray,
        ref_intensity: np.ndarray,
        *,
        shift_bounds: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the estimated global m/z shift to a raw spectrum."""
        shift = self.estimate_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, ref_mz, ref_intensity, shift_bounds
        )
        return mz + shift, intensity


def _robust_linear_fit(
    sample: np.ndarray,
    ref: np.ndarray,
    residual_threshold: float = 3.0,
) -> tuple[float, float]:
    """Fit mz' = a*mz + b with MAD-based outlier rejection.

    Performs a least-squares fit, then rejects peak pairs whose residual
    is more than ``residual_threshold`` MAD-based standard deviations (the
    MAD divided by ``Phi^{-1}(3/4)``) from the median residual, and refits.

    Parameters
    ----------
    sample, ref : np.ndarray
        Matched peak positions (sample and reference).
    residual_threshold : float, default=3.0
        Number of MAD-based sigma units for outlier rejection.

    Returns
    -------
    a, b : float
        Coefficients of the linear fit ``ref = a * sample + b``.
    """
    A = np.vstack([sample, np.ones_like(sample)]).T
    a, b = np.linalg.lstsq(A, ref, rcond=1e-10)[0]

    if len(sample) > 2:
        residuals = ref - (a * sample + b)
        sigma = median_abs_deviation(residuals, scale="normal")
        if sigma > 0:
            cutoff = residual_threshold * sigma
            inlier_mask = np.abs(residuals - np.median(residuals)) <= cutoff
            if inlier_mask.sum() >= 2:
                A_inlier = A[inlier_mask]
                a, b = np.linalg.lstsq(A_inlier, ref[inlier_mask], rcond=1e-10)[0]

    return a, b


class _PeakFitStrategy(AlignmentStrategy):
    """Warp fitted on matched peak pairs around a global shift.

    The global shift (see :class:`ShiftStrategy`) is estimated first and
    serves as a prior: peaks are paired only when mutually nearest within
    ``max_shift`` of it. The fitted warp must be strictly increasing and,
    when ``min_support`` is given, must bring at least that share of the
    pairs within ``max_shift / 3`` of their reference partner; otherwise
    the shifted spectrum is returned.
    """

    _min_pairs = 2

    def __init__(
        self,
        max_shift: float,
        estimator: str = "peaks",
        min_gain: float | None = None,
        min_support: float | None = None,
    ) -> None:
        self.max_shift = max_shift
        self.min_support = min_support
        self._shift = ShiftStrategy(max_shift, estimator, min_gain)

    @property
    def estimator(self) -> str:
        return self._shift.estimator

    @property
    def min_gain(self) -> float | None:
        return self._shift.min_gain

    @abstractmethod
    def _positions(
        self, sample: np.ndarray, ref: np.ndarray, axis: np.ndarray, raw: bool
    ) -> np.ndarray | None:
        """Warped position of every point of ``axis``, or None if no fit."""

    def _supported(
        self,
        sample: np.ndarray,
        ref: np.ndarray,
        axis: np.ndarray,
        positions: np.ndarray,
    ) -> bool:
        """Whether the warp brings enough pairs close to their partner."""
        if self.min_support is None:
            return True
        residual = ref - np.interp(sample, axis, positions)
        close = np.abs(residual) <= self.max_shift / 3
        return bool(np.mean(close) >= self.min_support)

    def align_binned(
        self,
        row: np.ndarray,
        peaks: np.ndarray,
        ref_peaks: np.ndarray,
        mz_axis: np.ndarray,
        *,
        reference: np.ndarray | None = None,
        shift_bounds: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Warp a binned spectrum, resampling it so its intensity is preserved."""
        shift = self._shift.estimate_binned(
            row, peaks, ref_peaks, reference, shift_bounds
        )
        shifted = _shift_row(row, shift)
        sample, ref = _mutual_pairs(peaks, ref_peaks, shift, self.max_shift)
        if len(sample) < self._min_pairs:
            return shifted
        axis = np.asarray(mz_axis, dtype=np.float64)
        positions = self._positions(sample, ref, axis, raw=False)
        if positions is None:
            return shifted
        if not _strictly_increasing(positions):
            _warn_non_monotonic()
            return shifted
        if not self._supported(sample, ref, axis, positions):
            return shifted
        return _conservative_resample(axis, positions, row)

    def align_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray,
        ref_peaks_mz: np.ndarray,
        ref_mz: np.ndarray,
        ref_intensity: np.ndarray,
        *,
        shift_bounds: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp the m/z axis of a raw spectrum (intensities are unchanged)."""
        shift = self._shift.estimate_raw(
            mz, intensity, peaks_mz, ref_peaks_mz, ref_mz, ref_intensity, shift_bounds
        )
        shifted_mz = mz + shift
        sample, ref = _mutual_pairs(peaks_mz, ref_peaks_mz, shift, self.max_shift)
        if len(sample) < self._min_pairs:
            return shifted_mz, intensity
        axis = np.asarray(mz, dtype=np.float64)
        new_mz = self._positions(sample, ref, axis, raw=True)
        if new_mz is None:
            return shifted_mz, intensity
        if not _strictly_increasing(new_mz):
            _warn_non_monotonic()
            return shifted_mz, intensity
        if not self._supported(sample, ref, axis, new_mz):
            return shifted_mz, intensity
        return new_mz, intensity


class LinearStrategy(_PeakFitStrategy):
    """Least-squares linear transformation alignment with outlier rejection.

    Parameters
    ----------
    max_shift : float
        Largest shift considered (see :class:`ShiftStrategy`), which also
        bounds the distance between matched peak pairs.
    estimator, min_gain
        See :class:`ShiftStrategy` (they apply to the global shift used as
        a prior).
    min_support : float or None, default=None
        Share of the matched pairs that the warp must bring within
        ``max_shift / 3`` of their reference partner to be used instead of
        the global shift. ``None`` disables the check.
    """

    def _positions(self, sample, ref, axis, raw):
        a, b = _robust_linear_fit(sample, ref)
        return a * axis + b


class PiecewiseStrategy(_PeakFitStrategy):
    """Piecewise linear alignment with smoothed local shifts.

    The matched peaks are split into ``n_segments`` groups of equal size
    along the m/z axis. The shift of each group is the mean of the peak
    offsets lying within ``max_shift / 3`` of the group's median offset
    (robust to stray pairs, but finer than the whole bins a binned median
    would give), or the median itself when no offset lies that close, and
    the shifts are interpolated between the groups.

    Parameters
    ----------
    n_segments : int
        Number of groups of matched peaks.
    smooth_sigma : float
        Gaussian smoothing of the interpolated shifts (in bins for binned
        spectra and in Daltons for raw ones; 0 disables it).
    max_shift : float
        Largest shift considered (see :class:`ShiftStrategy`).
    estimator, min_gain
        See :class:`ShiftStrategy`.
    min_support : float or None, default=None
        See :class:`LinearStrategy`.
    """

    _min_pairs = 1

    def __init__(
        self,
        n_segments: int,
        smooth_sigma: float,
        max_shift: float,
        estimator: str = "peaks",
        min_gain: float | None = None,
        min_support: float | None = None,
    ) -> None:
        super().__init__(max_shift, estimator, min_gain, min_support)
        self.n_segments = n_segments
        self.smooth_sigma = smooth_sigma

    def _compute_segment_shifts(
        self, sample: np.ndarray, ref: np.ndarray
    ) -> tuple[list, list]:
        """Compute per-segment median positions and trimmed-mean shifts."""
        quantiles = np.linspace(0, 1, self.n_segments + 1)
        boundaries = np.quantile(sample, quantiles)

        seg_x, seg_shift = [], []
        for q in range(self.n_segments):
            if q == self.n_segments - 1:
                mask = (sample >= boundaries[q]) & (sample <= boundaries[q + 1])
            else:
                mask = (sample >= boundaries[q]) & (sample < boundaries[q + 1])

            if mask.sum() > 0:
                offsets = ref[mask] - sample[mask]
                centre = np.median(offsets)
                near = np.abs(offsets - centre) <= self.max_shift / 3
                seg_x.append(np.median(sample[mask]))
                seg_shift.append(offsets[near].mean() if near.any() else centre)

        return seg_x, seg_shift

    def _positions(self, sample, ref, axis, raw):
        seg_x, seg_shift = self._compute_segment_shifts(sample, ref)
        if len(seg_x) == 0:
            return None
        shift = np.interp(
            axis, seg_x, seg_shift, left=seg_shift[0], right=seg_shift[-1]
        )
        if raw:
            if len(axis) > 1:
                spacing = np.median(np.diff(axis))
                sigma = min(int(self.smooth_sigma / spacing), len(axis) // 4)
                if sigma > 1:
                    shift = gaussian_filter1d(shift, sigma=sigma, mode="nearest")
        elif self.smooth_sigma > 0:
            shift = gaussian_filter1d(shift, sigma=self.smooth_sigma, mode="nearest")
        return axis + shift


def _use_banded_dtw(query: np.ndarray, reference: np.ndarray, radius) -> bool:
    """Whether the banded DTW kernel applies (otherwise tslearn is used).

    It needs two finite 1-D series of the same length and a non-negative
    integer radius whose band (``2 * radius + 1``) is narrower than the
    series; tslearn handles every other case as before.
    """
    if isinstance(radius, bool) or not isinstance(radius, numbers.Integral):
        return False
    n = len(reference)
    return (
        query.ndim == 1
        and reference.ndim == 1
        and len(query) == n
        and n >= 1
        and 0 <= radius
        and 2 * radius + 1 <= n
        and bool(np.isfinite(query).all())
        and bool(np.isfinite(reference).all())
    )


class DTWStrategy(AlignmentStrategy):
    """Dynamic time warping alignment.

    DTW maps the sample onto the reference grid by averaging the sample
    points matched to each reference point, so the total intensity is not
    preserved (repeated or averaged points change it).

    Parameters
    ----------
    dtw_radius : int
        Sakoe-Chiba radius constraining the warping path.
    min_gain : float or None, default=None
        When set, the DTW result is kept only if it raises the correlation
        with the reference by at least ``min_gain`` times the correlation of
        the unwarped spectrum; otherwise the spectrum is returned unchanged.
    """

    def __init__(self, dtw_radius: int, min_gain: float | None = None) -> None:
        self.dtw_radius = dtw_radius
        self.min_gain = min_gain

    def align_to_reference(self, row: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """DTW-align a binned row to ``reference`` (subject to ``min_gain``)."""
        aligned = self._dtw_align(row, reference)
        if self.min_gain is not None and not _beats(
            aligned, row, reference, self.min_gain
        ):
            return np.asarray(row).copy()
        return aligned

    def _dtw_align(self, query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Core DTW alignment returning aligned intensity."""
        query = np.asarray(query)
        reference = np.asarray(reference)
        if _use_banded_dtw(query, reference, self.dtw_radius):
            # Lazy import: the kernel compiles with numba on first use.
            from ._dtw import _dtw_path

            path_i, path_j = _dtw_path(query, reference, self.dtw_radius)
        else:
            from tslearn.metrics import dtw_path

            path, _ = dtw_path(
                query,
                reference,
                global_constraint="sakoe_chiba",
                sakoe_chiba_radius=self.dtw_radius,
            )
            path = np.asarray(path, dtype=np.int64).reshape(-1, 2)
            path_i, path_j = path[:, 0], path[:, 1]

        aligned_sum = np.zeros_like(reference)
        aligned_count = np.zeros_like(reference)
        # Unbuffered, in path order: the same sums as a loop over the path.
        np.add.at(aligned_sum, path_j, query[path_i])
        np.add.at(aligned_count, path_j, 1)

        aligned = np.zeros_like(reference)
        mask = aligned_count > 0
        aligned[mask] = aligned_sum[mask] / aligned_count[mask]
        return aligned

    def align_binned(
        self,
        row: np.ndarray,
        peaks: np.ndarray,
        ref_peaks: np.ndarray,
        mz_axis: np.ndarray,
    ) -> np.ndarray:
        """Raise because DTW binned alignment requires the full reference spectrum.

        DTW binned alignment is handled directly by the Warping class
        which has access to the stored reference spectrum.
        """
        raise NotImplementedError(
            "DTW binned alignment is handled directly by the Warping class"
        )

    def align_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray,
        ref_peaks_mz: np.ndarray,
        ref_mz: np.ndarray,
        ref_intensity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply DTW alignment to a raw spectrum."""
        query_intensity = np.interp(ref_mz, mz, intensity)
        aligned_intensity = self._dtw_align(query_intensity, ref_intensity)
        if self.min_gain is not None and not _beats(
            aligned_intensity, query_intensity, ref_intensity, self.min_gain
        ):
            return mz, intensity
        return ref_mz, aligned_intensity


class PolynomialStrategy(_PeakFitStrategy):
    """Polynomial alignment via ``numpy.polyfit`` on matched peak pairs.

    Used for both the ``"quadratic"`` (``degree=2``) and ``"cubic"``
    (``degree=3``) warping methods. When fewer than ``degree + 1`` matched
    peak pairs are available, the global shift is used instead.

    Parameters
    ----------
    max_shift : float
        Largest shift considered (see :class:`ShiftStrategy`).
    degree : int
        Polynomial degree. ``2`` for quadratic, ``3`` for cubic.
    estimator, min_gain
        See :class:`ShiftStrategy`.
    min_support : float or None, default=None
        See :class:`LinearStrategy`.
    """

    def __init__(
        self,
        max_shift: float,
        degree: int,
        estimator: str = "peaks",
        min_gain: float | None = None,
        min_support: float | None = None,
    ) -> None:
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        super().__init__(max_shift, estimator, min_gain, min_support)
        self.degree = degree
        self._min_pairs = degree + 1

    def _positions(self, sample, ref, axis, raw):
        return np.polyval(np.polyfit(sample, ref, self.degree), axis)


def _lowess_fit(
    x: np.ndarray,
    y: np.ndarray,
    frac: float,
    it: int,
) -> np.ndarray:
    """Locally-weighted scatterplot smoothing (Cleveland 1979).

    Lightweight pure-numpy implementation that fits a local linear
    regression at each data point using a tricube kernel on the
    ``frac * n`` nearest neighbours, with ``it`` robustness reweighting
    iterations using bisquare residual weights. Input ``x`` must be
    sorted in ascending order.

    The iterations stop once the residuals are negligible next to ``y``,
    and a point whose window the robustness weights leave (almost) empty
    keeps its fit from the previous iteration, so the fit cannot jump to
    a distant point.

    Parameters
    ----------
    x, y : np.ndarray
        One-dimensional arrays of the same length; ``x`` sorted.
    frac : float
        Smoothing bandwidth, fraction of data used for each local fit,
        in the half-open interval ``(0, 1]``.
    it : int
        Number of robustness reweighting iterations. ``0`` disables it.

    Returns
    -------
    np.ndarray
        Smoothed ``y`` values at each ``x``.
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=float)
    k = max(int(np.ceil(frac * n)), 2)
    k = min(k, n)

    fitted = np.asarray(y, dtype=float).copy()
    residual_weights = np.ones(n, dtype=float)
    # As in Cleveland's algorithm, residuals this small count as an exact fit.
    tolerance = 1e-7 * np.mean(np.abs(y))

    for _ in range(it + 1):
        previous = fitted.copy()
        for i in range(n):
            distances = np.abs(x - x[i])
            nearest = np.argsort(distances)[:k]
            h = distances[nearest].max()
            if h == 0:
                fitted[i] = np.average(y[nearest], weights=residual_weights[nearest])
                continue
            u = np.clip(distances[nearest] / h, 0.0, 1.0)
            kernel = (1.0 - u**3) ** 3
            w = kernel * residual_weights[nearest]
            total = w.sum()
            if total <= 1e-3 * kernel.sum():
                # The robustness weights left (almost) nothing in the window.
                fitted[i] = previous[i]
                continue
            xn = x[nearest]
            yn = y[nearest]
            mean_x = np.sum(w * xn) / total
            mean_y = np.sum(w * yn) / total
            dx = xn - mean_x
            denom = np.sum(w * dx * dx)
            if denom <= 0:
                fitted[i] = mean_y
            else:
                beta = np.sum(w * dx * (yn - mean_y)) / denom
                fitted[i] = mean_y + beta * (x[i] - mean_x)

        residuals = y - fitted
        s = np.median(np.abs(residuals))
        if 6.0 * s <= tolerance:
            break
        u = np.clip(residuals / (6.0 * s), -1.0, 1.0)
        residual_weights = (1.0 - u**2) ** 2

    return fitted


class LOWESSStrategy(_PeakFitStrategy):
    """Non-linear LOWESS alignment on matched peak pairs.

    Fits a LOWESS (Cleveland 1979) regression of the shift on the matched
    peak pairs, interpolates it over the whole axis and keeps it constant
    beyond the first and last matched peak.

    Parameters
    ----------
    max_shift : float
        Largest shift considered (see :class:`ShiftStrategy`).
    frac : float
        LOWESS smoothing bandwidth (fraction of matched peaks used for
        each local fit). Must be in ``(0, 1]``.
    it : int
        Number of LOWESS robustness iterations. Must be ``>= 0``.
    estimator, min_gain
        See :class:`ShiftStrategy`.
    min_support : float or None, default=None
        See :class:`LinearStrategy`.
    """

    _min_pairs = 3

    def __init__(
        self,
        max_shift: float,
        frac: float,
        it: int,
        estimator: str = "peaks",
        min_gain: float | None = None,
        min_support: float | None = None,
    ) -> None:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"frac must be in (0, 1], got {frac}")
        if it < 0:
            raise ValueError(f"it must be >= 0, got {it}")
        super().__init__(max_shift, estimator, min_gain, min_support)
        self.frac = frac
        self.it = it

    def _fit_positions(
        self, sample: np.ndarray, ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (sorted_sample, fitted_ref) on the matched peak pairs."""
        order = np.argsort(sample)
        sample_sorted = sample[order].astype(float)
        ref_sorted = ref[order].astype(float)
        fitted = _lowess_fit(sample_sorted, ref_sorted, self.frac, self.it)
        return sample_sorted, fitted

    def _positions(self, sample, ref, axis, raw):
        sample_sorted, fitted = self._fit_positions(sample, ref)
        return axis + np.interp(axis, sample_sorted, fitted - sample_sorted)


_ALIGNMENT_REGISTRY: dict[str, type[AlignmentStrategy]] = {
    "shift": ShiftStrategy,
    "linear": LinearStrategy,
    "piecewise": PiecewiseStrategy,
    "dtw": DTWStrategy,
    "quadratic": PolynomialStrategy,
    "cubic": PolynomialStrategy,
    "lowess": LOWESSStrategy,
}
