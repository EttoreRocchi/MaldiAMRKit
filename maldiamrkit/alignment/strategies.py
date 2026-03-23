"""Shared alignment strategy classes for spectral warping.

Each strategy implements one alignment algorithm that can operate on either
binned (index-based) or raw (m/z-based) coordinate systems.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tslearn.metrics import dtw_path


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


def monotonic_interp(
    mz_axis: np.ndarray, new_positions: np.ndarray, row: np.ndarray
) -> np.ndarray:
    """Perform interpolation with monotonicity enforcement."""
    if np.all(np.diff(new_positions) > 0):
        return np.interp(mz_axis, new_positions, row, left=0.0, right=0.0)

    warnings.warn(
        "Warping produced non-monotonic m/z mapping for a sample. "
        "This may indicate poor alignment quality. "
        "Consider adjusting alignment parameters (e.g., reduce max_shift_da "
        "or increase n_segments).",
        UserWarning,
        stacklevel=4,
    )

    sort_idx = np.argsort(new_positions)
    new_positions_sorted = new_positions[sort_idx]
    row_sorted = row[sort_idx]

    unique_pos, inverse = np.unique(new_positions_sorted, return_inverse=True)
    counts = np.bincount(inverse)
    unique_intensities = (
        np.bincount(inverse, weights=row_sorted, minlength=len(unique_pos)) / counts
    )

    return np.interp(mz_axis, unique_pos, unique_intensities, left=0.0, right=0.0)


def _nearest_ref_indices(peaks: np.ndarray, ref_peaks: np.ndarray) -> np.ndarray:
    """For each peak, find the index of the nearest reference peak (O(P log R))."""
    idx = np.searchsorted(ref_peaks, peaks)
    idx = np.clip(idx, 0, len(ref_peaks) - 1)
    left = np.clip(idx - 1, 0, len(ref_peaks) - 1)
    use_left = np.abs(ref_peaks[left] - peaks) < np.abs(ref_peaks[idx] - peaks)
    idx[use_left] = left[use_left]
    return idx


def _match_peaks_to_ref(peaks: np.ndarray, ref_peaks: np.ndarray) -> np.ndarray:
    """For each peak, compute shift to nearest reference peak."""
    matched = ref_peaks[_nearest_ref_indices(peaks, ref_peaks)]
    return matched - peaks


def _match_peak_pairs(
    peaks: np.ndarray, ref_peaks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match peaks to nearest reference peaks. Returns (sample, ref) arrays."""
    matched = ref_peaks[_nearest_ref_indices(peaks, ref_peaks)]
    return peaks.copy(), matched


class ShiftStrategy(AlignmentStrategy):
    """Global median shift alignment."""

    def __init__(self, max_shift: float) -> None:
        self.max_shift = max_shift

    def align_binned(self, row, peaks, ref_peaks, mz_axis):
        """Apply global median shift to a binned spectrum."""
        if len(peaks) == 0 or len(ref_peaks) == 0:
            return row

        shifts = _match_peaks_to_ref(peaks, ref_peaks)
        shift = int(np.median(shifts)) if len(shifts) else 0
        shift = np.clip(shift, -self.max_shift, self.max_shift)

        if shift > 0:
            aligned = np.zeros_like(row)
            aligned[shift:] = row[:-shift]
        elif shift < 0:
            aligned = np.zeros_like(row)
            aligned[:shift] = row[-shift:]
        else:
            aligned = row.copy()
        return aligned

    def align_raw(self, mz, intensity, peaks_mz, ref_peaks_mz, ref_mz, ref_intensity):
        """Apply global m/z shift to a raw spectrum."""
        if len(peaks_mz) == 0 or len(ref_peaks_mz) == 0:
            return mz, intensity

        shifts = _match_peaks_to_ref(peaks_mz, ref_peaks_mz)
        shift_da = np.median(shifts) if len(shifts) else 0.0
        shift_da = np.clip(shift_da, -self.max_shift, self.max_shift)
        return mz + shift_da, intensity


class LinearStrategy(AlignmentStrategy):
    """Least-squares linear transformation alignment."""

    def __init__(self, max_shift: float) -> None:
        self.max_shift = max_shift
        self._fallback = ShiftStrategy(max_shift)

    def align_binned(self, row, peaks, ref_peaks, mz_axis):
        """Apply linear transformation alignment to a binned spectrum."""
        if len(peaks) < 2 or len(ref_peaks) < 2:
            return self._fallback.align_binned(row, peaks, ref_peaks, mz_axis)

        sample, ref = _match_peak_pairs(peaks, ref_peaks)
        A = np.vstack([sample, np.ones_like(sample)]).T
        a, b = np.linalg.lstsq(A, ref, rcond=None)[0]
        new_positions = a * mz_axis + b
        return monotonic_interp(mz_axis, new_positions, row)

    def align_raw(self, mz, intensity, peaks_mz, ref_peaks_mz, ref_mz, ref_intensity):
        """Apply linear m/z transformation to a raw spectrum."""
        if len(peaks_mz) < 2 or len(ref_peaks_mz) < 2:
            return self._fallback.align_raw(
                mz, intensity, peaks_mz, ref_peaks_mz, ref_mz, ref_intensity
            )

        sample, ref = _match_peak_pairs(peaks_mz, ref_peaks_mz)
        A = np.vstack([sample, np.ones_like(sample)]).T
        a, b = np.linalg.lstsq(A, ref, rcond=None)[0]
        return a * mz + b, intensity


class PiecewiseStrategy(AlignmentStrategy):
    """Piecewise linear alignment with smoothed local shifts."""

    def __init__(self, n_segments: int, smooth_sigma: float, max_shift: float) -> None:
        self.n_segments = n_segments
        self.smooth_sigma = smooth_sigma
        self.max_shift = max_shift

    def _compute_segment_shifts(
        self, sample: np.ndarray, ref: np.ndarray
    ) -> tuple[list, list]:
        """Compute per-segment median positions and shifts."""
        quantiles = np.linspace(0, 1, self.n_segments + 1)
        boundaries = np.quantile(sample, quantiles)

        seg_x, seg_shift = [], []
        for q in range(self.n_segments):
            if q == self.n_segments - 1:
                mask = (sample >= boundaries[q]) & (sample <= boundaries[q + 1])
            else:
                mask = (sample >= boundaries[q]) & (sample < boundaries[q + 1])

            if mask.sum() > 0:
                seg_x.append(np.median(sample[mask]))
                seg_shift.append(np.median(ref[mask] - sample[mask]))

        return seg_x, seg_shift

    def align_binned(self, row, peaks, ref_peaks, mz_axis):
        """Apply piecewise alignment to a binned spectrum."""
        if len(peaks) == 0 or len(ref_peaks) == 0:
            return row

        sample, ref = _match_peak_pairs(peaks, ref_peaks)
        seg_x, seg_shift = self._compute_segment_shifts(sample, ref)
        if len(seg_x) == 0:
            return row

        shift_interp = np.interp(
            mz_axis, seg_x, seg_shift, left=seg_shift[0], right=seg_shift[-1]
        )
        if self.smooth_sigma > 0:
            shift_interp = gaussian_filter1d(
                shift_interp, sigma=self.smooth_sigma, mode="nearest"
            )

        new_positions = mz_axis + shift_interp
        return monotonic_interp(mz_axis, new_positions, row)

    def align_raw(self, mz, intensity, peaks_mz, ref_peaks_mz, ref_mz, ref_intensity):
        """Apply piecewise m/z transformation to a raw spectrum."""
        if len(peaks_mz) == 0 or len(ref_peaks_mz) == 0:
            return mz, intensity

        sample, ref = _match_peak_pairs(peaks_mz, ref_peaks_mz)
        seg_x, seg_shift = self._compute_segment_shifts(sample, ref)
        if len(seg_x) == 0:
            return mz, intensity

        shift_interp = np.interp(
            mz, seg_x, seg_shift, left=seg_shift[0], right=seg_shift[-1]
        )
        if self.smooth_sigma > 0:
            mz_spacing = np.median(np.diff(mz))
            sigma_points = min(int(self.smooth_sigma / mz_spacing), len(mz) // 4)
            if sigma_points > 1:
                shift_interp = gaussian_filter1d(
                    shift_interp, sigma=sigma_points, mode="nearest"
                )

        return mz + shift_interp, intensity


class DTWStrategy(AlignmentStrategy):
    """Dynamic time warping alignment."""

    def __init__(self, dtw_radius: int) -> None:
        self.dtw_radius = dtw_radius

    def _dtw_align(self, query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Core DTW alignment returning aligned intensity."""
        path, _ = dtw_path(
            query,
            reference,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=self.dtw_radius,
        )

        aligned_sum = np.zeros_like(reference)
        aligned_count = np.zeros_like(reference)

        for i, j in path:
            if 0 <= j < len(aligned_sum):
                aligned_sum[j] += query[i]
                aligned_count[j] += 1

        aligned = np.zeros_like(reference)
        mask = aligned_count > 0
        aligned[mask] = aligned_sum[mask] / aligned_count[mask]
        return aligned

    def align_binned(self, row, peaks, ref_peaks, mz_axis):
        """Raise because DTW binned alignment requires the full reference spectrum.

        DTW binned alignment is handled directly by the Warping class
        which has access to the stored reference spectrum.
        """
        raise NotImplementedError(
            "DTW binned alignment is handled directly by the Warping class"
        )

    def align_raw(self, mz, intensity, peaks_mz, ref_peaks_mz, ref_mz, ref_intensity):
        """Apply DTW alignment to a raw spectrum."""
        query_intensity = np.interp(ref_mz, mz, intensity)
        aligned_intensity = self._dtw_align(query_intensity, ref_intensity)
        return ref_mz, aligned_intensity


ALIGNMENT_REGISTRY: dict[str, type[AlignmentStrategy]] = {
    "shift": ShiftStrategy,
    "linear": LinearStrategy,
    "piecewise": PiecewiseStrategy,
    "dtw": DTWStrategy,
}
