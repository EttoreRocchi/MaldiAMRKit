"""Fit-free peak-m/z alignment for peak sets.

:func:`align_peaks` warps the m/z positions of a :class:`~maldiamrkit.detection.PeakSet`
onto a reference peak list using the shared alignment strategies. It is
**fit-free**: the reference peaks are supplied by the caller, which is expected
to estimate them on the training split only (in the consumer's fold). The
intensities are carried through unchanged.
"""

from __future__ import annotations

import numpy as np

from ..detection.peaklist import PeakSet
from .strategies import ALIGNMENT_REGISTRY, AlignmentMethod, AlignmentStrategy


def _build_strategy(
    method: AlignmentMethod,
    max_shift_da: float,
    n_segments: int,
    smooth_sigma: float,
    lowess_frac: float,
    lowess_it: int,
) -> AlignmentStrategy:
    """Instantiate the requested alignment strategy with its parameters."""
    cls = ALIGNMENT_REGISTRY[method]
    if method in (AlignmentMethod.shift, AlignmentMethod.linear):
        return cls(max_shift=max_shift_da)
    if method == AlignmentMethod.piecewise:
        return cls(
            n_segments=n_segments, smooth_sigma=smooth_sigma, max_shift=max_shift_da
        )
    if method == AlignmentMethod.quadratic:
        return cls(max_shift=max_shift_da, degree=2)
    if method == AlignmentMethod.cubic:
        return cls(max_shift=max_shift_da, degree=3)
    if method == AlignmentMethod.lowess:
        return cls(max_shift=max_shift_da, frac=lowess_frac, it=lowess_it)
    return cls()


def align_peaks(
    peaks: PeakSet,
    ref_peaks_mz: np.ndarray,
    *,
    method: str | AlignmentMethod = "shift",
    max_shift_da: float = 50.0,
    n_segments: int = 5,
    smooth_sigma: float = 2.0,
    lowess_frac: float = 0.3,
    lowess_it: int = 3,
) -> PeakSet:
    """Align a peak set's m/z positions to a reference peak list.

    Fit-free: ``ref_peaks_mz`` is supplied by the caller. Builds the requested
    warping transform from the matched (sample, reference) peak pairs and applies
    it to the peak m/z; intensities are unchanged.

    Parameters
    ----------
    peaks : PeakSet
        Peak set to align.
    ref_peaks_mz : array-like
        Reference peak m/z positions to align to.
    method : {"shift", "linear", "piecewise", "quadratic", "cubic", "lowess"}, default="shift"
        Warping strategy. ``"dtw"`` is unsupported for peak sets because it
        resamples onto a dense grid and would destroy the set structure.
    max_shift_da : float, default=50.0
        Maximum allowed shift in Daltons (used by the shift fallback and the
        rigid/linear strategies).
    n_segments : int, default=5
        Number of segments for ``method="piecewise"``.
    smooth_sigma : float, default=2.0
        Gaussian smoothing (Da) for piecewise transitions.
    lowess_frac : float, default=0.3
        LOWESS bandwidth for ``method="lowess"``.
    lowess_it : int, default=3
        LOWESS robustness iterations for ``method="lowess"``.

    Returns
    -------
    PeakSet
        A new peak set with warped m/z and unchanged intensities.

    Raises
    ------
    ValueError
        If ``method="dtw"``.

    Notes
    -----
    ``"shift"``, ``"linear"``, ``"quadratic"``, ``"cubic"`` and ``"lowess"``
    operate directly on the matched peak m/z and are the well-behaved choices
    for sparse peak sets. ``"piecewise"`` derives its Gaussian smoothing width
    from the median spacing of the input points; on a sparse peak set that
    spacing is large, so the smoothing is effectively inactive and the warp
    reduces to an unsmoothed piecewise-linear transform.
    """
    method = AlignmentMethod(method)
    if method == AlignmentMethod.dtw:
        raise ValueError(
            "method='dtw' is not supported for peak sets (it resamples onto a "
            "dense grid). Use 'shift', 'linear', 'piecewise', 'quadratic', "
            "'cubic', or 'lowess'."
        )

    ref_peaks_mz = np.asarray(ref_peaks_mz, dtype=float).ravel()
    if peaks.n_peaks == 0 or ref_peaks_mz.size == 0:
        return PeakSet(peaks.mz.copy(), peaks.intensity.copy())

    strategy = _build_strategy(
        method, max_shift_da, n_segments, smooth_sigma, lowess_frac, lowess_it
    )
    new_mz, _ = strategy.align_raw(
        peaks.mz,
        peaks.intensity,
        peaks.mz,
        ref_peaks_mz,
        ref_peaks_mz,
        np.ones_like(ref_peaks_mz),
    )
    return PeakSet(np.asarray(new_mz, dtype=float), peaks.intensity.copy())
