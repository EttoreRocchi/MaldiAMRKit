"""Spectral distance metrics and registry."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from maldiamrkit.spectrum import MaldiSpectrum


def _extract_mz_intensity(
    spec: MaldiSpectrum | pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Normalize input to ``(mz_array | None, intensity_array)``.

    Parameters
    ----------
    spec : MaldiSpectrum, DataFrame, or ndarray
        Spectrum input.  For :class:`MaldiSpectrum` or a DataFrame with
        ``mass`` and ``intensity`` columns the m/z axis is returned.  For a
        plain 1-D array (binned vector) ``mz`` is ``None``.

    Returns
    -------
    mz : ndarray or None
        m/z values, or ``None`` for binned vectors.
    intensity : ndarray
        Intensity values.
    """
    if hasattr(spec, "get_data"):
        df = spec.get_data(prefer="preprocessed")
        return np.asarray(df["mass"]), np.asarray(df["intensity"])

    if isinstance(spec, pd.DataFrame):
        if "mass" in spec.columns and "intensity" in spec.columns:
            return np.asarray(spec["mass"]), np.asarray(spec["intensity"])
        return None, np.asarray(spec.iloc[0])

    arr = np.asarray(spec, dtype=float)
    return None, arr


def _wasserstein_distance(
    spec_a: MaldiSpectrum | pd.DataFrame | np.ndarray,
    spec_b: MaldiSpectrum | pd.DataFrame | np.ndarray,
) -> float:
    """Earth-mover distance between two raw spectra.

    Uses m/z positions as the support and intensities as weights.
    Any negative intensity values (which would break scipy's
    non-negative weight precondition) are clipped to zero.
    """
    from scipy.stats import wasserstein_distance as _wd

    mz_a, int_a = _extract_mz_intensity(spec_a)
    mz_b, int_b = _extract_mz_intensity(spec_b)
    if mz_a is None or mz_b is None:
        raise TypeError(
            "Wasserstein distance requires raw spectra with m/z values, "
            "not binned vectors."
        )
    int_a = np.clip(np.asarray(int_a, dtype=float), 0.0, None)
    int_b = np.clip(np.asarray(int_b, dtype=float), 0.0, None)
    return float(_wd(mz_a, mz_b, int_a, int_b))


def _dtw_distance(
    spec_a: MaldiSpectrum | pd.DataFrame | np.ndarray,
    spec_b: MaldiSpectrum | pd.DataFrame | np.ndarray,
) -> float:
    """Dynamic time-warping distance between two raw spectra.

    Both spectra are interpolated onto a common m/z grid before computing
    DTW.  Pre-processed or trimmed input is recommended for performance.
    """
    from tslearn.metrics import dtw

    mz_a, int_a = _extract_mz_intensity(spec_a)
    mz_b, int_b = _extract_mz_intensity(spec_b)
    if mz_a is None or mz_b is None:
        raise TypeError(
            "DTW distance requires raw spectra with m/z values, not binned vectors."
        )

    # Interpolate to a common grid spanning the union range.
    lo = min(mz_a[0], mz_b[0])
    hi = max(mz_a[-1], mz_b[-1])
    n_points = max(len(mz_a), len(mz_b))
    common_mz = np.linspace(lo, hi, n_points)

    int_a_interp = np.interp(common_mz, mz_a, int_a).reshape(-1, 1)
    int_b_interp = np.interp(common_mz, mz_b, int_b).reshape(-1, 1)

    return float(dtw(int_a_interp, int_b_interp))


def _cosine_distance(
    spec_a: MaldiSpectrum | pd.DataFrame | np.ndarray,
    spec_b: MaldiSpectrum | pd.DataFrame | np.ndarray,
) -> float:
    """Cosine distance (``1 - cosine_similarity``) for binned vectors."""
    _, a = _extract_mz_intensity(spec_a)
    _, b = _extract_mz_intensity(spec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    _tiny = np.finfo(float).tiny
    if norm_a < _tiny or norm_b < _tiny:
        return 1.0
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(1.0 - cos_sim, 0.0, 2.0))


def _spectral_contrast_angle(
    spec_a: MaldiSpectrum | pd.DataFrame | np.ndarray,
    spec_b: MaldiSpectrum | pd.DataFrame | np.ndarray,
) -> float:
    """Spectral contrast angle distance for binned vectors.

    Defined as ``(2 / pi) * arccos(cosine_similarity)``.  Ranges from
    0 (identical) to 1 (orthogonal).
    """
    _, a = _extract_mz_intensity(spec_a)
    _, b = _extract_mz_intensity(spec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    _tiny = np.finfo(float).tiny
    if norm_a < _tiny or norm_b < _tiny:
        return 1.0
    cos_sim = float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
    return float((2.0 / np.pi) * np.arccos(cos_sim))


def _pearson_correlation(
    spec_a: MaldiSpectrum | pd.DataFrame | np.ndarray,
    spec_b: MaldiSpectrum | pd.DataFrame | np.ndarray,
) -> float:
    """Pearson-correlation distance (``1 - r``) for binned vectors.

    Returns a value in ``[0, 2]``: 0 for perfectly correlated spectra,
    1 for uncorrelated, and 2 for perfectly anti-correlated.
    """
    _, a = _extract_mz_intensity(spec_a)
    _, b = _extract_mz_intensity(spec_b)
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 1.0
    return float(1.0 - corr)


class SpectralMetric(str, Enum):
    """Supported spectral distance/similarity metrics.

    Attributes
    ----------
    wasserstein : str
        Earth mover's (Wasserstein-1) distance on raw spectra.
    dtw : str
        Dynamic time warping distance on raw spectra.
    cosine : str
        Cosine distance on binned intensity vectors.
    spectral_contrast_angle : str
        Spectral contrast angle on binned intensity vectors.
    pearson : str
        1 - Pearson correlation on binned intensity vectors.
    """

    wasserstein = "wasserstein"
    dtw = "dtw"
    cosine = "cosine"
    spectral_contrast_angle = "spectral_contrast_angle"
    pearson = "pearson"


METRIC_REGISTRY: dict[str, Callable] = {
    "wasserstein": _wasserstein_distance,
    "dtw": _dtw_distance,
    "cosine": _cosine_distance,
    "spectral_contrast_angle": _spectral_contrast_angle,
    "pearson": _pearson_correlation,
}


def spectral_distance(
    spec_a: MaldiSpectrum | pd.DataFrame | np.ndarray,
    spec_b: MaldiSpectrum | pd.DataFrame | np.ndarray,
    metric: str | SpectralMetric = SpectralMetric.wasserstein,
) -> float:
    """Compute distance between two spectra.

    Parameters
    ----------
    spec_a, spec_b : MaldiSpectrum, DataFrame, or ndarray
        For non-binned metrics (``"wasserstein"``, ``"dtw"``):
        :class:`~maldiamrkit.spectrum.MaldiSpectrum` or DataFrame with
        ``mass`` and ``intensity`` columns.
        For binned metrics (``"cosine"``, ``"spectral_contrast_angle"``,
        ``"pearson"``): 1-D intensity arrays.
    metric : str or SpectralMetric, default="wasserstein"
        Key in :data:`METRIC_REGISTRY`.

    Returns
    -------
    float
        Distance (or ``1 - similarity`` for correlation-based metrics).

    Raises
    ------
    ValueError
        If *metric* is not in :data:`METRIC_REGISTRY`.
    """
    metric = SpectralMetric(metric)
    return METRIC_REGISTRY[metric](spec_a, spec_b)
