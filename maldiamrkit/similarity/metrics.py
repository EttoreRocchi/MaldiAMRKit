"""Spectral distance metrics and registry."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from maldiamrkit._registry import _ComponentRegistry

if TYPE_CHECKING:
    from maldiamrkit.spectrum import MaldiSpectrum


def extract_mz_intensity(
    spec: MaldiSpectrum | pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Normalize a spectrum input to ``(mz_array | None, intensity_array)``.

    Every metric in this module accepts the same union of input types, so a
    custom metric registered with :func:`register_spectral_metric` should
    normalize its two arguments through this helper rather than assuming a
    single shape.

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

    Examples
    --------
    >>> import numpy as np
    >>> from maldiamrkit.similarity import extract_mz_intensity
    >>> def manhattan(spec_a, spec_b):
    ...     _, a = extract_mz_intensity(spec_a)
    ...     _, b = extract_mz_intensity(spec_b)
    ...     return float(np.abs(np.asarray(a) - np.asarray(b)).sum())
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

    mz_a, int_a = extract_mz_intensity(spec_a)
    mz_b, int_b = extract_mz_intensity(spec_b)
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

    mz_a, int_a = extract_mz_intensity(spec_a)
    mz_b, int_b = extract_mz_intensity(spec_b)
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
    _, a = extract_mz_intensity(spec_a)
    _, b = extract_mz_intensity(spec_b)
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
    _, a = extract_mz_intensity(spec_a)
    _, b = extract_mz_intensity(spec_b)
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
    _, a = extract_mz_intensity(spec_a)
    _, b = extract_mz_intensity(spec_b)
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


def _require_callable(fn: object) -> None:
    """Reject non-callable metric candidates at registration time."""
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}.")


_METRIC_REGISTRY = _ComponentRegistry(
    kind="spectral metric",
    short="metric",
    register_fn="register_spectral_metric",
    entries={
        "wasserstein": _wasserstein_distance,
        "dtw": _dtw_distance,
        "cosine": _cosine_distance,
        "spectral_contrast_angle": _spectral_contrast_angle,
        "pearson": _pearson_correlation,
    },
    validate=_require_callable,
)
"""Mapping of metric name to its pairwise spectral-distance function.

Keys are the values of :class:`~maldiamrkit.similarity.SpectralMetric`
(``"wasserstein"``, ``"dtw"``, ``"cosine"``, ``"spectral_contrast_angle"``,
``"pearson"``).

Register custom metrics with :func:`register_spectral_metric` rather than
mutating this mapping directly.
"""

_DEFAULT_SPECTRAL_METRICS: frozenset[str] = frozenset(_METRIC_REGISTRY.defaults)
"""Names of the built-in metrics, which cannot be removed (only overridden)."""


def _resolve_metric(metric: str | SpectralMetric) -> tuple[str, Callable]:
    """Resolve a metric name or enum member to ``(registry key, function)``.

    Accepts a :class:`SpectralMetric` member, one of its string values, or
    the name of a metric added with :func:`register_spectral_metric`.

    Raises
    ------
    ValueError
        If no metric of that name is registered.
    """
    key = _METRIC_REGISTRY.resolve_key(metric, SpectralMetric)
    return key, _METRIC_REGISTRY[key]


def register_spectral_metric(
    name: str,
    fn: Callable[..., float],
    *,
    override: bool = False,
) -> None:
    """Register a custom metric for use with the spectral-distance functions.

    After registration, pass ``metric=name`` to :func:`spectral_distance` or
    :func:`~maldiamrkit.similarity.pairwise_distances` (and to any consumer
    that forwards a metric name, such as
    :class:`~maldiamrkit.drift.DriftMonitor`).

    Parameters
    ----------
    name : str
        Metric name to expose. Re-registering a custom name replaces it
        silently; replacing a built-in requires ``override=True``.
    fn : callable
        Distance function ``fn(spec_a, spec_b) -> float``. It receives the
        two spectra exactly as the caller passed them (a
        :class:`~maldiamrkit.spectrum.MaldiSpectrum`, a ``(mass, intensity)``
        DataFrame, or a 1-D intensity array), so normalize both arguments
        through :func:`extract_mz_intensity`.
    override : bool, default=False
        Allow replacing one of the built-in metrics. The replacement is
        undone by :func:`unregister_spectral_metric`, which restores the
        default implementation.

    Raises
    ------
    TypeError
        If ``fn`` is not callable.
    ValueError
        If ``name`` is a built-in metric and ``override`` is False.

    Notes
    -----
    :func:`~maldiamrkit.similarity.pairwise_distances` resolves the function
    in the calling process and hands it to its workers, so a custom metric
    works at any ``n_jobs`` provided ``fn`` is serialisable (module-level
    functions and functions defined in a notebook both are).

    Examples
    --------
    >>> import numpy as np
    >>> from maldiamrkit.similarity import (
    ...     extract_mz_intensity,
    ...     register_spectral_metric,
    ...     spectral_distance,
    ... )
    >>> def manhattan(spec_a, spec_b):
    ...     _, a = extract_mz_intensity(spec_a)
    ...     _, b = extract_mz_intensity(spec_b)
    ...     return float(np.abs(np.asarray(a) - np.asarray(b)).sum())
    >>> register_spectral_metric("manhattan", manhattan)
    >>> spectral_distance(np.array([1.0, 2.0]), np.array([1.0, 5.0]), "manhattan")
    3.0
    >>> unregister_spectral_metric("manhattan")
    """
    _METRIC_REGISTRY.register(name, fn, override=override)


def unregister_spectral_metric(name: str) -> None:
    """Remove a custom metric added with :func:`register_spectral_metric`.

    For a built-in name that was replaced via ``override=True``, this
    restores the default implementation instead of removing the name.

    Parameters
    ----------
    name : str
        Metric name to remove (or, for an overridden built-in, to restore).

    Raises
    ------
    ValueError
        If ``name`` is one of the built-in metrics ('wasserstein', 'dtw',
        'cosine', 'spectral_contrast_angle', 'pearson') and has not been
        overridden; the built-in names cannot be removed.
    KeyError
        If no metric named ``name`` is registered.

    Examples
    --------
    >>> from maldiamrkit.similarity import (
    ...     register_spectral_metric,
    ...     unregister_spectral_metric,
    ... )
    >>> def always_zero(spec_a, spec_b):
    ...     return 0.0
    >>> register_spectral_metric("always_zero", always_zero)
    >>> unregister_spectral_metric("always_zero")
    """
    _METRIC_REGISTRY.unregister(name)


def list_spectral_metrics() -> list[str]:
    """List every registered spectral metric name, sorted.

    Returns
    -------
    list of str
        Built-in metric names plus any added with
        :func:`register_spectral_metric`.

    Examples
    --------
    >>> from maldiamrkit.similarity import list_spectral_metrics
    >>> list_spectral_metrics()
    ['cosine', 'dtw', 'pearson', 'spectral_contrast_angle', 'wasserstein']
    """
    return _METRIC_REGISTRY.names()


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
        One of the values of :class:`~maldiamrkit.similarity.SpectralMetric`,
        or a custom name registered with :func:`register_spectral_metric`.

    Returns
    -------
    float
        Distance (or ``1 - similarity`` for correlation-based metrics).

    Raises
    ------
    ValueError
        If *metric* is neither a recognised
        :class:`~maldiamrkit.similarity.SpectralMetric` nor a registered
        custom metric.
    """
    _, metric_fn = _resolve_metric(metric)
    return metric_fn(spec_a, spec_b)
