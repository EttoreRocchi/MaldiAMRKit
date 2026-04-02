"""Spectrum binning functions with multiple binning strategies."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _uniform_edges(
    mz_min: float,
    mz_max: float,
    bin_width: float,
) -> np.ndarray:
    """
    Generate uniform bin edges with fixed width.

    Parameters
    ----------
    mz_min : float
        Lower m/z bound.
    mz_max : float
        Upper m/z bound.
    bin_width : float
        Width of each bin in Daltons.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    return np.arange(mz_min, mz_max + bin_width, bin_width)


def _proportional_edges(
    mz_min: float,
    mz_max: float,
    bin_width: float,
) -> np.ndarray:
    """
    Generate proportional-width bin edges.

    Bin width increases linearly with m/z:
    ``w(mz) = bin_width * (mz / mz_min)``.

    .. note::

       This is *not* true logarithmic (geometric) spacing.  For
       geometric spacing the edges would be ``mz_min * r**k``.  The
       proportional scheme gives narrower bins at low m/z and wider
       bins at high m/z, which is often desirable for MALDI-TOF data,
       but the scaling is linear, not exponential.

    Parameters
    ----------
    mz_min : float
        Lower m/z bound.
    mz_max : float
        Upper m/z bound.
    bin_width : float
        Reference bin width at mz_min in Daltons.

    Returns
    -------
    np.ndarray
        Array of bin edges with width scaling as w(mz) = bin_width * (mz / mz_min).
    """
    edges_list: list[float] = [mz_min]
    current = mz_min

    while current < mz_max:
        # Width at current position
        width = max(1.0, bin_width * (current / mz_min))
        current += width
        edges_list.append(current)

    result = np.array(edges_list)
    # Ensure last edge covers mz_max
    if result[-1] < mz_max:
        result = np.append(result, mz_max)

    return result


def _estimate_peak_density(
    peak_mz: np.ndarray,
    mz_range: np.ndarray,
    kde_bandwidth: float | None = None,
    mz_min: float = 2000,
    mz_max: float = 20000,
) -> tuple[np.ndarray, float]:
    """Estimate local peak density via Gaussian KDE.

    Parameters
    ----------
    peak_mz : np.ndarray
        Detected peak m/z positions.
    mz_range : np.ndarray
        Grid of m/z values to evaluate density on.
    kde_bandwidth : float or None
        Bandwidth for the Gaussian KDE. If None, uses Silverman's rule.
    mz_min, mz_max : float
        m/z bounds (used for fallback bandwidth).

    Returns
    -------
    density : np.ndarray
        Normalised density in [0, 1] on ``mz_range``.
    bandwidth : float
        Bandwidth used.
    """
    if kde_bandwidth is None:
        n = len(peak_mz)
        std = np.std(peak_mz)
        iqr = np.subtract(*np.percentile(peak_mz, [75, 25]))
        if n > 1 and std > 0 and iqr > 0:
            kde_bandwidth = 0.9 * min(std, iqr / 1.34) * n ** (-0.2)
        else:
            kde_bandwidth = (mz_max - mz_min) / 50

    # Vectorised KDE computation
    density = np.exp(
        -0.5 * ((mz_range[:, None] - peak_mz[None, :]) / kde_bandwidth) ** 2
    ).sum(axis=1)

    # Normalize to [0, 1]
    if density.max() > 0:
        density = density / density.max()

    return density, kde_bandwidth


def _adaptive_edges(
    df: pd.DataFrame,
    mz_min: float,
    mz_max: float,
    min_width: float = 1.0,
    max_width: float = 10.0,
    peak_prominence: float | None = None,
    kde_bandwidth: float | None = None,
) -> np.ndarray:
    """
    Generate bin edges based on local peak density.

    Parameters
    ----------
    df : pd.DataFrame
        Spectrum with 'mass' and 'intensity' columns.
    mz_min : float
        Lower m/z bound.
    mz_max : float
        Upper m/z bound.
    min_width : float, default=1.0
        Minimum bin width in Daltons.
    max_width : float, default=10.0
        Maximum bin width in Daltons.
    peak_prominence : float or None, default=None
        Minimum prominence for peak detection.  If ``None``, uses a
        MAD-based estimate (3-sigma equivalent), which is more robust
        to outliers than the standard deviation.
    kde_bandwidth : float or None, default=None
        Bandwidth for the Gaussian KDE used to estimate local peak
        density.  If ``None``, uses Silverman's rule of thumb
        computed from the detected peak m/z positions.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    mz = df["mass"].values
    intensity = df["intensity"].values

    # Detect peaks to identify regions of interest
    if peak_prominence is None:
        # MAD-based prominence: robust to outliers
        mad = np.median(np.abs(intensity - np.median(intensity)))
        # 1.4826 converts MAD to sigma under Gaussian assumption
        # (Rousseeuw & Croux 1993); * 3 gives 3-sigma equivalent.
        peak_prominence = 1.4826 * mad * 3
    peaks, _ = find_peaks(intensity, prominence=peak_prominence)

    if len(peaks) == 0:
        # No peaks found, use uniform binning
        return _uniform_edges(mz_min, mz_max, max_width)

    peak_mz = mz[peaks]

    # Calculate local peak density using kernel density estimation
    mz_range = np.linspace(mz_min, mz_max, 1000)
    density, bandwidth = _estimate_peak_density(
        peak_mz, mz_range, kde_bandwidth, mz_min, mz_max
    )

    # Map density to bin width: high density -> small bins
    width_at_mz = max_width - density * (max_width - min_width)

    # Generate edges based on variable widths
    edges_list: list[float] = [mz_min]
    current = mz_min
    idx = 0

    while current < mz_max:
        # Find width at current position
        while idx < len(mz_range) - 1 and mz_range[idx] < current:
            idx += 1
        width = width_at_mz[min(idx, len(width_at_mz) - 1)]
        width = max(min_width, min(max_width, width))

        current += width
        if current <= mz_max + max_width:
            edges_list.append(current)

    result = np.array(edges_list)
    # Ensure last edge covers mz_max
    if result[-1] < mz_max:
        result = np.append(result, mz_max)

    return result


def _validate_custom_edges(
    edges: np.ndarray | list,
    mz_min: float,
    mz_max: float,
) -> np.ndarray:
    """
    Validate user-provided bin edges.

    Parameters
    ----------
    edges : array-like
        User-provided bin edges.
    mz_min : float
        Lower m/z bound.
    mz_max : float
        Upper m/z bound.

    Returns
    -------
    np.ndarray
        Validated bin edges.

    Raises
    ------
    ValueError
        If edges are not sorted, don't cover trim range, or have fewer than 2 elements.
    """
    edges = np.asarray(edges, dtype=float)

    if len(edges) < 2:
        raise ValueError("Custom edges must have at least 2 elements.")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("Custom edges must be sorted in ascending order.")

    if edges[0] > mz_min:
        raise ValueError(f"First edge ({edges[0]}) must be <= mz_min ({mz_min}).")

    if edges[-1] < mz_max:
        raise ValueError(f"Last edge ({edges[-1]}) must be >= mz_max ({mz_max}).")

    # Check minimum bin width of 1 Da
    min_width = np.diff(edges).min()
    if min_width < 1.0:
        raise ValueError(f"Minimum bin width is 1 Dalton, but got {min_width:.3f}.")

    return edges


def get_bin_metadata(edges: np.ndarray) -> pd.DataFrame:
    """
    Generate bin metadata from edges.

    Parameters
    ----------
    edges : np.ndarray
        Array of bin edges.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bin_index, bin_start, bin_end, bin_width.
    """
    bin_starts = edges[:-1]
    bin_ends = edges[1:]
    bin_widths = bin_ends - bin_starts

    return pd.DataFrame(
        {
            "bin_index": np.arange(len(bin_starts)),
            "bin_start": bin_starts,
            "bin_end": bin_ends,
            "bin_width": bin_widths,
        }
    )


def _uniform_edge_fn(*, mz_min, mz_max, bin_width, **_kwargs) -> np.ndarray:
    """Generate uniform bin edges with validation."""
    if bin_width < 1.0:
        raise ValueError(f"bin_width must be >= 1 Dalton, got {bin_width}.")
    return _uniform_edges(mz_min, mz_max, bin_width)


def _proportional_edge_fn(*, mz_min, mz_max, bin_width, **_kwargs) -> np.ndarray:
    """Generate proportional-width bin edges with validation."""
    if bin_width < 1.0:
        raise ValueError(f"bin_width must be >= 1 Dalton, got {bin_width}.")
    return _proportional_edges(mz_min, mz_max, bin_width)


def _adaptive_edge_fn(
    *,
    df,
    mz_min,
    mz_max,
    adaptive_min_width=1.0,
    adaptive_max_width=10.0,
    adaptive_peak_prominence=None,
    adaptive_kde_bandwidth=None,
    **_kwargs,
) -> np.ndarray:
    """Generate adaptive bin edges with validation."""
    if adaptive_min_width < 1.0:
        raise ValueError(
            f"adaptive_min_width must be >= 1 Dalton, got {adaptive_min_width}."
        )
    return _adaptive_edges(
        df,
        mz_min,
        mz_max,
        adaptive_min_width,
        adaptive_max_width,
        peak_prominence=adaptive_peak_prominence,
        kde_bandwidth=adaptive_kde_bandwidth,
    )


def _custom_edge_fn(*, mz_min, mz_max, custom_edges=None, **_kwargs) -> np.ndarray:
    """Validate and return custom bin edges."""
    if custom_edges is None:
        raise ValueError("custom_edges is required when method='custom'.")
    return _validate_custom_edges(custom_edges, mz_min, mz_max)


BINNING_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "uniform": _uniform_edge_fn,
    "proportional": _proportional_edge_fn,
    "adaptive": _adaptive_edge_fn,
    "custom": _custom_edge_fn,
}
"""Registry mapping binning method names to edge-generator functions.

To add a custom binning method::

    from maldiamrkit.preprocessing.binning import BINNING_REGISTRY

    def my_edges(*, mz_min, mz_max, **kwargs):
        return np.array([mz_min, (mz_min + mz_max) / 2, mz_max])

    BINNING_REGISTRY["my_method"] = my_edges
"""


def bin_spectrum(
    df: pd.DataFrame,
    mz_min: int = 2000,
    mz_max: int = 20000,
    bin_width: int | float = 3,
    method: str = "uniform",
    custom_edges: np.ndarray | list | None = None,
    adaptive_min_width: float = 1.0,
    adaptive_max_width: float = 10.0,
    adaptive_peak_prominence: float | None = None,
    adaptive_kde_bandwidth: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin spectrum intensities into m/z intervals.

    Supports multiple binning strategies: uniform (fixed width), proportional
    (width scales linearly with m/z), adaptive (smaller bins in peak-dense
    regions), and custom (user-defined edges).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed spectrum with columns 'mass' and 'intensity'.
    mz_min : int, default=2000
        Lower m/z bound in Daltons.
    mz_max : int, default=20000
        Upper m/z bound in Daltons.
    bin_width : int or float, default=3
        Width of each bin in Daltons. For 'uniform', this is the fixed width.
        For 'proportional', this is the reference width at mz_min.
        Ignored for 'adaptive' and 'custom' methods.
    method : str, default='uniform'
        Binning method. One of 'uniform', 'proportional', 'adaptive',
        'custom'.
    custom_edges : array-like, optional
        User-provided bin edges. Required if method='custom'.
    adaptive_min_width : float, default=1.0
        Minimum bin width in Daltons for adaptive binning.
    adaptive_max_width : float, default=10.0
        Maximum bin width in Daltons for adaptive binning.
    adaptive_peak_prominence : float or None, default=None
        Minimum prominence for peak detection in adaptive binning.
        If ``None``, uses a MAD-based estimate (robust to outliers).
    adaptive_kde_bandwidth : float or None, default=None
        Bandwidth for the Gaussian KDE in adaptive binning.
        If ``None``, uses Silverman's rule of thumb.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (binned_spectrum, bin_metadata).
        binned_spectrum has columns 'mass' (bin start) and 'intensity'.
        bin_metadata has columns 'bin_index', 'bin_start', 'bin_end', 'bin_width'.

    Raises
    ------
    ValueError
        If method is invalid, custom_edges is missing for 'custom' method,
        or bin_width < 1.

    Examples
    --------
    >>> from maldiamrkit.preprocessing import bin_spectrum
    >>>
    >>> # Uniform binning (default)
    >>> binned, metadata = bin_spectrum(df, bin_width=3)
    >>>
    >>> # Proportional binning (width grows with m/z)
    >>> binned, metadata = bin_spectrum(df, bin_width=3, method='proportional')
    >>>
    >>> # Adaptive binning
    >>> binned, metadata = bin_spectrum(df, method='adaptive')
    >>>
    >>> # Custom binning
    >>> edges = [2000, 5000, 10000, 15000, 20000]
    >>> binned, metadata = bin_spectrum(df, method='custom', custom_edges=edges)
    """
    if method not in BINNING_REGISTRY:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {tuple(BINNING_REGISTRY)}."
        )

    if mz_min >= mz_max:
        raise ValueError(f"mz_min ({mz_min}) must be less than mz_max ({mz_max}).")

    if method == "proportional" and mz_min <= 0:
        raise ValueError(f"mz_min must be > 0 for {method} binning, got {mz_min}")

    # Generate bin edges via registry
    edge_fn = BINNING_REGISTRY[method]
    edges = edge_fn(
        df=df,
        mz_min=mz_min,
        mz_max=mz_max,
        bin_width=bin_width,
        custom_edges=custom_edges,
        adaptive_min_width=adaptive_min_width,
        adaptive_max_width=adaptive_max_width,
        adaptive_peak_prominence=adaptive_peak_prominence,
        adaptive_kde_bandwidth=adaptive_kde_bandwidth,
    )

    # Generate bin metadata
    metadata = get_bin_metadata(edges)

    # Perform binning
    labels = metadata["bin_start"].astype(str).values
    binned = (
        df.assign(bins=pd.cut(df.mass, edges, labels=labels, include_lowest=True))
        .groupby("bins", observed=True)["intensity"]
        .sum()
        .reindex(labels, fill_value=0.0)
        .reset_index()
        .rename(columns={"bins": "mass"})
    )

    return binned, metadata
