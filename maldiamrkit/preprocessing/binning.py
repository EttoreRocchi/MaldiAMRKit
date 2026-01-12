"""Spectrum binning functions with multiple binning strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..core.config import PreprocessingSettings


def _uniform_edges(
    cfg: PreprocessingSettings,
    bin_width: float,
) -> np.ndarray:
    """
    Generate uniform bin edges with fixed width.

    Parameters
    ----------
    cfg : PreprocessingSettings
        Configuration with trim boundaries.
    bin_width : float
        Width of each bin in Daltons.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    return np.arange(cfg.trim_from, cfg.trim_to + bin_width, bin_width)


def _logarithmic_edges(
    cfg: PreprocessingSettings,
    bin_width: float,
) -> np.ndarray:
    """
    Generate log-scaled bin edges.

    Parameters
    ----------
    cfg : PreprocessingSettings
        Configuration with trim boundaries.
    bin_width : float
        Reference bin width at mz_start in Daltons.

    Returns
    -------
    np.ndarray
        Array of bin edges with width scaling as w(mz) = bin_width * (mz / mz_start).
    """
    mz_start = cfg.trim_from
    mz_end = cfg.trim_to

    # Calculate scaling factor: bin_width at mz_start, grows proportionally
    # w(mz) = bin_width * (mz / mz_start)
    # Integral gives: mz_start * ln(mz/mz_start) = sum of widths
    # Solve for edges using geometric progression

    edges = [mz_start]
    current = mz_start

    while current < mz_end:
        # Width at current position
        width = max(1.0, bin_width * (current / mz_start))
        current += width
        edges.append(min(current, mz_end + width))

    edges = np.array(edges)
    # Ensure last edge covers mz_end
    if edges[-1] < mz_end:
        edges = np.append(edges, mz_end)

    return edges


def _adaptive_edges(
    df: pd.DataFrame,
    cfg: PreprocessingSettings,
    min_width: float = 1.0,
    max_width: float = 10.0,
) -> np.ndarray:
    """
    Generate bin edges based on local peak density.

    Parameters
    ----------
    df : pd.DataFrame
        Spectrum with 'mass' and 'intensity' columns.
    cfg : PreprocessingSettings
        Configuration with trim boundaries.
    min_width : float, default=1.0
        Minimum bin width in Daltons.
    max_width : float, default=10.0
        Maximum bin width in Daltons.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    mz = df["mass"].values
    intensity = df["intensity"].values

    # Detect peaks to identify regions of interest
    peaks, _ = find_peaks(intensity, prominence=np.std(intensity) * 0.1)

    if len(peaks) == 0:
        # No peaks found, use uniform binning
        return _uniform_edges(cfg, max_width)

    peak_mz = mz[peaks]

    # Calculate local peak density using kernel density estimation
    mz_range = np.linspace(cfg.trim_from, cfg.trim_to, 1000)
    bandwidth = (cfg.trim_to - cfg.trim_from) / 50  # Adaptive bandwidth

    density = np.zeros_like(mz_range)
    for pm in peak_mz:
        density += np.exp(-0.5 * ((mz_range - pm) / bandwidth) ** 2)

    # Normalize density to [0, 1]
    if density.max() > 0:
        density = density / density.max()

    # Map density to bin width: high density -> small bins
    # width = max_width - density * (max_width - min_width)
    width_at_mz = max_width - density * (max_width - min_width)

    # Generate edges based on variable widths
    edges = [cfg.trim_from]
    current = cfg.trim_from
    idx = 0

    while current < cfg.trim_to:
        # Find width at current position
        while idx < len(mz_range) - 1 and mz_range[idx] < current:
            idx += 1
        width = width_at_mz[min(idx, len(width_at_mz) - 1)]
        width = max(min_width, min(max_width, width))

        current += width
        if current <= cfg.trim_to + max_width:
            edges.append(current)

    edges = np.array(edges)
    # Ensure last edge covers trim_to
    if edges[-1] < cfg.trim_to:
        edges = np.append(edges, cfg.trim_to)

    return edges


def _validate_custom_edges(
    edges: np.ndarray | list,
    cfg: PreprocessingSettings,
) -> np.ndarray:
    """
    Validate user-provided bin edges.

    Parameters
    ----------
    edges : array-like
        User-provided bin edges.
    cfg : PreprocessingSettings
        Configuration with trim boundaries.

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

    if edges[0] > cfg.trim_from:
        raise ValueError(
            f"First edge ({edges[0]}) must be <= trim_from ({cfg.trim_from})."
        )

    if edges[-1] < cfg.trim_to:
        raise ValueError(
            f"Last edge ({edges[-1]}) must be >= trim_to ({cfg.trim_to})."
        )

    # Check minimum bin width of 1 Da
    min_width = np.diff(edges).min()
    if min_width < 1.0:
        raise ValueError(
            f"Minimum bin width is 1 Dalton, but got {min_width:.3f}."
        )

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

    return pd.DataFrame({
        "bin_index": np.arange(len(bin_starts)),
        "bin_start": bin_starts,
        "bin_end": bin_ends,
        "bin_width": bin_widths,
    })


def bin_spectrum(
    df: pd.DataFrame,
    cfg: PreprocessingSettings,
    bin_width: int | float = 3,
    method: str = "uniform",
    custom_edges: np.ndarray | list | None = None,
    adaptive_min_width: float = 1.0,
    adaptive_max_width: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin spectrum intensities into m/z intervals.

    Supports multiple binning strategies: uniform (fixed width), logarithmic
    (width scales with m/z), adaptive (smaller bins in peak-dense regions),
    and custom (user-defined edges).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed spectrum with columns 'mass' and 'intensity'.
    cfg : PreprocessingSettings
        Configuration object with trim_from and trim_to boundaries.
    bin_width : int or float, default=3
        Width of each bin in Daltons. For 'uniform', this is the fixed width.
        For 'logarithmic', this is the reference width at mz_start.
        Ignored for 'adaptive' and 'custom' methods.
    method : str, default='uniform'
        Binning method. One of 'uniform', 'logarithmic', 'adaptive', 'custom'.
    custom_edges : array-like, optional
        User-provided bin edges. Required if method='custom'.
    adaptive_min_width : float, default=1.0
        Minimum bin width in Daltons for adaptive binning.
    adaptive_max_width : float, default=10.0
        Maximum bin width in Daltons for adaptive binning.

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
    >>> from maldiamrkit.core.config import PreprocessingSettings
    >>> cfg = PreprocessingSettings()

    >>> # Uniform binning (default)
    >>> binned, metadata = bin_spectrum(df, cfg, bin_width=3)

    >>> # Logarithmic binning
    >>> binned, metadata = bin_spectrum(df, cfg, bin_width=3, method='logarithmic')

    >>> # Adaptive binning
    >>> binned, metadata = bin_spectrum(df, cfg, method='adaptive')

    >>> # Custom binning
    >>> edges = [2000, 5000, 10000, 15000, 20000]
    >>> binned, metadata = bin_spectrum(df, cfg, method='custom', custom_edges=edges)
    """
    valid_methods = ("uniform", "logarithmic", "adaptive", "custom")
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")

    # Validate bin_width minimum
    if method in ("uniform", "logarithmic") and bin_width < 1.0:
        raise ValueError(f"bin_width must be >= 1 Dalton, got {bin_width}.")

    # Generate bin edges based on method
    if method == "uniform":
        edges = _uniform_edges(cfg, bin_width)
    elif method == "logarithmic":
        edges = _logarithmic_edges(cfg, bin_width)
    elif method == "adaptive":
        if adaptive_min_width < 1.0:
            raise ValueError(
                f"adaptive_min_width must be >= 1 Dalton, got {adaptive_min_width}."
            )
        edges = _adaptive_edges(df, cfg, adaptive_min_width, adaptive_max_width)
    elif method == "custom":
        if custom_edges is None:
            raise ValueError("custom_edges is required when method='custom'.")
        edges = _validate_custom_edges(custom_edges, cfg)

    # Generate bin metadata
    metadata = get_bin_metadata(edges)

    # Perform binning
    labels = metadata["bin_start"].astype(str).values
    binned = (
        df
        .assign(bins=pd.cut(df.mass, edges, labels=labels, include_lowest=True))
        .groupby("bins", observed=True)["intensity"]
        .sum()
        .reindex(labels, fill_value=0.0)
        .reset_index()
        .rename(columns={"bins": "mass"})
    )

    return binned, metadata
