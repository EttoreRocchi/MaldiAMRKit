"""Internal helpers for reference-similarity drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from maldiamrkit.similarity import spectral_distance


def _compute_reference_spectrum(X: pd.DataFrame) -> np.ndarray:
    """Element-wise median spectrum across baseline rows.

    Parameters
    ----------
    X : pd.DataFrame
        Baseline feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    ndarray
        1-D reference spectrum of length ``n_features``.
    """
    return X.median(axis=0).to_numpy(dtype=float)


def _reference_similarity_timeseries(
    X: pd.DataFrame,
    timestamps: pd.Series,
    reference: np.ndarray,
    window: str | pd.Timedelta,
    metric: str = "cosine",
    min_samples: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Reference-similarity timeseries by time window.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (baseline + post-baseline), indexed by
        spectrum ID.
    timestamps : pd.Series
        Parsed datetime Series aligned with ``X.index``.
    reference : ndarray
        Reference spectrum from :func:`_compute_reference_spectrum`.
    window : str or pd.Timedelta
        Time window size forwarded to :class:`pandas.Grouper` (e.g.
        ``"30D"``).
    metric : str, default="cosine"
        Metric name understood by :func:`spectral_distance`.
    min_samples : int, default=5
        Windows with fewer than this many spectra are skipped.

    Returns
    -------
    df : pd.DataFrame
        Columns ``window_start``, ``window_end``, ``n_spectra``,
        ``distance_to_reference``.
    n_skipped : int
        Count of non-empty windows skipped because they had fewer than
        ``min_samples`` spectra.
    """
    if not timestamps.index.equals(X.index):
        raise ValueError("'timestamps' must be aligned with 'X.index'.")

    ts_col = "_drift_ts"
    joined = X.copy()
    joined[ts_col] = timestamps.values

    offset = pd.tseries.frequencies.to_offset(pd.Timedelta(window))  # type: ignore[call-overload]
    rows: list[dict] = []
    n_skipped = 0
    grouper = pd.Grouper(key=ts_col, freq=offset)
    for window_start, group in joined.groupby(grouper):
        if group.empty:
            continue
        n = len(group)
        if n < min_samples:
            n_skipped += 1
            continue
        median_vec = group.drop(columns=ts_col).median(axis=0).to_numpy(dtype=float)
        distance = spectral_distance(median_vec, reference, metric=metric)
        window_end = window_start + offset
        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "n_spectra": int(n),
                "distance_to_reference": float(distance),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["window_start", "window_end", "n_spectra", "distance_to_reference"],
    )
    return df, n_skipped
