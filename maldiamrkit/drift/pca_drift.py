"""Internal helpers for PCA-space drift monitoring.

Not part of the public API; called by
:class:`maldiamrkit.drift.DriftMonitor`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pca_drift_timeseries(
    X: pd.DataFrame,
    timestamps: pd.Series,
    pca_model,
    window: str | pd.Timedelta,
    min_samples: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Project per-window spectra with a pre-fit PCA and summarise.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (baseline + post-baseline).
    timestamps : pd.Series
        Parsed datetime Series aligned with ``X.index``.
    pca_model : object
        Pre-fit reducer exposing a ``transform`` method (e.g. an
        ``sklearn.decomposition.PCA`` fitted on baseline rows).
    window : str or pd.Timedelta
        Time window size for :class:`pandas.Grouper`.
    min_samples : int, default=5
        Windows with fewer than this many spectra are skipped.

    Returns
    -------
    df : pd.DataFrame
        Columns ``window_start``, ``window_end``, ``centroid_pc1``,
        ``centroid_pc2``, ``dispersion``, ``n_spectra``.  ``dispersion``
        is the standard deviation of the window's projected spectra
        from their centroid, i.e.
        ``sqrt(mean(sum_k (proj_k - centroid_k)^2))``.  Equivalent to
        the square root of the trace of the per-window covariance
        matrix (Jolliffe 2002, §3).
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
        values = group.drop(columns=ts_col).to_numpy(dtype=float)
        projected = pca_model.transform(values)
        centroid = projected.mean(axis=0)
        squared = np.sum((projected - centroid) ** 2, axis=1)
        dispersion = float(np.sqrt(squared.mean()))
        window_end = window_start + offset
        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "centroid_pc1": float(centroid[0]),
                "centroid_pc2": float(centroid[1]) if projected.shape[1] > 1 else 0.0,
                "dispersion": dispersion,
                "n_spectra": int(n),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "window_start",
            "window_end",
            "centroid_pc1",
            "centroid_pc2",
            "dispersion",
            "n_spectra",
        ],
    )
    return df, n_skipped
