"""Internal helpers for peak-stability drift monitoring.

Not part of the public API; called by
:class:`maldiamrkit.drift.DriftMonitor`.
"""

from __future__ import annotations

import pandas as pd

from maldiamrkit.differential import DifferentialAnalysis
from maldiamrkit.differential.stats import _compute_effect_size


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets (``1.0`` when both are empty)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _top_peaks_set(analysis: DifferentialAnalysis, n_top: int) -> set:
    """Extract the top-k ``mz_bin`` values from a run analysis as a set."""
    top = analysis.top_peaks(n=n_top)
    return set(top["mz_bin"].to_numpy().tolist())


def _peak_stability_timeseries(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    baseline_peaks: set,
    window: str | pd.Timedelta,
    n_top: int = 20,
    test: str = "mann_whitney",
    correction: str = "fdr_bh",
    min_samples: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Jaccard stability of top-k peaks per window vs. baseline peaks.

    For each window we instantiate :class:`DifferentialAnalysis` on the
    window rows and take its top-``n_top`` peaks by adjusted p-value.
    Stability = Jaccard similarity with ``baseline_peaks``.

    Windows with fewer than ``min_samples`` in either class are skipped.
    """
    if not timestamps.index.equals(X.index):
        raise ValueError("'timestamps' must be aligned with 'X.index'.")
    if not y.index.equals(X.index):
        raise ValueError("'y' must be aligned with 'X.index'.")

    ts_col = "_drift_ts"
    joined = X.copy()
    joined[ts_col] = timestamps.values

    offset = pd.tseries.frequencies.to_offset(pd.Timedelta(window))  # type: ignore[call-overload]
    rows: list[dict[str, object]] = []
    n_skipped = 0
    grouper = pd.Grouper(key=ts_col, freq=offset)
    for window_start, group in joined.groupby(grouper):
        if group.empty:
            continue
        window_ids = group.index
        y_window = y.loc[window_ids]
        n = len(window_ids)
        n_r = int((y_window == 1).sum())
        n_s = int((y_window == 0).sum())
        if n_r < min_samples or n_s < min_samples:
            n_skipped += 1
            continue
        X_window = group.drop(columns=ts_col)
        analysis = DifferentialAnalysis(X_window, y_window).run(
            test=test, correction=correction
        )
        window_peaks = _top_peaks_set(analysis, n_top=n_top)
        score = _jaccard(window_peaks, baseline_peaks)
        rows.append(
            {
                "window_start": window_start,
                "stability_score": float(score),
                "n_spectra": int(n),
            }
        )

    df = pd.DataFrame(rows, columns=["window_start", "stability_score", "n_spectra"])
    return df, n_skipped


def _effect_size_timeseries(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    peaks: list[str],
    window: str | pd.Timedelta,
    min_samples: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Per-window Cohen's d of the listed peaks between R and S.

    Columns: ``window_start`` + one column per requested peak (the
    peak's ``mz_bin`` label as a string).  Windows with fewer than
    ``min_samples`` spectra in either class are skipped.
    """
    if not timestamps.index.equals(X.index):
        raise ValueError("'timestamps' must be aligned with 'X.index'.")
    if not y.index.equals(X.index):
        raise ValueError("'y' must be aligned with 'X.index'.")

    missing = [p for p in peaks if p not in X.columns]
    if missing:
        raise ValueError(
            f"Peaks not found in X columns: {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )

    ts_col = "_drift_ts"
    joined = X.copy()
    joined[ts_col] = timestamps.values

    offset = pd.tseries.frequencies.to_offset(pd.Timedelta(window))  # type: ignore[call-overload]
    rows: list[dict[str, object]] = []
    n_skipped = 0
    grouper = pd.Grouper(key=ts_col, freq=offset)
    for window_start, group in joined.groupby(grouper):
        if group.empty:
            continue
        window_ids = group.index
        y_window = y.loc[window_ids]
        n_r = int((y_window == 1).sum())
        n_s = int((y_window == 0).sum())
        if n_r < min_samples or n_s < min_samples:
            n_skipped += 1
            continue
        row: dict[str, object] = {"window_start": window_start}
        for peak in peaks:
            col_r = group.loc[y_window == 1, peak].to_numpy(dtype=float)
            col_s = group.loc[y_window == 0, peak].to_numpy(dtype=float)
            row[str(peak)] = float(_compute_effect_size(col_r, col_s))
        rows.append(row)

    columns = ["window_start"] + [str(p) for p in peaks]
    df = pd.DataFrame(rows, columns=columns)
    return df, n_skipped
