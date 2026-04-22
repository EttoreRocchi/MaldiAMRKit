"""DriftMonitor: orchestrator for temporal drift analysis over a MaldiSet."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from maldiamrkit.visualization.exploratory_plots import _reduce_dimensions

from .pca_drift import _pca_drift_timeseries
from .peak_drift import (
    _effect_size_timeseries,
    _peak_stability_timeseries,
    _top_peaks_set,
)
from .reference import _compute_reference_spectrum, _reference_similarity_timeseries

if TYPE_CHECKING:
    from maldiamrkit.dataset import MaldiSet
    from maldiamrkit.differential import DifferentialAnalysis


class DriftMonitor:
    """Monitor spectral drift over time using baseline-anchored metrics.

    Establishes a baseline from the earliest timestamps, then quantifies
    drift for later time windows via three complementary views:

    - reference similarity (distance to baseline median spectrum)
    - PCA centroid trajectory (baseline-fitted PCA space)
    - peak-selection stability (Jaccard overlap of top-k discriminative
      peaks per window vs. baseline) and Cohen's d tracking of specific
      peaks

    Output is data + plots only; no automated alerts.

    Parameters
    ----------
    time_column : str
        Metadata column containing timestamps (parsed via
        :func:`pandas.to_datetime`).
    window : str or pd.Timedelta, default="30D"
        Time window size for :class:`pandas.Grouper` (e.g. ``"30D"``,
        ``"7D"``).
    baseline_end : str, pd.Timestamp, or None, default=None
        End of the baseline period (inclusive).  If ``None``, defaults
        to the timestamp at the 20th percentile of sorted timestamps.
    metric : str, default="cosine"
        Distance metric for reference similarity (see
        :func:`maldiamrkit.similarity.spectral_distance`).
    n_components : int, default=2
        PCA components for PCA-drift monitoring.
    min_samples : int, default=5
        Skip time windows with fewer spectra than this (and, for
        peak-stability / effect-size monitoring, fewer than this many
        samples in either the R or S class).
    """

    def __init__(
        self,
        time_column: str,
        window: str | pd.Timedelta = "30D",
        baseline_end: str | pd.Timestamp | None = None,
        metric: str = "cosine",
        n_components: int = 2,
        min_samples: int = 5,
    ) -> None:
        self.time_column = time_column
        self.window = window
        self.baseline_end = baseline_end
        self.metric = metric
        self.n_components = n_components
        self.min_samples = min_samples

        self._reference: np.ndarray | None = None
        self._reducer: Any = None
        self._baseline_end_ts: pd.Timestamp | None = None

    def fit(self, maldi_set: MaldiSet) -> DriftMonitor:
        """Establish the baseline reference and PCA space.

        Parameters
        ----------
        maldi_set : MaldiSet
            Dataset exposing ``.X`` and ``.meta`` with ``time_column``.

        Returns
        -------
        DriftMonitor
            ``self``, for chaining.
        """
        X, ts = self._extract_X_and_timestamps(maldi_set)
        baseline_end = self._resolve_baseline_end(ts)
        baseline_mask = ts <= baseline_end
        n_baseline = int(baseline_mask.sum())
        if n_baseline < 2:
            raise ValueError(
                f"Baseline contains only {n_baseline} sample(s); need at least 2 "
                "to fit a reference and PCA."
            )

        X_base = X.loc[baseline_mask[baseline_mask].index]

        self._reference = _compute_reference_spectrum(X_base)
        n_components = min(self.n_components, X_base.shape[0], X_base.shape[1])
        _, self._reducer = _reduce_dimensions(
            X_base, method="pca", n_components=n_components
        )
        self._baseline_end_ts = baseline_end
        return self

    @property
    def reference_(self) -> np.ndarray:
        """Baseline reference spectrum (read-only)."""
        self._check_fitted()
        assert self._reference is not None
        return self._reference.copy()

    @property
    def baseline_end_(self) -> pd.Timestamp:
        """Timestamp used as the (inclusive) baseline cut-off."""
        self._check_fitted()
        assert self._baseline_end_ts is not None
        return self._baseline_end_ts

    def monitor(self, maldi_set: MaldiSet) -> pd.DataFrame:
        """Reference-similarity timeseries.

        Only spectra with ``timestamp > baseline_end_`` are monitored; the
        baseline is reserved as a reference and excluded from windows to
        avoid a self-reference artefact in the first window.

        Returns a DataFrame with columns ``window_start``,
        ``window_end``, ``n_spectra``, ``distance_to_reference``.
        """
        self._check_fitted()
        assert self._reference is not None
        X, ts = self._extract_X_and_timestamps(maldi_set)
        X, ts = self._drop_baseline(X, ts)
        df, n_skipped = _reference_similarity_timeseries(
            X,
            ts,
            self._reference,
            window=self.window,
            metric=self.metric,
            min_samples=self.min_samples,
        )
        self._warn_skipped(n_skipped, "monitor")
        return df

    def monitor_pca(self, maldi_set: MaldiSet) -> pd.DataFrame:
        """PCA centroid + dispersion timeseries.

        Only post-baseline spectra (``timestamp > baseline_end_``) are
        included.

        Returns a DataFrame with columns ``window_start``,
        ``window_end``, ``centroid_pc1``, ``centroid_pc2``,
        ``dispersion``, ``n_spectra``.
        """
        self._check_fitted()
        X, ts = self._extract_X_and_timestamps(maldi_set)
        X, ts = self._drop_baseline(X, ts)
        df, n_skipped = _pca_drift_timeseries(
            X,
            ts,
            self._reducer,
            window=self.window,
            min_samples=self.min_samples,
        )
        self._warn_skipped(n_skipped, "monitor_pca")
        return df

    def monitor_peak_stability(
        self,
        maldi_set: MaldiSet,
        differential_analysis: DifferentialAnalysis,
        antibiotic: str | None = None,
        n_top: int = 20,
    ) -> pd.DataFrame:
        """Peak-selection stability (Jaccard) timeseries.

        ``differential_analysis`` must already have been ``.run()``; its
        top-``n_top`` peaks define the baseline peak set.  Only
        post-baseline spectra (``timestamp > baseline_end_``) are
        included in the monitored windows.

        Returns a DataFrame with columns ``window_start``,
        ``stability_score``, ``n_spectra``.
        """
        self._check_fitted()
        X, ts = self._extract_X_and_timestamps(maldi_set)
        X, ts = self._drop_baseline(X, ts)
        y = maldi_set.get_y_single(antibiotic)
        y_numeric = pd.to_numeric(y, errors="coerce")
        valid = y_numeric.notna()
        X = X.loc[X.index.intersection(valid[valid].index)]
        ts = ts.loc[X.index]
        y_numeric = y_numeric.loc[X.index].astype(int)

        baseline_peaks = _top_peaks_set(differential_analysis, n_top=n_top)
        df, n_skipped = _peak_stability_timeseries(
            X,
            y_numeric,
            ts,
            baseline_peaks=baseline_peaks,
            window=self.window,
            n_top=n_top,
            min_samples=self.min_samples,
        )
        self._warn_skipped(n_skipped, "monitor_peak_stability")
        return df

    def monitor_effect_sizes(
        self,
        maldi_set: MaldiSet,
        peaks: list[str],
        antibiotic: str | None = None,
    ) -> pd.DataFrame:
        """Per-peak Cohen's d timeseries.

        Only post-baseline spectra (``timestamp > baseline_end_``) are
        included.

        Returns a DataFrame with ``window_start`` plus one column per
        requested peak (the peak's ``mz_bin`` label as a string).
        """
        self._check_fitted()
        X, ts = self._extract_X_and_timestamps(maldi_set)
        X, ts = self._drop_baseline(X, ts)
        y = maldi_set.get_y_single(antibiotic)
        y_numeric = pd.to_numeric(y, errors="coerce")
        valid = y_numeric.notna()
        X = X.loc[X.index.intersection(valid[valid].index)]
        ts = ts.loc[X.index]
        y_numeric = y_numeric.loc[X.index].astype(int)

        df, n_skipped = _effect_size_timeseries(
            X,
            y_numeric,
            ts,
            peaks=peaks,
            window=self.window,
            min_samples=self.min_samples,
        )
        self._warn_skipped(n_skipped, "monitor_effect_sizes")
        return df

    def _drop_baseline(
        self, X: pd.DataFrame, ts: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``(X, ts)`` restricted to ``timestamp > baseline_end_``."""
        assert self._baseline_end_ts is not None
        post_mask = ts > self._baseline_end_ts
        kept = post_mask[post_mask].index
        return X.loc[kept], ts.loc[kept]

    def _extract_X_and_timestamps(
        self, maldi_set: MaldiSet
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return aligned ``(X, parsed_timestamps)`` from a MaldiSet."""
        X = maldi_set.X
        if self.time_column not in maldi_set.meta.columns:
            raise ValueError(
                f"'{self.time_column}' not found in maldi_set.meta columns."
            )
        raw = maldi_set.meta.loc[X.index, self.time_column]
        try:
            ts = pd.to_datetime(raw, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Failed to parse metadata column '{self.time_column}' as "
                f"timestamps: {exc}"
            ) from exc
        if ts.isna().any():
            n_bad = int(ts.isna().sum())
            raise ValueError(
                f"'{self.time_column}' contains {n_bad} unparseable timestamp(s)."
            )
        ts.index = X.index
        return X, ts

    def _resolve_baseline_end(self, ts: pd.Series) -> pd.Timestamp:
        """Return the configured or defaulted baseline end timestamp.

        When ``baseline_end`` is not set, returns the 20th-percentile
        timestamp using the ``"higher"`` interpolation method so the
        baseline contains ``ceil(0.2 * n)`` samples (at least one).
        """
        if self.baseline_end is not None:
            return pd.Timestamp(self.baseline_end)
        ordered = ts.sort_values()
        n = len(ordered)
        idx = max(int(np.ceil(0.2 * n)) - 1, 0)
        return pd.Timestamp(ordered.iloc[idx])

    def _check_fitted(self) -> None:
        if self._reference is None or self._reducer is None:
            raise RuntimeError("DriftMonitor.fit() must be called first.")

    def _warn_skipped(self, n: int, name: str) -> None:
        if n > 0:
            warnings.warn(
                f"DriftMonitor.{name}: skipped {n} window(s) with fewer than "
                f"min_samples={self.min_samples} spectra "
                "(or, for peak-stability / effect-size monitors, fewer than "
                f"{self.min_samples} samples in either class).",
                UserWarning,
                stacklevel=2,
            )
