"""Tests for internal PCA-drift helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from maldiamrkit.drift.pca_drift import _pca_drift_timeseries


def _fit_baseline_pca(X: pd.DataFrame, ts: pd.Series, cutoff_days: int = 30) -> PCA:
    baseline = ts <= (ts.min() + pd.Timedelta(days=cutoff_days))
    pca = PCA(n_components=2)
    pca.fit(X.loc[baseline[baseline].index].to_numpy(dtype=float))
    return pca


class TestPcaDriftTimeseries:
    def test_schema(self, drift_set_flat):
        ds = drift_set_flat
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        pca = _fit_baseline_pca(ds.X, ts)
        df, _ = _pca_drift_timeseries(ds.X, ts, pca, window="30D", min_samples=3)
        assert list(df.columns) == [
            "window_start",
            "window_end",
            "centroid_pc1",
            "centroid_pc2",
            "dispersion",
            "n_spectra",
        ]
        assert (df["dispersion"] >= 0).all()
        assert (df["n_spectra"] >= 3).all()

    def test_detects_injected_shift(self, drift_set_shifted):
        """With drift injected on features 0-4 for the late half of the data,
        the post-baseline centroid should sit far from the baseline centroid
        in the PCA space fitted on baseline rows."""
        ds = drift_set_shifted
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        pca = _fit_baseline_pca(ds.X, ts, cutoff_days=30)
        df, _ = _pca_drift_timeseries(ds.X, ts, pca, window="30D", min_samples=5)
        assert len(df) >= 2
        # First window is part of the baseline → centroid near origin
        baseline_dist = np.hypot(df.iloc[0]["centroid_pc1"], df.iloc[0]["centroid_pc2"])
        # Last window is post-shift → centroid should have moved noticeably
        last = df.iloc[-1]
        shift_magnitude = np.hypot(
            last["centroid_pc1"] - df.iloc[0]["centroid_pc1"],
            last["centroid_pc2"] - df.iloc[0]["centroid_pc2"],
        )
        assert shift_magnitude > max(1e-4, 3 * baseline_dist)

    def test_min_samples_skips_sparse(self, drift_set_flat):
        ds = drift_set_flat
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        pca = _fit_baseline_pca(ds.X, ts)
        df, n_skipped = _pca_drift_timeseries(
            ds.X, ts, pca, window="30D", min_samples=1000
        )
        assert len(df) == 0
        assert n_skipped >= 1
