"""Tests for internal peak-drift helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from maldiamrkit.differential import DifferentialAnalysis
from maldiamrkit.drift.peak_drift import (
    _effect_size_timeseries,
    _jaccard,
    _peak_stability_timeseries,
    _top_peaks_set,
)


class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        # |A ∩ B| = {2, 3} = 2, |A ∪ B| = {1, 2, 3, 4, 5} = 5
        assert _jaccard({1, 2, 3}, {2, 3, 4, 5}) == 2 / 5

    def test_empty_pair(self):
        assert _jaccard(set(), set()) == 1.0


class TestStabilityTimeseries:
    def test_perfect_stability_when_same_peaks(self, drift_set_with_peaks):
        ds = drift_set_with_peaks
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        y = ds.meta["Drug"].astype(int)
        y.index = ds.X.index
        baseline = DifferentialAnalysis(ds.X, y).run()
        baseline_peaks = _top_peaks_set(baseline, n_top=10)
        df, _ = _peak_stability_timeseries(
            ds.X,
            y,
            ts,
            baseline_peaks=baseline_peaks,
            window="45D",
            n_top=10,
            min_samples=5,
        )
        assert len(df) >= 2
        # Injected R-specific peaks are present in every window. Random
        # overlap for k=10 drawn from 40 features is ~10/40 = 0.25 ->
        # Jaccard ~0.14, so requiring >= 0.30 is a clear positive signal.
        assert (df["stability_score"] >= 0.30).all()

    def test_schema(self, drift_set_with_peaks):
        ds = drift_set_with_peaks
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        y = ds.meta["Drug"].astype(int)
        y.index = ds.X.index
        df, _ = _peak_stability_timeseries(
            ds.X,
            y,
            ts,
            baseline_peaks=set(),
            window="45D",
            n_top=5,
            min_samples=5,
        )
        assert list(df.columns) == ["window_start", "stability_score", "n_spectra"]

    def test_skips_single_class_windows(self, drift_set_with_peaks):
        ds = drift_set_with_peaks
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        y_all_resistant = pd.Series(np.ones(len(ds.X), dtype=int), index=ds.X.index)
        df, n_skipped = _peak_stability_timeseries(
            ds.X,
            y_all_resistant,
            ts,
            baseline_peaks=set(),
            window="45D",
            min_samples=5,
        )
        assert len(df) == 0
        assert n_skipped >= 1


class TestEffectSizeTimeseries:
    def test_schema_and_values(self, drift_set_with_peaks):
        ds = drift_set_with_peaks
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        y = ds.meta["Drug"].astype(int)
        y.index = ds.X.index
        injected = list(ds.X.columns[:3])
        df, _ = _effect_size_timeseries(
            ds.X,
            y,
            ts,
            peaks=injected,
            window="45D",
            min_samples=5,
        )
        assert list(df.columns) == ["window_start", *[str(p) for p in injected]]
        # Injected R-peaks should give positive effect sizes in every window
        for peak in injected:
            assert (df[str(peak)] > 0).all()

    def test_unknown_peak_raises(self, drift_set_with_peaks):
        ds = drift_set_with_peaks
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        y = ds.meta["Drug"].astype(int)
        y.index = ds.X.index
        try:
            _effect_size_timeseries(
                ds.X,
                y,
                ts,
                peaks=["not_a_real_mz_bin"],
                window="45D",
            )
        except ValueError as exc:
            assert "not found" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")
