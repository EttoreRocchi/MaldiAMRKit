"""End-to-end tests for DriftMonitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.differential import DifferentialAnalysis
from maldiamrkit.drift import DriftMonitor


class TestFit:
    def test_returns_self(self, drift_set_flat):
        monitor = DriftMonitor(time_column="acquisition_date", window="45D")
        assert monitor.fit(drift_set_flat) is monitor

    def test_default_baseline_end_is_20th_percentile(self, drift_set_flat):
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=2
        )
        monitor.fit(drift_set_flat)
        ts = pd.to_datetime(drift_set_flat.meta["acquisition_date"]).sort_values()
        # The 20% cutoff should fall inside the 1st fifth of timestamps
        assert monitor.baseline_end_ <= ts.iloc[int(0.3 * len(ts))]
        assert monitor.baseline_end_ >= ts.iloc[0]

    def test_default_baseline_end_matches_ceil_percentile(self, drift_set_flat):
        """Baseline end = ordered.iloc[ceil(0.2 * n) - 1]."""
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=2
        ).fit(drift_set_flat)
        ordered = pd.to_datetime(drift_set_flat.meta["acquisition_date"]).sort_values()
        n = len(ordered)
        expected_idx = max(int(np.ceil(0.2 * n)) - 1, 0)
        assert monitor.baseline_end_ == pd.Timestamp(ordered.iloc[expected_idx])

    def test_explicit_baseline_end(self, drift_set_flat):
        explicit = pd.Timestamp("2025-02-15")
        monitor = DriftMonitor(
            time_column="acquisition_date",
            window="45D",
            baseline_end=explicit,
            min_samples=2,
        )
        monitor.fit(drift_set_flat)
        assert monitor.baseline_end_ == explicit

    def test_missing_time_column_raises(self, drift_set_flat):
        monitor = DriftMonitor(time_column="nope")
        with pytest.raises(ValueError, match="not found"):
            monitor.fit(drift_set_flat)


class TestMonitor:
    def test_reference_schema(self, drift_set_flat):
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=2
        ).fit(drift_set_flat)
        df = monitor.monitor(drift_set_flat)
        assert list(df.columns) == [
            "window_start",
            "window_end",
            "n_spectra",
            "distance_to_reference",
        ]
        assert len(df) >= 2

    def test_pca_schema(self, drift_set_flat):
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=2
        ).fit(drift_set_flat)
        df = monitor.monitor_pca(drift_set_flat)
        assert list(df.columns) == [
            "window_start",
            "window_end",
            "centroid_pc1",
            "centroid_pc2",
            "dispersion",
            "n_spectra",
        ]

    @pytest.mark.filterwarnings(r"ignore:DriftMonitor\.monitor_pca")
    def test_pca_detects_shift(self, drift_set_shifted):
        monitor = DriftMonitor(
            time_column="acquisition_date", window="30D", min_samples=5
        ).fit(drift_set_shifted)
        df = monitor.monitor_pca(drift_set_shifted)
        assert len(df) >= 2
        first_pc1 = df.iloc[0]["centroid_pc1"]
        last_pc1 = df.iloc[-1]["centroid_pc1"]
        assert abs(last_pc1 - first_pc1) > 1e-4

    @pytest.mark.filterwarnings(r"ignore:DriftMonitor\.monitor_peak_stability")
    def test_peak_stability(self, drift_set_with_peaks):
        y = drift_set_with_peaks.meta["Drug"].astype(int)
        y.index = drift_set_with_peaks.X.index
        baseline = DifferentialAnalysis(drift_set_with_peaks.X, y).run()
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=5
        ).fit(drift_set_with_peaks)
        df = monitor.monitor_peak_stability(
            drift_set_with_peaks,
            baseline,
            antibiotic="Drug",
            n_top=10,
        )
        assert list(df.columns) == ["window_start", "stability_score", "n_spectra"]
        assert df["stability_score"].between(0.0, 1.0).all()

    @pytest.mark.filterwarnings(r"ignore:DriftMonitor\.monitor_effect_sizes")
    def test_effect_sizes(self, drift_set_with_peaks):
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=5
        ).fit(drift_set_with_peaks)
        peaks = list(drift_set_with_peaks.X.columns[:3])
        df = monitor.monitor_effect_sizes(
            drift_set_with_peaks,
            peaks=peaks,
            antibiotic="Drug",
        )
        assert list(df.columns) == ["window_start", *peaks]

    def test_methods_require_fit(self, drift_set_flat):
        monitor = DriftMonitor(time_column="acquisition_date", window="45D")
        with pytest.raises(RuntimeError, match="fit"):
            monitor.monitor(drift_set_flat)
        with pytest.raises(RuntimeError, match="fit"):
            monitor.monitor_pca(drift_set_flat)

    def test_skipped_windows_warn(self, drift_set_flat):
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=1000
        ).fit(drift_set_flat)
        with pytest.warns(UserWarning, match="skipped"):
            df = monitor.monitor(drift_set_flat)
        assert len(df) == 0

    def test_monitor_excludes_baseline_window(self, drift_set_flat):
        """All monitored windows must start strictly after baseline_end_."""
        monitor = DriftMonitor(
            time_column="acquisition_date", window="30D", min_samples=2
        ).fit(drift_set_flat)
        df = monitor.monitor(drift_set_flat)
        df_pca = monitor.monitor_pca(drift_set_flat)
        ts = pd.to_datetime(drift_set_flat.meta["acquisition_date"])
        post_baseline_ts = ts[ts > monitor.baseline_end_]
        assert len(df) > 0
        assert (df["window_end"] > monitor.baseline_end_).all()
        assert (df_pca["window_end"] > monitor.baseline_end_).all()
        # No window should contain a baseline sample.
        assert df["n_spectra"].sum() <= len(post_baseline_ts)


def _baseline_rows(monitor, ds):
    ts = pd.to_datetime(ds.meta.loc[ds.X.index, "acquisition_date"])
    return ds.X.index[ts <= monitor.baseline_end_]


class TestPeakStabilityUsesTheBaselineSettings:
    """Each window is analysed like the baseline analysis."""

    def test_windows_reuse_test_correction_and_ranges(
        self, drift_set_with_peaks, monkeypatch
    ):
        ds = drift_set_with_peaks
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=5
        ).fit(ds)
        rows = _baseline_rows(monitor, ds)
        y = ds.meta.loc[rows, "Drug"].astype(int)
        baseline = DifferentialAnalysis(ds.X.loc[rows], y).run(
            test="t_test", correction="bonferroni", mz_ranges=(2000, 2060)
        )
        calls = []
        original = DifferentialAnalysis.run

        def spy(self, *args, **kwargs):
            calls.append(kwargs)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(DifferentialAnalysis, "run", spy)
        monitor.monitor_peak_stability(ds, baseline, antibiotic="Drug", n_top=5)
        assert calls
        for kwargs in calls:
            assert kwargs["test"] == "t_test"
            assert kwargs["correction"] == "bonferroni"
            assert kwargs["mz_ranges"] == (2000, 2060)


class TestBaselineAnalysisOnBaselineRows:
    """A baseline analysis that includes monitored spectra is flagged."""

    def test_analysis_on_all_rows_warns(self, drift_set_with_peaks):
        ds = drift_set_with_peaks
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=5
        ).fit(ds)
        y = ds.meta["Drug"].astype(int).loc[ds.X.index]
        everything = DifferentialAnalysis(ds.X, y).run()
        with pytest.warns(UserWarning, match="baseline_end_"):
            monitor.monitor_peak_stability(ds, everything, antibiotic="Drug", n_top=5)

    def test_analysis_on_baseline_rows_does_not_warn(self, drift_set_with_peaks):
        import warnings

        ds = drift_set_with_peaks
        monitor = DriftMonitor(
            time_column="acquisition_date", window="45D", min_samples=5
        ).fit(ds)
        rows = _baseline_rows(monitor, ds)
        y = ds.meta.loc[rows, "Drug"].astype(int)
        baseline = DifferentialAnalysis(ds.X.loc[rows], y).run()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings(
                "ignore", message="DriftMonitor.monitor_peak_stability: skipped"
            )
            monitor.monitor_peak_stability(ds, baseline, antibiotic="Drug", n_top=5)
