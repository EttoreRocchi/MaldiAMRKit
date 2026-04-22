"""Tests for internal reference-similarity drift helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from maldiamrkit.drift.reference import (
    _compute_reference_spectrum,
    _reference_similarity_timeseries,
)


class TestComputeReferenceSpectrum:
    def test_equals_elementwise_median(self, drift_set_flat):
        ds = drift_set_flat
        ref = _compute_reference_spectrum(ds.X)
        expected = ds.X.median(axis=0).to_numpy(dtype=float)
        np.testing.assert_allclose(ref, expected)

    def test_shape(self, drift_set_flat):
        ref = _compute_reference_spectrum(drift_set_flat.X)
        assert ref.shape == (drift_set_flat.X.shape[1],)


class TestReferenceSimilarityTimeseries:
    def test_schema_and_monotonic_windows(self, drift_set_flat):
        ds = drift_set_flat
        ref = _compute_reference_spectrum(ds.X)
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        df, n_skipped = _reference_similarity_timeseries(
            ds.X, ts, ref, window="30D", metric="cosine", min_samples=3
        )
        assert list(df.columns) == [
            "window_start",
            "window_end",
            "n_spectra",
            "distance_to_reference",
        ]
        assert (
            df["window_start"].sort_values().values == df["window_start"].values
        ).all()
        assert (df["n_spectra"] >= 3).all()
        assert df["distance_to_reference"].between(0.0, 2.0).all()
        assert n_skipped >= 0

    def test_index_misalignment_raises(self, drift_set_flat):
        ds = drift_set_flat
        ref = _compute_reference_spectrum(ds.X)
        bad_ts = pd.Series(
            pd.to_datetime(ds.meta["acquisition_date"]).values,
            index=[f"other_{i}" for i in range(len(ds.X))],
        )
        try:
            _reference_similarity_timeseries(
                ds.X, bad_ts, ref, window="30D", metric="cosine", min_samples=3
            )
        except ValueError as exc:
            assert "aligned" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")

    def test_min_samples_skips_sparse_windows(self, drift_set_flat):
        ds = drift_set_flat
        ref = _compute_reference_spectrum(ds.X)
        ts = pd.to_datetime(ds.meta["acquisition_date"])
        ts.index = ds.X.index
        df_loose, _ = _reference_similarity_timeseries(
            ds.X, ts, ref, window="30D", min_samples=1
        )
        df_strict, n_skipped = _reference_similarity_timeseries(
            ds.X, ts, ref, window="30D", min_samples=1000
        )
        assert len(df_strict) == 0
        assert n_skipped >= 1
        assert len(df_loose) > len(df_strict)
