"""Tests for MaldiSpectrum.get_data, is_binned, is_preprocessed, has_bin_metadata."""

from __future__ import annotations

import pandas as pd

from maldiamrkit import MaldiSpectrum


class TestGetData:
    """Tests for MaldiSpectrum.get_data()."""

    def test_raw_only_returns_raw(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        result = spec.get_data()
        pd.testing.assert_frame_equal(result, spec.raw)

    def test_after_preprocess_returns_preprocessed(
        self, synthetic_spectrum: pd.DataFrame
    ):
        spec = MaldiSpectrum(synthetic_spectrum)
        spec.preprocess()
        result = spec.get_data()
        pd.testing.assert_frame_equal(result, spec.preprocessed)

    def test_prefer_binned_returns_binned(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        spec.preprocess().bin(3)
        result = spec.get_data(prefer="binned")
        pd.testing.assert_frame_equal(result, spec.binned)

    def test_prefer_binned_falls_back_to_preprocessed(
        self, synthetic_spectrum: pd.DataFrame
    ):
        spec = MaldiSpectrum(synthetic_spectrum)
        spec.preprocess()
        result = spec.get_data(prefer="binned")
        pd.testing.assert_frame_equal(result, spec.preprocessed)

    def test_prefer_binned_falls_back_to_raw(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        result = spec.get_data(prefer="binned")
        pd.testing.assert_frame_equal(result, spec.raw)

    def test_returns_copy(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        result = spec.get_data()
        result["intensity"] = 0
        assert spec.raw["intensity"].sum() > 0


class TestStateProperties:
    """Tests for is_binned, is_preprocessed, has_bin_metadata."""

    def test_initial_state(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        assert not spec.is_preprocessed
        assert not spec.is_binned
        assert not spec.has_bin_metadata

    def test_after_preprocess(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        spec.preprocess()
        assert spec.is_preprocessed
        assert not spec.is_binned
        assert not spec.has_bin_metadata

    def test_after_bin(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        spec.preprocess().bin(3)
        assert spec.is_preprocessed
        assert spec.is_binned
        assert spec.has_bin_metadata
