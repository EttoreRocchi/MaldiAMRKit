"""Tests for MICEncoder."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.susceptibility import BreakpointTable, MICEncoder


@pytest.fixture
def bp() -> BreakpointTable:
    path = resources.files("maldiamrkit") / "data" / "breakpoints" / "example.yaml"
    return BreakpointTable.from_yaml(path)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Species": ["Klebsiella pneumoniae"] * 5,
            "Drug": ["Ceftriaxone"] * 5,
            "MIC": ["<=0.25", "0.5", "2", ">16", None],
        }
    )


class TestMICEncoderRegressionOnly:
    def test_no_breakpoints_log2_and_censored(self, df):
        enc = MICEncoder(mic_col="MIC")
        out = enc.fit_transform(df)
        np.testing.assert_allclose(
            out["log2_mic"].to_numpy()[:-1],
            [-2.0, -1.0, 1.0, 4.0],
        )
        assert pd.isna(out["log2_mic"].iloc[-1])
        assert list(out["censored"]) == [True, False, False, True, False]
        assert out["category"].isna().all()
        assert out["source"].isna().all()

    def test_output_columns(self, df):
        enc = MICEncoder(mic_col="MIC")
        out = enc.fit_transform(df)
        assert list(out.columns) == [
            "log2_mic",
            "censored",
            "category",
            "atu",
            "source",
        ]

    def test_get_feature_names_out(self):
        enc = MICEncoder()
        names = enc.get_feature_names_out()
        assert list(names) == [
            "log2_mic",
            "censored",
            "category",
            "atu",
            "source",
        ]


class TestMICEncoderWithBreakpoints:
    def test_single_drug_scalar(self, df, bp):
        enc = MICEncoder(
            breakpoints=bp,
            mic_col="MIC",
            species_col="Species",
            drug="Ceftriaxone",
        )
        out = enc.fit_transform(df)
        assert list(out["category"][:-1]) == ["S", "S", "I", "R"]
        assert pd.isna(out["category"].iloc[-1])
        assert out["source"].iloc[0] == bp.source

    def test_drug_column(self, df, bp):
        enc = MICEncoder(
            breakpoints=bp,
            mic_col="MIC",
            species_col="Species",
            drug_col="Drug",
        )
        out = enc.fit_transform(df)
        assert list(out["category"][:-1]) == ["S", "S", "I", "R"]

    def test_species_scalar(self, bp):
        df = pd.DataFrame({"MIC": ["0.5", "2", "8"]})
        enc = MICEncoder(
            breakpoints=bp,
            mic_col="MIC",
            species="Klebsiella pneumoniae",
            drug="Ceftriaxone",
        )
        out = enc.fit_transform(df)
        assert list(out["category"]) == ["S", "I", "R"]

    def test_missing_species_with_breakpoints_raises(self, df, bp):
        enc = MICEncoder(breakpoints=bp, mic_col="MIC", drug="Ceftriaxone")
        with pytest.raises(ValueError, match="species"):
            enc.fit(df)

    def test_atu_flag_propagates(self, bp):
        df = pd.DataFrame({"MIC": ["4", "8", "16"]})
        enc = MICEncoder(
            breakpoints=bp,
            mic_col="MIC",
            species="Klebsiella pneumoniae",
            drug="Meropenem",
        )
        out = enc.fit_transform(df)
        assert out["atu"].iloc[0] is True or out["atu"].iloc[0] == True  # noqa: E712
        assert out["atu"].iloc[1] is True or out["atu"].iloc[1] == True  # noqa: E712

    def test_both_drug_and_drug_col_raises(self, df, bp):
        enc = MICEncoder(
            breakpoints=bp,
            mic_col="MIC",
            species_col="Species",
            drug="X",
            drug_col="Drug",
        )
        with pytest.raises(ValueError, match="drug"):
            enc.fit(df)


class TestMICEncoderErrors:
    def test_missing_mic_column(self):
        df = pd.DataFrame({"X": [1, 2]})
        enc = MICEncoder(mic_col="MIC")
        with pytest.raises(KeyError, match="MIC"):
            enc.fit(df)
