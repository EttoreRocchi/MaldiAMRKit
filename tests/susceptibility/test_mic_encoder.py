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


class TestCensoredMICs:
    """A censored MIC gets S or R only when its bound decides the category."""

    TABLE = [
        ("Ceftriaxone", "<=1", "S", False),
        ("Ceftriaxone", "<1", "S", False),
        ("Ceftriaxone", "<=2", None, False),
        ("Ceftriaxone", "<2", None, False),
        ("Ceftriaxone", ">2", "R", False),
        ("Ceftriaxone", ">1", None, False),
        ("Ceftriaxone", ">=2", None, False),
        ("Ceftriaxone", ">=4", "R", False),
        ("Piperacillin-tazobactam", ">8", "R", False),
        ("Piperacillin-tazobactam", "<=16", None, False),
        ("Piperacillin-tazobactam", "<=8", "S", False),
        ("Piperacillin-tazobactam", ">=8", None, False),
        ("Piperacillin-tazobactam", ">=16", "R", False),
        ("Piperacillin-tazobactam", "8", "S", False),
        ("Meropenem", "<=4", None, False),
        ("Meropenem", ">4", None, False),
        ("Meropenem", ">8", "R", False),
        ("Meropenem", "4", "I", True),
    ]

    @pytest.mark.parametrize("drug, mic, category, atu", TABLE)
    def test_truth_table(self, bp, drug, mic, category, atu):
        df = pd.DataFrame({"MIC": [mic], "Drug": [drug]})
        enc = MICEncoder(
            breakpoints=bp,
            mic_col="MIC",
            species="Klebsiella pneumoniae",
            drug_col="Drug",
        )
        out = enc.fit_transform(df).iloc[0]
        if category is None:
            assert pd.isna(out["category"])
            assert out["source"] == bp.source
        else:
            assert out["category"] == category
        assert bool(out["atu"]) is atu

    def test_subclass_apply_batch_is_used(self, bp):
        calls = []

        class Spy(BreakpointTable):
            def apply_batch(self, species, drug, mic):
                calls.append(np.asarray(mic, dtype=float))
                return super().apply_batch(species, drug, mic)

        spy = Spy(bp.rows, guideline=bp.guideline, version=bp.version)
        df = pd.DataFrame({"MIC": [">8", "<=8"]})
        out = MICEncoder(
            breakpoints=spy,
            species="Klebsiella pneumoniae",
            drug="Piperacillin-tazobactam",
        ).fit_transform(df)
        assert calls
        assert list(out["category"]) == ["R", "S"]
