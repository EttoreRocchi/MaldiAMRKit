"""Tests for MIC parsing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.io.mic import parse_mic_column


class TestParseMicColumn:
    """Tests for parse_mic_column()."""

    def test_simple_numeric(self):
        s = pd.Series(["8"])
        result = parse_mic_column(s)
        assert result["value"].iloc[0] == 8.0
        assert result["qualifier"].iloc[0] == "="

    def test_qualifier_le(self):
        s = pd.Series(["<=8"])
        result = parse_mic_column(s)
        assert result["value"].iloc[0] == 8.0
        assert result["qualifier"].iloc[0] == "<="

    def test_qualifier_gt(self):
        s = pd.Series([">16"])
        result = parse_mic_column(s)
        assert result["value"].iloc[0] == 16.0
        assert result["qualifier"].iloc[0] == ">"

    def test_qualifier_ge(self):
        s = pd.Series([">=4"])
        result = parse_mic_column(s)
        assert result["value"].iloc[0] == 4.0
        assert result["qualifier"].iloc[0] == ">="

    def test_qualifier_lt(self):
        s = pd.Series(["<2"])
        result = parse_mic_column(s)
        assert result["value"].iloc[0] == 2.0
        assert result["qualifier"].iloc[0] == "<"

    def test_european_decimal(self):
        s = pd.Series(["0,5"])
        result = parse_mic_column(s)
        assert result["value"].iloc[0] == pytest.approx(0.5)
        assert result["qualifier"].iloc[0] == "="

    def test_european_decimal_with_qualifier(self):
        s = pd.Series(["<=0,25"])
        result = parse_mic_column(s)
        assert result["value"].iloc[0] == pytest.approx(0.25)
        assert result["qualifier"].iloc[0] == "<="

    def test_nan_input(self):
        s = pd.Series([None])
        result = parse_mic_column(s)
        assert np.isnan(result["value"].iloc[0])
        assert result["qualifier"].iloc[0] == ""

    def test_empty_string(self):
        s = pd.Series([""])
        result = parse_mic_column(s)
        assert np.isnan(result["value"].iloc[0])
        assert result["qualifier"].iloc[0] == ""

    def test_ris_category_ignored(self):
        """R/I/S categorical values should not parse as MIC."""
        s = pd.Series(["S", "R", "I"])
        result = parse_mic_column(s)
        assert result["value"].isna().all()

    def test_mixed_column(self):
        s = pd.Series(["<=8", ">16", "0,5", None, "S", ">=4", ""])
        result = parse_mic_column(s)
        assert len(result) == 7
        assert result["value"].iloc[0] == 8.0
        assert result["qualifier"].iloc[0] == "<="
        assert result["value"].iloc[1] == 16.0
        assert result["qualifier"].iloc[1] == ">"
        assert result["value"].iloc[2] == pytest.approx(0.5)
        assert np.isnan(result["value"].iloc[3])
        assert np.isnan(result["value"].iloc[4])  # "S" not parsed
        assert result["value"].iloc[5] == 4.0
        assert np.isnan(result["value"].iloc[6])

    def test_preserves_index(self):
        s = pd.Series(["<=8", ">16"], index=[10, 20])
        result = parse_mic_column(s)
        assert list(result.index) == [10, 20]

    def test_output_columns(self):
        s = pd.Series(["<=8"])
        result = parse_mic_column(s)
        assert list(result.columns) == ["value", "qualifier"]
