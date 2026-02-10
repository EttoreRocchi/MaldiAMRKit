"""Tests for file reading utilities."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from maldiamrkit.io import read_spectrum, sniff_delimiter


class TestSniffDelimiter:
    """Tests for delimiter detection."""

    def test_tab_delimiter(self, tmp_path):
        f = tmp_path / "tab.txt"
        f.write_text("2000\t1234\n2001\t1456\n2002\t1678\n")
        assert sniff_delimiter(f) == "\t"

    def test_comma_delimiter(self, tmp_path):
        f = tmp_path / "comma.csv"
        f.write_text("2000,1234\n2001,1456\n2002,1678\n")
        assert sniff_delimiter(f) == ","

    def test_semicolon_delimiter(self, tmp_path):
        f = tmp_path / "semi.csv"
        f.write_text("2000;1234\n2001;1456\n2002;1678\n")
        assert sniff_delimiter(f) == ";"

    def test_space_delimiter(self, tmp_path):
        f = tmp_path / "space.txt"
        f.write_text("2000 1234\n2001 1456\n2002 1678\n")
        assert sniff_delimiter(f) == " "

    def test_short_file(self, tmp_path):
        """File with fewer lines than sample_lines should not crash."""
        f = tmp_path / "short.txt"
        f.write_text("2000\t1234\n2001\t1456\n")
        assert sniff_delimiter(f, sample_lines=10) == "\t"

    def test_single_line_file(self, tmp_path):
        f = tmp_path / "single.txt"
        f.write_text("2000\t1234\n")
        assert sniff_delimiter(f, sample_lines=10) == "\t"

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises(csv.Error, match="empty"):
            sniff_delimiter(f)


class TestReadSpectrum:
    """Tests for spectrum file reading."""

    def test_tab_separated(self, tmp_path):
        f = tmp_path / "spec.txt"
        f.write_text("2000\t100\n2001\t200\n2002\t300\n")
        df = read_spectrum(f)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 3
        assert df["mass"].iloc[0] == 2000
        assert df["intensity"].iloc[-1] == 300

    def test_comma_separated(self, tmp_path):
        f = tmp_path / "spec.csv"
        f.write_text("2000,100\n2001,200\n2002,300\n")
        df = read_spectrum(f)
        assert len(df) == 3
        assert df["mass"].iloc[1] == 2001

    def test_comment_lines_skipped(self, tmp_path):
        f = tmp_path / "commented.txt"
        f.write_text("# comment line\n2000\t100\n2001\t200\n")
        df = read_spectrum(f)
        assert len(df) == 2
        assert df["mass"].iloc[0] == 2000

    def test_whitespace_fallback(self, tmp_path):
        """When sniffer fails, falls back to whitespace delimiter."""
        f = tmp_path / "odd.txt"
        f.write_text("2000  100\n2001  200\n")
        df = read_spectrum(f)
        assert len(df) == 2

    def test_returns_dataframe(self, tmp_path):
        f = tmp_path / "spec.txt"
        f.write_text("2000\t100\n")
        result = read_spectrum(f)
        assert isinstance(result, pd.DataFrame)

    def test_file_not_found_raises(self, tmp_path):
        """FileNotFoundError should not be masked by fallback."""
        with pytest.raises(FileNotFoundError):
            read_spectrum(tmp_path / "nonexistent.txt")

    def test_float_values(self, tmp_path):
        f = tmp_path / "floats.txt"
        f.write_text("2000.5\t100.123\n2001.5\t200.456\n")
        df = read_spectrum(f)
        assert df["mass"].iloc[0] == pytest.approx(2000.5)
        assert df["intensity"].iloc[1] == pytest.approx(200.456)
