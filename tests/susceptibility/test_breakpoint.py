"""Tests for BreakpointTable and BreakpointResult."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from maldiamrkit.susceptibility import BreakpointResult, BreakpointTable


@pytest.fixture
def example_yaml() -> Path:
    return resources.files("maldiamrkit") / "data" / "breakpoints" / "example.yaml"


@pytest.fixture
def bp(example_yaml) -> BreakpointTable:
    return BreakpointTable.from_yaml(example_yaml)


class TestBreakpointResult:
    def test_frozen(self):
        from dataclasses import FrozenInstanceError

        r = BreakpointResult(category="S", atu=False, source="EUCAST v16.0")
        with pytest.raises(FrozenInstanceError):
            r.category = "R"

    def test_none_category(self):
        r = BreakpointResult(category=None, atu=False, source=None)
        assert r.category is None


class TestBreakpointTableFromYaml:
    def test_loads_example(self, example_yaml):
        bp = BreakpointTable.from_yaml(example_yaml)
        assert len(bp) == 5
        assert "Klebsiella pneumoniae" in bp.species()
        assert "Ceftriaxone" in bp.drugs()
        assert bp.guideline == "EXAMPLE"
        assert bp.version == "0.0"
        assert bp.year == 2026

    def test_repr(self, bp):
        text = repr(bp)
        assert "EXAMPLE" in text
        assert "5 rows" in text

    def test_rows_returns_copy(self, bp):
        rows = bp.rows
        rows.loc[0, "s_le"] = 999.0
        assert bp.rows.loc[0, "s_le"] != 999.0

    def test_missing_required_columns(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "guideline: EXAMPLE\nversion: 0.0\nrows:\n  - species: foo\n    drug: bar\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing required columns"):
            BreakpointTable.from_yaml(bad)

    def test_s_le_above_r_gt_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "guideline: EXAMPLE\nversion: 0.0\nrows:\n"
            "  - species: foo\n    drug: bar\n    s_le: 8\n    r_gt: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="s_le > r_gt"):
            BreakpointTable.from_yaml(bad)

    def test_empty_rows_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("guideline: EXAMPLE\nversion: 0.0\nrows: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no 'rows' entries"):
            BreakpointTable.from_yaml(bad)


class TestApply:
    def test_susceptible(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Ceftriaxone", mic=0.5)
        assert r.category == "S"
        assert r.atu is False

    def test_intermediate(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Ceftriaxone", mic=2.0)
        assert r.category == "I"

    def test_resistant(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Ceftriaxone", mic=8.0)
        assert r.category == "R"

    def test_boundary_s_le(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Ceftriaxone", mic=1.0)
        assert r.category == "S"

    def test_boundary_r_gt(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Ceftriaxone", mic=2.0)
        assert r.category == "I"
        r2 = bp.apply("Klebsiella pneumoniae", "Ceftriaxone", mic=2.001)
        assert r2.category == "R"

    def test_no_i_zone_when_s_le_equals_r_gt(self, bp):
        r_s = bp.apply("Klebsiella pneumoniae", "Piperacillin-tazobactam", mic=8.0)
        r_r = bp.apply("Klebsiella pneumoniae", "Piperacillin-tazobactam", mic=9.0)
        assert r_s.category == "S"
        assert r_r.category == "R"

    def test_atu_range(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Meropenem", mic=4.0)
        assert r.category == "I"
        assert r.atu is True
        r2 = bp.apply("Klebsiella pneumoniae", "Meropenem", mic=2.0)
        assert r2.atu is False

    def test_atu_single_value(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Piperacillin-tazobactam", mic=16.0)
        assert r.atu is True

    def test_missing_pair(self, bp):
        r = bp.apply("Unknown sp.", "Ceftriaxone", mic=2.0)
        assert r.category is None
        assert r.source is None

    def test_nan_mic(self, bp):
        r = bp.apply("Klebsiella pneumoniae", "Ceftriaxone", mic=float("nan"))
        assert r.category is None
        assert r.source is not None

    def test_case_insensitive_lookup(self, bp):
        r1 = bp.apply("KLEBSIELLA PNEUMONIAE", "ceftriaxone", mic=0.5)
        r2 = bp.apply("klebsiella pneumoniae", "CEFTRIAXONE", mic=0.5)
        assert r1.category == "S"
        assert r2.category == "S"


class TestApplyBatch:
    def test_scalar_species_and_drug(self, bp):
        out = bp.apply_batch(
            "Klebsiella pneumoniae",
            "Ceftriaxone",
            [0.5, 2.0, 8.0],
        )
        assert list(out["category"]) == ["S", "I", "R"]

    def test_array_species_and_drug(self, bp):
        out = bp.apply_batch(
            ["Klebsiella pneumoniae", "Escherichia coli"],
            ["Ceftriaxone", "Ciprofloxacin"],
            [0.5, 1.0],
        )
        assert list(out["category"]) == ["S", "R"]

    def test_length_mismatch_raises(self, bp):
        with pytest.raises(ValueError, match="length"):
            bp.apply_batch(
                ["A", "B"],
                "Ceftriaxone",
                [1.0, 2.0, 3.0],
            )

    def test_nan_mic_in_batch(self, bp):
        out = bp.apply_batch(
            "Klebsiella pneumoniae",
            "Ceftriaxone",
            [0.5, np.nan, 8.0],
        )
        assert out["category"].iloc[0] == "S"
        assert out["category"].iloc[1] is None
        assert out["category"].iloc[2] == "R"


class TestRegistryConstructors:
    def test_list_available_returns_list(self):
        out = BreakpointTable.list_available()
        assert isinstance(out, list)

    def test_from_version_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="No bundled EUCAST"):
            BreakpointTable.from_version("999.0")

    def test_from_latest_raises_when_empty(self):
        if BreakpointTable.list_available():
            pytest.skip("vendored EUCAST data is present; cannot exercise empty path")
        with pytest.raises(FileNotFoundError, match="No bundled EUCAST"):
            BreakpointTable.from_latest()
