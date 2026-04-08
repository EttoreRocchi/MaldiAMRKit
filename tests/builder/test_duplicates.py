"""Tests for duplicate handling strategies."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from maldiamrkit.data.duplicates import (
    DuplicateStrategy,
    apply_index_strategy,
    apply_metadata_strategy,
)


class TestDuplicateStrategyEnum:
    """Tests for the DuplicateStrategy enum."""

    def test_all_values(self):
        assert set(DuplicateStrategy) == {
            DuplicateStrategy.first,
            DuplicateStrategy.last,
            DuplicateStrategy.drop,
            DuplicateStrategy.keep_all,
            DuplicateStrategy.average,
        }

    def test_string_coercion(self):
        assert DuplicateStrategy("first") is DuplicateStrategy.first
        assert DuplicateStrategy("keep_all") is DuplicateStrategy.keep_all

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DuplicateStrategy("invalid")


class TestApplyMetadataStrategy:
    """Tests for apply_metadata_strategy."""

    @pytest.fixture()
    def df_with_dups(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ID": ["A", "A", "B", "C", "C"],
                "value": [1, 2, 3, 4, 5],
                "pos": ["p1", "p2", "p1", "p1", "p2"],
            }
        )

    @pytest.fixture()
    def df_no_dups(self) -> pd.DataFrame:
        return pd.DataFrame({"ID": ["A", "B", "C"], "value": [1, 2, 3]})

    def test_no_duplicates_noop(self, df_no_dups):
        """All strategies should be no-ops when there are no duplicates."""
        for strategy in DuplicateStrategy:
            result = apply_metadata_strategy(df_no_dups, strategy)
            assert len(result) == 3

    def test_first_keeps_first_occurrence(self, df_with_dups):
        result = apply_metadata_strategy(df_with_dups, DuplicateStrategy.first)
        assert len(result) == 3
        assert result[result["ID"] == "A"]["value"].iloc[0] == 1
        assert result[result["ID"] == "C"]["value"].iloc[0] == 4

    def test_last_keeps_last_occurrence(self, df_with_dups):
        result = apply_metadata_strategy(df_with_dups, DuplicateStrategy.last)
        assert len(result) == 3
        assert result[result["ID"] == "A"]["value"].iloc[0] == 2
        assert result[result["ID"] == "C"]["value"].iloc[0] == 5

    def test_drop_removes_all_duplicates(self, df_with_dups):
        result = apply_metadata_strategy(df_with_dups, DuplicateStrategy.drop)
        assert len(result) == 1
        assert result["ID"].iloc[0] == "B"

    def test_keep_all_with_suffix_col(self, df_with_dups):
        result = apply_metadata_strategy(
            df_with_dups, DuplicateStrategy.keep_all, suffix_col="pos"
        )
        assert len(result) == 5
        assert "A_p1" in result["ID"].values
        assert "A_p2" in result["ID"].values
        assert "C_p1" in result["ID"].values
        assert "C_p2" in result["ID"].values

    def test_keep_all_without_suffix_col(self, df_with_dups):
        result = apply_metadata_strategy(df_with_dups, DuplicateStrategy.keep_all)
        assert len(result) == 5
        assert "A_rep1" in result["ID"].values
        assert "A_rep2" in result["ID"].values

    def test_average_keeps_all_rows_and_adds_original_id(self, df_with_dups):
        result = apply_metadata_strategy(
            df_with_dups, DuplicateStrategy.average, suffix_col="pos"
        )
        assert len(result) == 5
        assert "_original_id" in result.columns
        assert set(result["_original_id"]) == {"A", "B", "C"}

    def test_average_without_suffix_col(self, df_with_dups):
        result = apply_metadata_strategy(df_with_dups, DuplicateStrategy.average)
        assert "_original_id" in result.columns
        assert len(result) == 5

    def test_string_strategy_accepted(self, df_with_dups):
        result = apply_metadata_strategy(df_with_dups, "first")
        assert len(result) == 3

    def test_logs_warning_on_duplicates(self, df_with_dups, caplog):
        with caplog.at_level(logging.WARNING, logger="maldiamrkit.data.duplicates"):
            apply_metadata_strategy(df_with_dups, DuplicateStrategy.first)
        assert "duplicate" in caplog.text.lower()

    def test_no_warning_without_duplicates(self, df_no_dups, caplog):
        with caplog.at_level(logging.WARNING, logger="maldiamrkit.data.duplicates"):
            apply_metadata_strategy(df_no_dups, DuplicateStrategy.first)
        assert caplog.text == ""

    def test_all_rows_duplicate(self):
        df = pd.DataFrame({"ID": ["A", "A", "A"], "v": [1, 2, 3]})
        result = apply_metadata_strategy(df, DuplicateStrategy.drop)
        assert len(result) == 0

    def test_single_row(self):
        df = pd.DataFrame({"ID": ["A"], "v": [1]})
        for strategy in DuplicateStrategy:
            result = apply_metadata_strategy(df, strategy)
            assert len(result) == 1


class TestApplyIndexStrategy:
    """Tests for apply_index_strategy."""

    @pytest.fixture()
    def df_with_dups(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"path": ["/a1", "/a2", "/b", "/c1", "/c2"]},
            index=["A", "A", "B", "C", "C"],
        )

    @pytest.fixture()
    def df_no_dups(self) -> pd.DataFrame:
        return pd.DataFrame({"path": ["/a", "/b", "/c"]}, index=["A", "B", "C"])

    def test_no_duplicates_noop(self, df_no_dups):
        for strategy in DuplicateStrategy:
            result = apply_index_strategy(df_no_dups, strategy)
            assert len(result) == 3

    def test_first(self, df_with_dups):
        result = apply_index_strategy(df_with_dups, DuplicateStrategy.first)
        assert len(result) == 3
        assert result.loc["A", "path"] == "/a1"

    def test_last(self, df_with_dups):
        result = apply_index_strategy(df_with_dups, DuplicateStrategy.last)
        assert len(result) == 3
        assert result.loc["A", "path"] == "/a2"

    def test_drop(self, df_with_dups):
        result = apply_index_strategy(df_with_dups, DuplicateStrategy.drop)
        assert len(result) == 1
        assert result.index[0] == "B"

    def test_keep_all(self, df_with_dups):
        result = apply_index_strategy(df_with_dups, DuplicateStrategy.keep_all)
        assert len(result) == 5
        assert "A_rep1" in result.index
        assert "A_rep2" in result.index
        assert "B_rep1" in result.index

    def test_average(self, df_with_dups):
        result = apply_index_strategy(df_with_dups, DuplicateStrategy.average)
        assert len(result) == 5
        assert "_original_id" in result.columns
        assert set(result["_original_id"]) == {"A", "B", "C"}

    def test_logs_warning(self, df_with_dups, caplog):
        with caplog.at_level(logging.WARNING, logger="maldiamrkit.data.duplicates"):
            apply_index_strategy(df_with_dups, DuplicateStrategy.first)
        assert "duplicate" in caplog.text.lower()
