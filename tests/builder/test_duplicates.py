"""Tests for duplicate handling strategies."""

from __future__ import annotations

import logging
import warnings

import pandas as pd
import pytest

from maldiamrkit.data.duplicates import (
    DuplicateStrategy,
    _merge_spectrum_rows,
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

    def test_keep_all_isolate_from_string_ids_without_warning(self, df_with_dups):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = apply_metadata_strategy(df_with_dups, DuplicateStrategy.keep_all)
        assert list(result["_isolate"]) == ["A", "A", "B", "C", "C"]

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
        assert "B" in result.index

    def test_average(self, df_with_dups):
        result = apply_index_strategy(df_with_dups, DuplicateStrategy.average)
        assert len(result) == 5
        assert "_original_id" in result.columns
        assert set(result["_original_id"]) == {"A", "B", "C"}

    def test_logs_warning(self, df_with_dups, caplog):
        with caplog.at_level(logging.WARNING, logger="maldiamrkit.data.duplicates"):
            apply_index_strategy(df_with_dups, DuplicateStrategy.first)
        assert "duplicate" in caplog.text.lower()


class TestSuffixOnlyDuplicatedIds:
    """keep_all/average suffix only the duplicated IDs."""

    @pytest.fixture()
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({"ID": ["x", "y", "y"], "pos": ["p1", "p1", "p2"]})

    @pytest.mark.parametrize("strategy", ["keep_all", "average"])
    def test_unique_ids_unchanged(self, df, strategy):
        out = apply_metadata_strategy(df, strategy)
        assert out["ID"].tolist() == ["x", "y_rep1", "y_rep2"]
        out = apply_metadata_strategy(df, strategy, suffix_col="pos")
        assert out["ID"].tolist() == ["x", "y_p1", "y_p2"]

    def test_average_tags_every_row(self, df):
        out = apply_metadata_strategy(df, "average")
        assert out["_original_id"].tolist() == ["x", "y", "y"]

    def test_keep_all_writes_no_original_id(self, df):
        assert "_original_id" not in apply_metadata_strategy(df, "keep_all").columns

    def test_suffix_collision_raises(self):
        df = pd.DataFrame({"ID": ["y_rep1", "y", "y"]})
        with pytest.raises(ValueError, match="y_rep1"):
            apply_metadata_strategy(df, "keep_all")

    def test_index_keep_all_keeps_unique_ids(self):
        df = pd.DataFrame({"path": ["/x", "/y1", "/y2"]}, index=["x", "y", "y"])
        out = apply_index_strategy(df, "keep_all")
        assert list(out.index) == ["x", "y_rep1", "y_rep2"]
        assert "_original_id" not in out.columns


class TestMergeSpectrumRows:
    """Rows of one spectrum (same ID, target position and path) are merged."""

    @staticmethod
    def _merge(df):
        merged, _ = _merge_spectrum_rows(
            df, path_column="path", target_position_column="pos"
        )
        return merged

    @staticmethod
    def _rows(**columns):
        base = {
            "ID": ["A", "A", "B"],
            "pos": ["p1", "p1", "p1"],
            "path": ["a", "a", "b"],
        }
        return pd.DataFrame({**base, **columns})

    def test_calls_that_disagree_become_missing(self):
        """S against R becomes missing; other spectra keep their call."""
        merged = self._merge(self._rows(Cipro=["S", "R", "S"]))
        assert merged["ID"].tolist() == ["A", "B"]
        assert pd.isna(merged["Cipro"].iloc[0])
        assert merged["Cipro"].iloc[1] == "S"

    def test_starred_and_plain_calls_agree_keeping_the_first(self):
        """R* and R (S and ' S* ') agree; the first value is kept."""
        merged = self._merge(self._rows(Amp=["R*", "R", "S"], Levo=["S", " S* ", "R"]))
        assert merged["Amp"].tolist() == ["R*", "S"]
        assert merged["Levo"].tolist() == ["S", "R"]

    def test_other_values_in_call_columns_disagree_with_calls(self):
        """ESBL against R in a call column becomes missing."""
        merged = self._merge(self._rows(Cefotaxime=["ESBL", "R", "S"]))
        assert pd.isna(merged["Cefotaxime"].iloc[0])

    def test_other_columns_join_distinct_values_in_order(self):
        """Text and MIC values are joined with '; ' in order."""
        merged = self._merge(
            self._rows(Mechanism=["ESBL", "AmpC", None], MIC=["32", ">32", "4"])
        )
        assert merged["Mechanism"].iloc[0] == "ESBL; AmpC"
        assert merged["MIC"].iloc[0] == "32; >32"
        assert pd.isna(merged["Mechanism"].iloc[1])

    def test_empty_cells_are_ignored(self):
        """An empty cell neither joins nor disagrees."""
        merged = self._merge(
            self._rows(Mechanism=[None, "ESBL", None], Cipro=["", "R", "S"])
        )
        assert merged["Mechanism"].iloc[0] == "ESBL"
        assert merged["Cipro"].iloc[0] == "R"

    def test_replicates_at_other_positions_are_kept(self):
        """Rows at other target positions are replicates, not merged."""
        df = pd.DataFrame(
            {
                "ID": ["A", "A"],
                "pos": ["p1", "p2"],
                "path": ["a1", "a2"],
                "Cipro": ["S", "R"],
            }
        )
        pd.testing.assert_frame_equal(self._merge(df), df)

    def test_missing_target_position_column_merges_on_id_and_path(self):
        """Without a target position column, ID and path are the key."""
        df = self._rows(Cipro=["S", "S", "R"]).drop(columns="pos")
        assert self._merge(df)["ID"].tolist() == ["A", "B"]

    def test_missing_path_column_does_not_merge(self):
        """ID and target position alone may name different spectra."""
        df = self._rows(Cipro=["S", "R", "S"]).drop(columns="path")
        pd.testing.assert_frame_equal(self._merge(df), df)

    def test_id_alone_is_not_a_spectrum_key(self):
        """Without target position and path nothing is merged."""
        df = pd.DataFrame({"ID": ["A", "A"], "Cipro": ["S", "R"]})
        pd.testing.assert_frame_equal(self._merge(df), df)

    def test_rows_keep_their_order(self):
        """Merged rows stay at the position of their first row."""
        df = pd.DataFrame(
            {
                "ID": ["B", "A", "C", "A"],
                "pos": ["p", "p", "p", "p"],
                "path": ["b", "a", "c", "a"],
                "Cipro": ["S", "R", "I", "R*"],
            }
        )
        merged = self._merge(df)
        assert merged["ID"].tolist() == ["B", "A", "C"]
        assert merged["Cipro"].tolist() == ["S", "R", "I"]

    def test_counts_rows_spectra_merges_and_conflicting_calls(self):
        """3 rows give 2 spectra; A had 2 rows and 1 call they disagree on."""
        df = self._rows(Cipro=["S", "R", "S"], Amp=["R*", "R", "S"])
        _, counts = _merge_spectrum_rows(
            df, path_column="path", target_position_column="pos"
        )
        assert counts == {
            "rows": 3,
            "spectra": 2,
            "merged_spectra": 1,
            "conflicting_calls": 1,
        }

    def test_counts_without_repeated_rows(self):
        """Nothing to merge: every row is a spectrum."""
        df = pd.DataFrame({"ID": ["A", "B"], "pos": ["p", "p"], "path": ["a", "b"]})
        _, counts = _merge_spectrum_rows(
            df, path_column="path", target_position_column="pos"
        )
        assert counts == {
            "rows": 2,
            "spectra": 2,
            "merged_spectra": 0,
            "conflicting_calls": 0,
        }

    def test_one_warning_counts_spectra_rows_and_missing_calls(self, caplog):
        """One warning gives the spectra, rows and missing calls."""
        df = self._rows(Cipro=["S", "R", "S"], Amp=["R", "S", "R"])
        with caplog.at_level(logging.WARNING):
            self._merge(df)
        assert caplog.text.count("several metadata rows") == 1
        assert "1 spectra had several metadata rows (2 rows)" in caplog.text
        assert "2 susceptibility calls" in caplog.text
