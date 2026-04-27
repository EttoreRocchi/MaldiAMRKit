"""Tests for dataset layout adapters (DRIAMSLayout, MARISMaLayout, helpers)."""

from __future__ import annotations

import struct
from pathlib import Path

import pandas as pd
import pytest

from maldiamrkit.data.dataset_layouts import (
    DRIAMSLayout,
    MARISMaLayout,
    _detect_id_column,
    _discover_driams_metadata,
)


def _make_bruker_dir(root: Path, rel_path: str) -> Path:
    """Create a minimal Bruker directory with acqus + fid."""
    bruker_dir = root / rel_path / "1" / "1SLin"
    bruker_dir.mkdir(parents=True, exist_ok=True)
    acqus_content = (
        "##TITLE= Test\n"
        "##$TD= 8\n"
        "##$DELAY= 0\n"
        "##$DW= 1\n"
        "##$ML1= 1.0\n"
        "##$ML2= 0.0\n"
        "##$ML3= 0.0\n"
        "##$BYTORDA= 0\n"
        "##END=\n"
    )
    (bruker_dir / "acqus").write_text(acqus_content)
    (bruker_dir / "fid").write_bytes(struct.pack("<8i", *range(1, 9)))
    return root / rel_path


class TestDetectIdColumn:
    """Tests for the _detect_id_column helper."""

    def test_code_present_returns_code(self):
        """Verify 'code' column is preferred."""
        meta = pd.DataFrame({"code": [1], "ID": [2], "other": [3]})
        assert _detect_id_column(meta) == "code"

    def test_id_present_returns_id(self):
        """Verify 'ID' column is second choice."""
        meta = pd.DataFrame({"ID": [1], "other": [2]})
        assert _detect_id_column(meta) == "ID"

    def test_neither_returns_first_column(self):
        """Verify first column is fallback."""
        meta = pd.DataFrame({"specimen": [1], "year": [2]})
        assert _detect_id_column(meta) == "specimen"


class TestDiscoverDriamsMetadata:
    """Tests for the _discover_driams_metadata helper function."""

    def test_specific_year_loads_csv(self, tmp_path):
        """Verify loading metadata for a specific year."""
        year_dir = tmp_path / "2024"
        year_dir.mkdir()
        pd.DataFrame({"ID": ["a", "b"]}).to_csv(
            year_dir / "2024_clean.csv", index=False
        )
        meta, years = _discover_driams_metadata(tmp_path, "2024")
        assert len(meta) == 2
        assert years == ["2024"]

    def test_specific_year_missing_raises(self, tmp_path):
        """Verify FileNotFoundError when year CSV is missing."""
        year_dir = tmp_path / "2024"
        year_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            _discover_driams_metadata(tmp_path, "2024")

    def test_year_dirs_merge(self, tmp_path):
        """Verify multiple year directories are merged."""
        for yr in ("2023", "2024"):
            yd = tmp_path / yr
            yd.mkdir()
            pd.DataFrame({"ID": [f"{yr}_a"]}).to_csv(
                yd / f"{yr}_clean.csv", index=False
            )
        meta, years = _discover_driams_metadata(tmp_path, None)
        assert len(meta) == 2
        assert set(years) == {"2023", "2024"}

    def test_year_dirs_no_csvs_raises(self, tmp_path):
        """Verify FileNotFoundError when year dirs exist but have no CSVs."""
        (tmp_path / "2024").mkdir()
        with pytest.raises(FileNotFoundError, match="no.*_clean.csv"):
            _discover_driams_metadata(tmp_path, None)

    def test_no_year_dirs_suffix_csv(self, tmp_path):
        """Verify fallback to *_clean.csv in flat directory."""
        pd.DataFrame({"ID": ["a"]}).to_csv(tmp_path / "dataset_clean.csv", index=False)
        meta, years = _discover_driams_metadata(tmp_path, None)
        assert len(meta) == 1
        assert years == []

    def test_no_year_dirs_generic_csv(self, tmp_path):
        """Verify fallback to any .csv file when no *_clean.csv exists."""
        pd.DataFrame({"ID": ["a"]}).to_csv(tmp_path / "metadata.csv", index=False)
        meta, years = _discover_driams_metadata(tmp_path, None)
        assert len(meta) == 1
        assert years == []

    def test_no_csvs_raises(self, tmp_path):
        """Verify FileNotFoundError when no CSV files exist at all."""
        with pytest.raises(FileNotFoundError, match="No metadata CSV files"):
            _discover_driams_metadata(tmp_path, None)


class TestDRIAMSLayout:
    """Tests for DRIAMSLayout navigation."""

    def test_discover_metadata_no_id_dir_raises(self, tmp_path):
        """Verify FileNotFoundError when metadata dir is missing."""
        layout = DRIAMSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="Metadata directory"):
            layout.discover_metadata()

    def test_discover_metadata_auto_detect_id_column(self, tmp_path):
        """Verify auto-detection of ID column (code > ID > first)."""
        id_dir = tmp_path / "id"
        id_dir.mkdir()
        pd.DataFrame({"code": ["a"], "Species": ["E. coli"]}).to_csv(
            id_dir / "data_clean.csv", index=False
        )
        layout = DRIAMSLayout(tmp_path)
        meta = layout.discover_metadata()
        assert "ID" in meta.columns
        assert meta["ID"].iloc[0] == "a"

    def test_detect_stage_binned(self, tmp_path):
        """Verify binned_* directories are detected first."""
        (tmp_path / "binned_3").mkdir()
        (tmp_path / "preprocessed").mkdir()
        (tmp_path / "id").mkdir()
        layout = DRIAMSLayout(tmp_path)
        assert layout.detect_stage() == "binned_3"

    def test_detect_stage_preprocessed(self, tmp_path):
        """Verify preprocessed is detected when no binned dirs exist."""
        (tmp_path / "preprocessed").mkdir()
        (tmp_path / "id").mkdir()
        layout = DRIAMSLayout(tmp_path)
        assert layout.detect_stage() == "preprocessed"

    def test_detect_stage_raw(self, tmp_path):
        """Verify raw is detected when no binned or preprocessed exist."""
        (tmp_path / "raw").mkdir()
        (tmp_path / "id").mkdir()
        layout = DRIAMSLayout(tmp_path)
        assert layout.detect_stage() == "raw"

    def test_detect_stage_none_raises(self, tmp_path):
        """Verify FileNotFoundError when no recognized stage exists."""
        (tmp_path / "id").mkdir()
        (tmp_path / "other").mkdir()
        layout = DRIAMSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="No recognised stage"):
            layout.detect_stage()

    def test_collect_spectrum_files_year_subdirs(self, tmp_path):
        """Verify files are found in year subdirectories."""
        stage_dir = tmp_path / "binned_3"
        year_dir = stage_dir / "2024"
        year_dir.mkdir(parents=True)
        (year_dir / "s1.txt").write_text("1000 100\n")
        (tmp_path / "id").mkdir()
        layout = DRIAMSLayout(tmp_path)
        files = layout.collect_spectrum_files("binned_3", None)
        assert len(files) == 1

    def test_collect_spectrum_files_flat(self, tmp_path):
        """Verify files are found in flat stage directory."""
        stage_dir = tmp_path / "preprocessed"
        stage_dir.mkdir()
        (stage_dir / "s1.txt").write_text("1000 100\n")
        layout = DRIAMSLayout(tmp_path)
        files = layout.collect_spectrum_files("preprocessed", None)
        assert len(files) == 1

    def test_collect_spectrum_files_specific_year(self, tmp_path):
        """Verify year parameter filters to specific year subdir."""
        stage_dir = tmp_path / "binned_3"
        for yr in ("2023", "2024"):
            yd = stage_dir / yr
            yd.mkdir(parents=True)
            (yd / f"s_{yr}.txt").write_text("1000 100\n")
        layout = DRIAMSLayout(tmp_path)
        files = layout.collect_spectrum_files("binned_3", "2024")
        assert len(files) == 1
        assert "2024" in str(files[0])

    def test_collect_spectrum_files_missing_stage_raises(self, tmp_path):
        """Verify FileNotFoundError when stage dir doesn't exist."""
        layout = DRIAMSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="Stage folder"):
            layout.collect_spectrum_files("binned_3", None)


class TestDRIAMSLayoutIdTransform:
    """``id_transform`` collapses technical replicates at load time.

    Regression guard for the replicate-leakage footgun: DRIAMS files
    carry a ``UUID_MALDI<N>`` suffix that makes each replicate a
    distinct ID under the default ``duplicate_strategy``. Callers
    who CV-split the resulting feature matrix leak replicates across
    folds. ``id_transform`` provides the opt-in fix.
    """

    def _make_metadata(self, tmp_path: Path, codes: list[str]) -> DRIAMSLayout:
        """Write a minimal ``id/data_clean.csv`` with the given codes."""
        id_dir = tmp_path / "id"
        id_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"code": codes, "Species": ["E. coli"] * len(codes)}).to_csv(
            id_dir / "data_clean.csv", index=False
        )
        return tmp_path

    def test_collapses_maldi_replicates_via_first(self, tmp_path):
        """``id_transform`` + ``duplicate_strategy='first'`` keeps one per UUID."""
        import re as _re

        self._make_metadata(
            tmp_path,
            codes=[
                "uuid-a_MALDI1",
                "uuid-a_MALDI2",
                "uuid-b_MALDI1",
                "uuid-c_MALDI1",
                "uuid-c_MALDI2",
            ],
        )
        layout = DRIAMSLayout(
            tmp_path,
            id_transform=lambda s: _re.sub(r"_MALDI\d+$", "", s),
            duplicate_strategy="first",
        )
        meta = layout.discover_metadata()
        assert len(meta) == 3
        assert set(meta["ID"]) == {
            "uuid-a_MALDI1",
            "uuid-b_MALDI1",
            "uuid-c_MALDI1",
        }
        # ``_canonical_id`` is an internal scratch column - must not leak.
        assert "_canonical_id" not in meta.columns

    def test_preserves_raw_ids_for_loader_matching(self, tmp_path):
        """Raw ``ID`` column untouched so the loader can still match files."""
        self._make_metadata(tmp_path, codes=["uuid-a_MALDI1", "uuid-a_MALDI2"])
        layout = DRIAMSLayout(
            tmp_path,
            id_transform=lambda s: s.split("_")[0],
            duplicate_strategy="first",
        )
        meta = layout.discover_metadata()
        assert meta["ID"].iloc[0] == "uuid-a_MALDI1"

    def test_none_preserves_legacy_behaviour(self, tmp_path):
        """Without ``id_transform``, per-replicate IDs are kept (legacy)."""
        self._make_metadata(tmp_path, codes=["uuid-a_MALDI1", "uuid-a_MALDI2"])
        layout = DRIAMSLayout(tmp_path)
        meta = layout.discover_metadata()
        assert len(meta) == 2

    def test_replicate_warning_fires_when_transform_omitted(self, tmp_path, caplog):
        """A one-time log warning points at the fix when replicates are detected."""
        import logging as _logging

        self._make_metadata(
            tmp_path,
            codes=[
                "uuid-a_MALDI1",
                "uuid-a_MALDI2",
                "uuid-b_MALDI1",
            ],
        )
        caplog.clear()
        with caplog.at_level(
            _logging.WARNING, logger="maldiamrkit.data.dataset_layouts"
        ):
            DRIAMSLayout(tmp_path).discover_metadata()
        msgs = [r.message for r in caplog.records if r.levelno >= _logging.WARNING]
        assert any("_MALDI" in m for m in msgs)

    def test_replicate_warning_silenced_by_identity_transform(self, tmp_path, caplog):
        """Passing any ``id_transform`` (even ``str``) acknowledges the issue."""
        import logging as _logging

        self._make_metadata(
            tmp_path,
            codes=["uuid-a_MALDI1", "uuid-a_MALDI2"],
        )
        caplog.clear()
        with caplog.at_level(
            _logging.WARNING, logger="maldiamrkit.data.dataset_layouts"
        ):
            # ``str`` is an identity transform on strings - opts out
            # of deduplication but silences the warning.
            DRIAMSLayout(tmp_path, id_transform=str).discover_metadata()
        msgs = [r.message for r in caplog.records]
        assert not any("_MALDI" in m for m in msgs)

    def test_no_warning_when_ids_are_replicate_free(self, tmp_path, caplog):
        """A dataset without ``_MALDI<N>`` suffixes produces no warning."""
        import logging as _logging

        self._make_metadata(tmp_path, codes=["sample-a", "sample-b", "sample-c"])
        caplog.clear()
        with caplog.at_level(
            _logging.WARNING, logger="maldiamrkit.data.dataset_layouts"
        ):
            DRIAMSLayout(tmp_path).discover_metadata()
        msgs = [r.message for r in caplog.records]
        assert not any("_MALDI" in m for m in msgs)


class TestMARISMaLayout:
    """Tests for MARISMaLayout navigation."""

    def test_discover_metadata_strategy_first(self, tmp_path):
        """Verify duplicate_strategy='first' drops duplicates by ID."""
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A", "A", "B"],
                "Path": ["p1", "p2", "p3"],
                "target_position": ["0_A1", "0_A2", "0_B1"],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(tmp_path, csv_path)
        meta = layout.discover_metadata()
        assert len(meta) == 2

    def test_discover_metadata_strategy_keep_all(self, tmp_path):
        """Verify duplicate_strategy='keep_all' creates combined IDs."""
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A", "A"],
                "Path": ["p1", "p2"],
                "target_position": ["0_A1", "0_A2"],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(tmp_path, csv_path, duplicate_strategy="keep_all")
        meta = layout.discover_metadata()
        assert "A_0_A1" in meta["ID"].values
        assert "A_0_A2" in meta["ID"].values

    def test_discover_metadata_year_filter(self, tmp_path):
        """Verify year parameter filters metadata rows."""
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A", "B"],
                "Path": ["p1", "p2"],
                "target_position": ["0_A1", "0_B1"],
                "Year": [2023, 2024],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(tmp_path, csv_path, year=2024)
        meta = layout.discover_metadata()
        assert len(meta) == 1
        assert meta["ID"].iloc[0] == "B"

    def test_discover_metadata_missing_id_col_raises(self, tmp_path):
        """Verify ValueError when ID column is missing."""
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame({"Name": ["A"]}).to_csv(csv_path, index=False)
        layout = MARISMaLayout(tmp_path, csv_path)
        with pytest.raises(ValueError, match="ID column 'Identifier' not in"):
            layout.discover_metadata()

    def test_discover_metadata_renames_id_column(self, tmp_path):
        """Verify non-default id_column is renamed to 'ID'."""
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A"],
                "Path": ["p1"],
                "target_position": ["0_A1"],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(tmp_path, csv_path, id_column="Identifier")
        meta = layout.discover_metadata()
        assert "ID" in meta.columns

    def test_collect_spectrum_files_resolves_bruker(self, tmp_path):
        """Verify Bruker directories are resolved and missing dirs skipped."""
        root = tmp_path / "data"
        _make_bruker_dir(root, "specimen_A")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A", "B"],
                "Path": ["specimen_A", "nonexistent"],
                "target_position": ["0_A1", "0_B1"],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(root, csv_path)
        files = layout.collect_spectrum_files(None, None)
        assert len(files) == 1

    def test_collect_spectrum_files_year_filter(self, tmp_path):
        """Verify year parameter filters spectrum files."""
        root = tmp_path / "data"
        _make_bruker_dir(root, "specimen_A")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A", "B"],
                "Path": ["specimen_A", "specimen_A"],
                "target_position": ["0_A1", "0_B1"],
                "Year": [2023, 2024],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(root, csv_path)
        files = layout.collect_spectrum_files(None, "2024")
        assert len(files) <= 1

    def test_detect_stage_returns_raw(self, tmp_path):
        """Verify detect_stage always returns 'raw'."""
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame({"Identifier": []}).to_csv(csv_path, index=False)
        layout = MARISMaLayout(tmp_path, csv_path)
        assert layout.detect_stage() == "raw"

    def test_collect_overlapping_prefix(self, tmp_path):
        """Metadata paths with leading segment duplicating root_dir.name."""
        root = tmp_path / "MARISMa"
        _make_bruker_dir(root, "2024/Staphylococcus/specimen_A")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A"],
                "Path": ["/MARISMa/2024/Staphylococcus/specimen_A"],
                "target_position": ["0_A1"],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(root, csv_path)
        files = layout.collect_spectrum_files(None, None)
        assert len(files) == 1

    def test_collect_no_overlap(self, tmp_path):
        """Metadata paths without overlapping prefix still work."""
        root = tmp_path / "data"
        _make_bruker_dir(root, "specimen_A")
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame(
            {
                "Identifier": ["A"],
                "Path": ["specimen_A"],
                "target_position": ["0_A1"],
            }
        ).to_csv(csv_path, index=False)
        layout = MARISMaLayout(root, csv_path)
        files = layout.collect_spectrum_files(None, None)
        assert len(files) == 1
