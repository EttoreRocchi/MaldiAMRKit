"""Tests for input layout adapters (FlatLayout, BrukerTreeLayout, _extract_year)."""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from maldiamrkit.data.input_layouts import (
    BrukerTreeLayout,
    FlatLayout,
    _extract_year,
)


def _make_bruker_dir(
    root: Path,
    rel_path: str,
    *,
    fid_content: bytes | None = None,
) -> Path:
    """Create a minimal Bruker directory tree with acqus + fid files.

    Structure: root / rel_path / 1 / 1SLin / acqus + fid
    Returns the *outer* directory (root / rel_path).
    """
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

    if fid_content is None:
        # 8 non-zero int32 values
        fid_content = struct.pack("<8i", *range(1, 9))
    (bruker_dir / "fid").write_bytes(fid_content)

    return root / rel_path


def _make_bruker_metadata(
    csv_path: Path,
    entries: list[dict],
    *,
    id_column: str = "Identifier",
) -> None:
    """Write a Bruker-compatible metadata CSV."""
    pd.DataFrame(entries).to_csv(csv_path, index=False)


class TestExtractYear:
    """Tests for the _extract_year helper function."""

    def test_datetime_object_returns_year_string(self):
        """Verify datetime objects are converted to year strings."""
        assert _extract_year(datetime(2024, 6, 15)) == "2024"

    def test_int_returns_string(self):
        """Verify integer years are converted to strings."""
        assert _extract_year(2024) == "2024"

    def test_float_returns_truncated_string(self):
        """Verify float years are truncated and converted."""
        assert _extract_year(2024.0) == "2024"

    def test_date_string_dash_separator(self):
        """Verify dash-separated date strings extract the year."""
        assert _extract_year("2024-01-15") == "2024"

    def test_date_string_slash_separator(self):
        """Verify slash-separated date strings extract the year."""
        assert _extract_year("2024/03/01") == "2024"

    def test_invalid_string_raises(self):
        """Verify non-parsable strings raise ValueError."""
        with pytest.raises(ValueError, match="Cannot extract year"):
            _extract_year("abc")

    def test_short_digit_string_raises(self):
        """Verify non-4-digit numeric strings raise ValueError."""
        with pytest.raises(ValueError, match="Cannot extract year"):
            _extract_year("24")


class TestFlatLayout:
    """Tests for FlatLayout discovery and ID/year mapping."""

    def test_discover_spectra_no_files_raises(self, tmp_path):
        """Verify empty directory raises ValueError."""
        spectra_dir = tmp_path / "empty"
        spectra_dir.mkdir()
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["a"]}).to_csv(meta_csv, index=False)
        layout = FlatLayout(spectra_dir, meta_csv)
        with pytest.raises(ValueError, match="No .txt spectrum files"):
            layout.discover_spectra()

    def test_discover_spectra_year_subfolders(self, tmp_path):
        """Verify fallback glob finds files in year subfolders."""
        spectra_dir = tmp_path / "spectra"
        year_dir = spectra_dir / "2024"
        year_dir.mkdir(parents=True)
        (year_dir / "s1.txt").write_text("1000 100\n")
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["s1"]}).to_csv(meta_csv, index=False)
        layout = FlatLayout(spectra_dir, meta_csv)
        found = layout.discover_spectra()
        assert len(found) == 1
        assert found[0].name == "s1.txt"

    def test_discover_metadata_missing_id_col_raises(self, tmp_path):
        """Verify missing ID column raises ValueError."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "s1.txt").write_text("1000 100\n")
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"Name": ["s1"]}).to_csv(meta_csv, index=False)
        layout = FlatLayout(spectra_dir, meta_csv)
        with pytest.raises(ValueError, match="ID column 'ID' not found"):
            layout.discover_metadata()

    def test_discover_metadata_renames_id_column(self, tmp_path):
        """Verify non-default id_column is renamed to 'ID'."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "s1.txt").write_text("1000 100\n")
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"code": ["s1"], "Species": ["E. coli"]}).to_csv(
            meta_csv, index=False
        )
        layout = FlatLayout(spectra_dir, meta_csv, id_column="code")
        meta = layout.discover_metadata()
        assert "ID" in meta.columns
        assert meta["ID"].iloc[0] == "s1"

    def test_discover_metadata_year_column_missing_raises(self, tmp_path):
        """Verify missing year_column raises ValueError."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "s1.txt").write_text("1000 100\n")
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["s1"]}).to_csv(meta_csv, index=False)
        layout = FlatLayout(spectra_dir, meta_csv, year_column="Year")
        with pytest.raises(ValueError, match="year_column 'Year' not found"):
            layout.discover_metadata()

    def test_discover_metadata_builds_year_map(self, tmp_path):
        """Verify year_column populates the internal year map."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "s1.txt").write_text("1000 100\n")
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["s1"], "Year": [2024]}).to_csv(meta_csv, index=False)
        layout = FlatLayout(spectra_dir, meta_csv, year_column="Year")
        layout.discover_metadata()
        assert layout.get_year("s1") == "2024"

    def test_get_year_no_map_returns_none(self, tmp_path):
        """Verify get_year returns None when no year_column is set."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "s1.txt").write_text("1000 100\n")
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["s1"]}).to_csv(meta_csv, index=False)
        layout = FlatLayout(spectra_dir, meta_csv)
        assert layout.get_year("s1") is None

    def test_get_year_returns_mapped_value(self, tmp_path):
        """Verify get_year returns the correct mapped year."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "s1.txt").write_text("1000 100\n")
        meta_csv = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["s1"], "Year": [2023]}).to_csv(meta_csv, index=False)
        layout = FlatLayout(spectra_dir, meta_csv, year_column="Year")
        layout.discover_metadata()
        assert layout.get_year("s1") == "2023"

    def test_get_id_returns_stem(self, tmp_path):
        """Verify get_id returns the file stem."""
        layout = FlatLayout(tmp_path, tmp_path / "meta.csv")
        assert layout.get_id(Path("/some/dir/spectrum_1.txt")) == "spectrum_1"


class TestBrukerTreeLayout:
    """Tests for BrukerTreeLayout discovery and validation."""

    def test_init_stores_params(self, tmp_path):
        """Verify constructor stores all parameters."""
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame({"Identifier": []}).to_csv(csv_path, index=False)
        layout = BrukerTreeLayout(
            tmp_path,
            csv_path,
            id_column="Identifier",
            year_column="Year",
            path_column="Path",
            target_position_column="target_position",
            duplicate_strategy="keep_all",
            validate=False,
        )
        assert layout.root_dir == tmp_path
        assert layout.duplicate_strategy.value == "keep_all"
        assert layout.validate is False
        assert layout._year_map == {}
        assert layout._id_to_path == {}

    def test_discover_spectra_strategy_first(self, tmp_path):
        """Verify duplicate_strategy='first' keeps first spectrum per ID."""
        root = tmp_path / "root"
        _make_bruker_dir(root, "specimen_A/0_A1")
        _make_bruker_dir(
            root, "specimen_A/0_A2", fid_content=struct.pack("<8i", *range(10, 18))
        )
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A/0_A1",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A/0_A2",
                    "target_position": "0_A2",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        paths = layout.discover_spectra()
        assert len(paths) == 1

    def test_discover_spectra_strategy_keep_all(self, tmp_path):
        """Verify duplicate_strategy='keep_all' creates combined IDs."""
        root = tmp_path / "root"
        _make_bruker_dir(root, "specimen_A/0_A1")
        _make_bruker_dir(
            root, "specimen_A/0_A2", fid_content=struct.pack("<8i", *range(10, 18))
        )
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A/0_A1",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A/0_A2",
                    "target_position": "0_A2",
                },
            ],
        )
        layout = BrukerTreeLayout(
            root, csv_path, duplicate_strategy="keep_all", validate=False
        )
        paths = layout.discover_spectra()
        assert len(paths) == 2
        # IDs should be combined
        assert "specimen_A_0_A1" in layout._id_to_path
        assert "specimen_A_0_A2" in layout._id_to_path

    def test_discover_spectra_missing_dir_skipped(self, tmp_path):
        """Verify missing directory is skipped with warning."""
        root = tmp_path / "root"
        root.mkdir()
        csv_path = tmp_path / "meta.csv"
        _make_bruker_dir(root, "valid_specimen")
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "valid",
                    "Year": "2024",
                    "Path": "valid_specimen",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "missing",
                    "Year": "2024",
                    "Path": "nonexistent",
                    "target_position": "0_A2",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        paths = layout.discover_spectra()
        assert len(paths) == 1

    def test_discover_spectra_no_acqus_skipped(self, tmp_path):
        """Verify directory without acqus is skipped."""
        root = tmp_path / "root"
        no_acqus_dir = root / "no_acqus"
        no_acqus_dir.mkdir(parents=True)
        # Create a valid one too
        _make_bruker_dir(root, "valid_specimen")
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "valid",
                    "Year": "2024",
                    "Path": "valid_specimen",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "noacq",
                    "Year": "2024",
                    "Path": "no_acqus",
                    "target_position": "0_A2",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        paths = layout.discover_spectra()
        assert len(paths) == 1

    def test_discover_spectra_validate_empty_fid_skipped(self, tmp_path):
        """Verify all-zero fid file is skipped when validate=True."""
        root = tmp_path / "root"
        zero_fid = b"\x00" * 32
        _make_bruker_dir(root, "empty_fid_specimen", fid_content=zero_fid)
        _make_bruker_dir(root, "good_specimen")
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "empty",
                    "Year": "2024",
                    "Path": "empty_fid_specimen",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "good",
                    "Year": "2024",
                    "Path": "good_specimen",
                    "target_position": "0_A2",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=True)
        paths = layout.discover_spectra()
        assert len(paths) == 1

    def test_discover_spectra_validate_duplicate_fid_warns(self, tmp_path):
        """Verify duplicate fid hashes produce a warning."""
        root = tmp_path / "root"
        same_content = struct.pack("<8i", *range(1, 9))
        _make_bruker_dir(root, "dup_A", fid_content=same_content)
        _make_bruker_dir(root, "dup_B", fid_content=same_content)
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "dup_A",
                    "Year": "2024",
                    "Path": "dup_A",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "dup_B",
                    "Year": "2024",
                    "Path": "dup_B",
                    "target_position": "0_A2",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=True)
        # Should succeed but log a warning about duplicate
        paths = layout.discover_spectra()
        assert len(paths) == 2

    def test_discover_spectra_no_valid_raises(self, tmp_path):
        """Verify ValueError when no valid Bruker paths exist."""
        root = tmp_path / "root"
        root.mkdir()
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "missing",
                    "Year": "2024",
                    "Path": "nonexistent",
                    "target_position": "0_A1",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        with pytest.raises(ValueError, match="No valid Bruker spectra"):
            layout.discover_spectra()

    def test_discover_spectra_validate_false_skips_checks(self, tmp_path):
        """Verify validate=False skips fid validation."""
        root = tmp_path / "root"
        zero_fid = b"\x00" * 32
        _make_bruker_dir(root, "specimen_A", fid_content=zero_fid)
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A",
                    "target_position": "0_A1",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        paths = layout.discover_spectra()
        assert len(paths) == 1

    def test_discover_metadata_filters_by_id_to_path(self, tmp_path):
        """Verify metadata is filtered to only discovered spectra."""
        root = tmp_path / "root"
        _make_bruker_dir(root, "specimen_A")
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "specimen_B",
                    "Year": "2024",
                    "Path": "nonexistent",
                    "target_position": "0_B1",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        layout.discover_spectra()
        meta = layout.discover_metadata()
        assert len(meta) == 1
        assert meta["ID"].iloc[0] == "specimen_A"

    def test_discover_metadata_strategy_keep_all(self, tmp_path):
        """Verify duplicate_strategy='keep_all' creates combined IDs in metadata."""
        root = tmp_path / "root"
        _make_bruker_dir(root, "specimen_A/0_A1")
        _make_bruker_dir(
            root, "specimen_A/0_A2", fid_content=struct.pack("<8i", *range(10, 18))
        )
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A/0_A1",
                    "target_position": "0_A1",
                },
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A/0_A2",
                    "target_position": "0_A2",
                },
            ],
        )
        layout = BrukerTreeLayout(
            root, csv_path, duplicate_strategy="keep_all", validate=False
        )
        layout.discover_spectra()
        meta = layout.discover_metadata()
        assert "specimen_A_0_A1" in meta["ID"].values

    def test_get_id_from_mapping(self, tmp_path):
        """Verify get_id returns mapped ID when path is known."""
        root = tmp_path / "root"
        bruker_path = _make_bruker_dir(root, "specimen_A")
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A",
                    "target_position": "0_A1",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        layout.discover_spectra()
        assert layout.get_id(bruker_path) == "specimen_A"

    def test_get_id_fallback(self, tmp_path):
        """Verify get_id falls back to grandparent name for unknown paths."""
        root = tmp_path / "root"
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame({"Identifier": []}).to_csv(csv_path, index=False)
        layout = BrukerTreeLayout(root, csv_path)
        unknown_path = Path("/a/b/c/d")
        # parent.parent.parent.name = /a/b/c -> /a/b -> /a -> "a"
        assert layout.get_id(unknown_path) == "a"

    def test_get_year_returns_value(self, tmp_path):
        """Verify get_year returns the mapped year."""
        root = tmp_path / "root"
        _make_bruker_dir(root, "specimen_A")
        csv_path = tmp_path / "meta.csv"
        _make_bruker_metadata(
            csv_path,
            [
                {
                    "Identifier": "specimen_A",
                    "Year": "2024",
                    "Path": "specimen_A",
                    "target_position": "0_A1",
                },
            ],
        )
        layout = BrukerTreeLayout(root, csv_path, validate=False)
        layout.discover_spectra()
        assert layout.get_year("specimen_A") == "2024"

    def test_read_raw_metadata_missing_id_col_raises(self, tmp_path):
        """Verify missing id_column in CSV raises ValueError."""
        root = tmp_path / "root"
        root.mkdir()
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame({"Name": ["a"]}).to_csv(csv_path, index=False)
        layout = BrukerTreeLayout(root, csv_path, id_column="Identifier")
        with pytest.raises(ValueError, match="ID column 'Identifier' not in"):
            layout._read_raw_metadata()

    def test_read_raw_metadata_renames_id_column(self, tmp_path):
        """Verify non-default id_column is renamed to 'ID'."""
        root = tmp_path / "root"
        root.mkdir()
        csv_path = tmp_path / "meta.csv"
        pd.DataFrame({"MyID": ["a"], "Year": [2024]}).to_csv(csv_path, index=False)
        layout = BrukerTreeLayout(root, csv_path, id_column="MyID")
        meta = layout._read_raw_metadata()
        assert "ID" in meta.columns
