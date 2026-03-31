"""Tests for the DRIAMS dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.builder import build_driams_dataset
from maldiamrkit.loader import (
    _detect_id_column,
    _detect_stage,
    _discover_metadata,
    load_driams_dataset,
)

# Import shared helper
from tests.conftest import _generate_synthetic_spectrum


@pytest.fixture()
def synthetic_spectra_dir(tmp_path: Path) -> Path:
    """Create a temp directory with 5 synthetic spectrum .txt files."""
    spectra_dir = tmp_path / "spectra"
    spectra_dir.mkdir()
    for i in range(5):
        df = _generate_synthetic_spectrum(random_state=42 + i)
        out = spectra_dir / f"sample_{i}.txt"
        np.savetxt(
            out,
            df[["mass", "intensity"]].values,
            header="mass intensity",
            comments="# ",
            fmt="%.6f",
        )
    return spectra_dir


@pytest.fixture()
def synthetic_metadata(tmp_path: Path) -> Path:
    """Create a metadata CSV matching the 5 synthetic spectra."""
    meta = pd.DataFrame(
        {
            "ID": [f"sample_{i}" for i in range(5)],
            "Species": [
                "Escherichia coli",
                "Escherichia coli",
                "Klebsiella pneumoniae",
                "Klebsiella pneumoniae",
                "Escherichia coli",
            ],
            "Ceftriaxone": ["S", "R", "S", "R", "S"],
        }
    )
    path = tmp_path / "metadata.csv"
    meta.to_csv(path, index=False)
    return path


@pytest.fixture()
def synthetic_metadata_with_years(tmp_path: Path) -> Path:
    """Metadata with acquisition_date spanning 2 years."""
    meta = pd.DataFrame(
        {
            "ID": [f"sample_{i}" for i in range(5)],
            "Species": [
                "Escherichia coli",
                "Escherichia coli",
                "Klebsiella pneumoniae",
                "Klebsiella pneumoniae",
                "Escherichia coli",
            ],
            "Ceftriaxone": ["S", "R", "S", "R", "S"],
            "acquisition_date": [
                "2015-01-10",
                "2015-06-15",
                "2016-03-20",
                "2016-09-01",
                "2015-12-25",
            ],
        }
    )
    path = tmp_path / "metadata_years.csv"
    meta.to_csv(path, index=False)
    return path


@pytest.fixture()
def built_dataset(tmp_path, synthetic_spectra_dir, synthetic_metadata) -> Path:
    """Build a DRIAMS dataset and return the output directory."""
    out = tmp_path / "driams_out"
    build_driams_dataset(synthetic_spectra_dir, synthetic_metadata, out, n_jobs=1)
    return out


@pytest.fixture()
def built_dataset_with_years(
    tmp_path, synthetic_spectra_dir, synthetic_metadata_with_years
) -> Path:
    """Build a DRIAMS dataset with year subfolders."""
    out = tmp_path / "driams_years"
    build_driams_dataset(
        synthetic_spectra_dir,
        synthetic_metadata_with_years,
        out,
        year_column="acquisition_date",
        n_jobs=1,
    )
    return out


class TestDetectStage:
    """Tests for auto-detecting the processing stage."""

    def test_prefers_binned(self, built_dataset):
        stage = _detect_stage(built_dataset)
        assert stage.startswith("binned_")

    def test_falls_back_to_preprocessed(self, tmp_path):
        (tmp_path / "preprocessed").mkdir()
        assert _detect_stage(tmp_path) == "preprocessed"

    def test_falls_back_to_raw(self, tmp_path):
        (tmp_path / "raw").mkdir()
        assert _detect_stage(tmp_path) == "raw"

    def test_raises_when_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No recognised"):
            _detect_stage(tmp_path)


class TestDetectIdColumn:
    """Tests for auto-detecting the ID column."""

    def test_detects_code(self):
        meta = pd.DataFrame({"code": ["a"], "Species": ["E. coli"]})
        assert _detect_id_column(meta) == "code"

    def test_detects_id(self):
        meta = pd.DataFrame({"ID": ["a"], "Species": ["E. coli"]})
        assert _detect_id_column(meta) == "ID"

    def test_code_takes_priority(self):
        meta = pd.DataFrame({"code": ["a"], "ID": ["a"]})
        assert _detect_id_column(meta) == "code"

    def test_falls_back_to_first_column(self):
        meta = pd.DataFrame({"specimen_id": ["a"], "Species": ["E. coli"]})
        assert _detect_id_column(meta) == "specimen_id"


class TestDiscoverMetadata:
    """Tests for metadata CSV discovery."""

    def test_flat_layout(self, built_dataset):
        id_dir = built_dataset / "id"
        meta, years = _discover_metadata(id_dir, year=None)
        assert len(meta) == 5
        assert years == []

    def test_year_layout(self, built_dataset_with_years):
        id_dir = built_dataset_with_years / "id"
        meta, years = _discover_metadata(id_dir, year=None)
        assert len(meta) == 5
        assert set(years) == {"2015", "2016"}

    def test_specific_year(self, built_dataset_with_years):
        id_dir = built_dataset_with_years / "id"
        meta, years = _discover_metadata(id_dir, year=2015)
        assert all(meta["code"].astype(str).isin({"sample_0", "sample_1", "sample_4"}))
        assert years == ["2015"]

    def test_missing_year_raises(self, built_dataset_with_years):
        id_dir = built_dataset_with_years / "id"
        with pytest.raises(FileNotFoundError):
            _discover_metadata(id_dir, year=9999)

    def test_empty_id_dir_raises(self, tmp_path):
        id_dir = tmp_path / "id"
        id_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No metadata"):
            _discover_metadata(id_dir, year=None)


class TestLoadDriamsDataset:
    """End-to-end tests: build -> load -> verify."""

    def test_round_trip_flat(self, built_dataset):
        ds = load_driams_dataset(built_dataset, n_jobs=1)
        assert len(ds.spectra) == 5
        assert "Ceftriaxone" in ds.meta.columns

    def test_round_trip_with_years(self, built_dataset_with_years):
        ds = load_driams_dataset(built_dataset_with_years, n_jobs=1)
        assert len(ds.spectra) == 5

    def test_load_specific_year(self, built_dataset_with_years):
        ds = load_driams_dataset(built_dataset_with_years, year=2015, n_jobs=1)
        assert len(ds.spectra) == 3

    def test_load_raw_stage(self, built_dataset):
        ds = load_driams_dataset(built_dataset, stage="raw", n_jobs=1)
        assert len(ds.spectra) == 5

    def test_load_preprocessed_stage(self, built_dataset):
        ds = load_driams_dataset(built_dataset, stage="preprocessed", n_jobs=1)
        assert len(ds.spectra) == 5

    def test_explicit_id_column(self, built_dataset):
        ds = load_driams_dataset(built_dataset, id_column="code", n_jobs=1)
        assert len(ds.spectra) == 5

    def test_aggregate_by(self, built_dataset):
        ds = load_driams_dataset(
            built_dataset,
            aggregate_by=dict(antibiotics="Ceftriaxone"),
            n_jobs=1,
        )
        assert ds.antibiotics == ["Ceftriaxone"]

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_driams_dataset(tmp_path / "does_not_exist")

    def test_nonexistent_stage_raises(self, built_dataset):
        with pytest.raises(FileNotFoundError, match="Stage folder"):
            load_driams_dataset(built_dataset, stage="nonexistent")

    def test_missing_id_dir_raises(self, tmp_path):
        (tmp_path / "raw").mkdir()
        with pytest.raises(FileNotFoundError, match="id/"):
            load_driams_dataset(tmp_path, stage="raw")
