"""Tests for the dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.data import (
    DatasetBuilder,
    DatasetLoader,
    DRIAMSLayout,
    FlatLayout,
)
from maldiamrkit.data.dataset_layouts import (
    _detect_id_column,
    _discover_driams_metadata,
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
    """Build a dataset and return the output directory."""
    out = tmp_path / "driams_out"
    DatasetBuilder(
        FlatLayout(synthetic_spectra_dir, synthetic_metadata),
        out,
        n_jobs=1,
    ).build()
    return out


@pytest.fixture()
def built_dataset_with_years(
    tmp_path, synthetic_spectra_dir, synthetic_metadata_with_years
) -> Path:
    """Build a dataset with year subfolders."""
    out = tmp_path / "driams_years"
    DatasetBuilder(
        FlatLayout(
            synthetic_spectra_dir,
            synthetic_metadata_with_years,
            year_column="acquisition_date",
        ),
        out,
        n_jobs=1,
    ).build()
    return out


class TestDetectStage:
    """Tests for auto-detecting the processing stage."""

    def test_prefers_binned(self, built_dataset):
        layout = DRIAMSLayout(built_dataset)
        stage = layout.detect_stage()
        assert stage.startswith("binned_")

    def test_falls_back_to_preprocessed(self, tmp_path):
        (tmp_path / "preprocessed").mkdir()
        layout = DRIAMSLayout(tmp_path)
        assert layout.detect_stage() == "preprocessed"

    def test_falls_back_to_raw(self, tmp_path):
        (tmp_path / "raw").mkdir()
        layout = DRIAMSLayout(tmp_path)
        assert layout.detect_stage() == "raw"

    def test_raises_when_empty(self, tmp_path):
        layout = DRIAMSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="No recognised"):
            layout.detect_stage()


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
        meta, years = _discover_driams_metadata(id_dir, year=None)
        assert len(meta) == 5
        assert years == []

    def test_year_layout(self, built_dataset_with_years):
        id_dir = built_dataset_with_years / "id"
        meta, years = _discover_driams_metadata(id_dir, year=None)
        assert len(meta) == 5
        assert set(years) == {"2015", "2016"}

    def test_specific_year(self, built_dataset_with_years):
        id_dir = built_dataset_with_years / "id"
        meta, years = _discover_driams_metadata(id_dir, year="2015")
        assert all(meta["code"].astype(str).isin({"sample_0", "sample_1", "sample_4"}))
        assert years == ["2015"]

    def test_missing_year_raises(self, built_dataset_with_years):
        id_dir = built_dataset_with_years / "id"
        with pytest.raises(FileNotFoundError):
            _discover_driams_metadata(id_dir, year="9999")

    def test_empty_id_dir_raises(self, tmp_path):
        id_dir = tmp_path / "id"
        id_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No metadata"):
            _discover_driams_metadata(id_dir, year=None)


class TestDatasetLoader:
    """End-to-end tests: build -> load -> verify."""

    def test_round_trip_flat(self, built_dataset):
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1).load()
        assert len(ds.spectra) == 5
        assert "Ceftriaxone" in ds.meta.columns

    def test_round_trip_with_years(self, built_dataset_with_years):
        ds = DatasetLoader(DRIAMSLayout(built_dataset_with_years), n_jobs=1).load()
        assert len(ds.spectra) == 5

    def test_load_specific_year(self, built_dataset_with_years):
        ds = DatasetLoader(
            DRIAMSLayout(built_dataset_with_years, year=2015), n_jobs=1
        ).load()
        assert len(ds.spectra) == 3

    def test_load_raw_stage(self, built_dataset):
        ds = DatasetLoader(DRIAMSLayout(built_dataset), stage="raw", n_jobs=1).load()
        assert len(ds.spectra) == 5

    def test_load_preprocessed_stage(self, built_dataset):
        ds = DatasetLoader(
            DRIAMSLayout(built_dataset), stage="preprocessed", n_jobs=1
        ).load()
        assert len(ds.spectra) == 5

    def test_explicit_id_column(self, built_dataset):
        ds = DatasetLoader(
            DRIAMSLayout(built_dataset, id_column="code"), n_jobs=1
        ).load()
        assert len(ds.spectra) == 5

    def test_aggregate_by(self, built_dataset):
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1).load(
            aggregate_by=dict(antibiotics="Ceftriaxone"),
        )
        assert ds.antibiotics == ["Ceftriaxone"]

    def test_nonexistent_stage_raises(self, built_dataset):
        with pytest.raises(FileNotFoundError, match="Stage folder"):
            DatasetLoader(
                DRIAMSLayout(built_dataset), stage="nonexistent", n_jobs=1
            ).load()

    def test_missing_id_dir_raises(self, tmp_path):
        (tmp_path / "raw").mkdir()
        with pytest.raises(FileNotFoundError, match="id/"):
            DatasetLoader(DRIAMSLayout(tmp_path), stage="raw", n_jobs=1).load()

    def test_no_spectrum_files_raises(self, tmp_path):
        """Verify FileNotFoundError when stage dir has no spectrum files."""
        # Create structure with metadata but empty stage dir
        id_dir = tmp_path / "id"
        id_dir.mkdir()
        pd.DataFrame({"ID": ["s1"]}).to_csv(id_dir / "data_clean.csv", index=False)
        (tmp_path / "raw").mkdir()
        with pytest.raises(FileNotFoundError, match="No spectrum files"):
            DatasetLoader(DRIAMSLayout(tmp_path), stage="raw", n_jobs=1).load()

    def test_no_matched_ids_raises(self, tmp_path):
        """Verify ValueError when no spectrum files match metadata IDs."""
        id_dir = tmp_path / "id"
        id_dir.mkdir()
        pd.DataFrame({"ID": ["no_match"]}).to_csv(
            id_dir / "data_clean.csv", index=False
        )
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "different_name.txt").write_text("1000 100\n2000 200\n")
        with pytest.raises(ValueError, match="No spectrum files matched"):
            DatasetLoader(DRIAMSLayout(tmp_path), stage="raw", n_jobs=1).load()

    def test_partial_match_logs_info(self, tmp_path):
        """Verify info log when some spectra not in metadata."""
        import numpy as np

        from tests.conftest import _generate_synthetic_spectrum

        id_dir = tmp_path / "id"
        id_dir.mkdir()
        pd.DataFrame({"ID": ["s1"]}).to_csv(id_dir / "data_clean.csv", index=False)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # Create 2 spectra but only 1 in metadata
        for name in ("s1", "s2"):
            df = _generate_synthetic_spectrum(random_state=42)
            np.savetxt(
                raw_dir / f"{name}.txt",
                df[["mass", "intensity"]].values,
                fmt="%.6f",
            )
        loader = DatasetLoader(DRIAMSLayout(tmp_path), stage="raw", n_jobs=1)
        ds = loader.load()
        assert len(ds.spectra) == 1


class TestPrefilterMetadata:
    """Tests for metadata pre-filtering in DatasetLoader.load()."""

    def test_prefilter_by_species(self, built_dataset):
        """Pre-filter keeps only spectra matching the requested species."""
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1).load(
            aggregate_by=dict(species="Escherichia coli"),
        )
        # Fixture has 3 E. coli and 2 K. pneumoniae
        assert len(ds.spectra) == 3
        assert all(ds.meta["Species"] == "Escherichia coli")

    def test_prefilter_by_antibiotics(self, built_dataset):
        """Pre-filter keeps rows where antibiotic column is not null."""
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1).load(
            aggregate_by=dict(antibiotics="Ceftriaxone"),
        )
        # All 5 samples have non-null Ceftriaxone in fixture
        assert len(ds.spectra) == 5
        assert ds.antibiotics == ["Ceftriaxone"]

    def test_prefilter_combined(self, built_dataset):
        """Pre-filter by both species and antibiotics together."""
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1).load(
            aggregate_by=dict(
                species="Klebsiella pneumoniae", antibiotics="Ceftriaxone"
            ),
        )
        assert len(ds.spectra) == 2

    def test_prefilter_no_aggregate_by(self, built_dataset):
        """Without aggregate_by, all spectra are loaded."""
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1).load()
        assert len(ds.spectra) == 5

    def test_verbose_flag(self, built_dataset):
        """verbose=True is passed through to MaldiSet."""
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1, verbose=True).load()
        assert ds.verbose is True

    def test_verbose_n_jobs_1_tqdm(self, built_dataset):
        """verbose=True with n_jobs=1 uses tqdm loop without error."""
        ds = DatasetLoader(DRIAMSLayout(built_dataset), n_jobs=1, verbose=True).load()
        assert len(ds.spectra) == 5


class TestAverageReplicates:
    """Tests for DatasetLoader._average_replicates."""

    def test_groups_averaged_correctly(self):
        """Verify replicate groups are averaged and metadata is cleaned."""
        from maldiamrkit import MaldiSpectrum
        from maldiamrkit.data.loader import DatasetLoader

        mz = np.linspace(2000, 20000, 100)
        s0 = MaldiSpectrum(pd.DataFrame({"mass": mz, "intensity": np.ones(100) * 10}))
        s1 = MaldiSpectrum(pd.DataFrame({"mass": mz, "intensity": np.ones(100) * 20}))
        s2 = MaldiSpectrum(pd.DataFrame({"mass": mz, "intensity": np.ones(100) * 30}))

        meta = pd.DataFrame(
            {
                "ID": ["s0_rep1", "s0_rep2", "s1_rep1"],
                "Species": ["E. coli", "E. coli", "K. pneumoniae"],
                "_original_id": ["s0", "s0", "s1"],
            }
        )

        spectra = [s0, s1, s2]
        avg_specs, avg_meta = DatasetLoader._average_replicates(spectra, meta)

        assert len(avg_specs) == 2
        assert len(avg_meta) == 2
        assert "_original_id" not in avg_meta.columns
        assert set(avg_meta["ID"]) == {"s0", "s1"}

    def test_single_member_group_passthrough(self):
        """Verify single-member groups pass through without averaging."""
        from maldiamrkit import MaldiSpectrum
        from maldiamrkit.data.loader import DatasetLoader

        mz = np.linspace(2000, 20000, 100)
        s0 = MaldiSpectrum(pd.DataFrame({"mass": mz, "intensity": np.ones(100) * 10}))

        meta = pd.DataFrame(
            {
                "ID": ["s0_rep1"],
                "Species": ["E. coli"],
                "_original_id": ["s0"],
            }
        )

        avg_specs, avg_meta = DatasetLoader._average_replicates([s0], meta)
        assert len(avg_specs) == 1
        assert avg_meta["ID"].iloc[0] == "s0"
