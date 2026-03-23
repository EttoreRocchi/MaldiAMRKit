"""Tests for the DRIAMS-like dataset builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.builder import (
    BuildReport,
    ProcessingHandler,
    _extract_year,
    build_driams_dataset,
)
from maldiamrkit.preprocessing import PreprocessingPipeline
from maldiamrkit.preprocessing.transformers import (
    ClipNegatives,
    MzTrimmer,
    SqrtTransform,
    TICNormalizer,
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


class TestDirectoryStructure:
    """Tests for output directory layout."""

    def test_creates_directory_structure(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        report = build_driams_dataset(
            synthetic_spectra_dir, synthetic_metadata, out, n_jobs=1
        )
        assert (out / "raw").is_dir()
        assert (out / "preprocessed").is_dir()
        # Default bin_width=3, mz range 2000-20000 -> 6001 edges -> 6000 bins
        assert (out / "binned_6000").is_dir()
        assert (out / "id").is_dir()
        assert isinstance(report, BuildReport)

    def test_dynamic_binned_folder_name(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        report = build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            bin_width=6,
            n_jobs=1,
        )
        assert (out / "binned_3000").is_dir()
        assert "binned_3000" in report.folders_created

    def test_custom_folder_names_via_extra_handlers(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            extra_handlers=[
                ProcessingHandler("preprocessed_sqrt", "preprocessed"),
            ],
            n_jobs=1,
        )
        assert (out / "preprocessed_sqrt").is_dir()
        # Default folders should still exist
        assert (out / "preprocessed").is_dir()


class TestOutputFormats:
    """Tests for spectrum file content and format."""

    def test_raw_files_format(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        build_driams_dataset(synthetic_spectra_dir, synthetic_metadata, out, n_jobs=1)
        raw_files = list((out / "raw").glob("*.txt"))
        assert len(raw_files) == 5
        # Check content is space-separated with header
        content = raw_files[0].read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "# mass intensity"
        # Data lines should have 2 space-separated floats
        parts = lines[1].split()
        assert len(parts) == 2
        float(parts[0])
        float(parts[1])

    def test_preprocessed_files_format(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        build_driams_dataset(synthetic_spectra_dir, synthetic_metadata, out, n_jobs=1)
        pp_files = list((out / "preprocessed").glob("*.txt"))
        assert len(pp_files) == 5
        content = pp_files[0].read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "# mass intensity"

    def test_binned_format(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        out = tmp_path / "output"
        build_driams_dataset(synthetic_spectra_dir, synthetic_metadata, out, n_jobs=1)
        binned_files = list((out / "binned_6000").glob("*.txt"))
        assert len(binned_files) == 5
        content = binned_files[0].read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "bin_index binned_intensity"
        # Should have ~6000 data lines
        n_data = len(lines) - 1
        assert 5990 <= n_data <= 6010
        # First data line: integer index + float
        parts = lines[1].split()
        assert int(parts[0]) == 0
        float(parts[1])


class TestMetadata:
    """Tests for metadata output."""

    def test_metadata_output(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            name="test",
            n_jobs=1,
        )
        meta_path = out / "id" / "test_clean.csv"
        assert meta_path.exists()
        df = pd.read_csv(meta_path)
        assert len(df) == 5
        assert "code" in df.columns
        assert "ID" not in df.columns

    def test_id_column_configurable(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            id_column="sample_id",
            name="test",
            n_jobs=1,
        )
        df = pd.read_csv(out / "id" / "test_clean.csv")
        assert "sample_id" in df.columns
        assert "ID" not in df.columns
        assert "code" not in df.columns


class TestYearStructure:
    """Tests for year-based subfolder splitting."""

    def test_year_subfolders(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata_with_years
    ):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata_with_years,
            out,
            year_column="acquisition_date",
            n_jobs=1,
        )
        assert (out / "raw" / "2015").is_dir()
        assert (out / "raw" / "2016").is_dir()
        assert (out / "preprocessed" / "2015").is_dir()
        assert (out / "preprocessed" / "2016").is_dir()
        # Count files per year
        raw_2015 = list((out / "raw" / "2015").glob("*.txt"))
        raw_2016 = list((out / "raw" / "2016").glob("*.txt"))
        assert len(raw_2015) == 3  # samples 0, 1, 4
        assert len(raw_2016) == 2  # samples 2, 3

    def test_year_metadata_split(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata_with_years
    ):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata_with_years,
            out,
            year_column="acquisition_date",
            n_jobs=1,
        )
        meta_2015 = pd.read_csv(out / "id" / "2015" / "2015_clean.csv")
        meta_2016 = pd.read_csv(out / "id" / "2016" / "2016_clean.csv")
        assert len(meta_2015) == 3
        assert len(meta_2016) == 2
        assert "code" in meta_2015.columns

    def test_no_year_flat(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            name="test",
            n_jobs=1,
        )
        # No year subfolders - files directly in raw/
        raw_files = list((out / "raw").glob("*.txt"))
        assert len(raw_files) == 5
        # No year directories
        subdirs = [p for p in (out / "raw").iterdir() if p.is_dir()]
        assert len(subdirs) == 0


class TestExtraHandlers:
    """Tests for additional processing handlers."""

    def test_extra_handler_preprocessed(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            extra_handlers=[
                ProcessingHandler("preprocessed_sqrt", "preprocessed"),
            ],
            n_jobs=1,
        )
        pp_files = list((out / "preprocessed_sqrt").glob("*.txt"))
        assert len(pp_files) == 5

    def test_extra_handler_binned(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            extra_handlers=[
                ProcessingHandler("binned_3000", "binned", bin_width=6),
            ],
            n_jobs=1,
        )
        binned_files = list((out / "binned_3000").glob("*.txt"))
        assert len(binned_files) == 5
        # Check bin count is different from default
        content = binned_files[0].read_text()
        lines = content.strip().split("\n")
        n_data = len(lines) - 1  # exclude header
        assert 2990 <= n_data <= 3010  # ~3000 bins for 6Da width

    def test_extra_handlers_with_years(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata_with_years
    ):
        out = tmp_path / "output"
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata_with_years,
            out,
            year_column="acquisition_date",
            extra_handlers=[
                ProcessingHandler("preprocessed_alt", "preprocessed"),
            ],
            n_jobs=1,
        )
        assert (out / "preprocessed_alt" / "2015").is_dir()
        assert (out / "preprocessed_alt" / "2016").is_dir()
        alt_2015 = list((out / "preprocessed_alt" / "2015").glob("*.txt"))
        assert len(alt_2015) == 3

    def test_extra_handler_custom_pipeline(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        # A minimal pipeline with no smoothing
        minimal_pipe = PreprocessingPipeline(
            [
                ("clip", ClipNegatives()),
                ("sqrt", SqrtTransform()),
                ("trim", MzTrimmer(mz_min=2000, mz_max=20000)),
                ("norm", TICNormalizer()),
            ]
        )
        build_driams_dataset(
            synthetic_spectra_dir,
            synthetic_metadata,
            out,
            extra_handlers=[
                ProcessingHandler(
                    "preprocessed_minimal",
                    "preprocessed",
                    pipeline=minimal_pipe,
                ),
            ],
            n_jobs=1,
        )
        # Both default and custom preprocessed should exist
        default_files = list((out / "preprocessed").glob("*.txt"))
        custom_files = list((out / "preprocessed_minimal").glob("*.txt"))
        assert len(default_files) == 5
        assert len(custom_files) == 5
        # Content should differ (different pipeline)
        default_content = default_files[0].read_text()
        # Find matching file in custom
        custom_file = out / "preprocessed_minimal" / default_files[0].name
        custom_content = custom_file.read_text()
        assert default_content != custom_content


class TestValidation:
    """Tests for input validation and error handling."""

    def test_id_mismatch_warns(self, tmp_path, synthetic_spectra_dir, caplog):
        # Metadata with extra IDs not matching spectra
        meta = pd.DataFrame(
            {
                "ID": [f"sample_{i}" for i in range(8)],  # 5 match, 3 don't
                "Species": ["E. coli"] * 8,
            }
        )
        meta_path = tmp_path / "meta.csv"
        meta.to_csv(meta_path, index=False)
        out = tmp_path / "output"
        with caplog.at_level("WARNING"):
            report = build_driams_dataset(
                synthetic_spectra_dir, meta_path, out, n_jobs=1
            )
        assert report.total == 5
        assert report.succeeded == 5

    def test_missing_id_column_raises(self, tmp_path, synthetic_spectra_dir):
        meta = pd.DataFrame({"Name": ["a"], "Species": ["E. coli"]})
        meta_path = tmp_path / "meta.csv"
        meta.to_csv(meta_path, index=False)
        with pytest.raises(ValueError, match="ID"):
            build_driams_dataset(
                synthetic_spectra_dir, meta_path, tmp_path / "out", n_jobs=1
            )

    def test_empty_dir_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        meta = pd.DataFrame({"ID": ["a"], "Species": ["E. coli"]})
        meta_path = tmp_path / "meta.csv"
        meta.to_csv(meta_path, index=False)
        with pytest.raises(ValueError, match="No .txt"):
            build_driams_dataset(empty_dir, meta_path, tmp_path / "out", n_jobs=1)

    def test_duplicate_folder_raises(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        with pytest.raises(ValueError, match="Duplicate folder"):
            build_driams_dataset(
                synthetic_spectra_dir,
                synthetic_metadata,
                tmp_path / "out",
                extra_handlers=[
                    ProcessingHandler("preprocessed", "preprocessed"),
                ],
                n_jobs=1,
            )

    def test_corrupt_spectrum_skipped(self, tmp_path, synthetic_metadata):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        # Create 4 good spectra
        for i in range(4):
            df = _generate_synthetic_spectrum(random_state=42 + i)
            np.savetxt(
                spectra_dir / f"sample_{i}.txt",
                df[["mass", "intensity"]].values,
                header="mass intensity",
                comments="# ",
                fmt="%.6f",
            )
        # Create 1 corrupt spectrum
        (spectra_dir / "sample_4.txt").write_text("this is not a spectrum\n!@#$%\n")

        out = tmp_path / "output"
        report = build_driams_dataset(spectra_dir, synthetic_metadata, out, n_jobs=1)
        assert report.succeeded == 4
        assert report.failed == 1
        assert "sample_4" in report.failed_ids


class TestBuildReport:
    """Tests for BuildReport contents."""

    def test_report_fields(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        out = tmp_path / "output"
        report = build_driams_dataset(
            synthetic_spectra_dir, synthetic_metadata, out, n_jobs=1
        )
        assert report.total == 5
        assert report.succeeded == 5
        assert report.failed == 0
        assert report.failed_ids == []
        assert report.output_dir == out
        assert "raw" in report.folders_created
        assert "preprocessed" in report.folders_created
        assert "binned_6000" in report.folders_created


class TestProcessingHandlerSerialization:
    """Tests for ProcessingHandler to_dict / from_dict."""

    def test_roundtrip_no_pipeline(self):
        handler = ProcessingHandler("binned_3000", "binned", bin_width=6)
        d = handler.to_dict()
        restored = ProcessingHandler.from_dict(d)
        assert restored.folder_name == handler.folder_name
        assert restored.kind == handler.kind
        assert restored.pipeline is None
        assert restored.bin_width == handler.bin_width

    def test_roundtrip_with_pipeline(self):
        pipe = PreprocessingPipeline.default()
        handler = ProcessingHandler("pp_custom", "preprocessed", pipeline=pipe)
        d = handler.to_dict()
        restored = ProcessingHandler.from_dict(d)
        assert restored.folder_name == "pp_custom"
        assert restored.kind == "preprocessed"
        assert len(restored.pipeline) == len(pipe)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="Invalid kind"):
            ProcessingHandler("folder", "invalid_kind")


class TestExtractYear:
    """Tests for the year extraction helper."""

    def test_date_string(self):
        assert _extract_year("2015-01-12") == "2015"

    def test_integer(self):
        assert _extract_year(2016) == "2016"

    def test_float(self):
        assert _extract_year(2017.0) == "2017"

    def test_plain_year_string(self):
        assert _extract_year("2018") == "2018"

    def test_timestamp(self):
        ts = pd.Timestamp("2015-06-15")
        assert _extract_year(ts) == "2015"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot extract year"):
            _extract_year("not-a-date")
