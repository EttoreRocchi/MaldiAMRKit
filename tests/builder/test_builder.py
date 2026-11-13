"""Tests for the DRIAMS-like dataset builder."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.data import (
    BrukerTreeLayout,
    BuildReport,
    DatasetBuilder,
    DatasetLoader,
    DRIAMSLayout,
    FlatLayout,
    ProcessingHandler,
    read_site_info,
)
from maldiamrkit.data.input_layouts import _extract_year
from maldiamrkit.io import read_spectrum
from maldiamrkit.preprocessing import PreprocessingPipeline
from maldiamrkit.preprocessing.binning import bin_spectrum
from maldiamrkit.preprocessing.pipeline import preprocess
from maldiamrkit.preprocessing.transformers import (
    ClipNegatives,
    MzTrimmer,
    SqrtTransform,
    TICNormalizer,
)

# Import shared helper
from tests.conftest import DATA_DIR, _generate_synthetic_spectrum


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
        report = DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
        ).build()
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
        report = DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            bin_width=6,
            n_jobs=1,
        ).build()
        assert (out / "binned_3000").is_dir()
        assert "binned_3000" in report.folders_created

    def test_custom_folder_names_via_extra_handlers(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            extra_handlers=[
                ProcessingHandler("preprocessed_sqrt", "preprocessed"),
            ],
            n_jobs=1,
        ).build()
        assert (out / "preprocessed_sqrt").is_dir()
        # Default folders should still exist
        assert (out / "preprocessed").is_dir()


class TestOutputFormats:
    """Tests for spectrum file content and format."""

    def test_raw_files_format(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
        ).build()
        pp_files = list((out / "preprocessed").glob("*.txt"))
        assert len(pp_files) == 5
        content = pp_files[0].read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "# mass intensity"

    def test_binned_format(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        out = tmp_path / "output"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            name="test",
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            id_column="sample_id",
            name="test",
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(
                synthetic_spectra_dir,
                synthetic_metadata_with_years,
                year_column="acquisition_date",
            ),
            out,
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(
                synthetic_spectra_dir,
                synthetic_metadata_with_years,
                year_column="acquisition_date",
            ),
            out,
            n_jobs=1,
        ).build()
        meta_2015 = pd.read_csv(out / "id" / "2015" / "2015_clean.csv")
        meta_2016 = pd.read_csv(out / "id" / "2016" / "2016_clean.csv")
        assert len(meta_2015) == 3
        assert len(meta_2016) == 2
        assert "code" in meta_2015.columns

    def test_no_year_flat(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        out = tmp_path / "output"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            name="test",
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            extra_handlers=[
                ProcessingHandler("preprocessed_sqrt", "preprocessed"),
            ],
            n_jobs=1,
        ).build()
        pp_files = list((out / "preprocessed_sqrt").glob("*.txt"))
        assert len(pp_files) == 5

    def test_extra_handler_binned(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        out = tmp_path / "output"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            extra_handlers=[
                ProcessingHandler("binned_3000", "binned", bin_width=6),
            ],
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(
                synthetic_spectra_dir,
                synthetic_metadata_with_years,
                year_column="acquisition_date",
            ),
            out,
            extra_handlers=[
                ProcessingHandler("preprocessed_alt", "preprocessed"),
            ],
            n_jobs=1,
        ).build()
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
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            extra_handlers=[
                ProcessingHandler(
                    "preprocessed_minimal",
                    "preprocessed",
                    pipeline=minimal_pipe,
                ),
            ],
            n_jobs=1,
        ).build()
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
            report = DatasetBuilder(
                FlatLayout(synthetic_spectra_dir, meta_path),
                out,
                n_jobs=1,
            ).build()
        assert report.total == 5
        assert report.succeeded == 5

    def test_missing_id_column_raises(self, tmp_path, synthetic_spectra_dir):
        meta = pd.DataFrame({"Name": ["a"], "Species": ["E. coli"]})
        meta_path = tmp_path / "meta.csv"
        meta.to_csv(meta_path, index=False)
        with pytest.raises(ValueError, match="ID"):
            DatasetBuilder(
                FlatLayout(synthetic_spectra_dir, meta_path),
                tmp_path / "out",
                n_jobs=1,
            ).build()

    def test_empty_dir_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        meta = pd.DataFrame({"ID": ["a"], "Species": ["E. coli"]})
        meta_path = tmp_path / "meta.csv"
        meta.to_csv(meta_path, index=False)
        with pytest.raises(ValueError, match="No .txt"):
            DatasetBuilder(
                FlatLayout(empty_dir, meta_path),
                tmp_path / "out",
                n_jobs=1,
            ).build()

    def test_duplicate_folder_raises(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        with pytest.raises(ValueError, match="Duplicate folder"):
            DatasetBuilder(
                FlatLayout(synthetic_spectra_dir, synthetic_metadata),
                tmp_path / "out",
                extra_handlers=[
                    ProcessingHandler("preprocessed", "preprocessed"),
                ],
                n_jobs=1,
            ).build()

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
        report = DatasetBuilder(
            FlatLayout(spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
        ).build()
        assert report.succeeded == 4
        assert report.failed == 1
        assert "sample_4" in report.failed_ids


class TestBuildReport:
    """Tests for BuildReport contents."""

    def test_report_fields(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        out = tmp_path / "output"
        report = DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
        ).build()
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


class TestProcessingHandlerFromDict:
    """Tests for ProcessingHandler.from_dict with various pipeline types."""

    def test_string_pipeline_json(self, tmp_path):
        """Verify pipeline as JSON path string."""
        pipe = PreprocessingPipeline.default()
        json_path = tmp_path / "pipe.json"
        pipe.to_json(str(json_path))
        d = {
            "folder_name": "custom",
            "kind": "preprocessed",
            "pipeline": str(json_path),
        }
        handler = ProcessingHandler.from_dict(d)
        assert handler.pipeline is not None

    def test_string_pipeline_yaml(self, tmp_path):
        """Verify pipeline as YAML path string."""
        pipe = PreprocessingPipeline.default()
        yaml_path = tmp_path / "pipe.yaml"
        pipe.to_yaml(str(yaml_path))
        d = {
            "folder_name": "custom",
            "kind": "preprocessed",
            "pipeline": str(yaml_path),
        }
        handler = ProcessingHandler.from_dict(d)
        assert handler.pipeline is not None

    def test_unsupported_type_raises(self):
        """Verify unsupported pipeline type raises TypeError."""
        d = {"folder_name": "custom", "kind": "preprocessed", "pipeline": 42}
        with pytest.raises(TypeError, match="Unsupported pipeline"):
            ProcessingHandler.from_dict(d)


class TestBuilderEdgeCases:
    """Tests for DatasetBuilder error paths."""

    def test_no_matching_ids_raises(self, tmp_path):
        """Verify ValueError when no IDs match between metadata and spectra."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        df = _generate_synthetic_spectrum(random_state=42)
        np.savetxt(
            spectra_dir / "spec_X.txt",
            df[["mass", "intensity"]].values,
            header="mass intensity",
            comments="# ",
            fmt="%.6f",
        )
        meta_path = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["spec_Y"], "Drug": ["R"]}).to_csv(meta_path, index=False)
        layout = FlatLayout(spectra_dir, meta_path)
        builder = DatasetBuilder(layout, tmp_path / "out")
        with pytest.raises(ValueError, match="No matching IDs"):
            builder.build()

    def test_on_error_raise(self, tmp_path):
        """Verify on_error='raise' raises RuntimeError on failures."""
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        # Create an empty file that will fail to process
        (spectra_dir / "bad.txt").write_text("")
        # Also create a valid one so we get past ID matching
        df = _generate_synthetic_spectrum(random_state=42)
        np.savetxt(
            spectra_dir / "good.txt",
            df[["mass", "intensity"]].values,
            header="mass intensity",
            comments="# ",
            fmt="%.6f",
        )
        meta_path = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["bad", "good"], "Drug": ["R", "S"]}).to_csv(
            meta_path, index=False
        )
        layout = FlatLayout(spectra_dir, meta_path)
        builder = DatasetBuilder(layout, tmp_path / "out", on_error="raise")
        with pytest.raises(RuntimeError, match="failed"):
            builder.build()


class TestFullPrecisionOutput:
    """Builder outputs keep full float64 precision and reload exactly."""

    @pytest.fixture()
    def precise_build(self, tmp_path):
        src = tmp_path / "precise"
        src.mkdir()
        rng = np.random.default_rng(7)
        for i in range(2):
            df = pd.DataFrame(
                {
                    "mass": np.sort(rng.uniform(2000, 20000, 18000)),
                    "intensity": rng.random(18000) * 1000,
                }
            )
            np.savetxt(
                src / f"p{i}.txt",
                df.values,
                header="mass intensity",
                comments="# ",
                fmt="%.17g",
            )
        meta = tmp_path / "pmeta.csv"
        pd.DataFrame(
            {"ID": ["p0", "p1"], "Species": "Escherichia coli", "Drug": ["S", "R"]}
        ).to_csv(meta, index=False)
        out = tmp_path / "out"
        DatasetBuilder(FlatLayout(src, meta), out, n_jobs=1).build()
        return src, out

    def test_raw_and_preprocessed_reload_exactly(self, precise_build):
        src, out = precise_build
        for sid in ("p0", "p1"):
            source = read_spectrum(src / f"{sid}.txt")
            raw = read_spectrum(out / "raw" / f"{sid}.txt")
            np.testing.assert_array_equal(raw.to_numpy(), source.to_numpy())
            expected = preprocess(source, PreprocessingPipeline.default())
            pre = read_spectrum(out / "preprocessed" / f"{sid}.txt")
            np.testing.assert_array_equal(
                pre.to_numpy(), expected[["mass", "intensity"]].to_numpy()
            )

    def test_binned_stage_reloads_exactly(self, precise_build):
        src, out = precise_build
        source = read_spectrum(src / "p0.txt")
        expected, _ = bin_spectrum(
            preprocess(source, PreprocessingPipeline.default()),
            mz_min=2000,
            mz_max=20000,
            bin_width=3,
        )
        binned = read_spectrum(out / "binned_6000" / "p0.txt")
        np.testing.assert_array_equal(
            binned["intensity"].to_numpy(), expected["intensity"].to_numpy()
        )


class TestZeroPaddedIds:
    """Zero-padded metadata IDs survive a build."""

    def test_flat_layout_keeps_zero_padded_ids(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        for name, seed in (("007", 1), ("12", 2)):
            df = _generate_synthetic_spectrum(random_state=seed)
            np.savetxt(
                src / f"{name}.txt",
                df[["mass", "intensity"]].values,
                header="mass intensity",
                comments="# ",
                fmt="%.6f",
            )
        meta = tmp_path / "meta.csv"
        meta.write_text("ID,Drug\n007,R\n12,S\n")
        out = tmp_path / "out"
        report = DatasetBuilder(FlatLayout(src, meta), out, n_jobs=1).build()
        assert report.succeeded == 2
        assert (out / "preprocessed" / "007.txt").exists()


BRUKER_SAMPLE = DATA_DIR / "bruker_sample"


def _write_text_spectrum(path: Path, seed: int) -> None:
    """Write one synthetic text spectrum, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _generate_synthetic_spectrum(random_state=seed)
    np.savetxt(path, df[["mass", "intensity"]].values, fmt="%.6f")


def _bruker_tree(
    root: Path, rows: list[tuple[str, str]], *, without_pdata: bool = False
) -> Path:
    """Copy the sample Bruker spectrum into ``root`` and write its metadata.

    ``rows`` holds ``(identifier, relative folder)`` pairs; a folder named
    twice is copied once. ``without_pdata`` leaves only the raw ``fid``.
    """
    if not BRUKER_SAMPLE.exists():
        pytest.skip("No bruker_sample data available")
    for _, folder in rows:
        if not (root / folder).exists():
            shutil.copytree(BRUKER_SAMPLE / "922145" / "0_G10", root / folder)
    if without_pdata:
        for pdata in list(root.rglob("pdata")):
            shutil.rmtree(pdata)
    csv = root.parent / f"{root.name}.csv"
    pd.DataFrame(
        {
            "Identifier": [ident for ident, _ in rows],
            "Year": 2024,
            "Path": [folder for _, folder in rows],
            "target_position": "0_G10",
            "Drug": "R",
        }
    ).to_csv(csv, index=False)
    return csv


class TestCollidingInputs:
    """IDs that name several inputs, or inputs named by several IDs, fail."""

    @pytest.fixture()
    def flat_collision(self, tmp_path: Path) -> tuple[Path, Path]:
        spectra = tmp_path / "spectra"
        _write_text_spectrum(spectra / "2018" / "x.txt", seed=1)
        _write_text_spectrum(spectra / "2019" / "x.txt", seed=2)
        _write_text_spectrum(spectra / "2019" / "y.txt", seed=3)
        meta = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["x", "y"], "Drug": ["R", "S"]}).to_csv(meta, index=False)
        return spectra, meta

    def test_id_in_two_folders_fails_and_is_not_built(
        self, tmp_path, flat_collision, caplog
    ):
        """Neither file of an ID found in two folders is built; the ID fails."""
        spectra, meta = flat_collision
        out = tmp_path / "out"
        report = DatasetBuilder(FlatLayout(spectra, meta), out, n_jobs=1).build()
        assert (report.total, report.succeeded, report.failed) == (2, 1, 1)
        assert report.failed_ids == ["x"]
        assert not list((out / "raw").rglob("x.txt"))
        assert (
            "Spectrum ID 'x' names 2 input files (2018/x.txt, 2019/x.txt)"
            in caplog.text
        )
        written = pd.concat(pd.read_csv(f) for f in (out / "id").rglob("*.csv"))
        assert written["code"].tolist() == ["y"]

    def test_id_not_in_metadata_is_not_a_failure(self, tmp_path, flat_collision):
        """Files sharing an ID the metadata lacks are skipped, not failed."""
        spectra, meta = flat_collision
        _write_text_spectrum(spectra / "2018" / "z.txt", seed=4)
        _write_text_spectrum(spectra / "2019" / "z.txt", seed=5)
        report = DatasetBuilder(
            FlatLayout(spectra, meta), tmp_path / "out", n_jobs=1
        ).build()
        assert report.failed_ids == ["x"]
        assert report.total == 2

    def test_on_error_raise_stops_before_writing(self, tmp_path, flat_collision):
        """With on_error='raise' nothing is written for a colliding ID."""
        spectra, meta = flat_collision
        out = tmp_path / "out"
        builder = DatasetBuilder(FlatLayout(spectra, meta), out, on_error="raise")
        with pytest.raises(RuntimeError, match="'x'"):
            builder.build()
        assert not out.exists()

    def test_bruker_folder_named_by_two_ids_fails(self, tmp_path, caplog):
        """Two IDs naming one Bruker folder both fail; the others build."""
        root = tmp_path / "tree"
        csv = _bruker_tree(root, [("A", "2024/AB"), ("B", "2024/AB"), ("C", "2024/C")])
        layout = BrukerTreeLayout(root, csv, validate=False)
        out = tmp_path / "out"
        report = DatasetBuilder(layout, out, n_jobs=1).build()
        assert report.failed_ids == ["A", "B"]
        assert (report.total, report.succeeded) == (3, 1)
        assert sorted(p.name for p in (out / "raw").rglob("*.txt")) == ["C.txt"]
        assert "named by 2 spectrum IDs (A, B)" in caplog.text


class TestBrukerSource:
    """The builder reads the Bruker binary chosen with bruker_source."""

    @staticmethod
    def _tree_with_distinct_1r(tmp_path: Path) -> tuple[Path, Path, Path]:
        """Bruker tree whose pdata/1/1r differs from its fid (fid halved)."""
        root = tmp_path / "tree"
        csv = _bruker_tree(root, [("A", "2024/A")])
        acqus_dir = next((root / "2024" / "A").rglob("acqus")).parent
        fid = np.fromfile(acqus_dir / "fid", dtype="<i4")
        (fid // 2).astype("<i4").tofile(acqus_dir / "pdata" / "1" / "1r")
        return root, csv, root / "2024" / "A"

    @pytest.mark.parametrize("source", ["1r", "fid"])
    def test_raw_stage_holds_the_chosen_binary(self, tmp_path, source):
        """raw/ holds read_spectrum(..., bruker_source=source) of each folder."""
        root, csv, folder = self._tree_with_distinct_1r(tmp_path)
        out = tmp_path / f"out_{source}"
        kwargs = {} if source == "1r" else {"bruker_source": source}
        report = DatasetBuilder(
            BrukerTreeLayout(root, csv, validate=False), out, n_jobs=1, **kwargs
        ).build()
        assert report.succeeded == 1
        written = read_spectrum(out / "raw" / "2024" / "A.txt")
        expected = read_spectrum(folder, bruker_source=source)
        np.testing.assert_array_equal(
            written["intensity"].to_numpy(), expected["intensity"].to_numpy()
        )

    def test_tree_without_1r_builds_from_fid(self, tmp_path):
        """A tree without pdata/1/1r builds with bruker_source='fid'."""
        root = tmp_path / "tree"
        csv = _bruker_tree(root, [("A", "2024/A")], without_pdata=True)
        layout = BrukerTreeLayout(root, csv, validate=False)
        report = DatasetBuilder(
            layout, tmp_path / "out", n_jobs=1, bruker_source="fid"
        ).build()
        assert (report.succeeded, report.failed) == (1, 0)

    def test_missing_1r_fails_naming_the_fid_option(self, tmp_path, caplog):
        """Without pdata/1/1r the default fails and points to bruker_source='fid'."""
        root = tmp_path / "tree"
        csv = _bruker_tree(root, [("A", "2024/A")], without_pdata=True)
        layout = BrukerTreeLayout(root, csv, validate=False)
        report = DatasetBuilder(layout, tmp_path / "out", n_jobs=1).build()
        assert report.failed_ids == ["A"]
        assert "bruker_source='fid'" in caplog.text

    def test_invalid_value_raises(self, tmp_path, synthetic_spectra_dir):
        """An unknown bruker_source raises ValueError at construction."""
        layout = FlatLayout(synthetic_spectra_dir, tmp_path / "meta.csv")
        with pytest.raises(ValueError, match="bruker_source"):
            DatasetBuilder(layout, tmp_path / "out", bruker_source="2rr")

    def test_manifest_records_the_source_of_bruker_inputs(self, tmp_path):
        """site_info.json records bruker_source for a Bruker build."""
        root = tmp_path / "tree"
        csv = _bruker_tree(root, [("A", "2024/A")])
        out = tmp_path / "out"
        DatasetBuilder(
            BrukerTreeLayout(root, csv, validate=False),
            out,
            n_jobs=1,
            bruker_source="fid",
        ).build()
        assert read_site_info(out).build_info.bruker_source == "fid"

    def test_manifest_has_no_source_for_text_inputs(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        """A build from text files records no bruker_source."""
        out = tmp_path / "out"
        layout = FlatLayout(synthetic_spectra_dir, synthetic_metadata)
        DatasetBuilder(layout, out, n_jobs=1).build()
        assert read_site_info(out).build_info.bruker_source is None


class TestManifestMetadataRows:
    """site_info.json records the counts of the Bruker metadata merge."""

    def test_bruker_build_records_the_merge_counts(self, tmp_path):
        """A repeated row of A with another call: 3 rows, 2 spectra, 1 conflict."""
        root = tmp_path / "tree"
        csv = _bruker_tree(root, [("A", "2024/A"), ("B", "2024/B")])
        meta = pd.read_csv(csv)
        pd.concat([meta, meta.iloc[[0]].assign(Drug="S")]).to_csv(csv, index=False)
        out = tmp_path / "out"
        DatasetBuilder(
            BrukerTreeLayout(root, csv, validate=False), out, n_jobs=1
        ).build()
        assert read_site_info(out).build_info.metadata_rows == {
            "rows": 3,
            "spectra": 2,
            "merged_spectra": 1,
            "conflicting_calls": 1,
        }

    def test_text_build_records_no_counts(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        """FlatLayout merges no rows, so nothing is recorded."""
        out = tmp_path / "out"
        layout = FlatLayout(synthetic_spectra_dir, synthetic_metadata)
        DatasetBuilder(layout, out, n_jobs=1).build()
        assert read_site_info(out).build_info.metadata_rows is None


class TestManifestSpectrumExtension:
    """site_info.json records the extension of the files the builder writes."""

    def test_dataset_built_from_tsv_inputs_loads(self, tmp_path):
        """A dataset built from .tsv inputs loads with DRIAMSLayout."""
        spectra = tmp_path / "spectra"
        spectra.mkdir()
        for i, sid in enumerate(("a", "b")):
            df = _generate_synthetic_spectrum(random_state=i)
            df[["mass", "intensity"]].to_csv(
                spectra / f"{sid}.tsv", sep="\t", index=False
            )
        meta = tmp_path / "meta.csv"
        pd.DataFrame({"ID": ["a", "b"], "Drug": ["R", "S"]}).to_csv(meta, index=False)
        out = tmp_path / "out"
        layout = FlatLayout(spectra, meta, spectrum_ext=".tsv")
        DatasetBuilder(layout, out, n_jobs=1).build()
        assert read_site_info(out).spectrum_ext == ".txt"
        ds = DatasetLoader(DRIAMSLayout(out), n_jobs=1).load()
        assert len(ds.meta) == 2
