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

    @staticmethod
    def _manifest(root: Path, bin_width: float) -> None:
        from maldiamrkit.data import SiteInfo, write_site_info

        write_site_info(
            root,
            SiteInfo(
                id_column="code",
                metadata_dir="id",
                metadata_suffix="_clean.csv",
                spectrum_ext=".txt",
                spectra_folders=sorted(p.name for p in root.iterdir() if p.is_dir()),
                mz_range=(2000.0, 20000.0),
                bin_width=bin_width,
            ),
        )

    def test_several_binned_prefers_manifest_width(self, tmp_path):
        for name in ("binned_18000", "binned_6000"):
            (tmp_path / name).mkdir()
        self._manifest(tmp_path, 3.0)
        assert DRIAMSLayout(tmp_path).detect_stage() == "binned_6000"

    def test_several_binned_without_manifest_raise(self, tmp_path):
        for name in ("binned_18000", "binned_6000"):
            (tmp_path / name).mkdir()
        with pytest.raises(ValueError, match="stage="):
            DRIAMSLayout(tmp_path).detect_stage()

    def test_several_binned_without_the_manifest_folder_raise(self, tmp_path):
        for name in ("binned_18000", "binned_9000"):
            (tmp_path / name).mkdir()
        self._manifest(tmp_path, 3.0)
        with pytest.raises(ValueError, match="stage="):
            DRIAMSLayout(tmp_path).detect_stage()


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

        for spec, sid in zip((s0, s1, s2), meta["ID"], strict=True):
            spec.id = sid
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

        s0.id = "s0_rep1"
        avg_specs, avg_meta = DatasetLoader._average_replicates([s0], meta)
        assert len(avg_specs) == 1
        assert avg_meta["ID"].iloc[0] == "s0"
        assert avg_specs[0].id == "s0"


class TestDRIAMSBinIndexConversion:
    """DRIAMS binned_N/*.txt files store `bin_index binned_intensity`.

    DRIAMSLayout must convert those bin indices to real m/z on load so
    downstream m/z-aware APIs work.
    """

    @pytest.fixture()
    def driams_binned_dataset(self, tmp_path: Path) -> Path:
        """A minimal DRIAMS-shaped dataset whose binned_10/ files are bin
        indices 0..9 (standing in for the real 6000-bin DRIAMS format)."""
        root = tmp_path / "driams_like"
        binned = root / "binned_10"
        binned.mkdir(parents=True)
        id_dir = root / "id"
        id_dir.mkdir()

        ids = [f"sample_{i}" for i in range(3)]
        for sid in ids:
            content = "bin_index binned_intensity\n" + "\n".join(
                f"{i} {0.01 * (i + 1):.6f}" for i in range(10)
            )
            (binned / f"{sid}.txt").write_text(content)

        meta = pd.DataFrame(
            {
                "code": ids,
                "species": ["Escherichia coli"] * 3,
                "Ceftriaxone": ["S", "R", "S"],
            }
        )
        meta.to_csv(id_dir / "all_clean.csv", index=False)
        return root

    def test_bin_index_rewritten_to_mz(self, driams_binned_dataset):
        """Loader rewrites bin_index mass column to real m/z in the
        [mz_min, mz_max] range defined by the layout."""
        layout = DRIAMSLayout(driams_binned_dataset, mz_min=2000.0, mz_max=19997.0)
        ds = DatasetLoader(layout, n_jobs=1).load()

        assert len(ds.spectra) == 3
        for spec in ds.spectra:
            mass = spec.raw["mass"].to_numpy()
            assert mass[0] == pytest.approx(2000.0)
            assert mass[-1] == pytest.approx(19997.0)
            assert spec.is_binned
            assert spec.has_bin_metadata

    def test_non_bin_index_files_untouched(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata
    ):
        """Non-binned stages keep their original (real) m/z values."""
        out = tmp_path / "driams_out"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
        ).build()

        ds = DatasetLoader(DRIAMSLayout(out), stage="preprocessed", n_jobs=1).load()
        for spec in ds.spectra:
            mass = spec.raw["mass"].to_numpy()
            # Real m/z, not bin indices
            assert mass.min() > 1000.0

    def test_idempotent_on_already_converted(self):
        """Calling postprocess twice on an already-converted spectrum
        is a no-op."""
        from maldiamrkit import MaldiSpectrum

        mz = np.linspace(2000.0, 19997.0, 10)
        spec = MaldiSpectrum(pd.DataFrame({"mass": mz, "intensity": np.ones(10)}))
        layout = DRIAMSLayout("/tmp/unused")
        out1 = layout.postprocess_spectrum(spec, stage="binned_10")
        out2 = layout.postprocess_spectrum(out1, stage="binned_10")
        np.testing.assert_allclose(out2.raw["mass"].to_numpy(), mz)

    def test_not_applied_when_stage_is_not_binned(self, tmp_path):
        """postprocess_spectrum is a no-op for `raw` / `preprocessed`."""
        from maldiamrkit import MaldiSpectrum

        bin_index_df = pd.DataFrame(
            {"mass": np.arange(10, dtype=float), "intensity": np.ones(10)}
        )
        spec = MaldiSpectrum(bin_index_df)
        layout = DRIAMSLayout("/tmp/unused")
        out = layout.postprocess_spectrum(spec, stage="preprocessed")
        np.testing.assert_allclose(
            out.raw["mass"].to_numpy(), np.arange(10, dtype=float)
        )

    def test_normalize_tic_rescales_to_unit_sum(self, driams_binned_dataset):
        """`normalize_tic=True` rescales each spectrum to sum=1."""
        layout_plain = DRIAMSLayout(
            driams_binned_dataset, mz_min=2000.0, mz_max=19997.0
        )
        layout_normed = DRIAMSLayout(
            driams_binned_dataset,
            mz_min=2000.0,
            mz_max=19997.0,
            normalize_tic=True,
        )

        ds_plain = DatasetLoader(layout_plain, n_jobs=1).load()
        ds_normed = DatasetLoader(layout_normed, n_jobs=1).load()

        # Without normalize_tic, the fixture's hand-written intensities
        # sum to 0.01 + 0.02 + ... + 0.10 = 0.55 per spectrum.
        plain_sums = [
            float(s.raw["intensity"].to_numpy().sum()) for s in ds_plain.spectra
        ]
        normed_sums = [
            float(s.raw["intensity"].to_numpy().sum()) for s in ds_normed.spectra
        ]

        assert all(abs(v - 0.55) < 1e-9 for v in plain_sums)
        assert all(abs(v - 1.0) < 1e-9 for v in normed_sums)

        # m/z axis must be unchanged by renormalization.
        np.testing.assert_allclose(
            ds_plain.spectra[0].raw["mass"].to_numpy(),
            ds_normed.spectra[0].raw["mass"].to_numpy(),
        )

    def test_normalize_tic_default_off(self, driams_binned_dataset):
        """Default is normalize_tic=False (faithful read)."""
        layout = DRIAMSLayout(driams_binned_dataset, mz_min=2000.0, mz_max=19997.0)
        assert layout.normalize_tic is False
        ds = DatasetLoader(layout, n_jobs=1).load()
        sums = [float(s.raw["intensity"].to_numpy().sum()) for s in ds.spectra]
        # Not normalized -> not 1.0
        assert all(abs(v - 1.0) > 1e-3 for v in sums)


class TestBuilderBinnedStageLabels:
    """Builder ``binned_N`` stages load with the labels of in-memory binning."""

    @staticmethod
    def _expected(src: Path, bin_width: float) -> pd.DataFrame:
        from maldiamrkit.io import read_spectrum
        from maldiamrkit.preprocessing import PreprocessingPipeline
        from maldiamrkit.preprocessing.binning import bin_spectrum
        from maldiamrkit.preprocessing.pipeline import preprocess

        pre = preprocess(read_spectrum(src), PreprocessingPipeline.default())
        binned, _ = bin_spectrum(pre, mz_min=2000, mz_max=20000, bin_width=bin_width)
        return binned

    @pytest.fixture()
    def built(self, tmp_path, synthetic_spectra_dir, synthetic_metadata):
        from maldiamrkit.data import ProcessingHandler

        out = tmp_path / "built"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata),
            out,
            n_jobs=1,
            extra_handlers=[
                ProcessingHandler(
                    folder_name="binned_3000", kind="binned", bin_width=6
                ),
                ProcessingHandler(
                    folder_name="binned_2572", kind="binned", bin_width=7
                ),
            ],
        ).build()
        return out

    def test_default_stage_matches_in_memory_binning(
        self, built, synthetic_spectra_dir
    ):
        ds = DatasetLoader(DRIAMSLayout(built), stage="binned_6000", n_jobs=1).load()
        expected = self._expected(synthetic_spectra_dir / "sample_0.txt", 3)
        X = ds.X
        assert X.columns.equals(pd.Index(expected["mass"]))
        np.testing.assert_array_equal(
            X.loc["sample_0"].to_numpy(), expected["intensity"].to_numpy()
        )
        assert ds.spectra[0]._raw["mass"].dtype == np.float64

    def test_same_columns_as_preprocessed_stage(self, built):
        X_binned = (
            DatasetLoader(DRIAMSLayout(built), stage="binned_6000", n_jobs=1).load().X
        )
        X_pre = (
            DatasetLoader(DRIAMSLayout(built), stage="preprocessed", n_jobs=1).load().X
        )
        assert X_binned.columns.equals(X_pre.columns)

    def test_labels_survive_tic_normalization(self, built, synthetic_spectra_dir):
        layout = DRIAMSLayout(built, normalize_tic=True)
        X = DatasetLoader(layout, stage="binned_6000", n_jobs=1).load().X
        expected = self._expected(synthetic_spectra_dir / "sample_0.txt", 3)
        assert X.columns.equals(pd.Index(expected["mass"]))
        np.testing.assert_allclose(X.sum(axis=1).to_numpy(), 1.0)

    def test_extra_folder_with_dividing_width(self, built, synthetic_spectra_dir):
        ds = DatasetLoader(DRIAMSLayout(built), stage="binned_3000", n_jobs=1).load()
        expected = self._expected(synthetic_spectra_dir / "sample_0.txt", 6)
        assert ds.X.columns.equals(pd.Index(expected["mass"]))

    def test_extra_folder_with_other_width_warns(self, built):
        with pytest.warns(UserWarning, match="bin width"):
            ds = DatasetLoader(
                DRIAMSLayout(built), stage="binned_2572", n_jobs=1
            ).load()
        assert ds.spectra[0].raw["mass"].iloc[0] == pytest.approx(2000.0)


class TestTextIds:
    """Metadata IDs are read as text, so zero-padded IDs keep their zeros."""

    @staticmethod
    def _copy_spectra(dest: Path, names: list[str]) -> None:
        import shutil

        data = Path(__file__).parent.parent.parent / "data"
        for f, name in zip(
            sorted(data.glob("*.txt"))[: len(names)], names, strict=True
        ):
            shutil.copy(f, dest / f"{name}.txt")

    def test_driams_layout_keeps_zero_padded_ids(self, tmp_path):
        root = tmp_path / "driams"
        (root / "preprocessed").mkdir(parents=True)
        (root / "id").mkdir()
        self._copy_spectra(root / "preprocessed", ["1", "2", "007"])
        (root / "id" / "all_clean.csv").write_text(
            "code,species,Drug\n1,E. coli,R\n2,E. coli,S\n007,E. coli,R\n"
        )
        ds = DatasetLoader(DRIAMSLayout(root), stage="preprocessed", n_jobs=1).load(
            aggregate_by={"antibiotics": "Drug"}
        )
        assert sorted(ds.X.index) == ["007", "1", "2"]

    def test_read_metadata_csv_detects_the_id_column(self, tmp_path):
        from maldiamrkit.data._metadata import _read_metadata_csv

        f = tmp_path / "meta.csv"
        f.write_text("code,ID,Drug\n007,10,R\n")
        detected = _read_metadata_csv(f)
        assert detected["code"].tolist() == ["007"]
        explicit = _read_metadata_csv(f, "ID")
        assert explicit["ID"].tolist() == ["10"]


class TestPreprocessedStageNotReprocessed:
    """Spectra loaded from the ``preprocessed`` stage are binned as they are."""

    def test_preprocessed_stage_is_binned_without_preprocessing(
        self, tmp_path, synthetic_spectra_dir, synthetic_metadata, monkeypatch
    ):
        from maldiamrkit.io import read_spectrum
        from maldiamrkit.preprocessing import PreprocessingPipeline
        from maldiamrkit.preprocessing.binning import bin_spectrum
        from maldiamrkit.preprocessing.pipeline import preprocess

        out = tmp_path / "built"
        DatasetBuilder(
            FlatLayout(synthetic_spectra_dir, synthetic_metadata), out, n_jobs=1
        ).build()
        expected = {}
        for i in range(5):
            pre = preprocess(
                read_spectrum(synthetic_spectra_dir / f"sample_{i}.txt"),
                PreprocessingPipeline.default(),
            )
            binned, _ = bin_spectrum(pre, mz_min=2000, mz_max=20000, bin_width=3)
            expected[f"sample_{i}"] = binned["intensity"].to_numpy()

        def fail(*args, **kwargs):
            raise AssertionError("preprocessed spectra were preprocessed again")

        monkeypatch.setattr("maldiamrkit.spectrum.preprocess", fail)
        X = DatasetLoader(DRIAMSLayout(out), stage="preprocessed", n_jobs=1).load().X
        for sid, values in expected.items():
            np.testing.assert_array_equal(X.loc[sid].to_numpy(), values)


class TestReplicateAveraging:
    """``duplicate_strategy="average"`` averages each isolate's own replicates."""

    @pytest.fixture()
    def driams_replicates(self, tmp_path: Path) -> Path:
        root = tmp_path / "driams_rep"
        (root / "binned_10").mkdir(parents=True)
        (root / "id").mkdir()
        base = {"a_MALDI1": 1.0, "a_MALDI2": 3.0, "b_MALDI1": 100.0}
        for sid, v in base.items():
            content = "bin_index binned_intensity\n" + "\n".join(
                f"{i} {v * (i + 1)!r}" for i in range(10)
            )
            (root / "binned_10" / f"{sid}.txt").write_text(content)
        pd.DataFrame(
            {
                "code": ["b_MALDI1", "a_MALDI1", "a_MALDI2", "c_MALDI1"],
                "species": "Escherichia coli",
                "Ceftriaxone": ["S", np.nan, "R", "S"],
            }
        ).to_csv(root / "id" / "all_clean.csv", index=False)
        return root

    @staticmethod
    def _load(root: Path, **kwargs):
        layout = DRIAMSLayout(root, mz_min=2000.0, mz_max=19997.0, **kwargs)
        return DatasetLoader(layout, stage="binned_10", n_jobs=1).load()

    def test_each_isolate_averages_its_own_replicates(self, driams_replicates):
        plain = self._load(driams_replicates).X
        ds = self._load(
            driams_replicates, collapse_replicates=True, duplicate_strategy="average"
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            X = ds.X
        assert sorted(X.index) == ["a", "b"]
        expected_a = (plain.loc["a_MALDI1"] + plain.loc["a_MALDI2"]).to_numpy() / 2
        np.testing.assert_array_equal(X.loc["a"].to_numpy(), expected_a)
        np.testing.assert_array_equal(
            X.loc["b"].to_numpy(), plain.loc["b_MALDI1"].to_numpy()
        )
        assert all(s.is_binned for s in ds.spectra)

    def test_first_replicate_row_kept_as_is(self, driams_replicates):
        ds = self._load(
            driams_replicates, collapse_replicates=True, duplicate_strategy="average"
        )
        assert pd.isna(ds.meta.loc["a", "Ceftriaxone"])

    def test_preprocessed_stage_average_stays_preprocessed(
        self, tmp_path, synthetic_spectra_dir
    ):
        meta = pd.DataFrame(
            {
                "ID": [f"x_MALDI{i}" for i in (1, 2)] + ["y_MALDI1"],
                "Species": "Escherichia coli",
                "Ceftriaxone": ["S", "S", "R"],
            }
        )
        src = tmp_path / "src"
        src.mkdir()
        for sid, i in zip(meta["ID"], range(3), strict=True):
            (src / f"{sid}.txt").write_bytes(
                (synthetic_spectra_dir / f"sample_{i}.txt").read_bytes()
            )
        meta_path = tmp_path / "meta.csv"
        meta.to_csv(meta_path, index=False)
        out = tmp_path / "built"
        DatasetBuilder(FlatLayout(src, meta_path), out, n_jobs=1).build()

        layout = DRIAMSLayout(
            out, collapse_replicates=True, duplicate_strategy="average"
        )
        ds = DatasetLoader(layout, stage="preprocessed", n_jobs=1).load()
        assert all(s.is_preprocessed for s in ds.spectra)

        stage = DatasetLoader(layout, stage="binned_6000", n_jobs=1).load()
        plain = DatasetLoader(DRIAMSLayout(out), stage="binned_6000", n_jobs=1).load().X
        X = stage.X
        assert sorted(X.index) == ["x", "y"]
        expected_x = (plain.loc["x_MALDI1"] + plain.loc["x_MALDI2"]).to_numpy() / 2
        np.testing.assert_array_equal(X.loc["x"].to_numpy(), expected_x)


class TestStrategyLoads:
    """Loads through DatasetLoader with keep_all and average-built metadata."""

    @staticmethod
    def _binned_stage(root: Path, values: dict[str, float]) -> None:
        (root / "binned_10").mkdir(parents=True)
        (root / "id").mkdir()
        for sid, v in values.items():
            content = "bin_index binned_intensity\n" + "\n".join(
                f"{i} {v * (i + 1)!r}" for i in range(10)
            )
            (root / "binned_10" / f"{sid}.txt").write_text(content)

    def test_keep_all_keeps_every_replicate(self, tmp_path):
        root = tmp_path / "keep"
        self._binned_stage(root, {"a_MALDI1": 1.0, "a_MALDI2": 3.0, "b_MALDI1": 5.0})
        pd.DataFrame(
            {
                "code": ["a_MALDI1", "a_MALDI2", "b_MALDI1"],
                "species": "Escherichia coli",
            }
        ).to_csv(root / "id" / "all_clean.csv", index=False)
        layout = DRIAMSLayout(
            root,
            mz_min=2000.0,
            mz_max=19997.0,
            collapse_replicates=True,
            duplicate_strategy="keep_all",
        )
        ds = DatasetLoader(layout, stage="binned_10", n_jobs=1).load()
        assert sorted(ds.X.index) == ["a_MALDI1", "a_MALDI2", "b_MALDI1"]

    def test_average_built_metadata_still_averages(self, tmp_path):
        root = tmp_path / "avg"
        self._binned_stage(root, {"x": 7.0, "y_rep1": 1.0, "y_rep2": 3.0})
        pd.DataFrame(
            {
                "code": ["x", "y_rep1", "y_rep2"],
                "species": "Escherichia coli",
                "_original_id": ["x", "y", "y"],
            }
        ).to_csv(root / "id" / "all_clean.csv", index=False)
        layout = DRIAMSLayout(root, mz_min=2000.0, mz_max=19997.0)
        X = DatasetLoader(layout, stage="binned_10", n_jobs=1).load().X
        assert sorted(X.index) == ["x", "y"]
        np.testing.assert_array_equal(
            X.loc["y"].to_numpy(), 2.0 * np.arange(1, 11, dtype=float)
        )


class TestDuplicatesAcrossYears:
    """``duplicate_strategy`` applies to the files of every year folder."""

    @staticmethod
    def _write(root: Path, year: str, sid: str, factor: float) -> None:
        folder = root / "binned_10" / year
        folder.mkdir(parents=True, exist_ok=True)
        content = "bin_index binned_intensity\n" + "\n".join(
            f"{i} {factor * (i + 1)!r}" for i in range(10)
        )
        (folder / f"{sid}.txt").write_text(content)

    @staticmethod
    def _meta(root: Path, year: str, rows: list[tuple[str, str]]) -> None:
        folder = root / "id" / year
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "code": [r[0] for r in rows],
                "species": "Escherichia coli",
                "Drug": [r[1] for r in rows],
            }
        ).to_csv(folder / f"{year}_clean.csv", index=False)

    @pytest.fixture()
    def two_years(self, tmp_path: Path) -> Path:
        root = tmp_path / "years"
        self._write(root, "2018", "1s", 1.0)
        self._write(root, "2018", "2s", 5.0)
        self._write(root, "2019", "1s", 3.0)
        self._meta(root, "2018", [("1s", "S"), ("2s", "S")])
        self._meta(root, "2019", [("1s", "R"), ("3s", "R")])
        return root

    @staticmethod
    def _load(root: Path, **kwargs):
        layout = DRIAMSLayout(root, mz_min=2000.0, mz_max=19997.0, **kwargs)
        return DatasetLoader(layout, stage="binned_10", n_jobs=1).load()

    @staticmethod
    def _values(factor: float) -> np.ndarray:
        return factor * np.arange(1, 11, dtype=float)

    def test_first_keeps_the_first_year(self, two_years):
        ds = self._load(two_years, duplicate_strategy="first")
        X = ds.X
        assert sorted(X.index) == ["1s", "2s"]
        np.testing.assert_array_equal(X.loc["1s"].to_numpy(), self._values(1.0))
        assert ds.meta.loc["1s", "Drug"] == "S"
        assert "_year" not in ds.meta.columns
        assert "_file_id" not in ds.meta.columns

    def test_last_keeps_the_last_year(self, two_years):
        ds = self._load(two_years, duplicate_strategy="last")
        np.testing.assert_array_equal(ds.X.loc["1s"].to_numpy(), self._values(3.0))
        assert ds.meta.loc["1s", "Drug"] == "R"

    def test_drop_removes_the_sample(self, two_years):
        assert sorted(self._load(two_years, duplicate_strategy="drop").X.index) == [
            "2s"
        ]

    def test_keep_all_keeps_each_file_in_one_group(self, two_years):
        ds = self._load(two_years, duplicate_strategy="keep_all")
        X = ds.X
        assert sorted(X.index) == ["1s_rep1", "1s_rep2", "2s"]
        np.testing.assert_array_equal(X.loc["1s_rep1"].to_numpy(), self._values(1.0))
        np.testing.assert_array_equal(X.loc["1s_rep2"].to_numpy(), self._values(3.0))
        assert ds.meta.loc["1s_rep2", "Drug"] == "R"
        groups = pd.Series(ds.groups, index=X.index)
        assert groups["1s_rep1"] == groups["1s_rep2"] != groups["2s"]

    def test_average_averages_both_years(self, two_years):
        X = self._load(two_years, duplicate_strategy="average").X
        assert sorted(X.index) == ["1s", "2s"]
        np.testing.assert_array_equal(X.loc["1s"].to_numpy(), self._values(2.0))

    def test_year_option_loads_one_year(self, two_years):
        X = self._load(two_years, year="2019").X
        assert sorted(X.index) == ["1s"]
        np.testing.assert_array_equal(X.loc["1s"].to_numpy(), self._values(3.0))

    def test_collapse_same_replicate_in_both_years(self, tmp_path):
        root = tmp_path / "collapse"
        self._write(root, "2018", "1s_MALDI1", 1.0)
        self._write(root, "2019", "1s_MALDI1", 3.0)
        self._write(root, "2018", "2s_MALDI1", 5.0)
        self._meta(root, "2018", [("1s_MALDI1", "S"), ("2s_MALDI1", "S")])
        self._meta(root, "2019", [("1s_MALDI1", "R")])
        X = self._load(root, collapse_replicates=True, duplicate_strategy="average").X
        assert sorted(X.index) == ["1s", "2s"]
        np.testing.assert_array_equal(X.loc["1s"].to_numpy(), self._values(2.0))

        ds = self._load(root, collapse_replicates=True, duplicate_strategy="keep_all")
        X = ds.X
        assert len(X) == 3 and X.index.is_unique
        groups = pd.Series(ds.groups, index=X.index)
        assert groups.nunique() == 2

    def test_two_rows_for_one_file_raise(self, tmp_path):
        root = tmp_path / "dup_rows"
        self._write(root, "2018", "1s", 1.0)
        self._write(root, "2019", "4s", 1.0)
        self._meta(root, "2018", [("1s", "S"), ("1s", "R")])
        self._meta(root, "2019", [("4s", "S")])
        with pytest.raises(ValueError, match="1s.txt"):
            self._load(root, duplicate_strategy="keep_all")


class TestFilesOfTheirRows:
    """Each metadata row is matched to its own file, whatever its ID."""

    BRUKER = Path(__file__).resolve().parents[2] / "data" / "bruker_sample"

    @classmethod
    def _bruker_tree(cls, root: Path, rows: list[tuple[str, str]]) -> Path:
        import shutil

        source = cls.BRUKER / "922145" / "0_G10"
        records = []
        for ident, pos in rows:
            shutil.copytree(source, root / "2024" / ident / pos)
            records.append(
                {
                    "Identifier": ident,
                    "Path": f"2024/{ident}/{pos}",
                    "target_position": pos,
                    "Year": 2024,
                    "Drug": "R",
                }
            )
        csv = root / "AMR.csv"
        pd.DataFrame(records).to_csv(csv, index=False)
        return csv

    @pytest.fixture()
    def marisma(self, tmp_path: Path) -> tuple[Path, Path]:
        root = tmp_path / "marisma"
        rows = [("A", "0_A1"), ("A", "0_A2"), ("B", "0_B1"), ("C", "0_C1")]
        return root, self._bruker_tree(root, rows)

    @staticmethod
    def _load_marisma(marisma, strategy):
        from maldiamrkit.data import MARISMaLayout

        root, csv = marisma
        layout = MARISMaLayout(root, csv, duplicate_strategy=strategy)
        return DatasetLoader(layout, n_jobs=1).load()

    def test_marisma_keep_all_loads_every_specimen(self, marisma):
        ds = self._load_marisma(marisma, "keep_all")
        assert sorted(ds.X.index) == ["A_0_A1", "A_0_A2", "B", "C"]
        groups = pd.Series(ds.groups, index=ds.X.index)
        assert groups["A_0_A1"] == groups["A_0_A2"]
        assert groups.nunique() == 3

    def test_marisma_first_loads_one_row_per_specimen(self, marisma):
        assert sorted(self._load_marisma(marisma, "first").X.index) == ["A", "B", "C"]

    def test_marisma_average_keeps_single_specimens(self, marisma):
        assert sorted(self._load_marisma(marisma, "average").X.index) == [
            "A",
            "B",
            "C",
        ]

    @pytest.mark.parametrize("strategy", ["keep_all", "average"])
    def test_flat_rows_sharing_a_file_raise(self, tmp_path, strategy):
        root = tmp_path / "flat"
        (root / "raw").mkdir(parents=True)
        (root / "id").mkdir()
        for k, sid in enumerate(["X", "Y", "Z"]):
            df = _generate_synthetic_spectrum(random_state=k)
            np.savetxt(root / "raw" / f"{sid}.txt", df[["mass", "intensity"]].values)
        pd.DataFrame(
            {"code": ["X", "X", "Y", "Z"], "species": "E", "Drug": ["R", "S", "R", "S"]}
        ).to_csv(root / "id" / "all_clean.csv", index=False)
        layout = DRIAMSLayout(root, duplicate_strategy=strategy)
        with pytest.raises(ValueError, match="X.txt"):
            DatasetLoader(layout, stage="raw", n_jobs=1).load()

    def test_keep_all_groups_follow_id_transform(self, tmp_path):
        root = tmp_path / "transform"
        (root / "raw").mkdir(parents=True)
        (root / "id").mkdir()
        ids = ["s1-r1", "s1-r2", "s2-r1", "s2-r2", "s3-r1"]
        for k, sid in enumerate(ids):
            df = _generate_synthetic_spectrum(random_state=k)
            np.savetxt(root / "raw" / f"{sid}.txt", df[["mass", "intensity"]].values)
        pd.DataFrame({"code": ids, "species": "E"}).to_csv(
            root / "id" / "all_clean.csv", index=False
        )
        layout = DRIAMSLayout(
            root,
            id_transform=lambda s: s.split("-")[0],
            duplicate_strategy="keep_all",
        )
        ds = DatasetLoader(layout, stage="raw", n_jobs=1).load()
        groups = pd.Series(ds.groups, index=ds.X.index)
        assert groups["s1-r1"] == groups["s1-r2"] != groups["s2-r1"]
        assert groups.nunique() == 3

    @pytest.mark.parametrize("strategy", ["average", "keep_all"])
    def test_built_numeric_ids_load_with_their_groups(self, tmp_path, strategy):
        from maldiamrkit.data import BrukerTreeLayout

        root = tmp_path / "tree"
        rows = [
            ("922145", "0_A1"),
            ("922145", "0_A2"),
            ("007", "0_B1"),
            ("007", "0_B2"),
            ("555", "0_C1"),
        ]
        csv = self._bruker_tree(root, rows)
        out = tmp_path / "built"
        layout = BrukerTreeLayout(
            root, csv, duplicate_strategy=strategy, validate=False
        )
        DatasetBuilder(layout, out, n_jobs=1).build()
        ds = DatasetLoader(DRIAMSLayout(out), stage="raw", n_jobs=1).load()
        if strategy == "average":
            assert sorted(ds.X.index) == ["007", "555", "922145"]
        else:
            assert len(ds.X) == 5
            groups = pd.Series(ds.groups, index=ds.X.index)
            assert sorted(groups.unique()) == ["007", "555", "922145"]

    def test_same_file_name_in_two_years_without_year_metadata_raises(self, tmp_path):
        root = tmp_path / "flat_meta"
        for year in ("2018", "2019"):
            (root / "raw" / year).mkdir(parents=True)
            df = _generate_synthetic_spectrum(random_state=int(year))
            np.savetxt(root / "raw" / year / "X.txt", df[["mass", "intensity"]].values)
        (root / "id").mkdir()
        pd.DataFrame({"code": ["X"], "species": "E"}).to_csv(
            root / "id" / "all_clean.csv", index=False
        )
        with pytest.raises(ValueError, match="no year"):
            DatasetLoader(DRIAMSLayout(root), stage="raw", n_jobs=1).load()

    def test_custom_layout_without_file_ids_matches_by_id(self, tmp_path):
        from maldiamrkit.data.dataset_layouts import DatasetLayout

        folder = tmp_path / "custom"
        folder.mkdir()
        for k, sid in enumerate(["a", "b", "c"]):
            df = _generate_synthetic_spectrum(random_state=k)
            np.savetxt(folder / f"{sid}.txt", df[["mass", "intensity"]].values)

        class Custom(DatasetLayout):
            def discover_metadata(self):
                return pd.DataFrame({"ID": ["a", "c", "z"], "Drug": ["R", "S", "R"]})

            def collect_spectrum_files(self, stage, year):
                return sorted(folder.glob("*.txt"))

            def detect_stage(self):
                return "raw"

        ds = DatasetLoader(Custom(), n_jobs=1).load()
        assert sorted(ds.X.index) == ["a", "c"]


class TestRowFileEdgeCases:
    """Row matching keeps v0.22.0 behaviour around the new file keys."""

    @staticmethod
    def _write(path: Path, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = _generate_synthetic_spectrum(random_state=seed)
        np.savetxt(path, df[["mass", "intensity"]].values)

    def _driams(self, root: Path, files: dict[str, int], meta: pd.DataFrame) -> Path:
        for rel, seed in files.items():
            self._write(root / "raw" / rel, seed)
        for year, rows in meta.groupby("year", dropna=False):
            folder = root / "id" if pd.isna(year) else root / "id" / str(year)
            folder.mkdir(parents=True, exist_ok=True)
            name = "all_clean.csv" if pd.isna(year) else f"{year}_clean.csv"
            rows.drop(columns="year").to_csv(folder / name, index=False)
        return root

    def test_rows_follow_the_file_order(self, tmp_path):
        root = self._driams(
            tmp_path / "order",
            {"a.txt": 0, "b.txt": 1, "c.txt": 2},
            pd.DataFrame({"code": ["c", "a", "b"], "species": "E", "year": np.nan}),
        )
        ds = DatasetLoader(DRIAMSLayout(root), stage="raw", n_jobs=1).load()
        assert list(ds.X.index) == ["a", "b", "c"]

    def test_file_in_another_year_folder_is_used_with_a_warning(self, tmp_path):
        root = self._driams(
            tmp_path / "years",
            {"2015/m.txt": 0, "2015/n.txt": 1},
            pd.DataFrame(
                {"code": ["m", "n"], "species": "E", "year": ["2015", "2016"]}
            ),
        )
        with pytest.warns(UserWarning, match="another year"):
            ds = DatasetLoader(DRIAMSLayout(root), stage="raw", n_jobs=1).load()
        assert sorted(ds.X.index) == ["m", "n"]

    def test_missing_file_of_a_kept_year_is_not_an_error(self, tmp_path):
        root = self._driams(
            tmp_path / "missing",
            {"2018/x.txt": 0},
            pd.DataFrame(
                {"code": ["x", "x"], "species": "E", "year": ["2018", "2019"]}
            ),
        )
        layout = DRIAMSLayout(root, duplicate_strategy="keep_all")
        ds = DatasetLoader(layout, stage="raw", n_jobs=1).load()
        assert len(ds.X) == 1

    def test_metadata_with_repeated_index_labels(self, tmp_path):
        from maldiamrkit.data.dataset_layouts import DatasetLayout

        root_a = self._driams(
            tmp_path / "a",
            {"a1.txt": 0, "a2.txt": 1},
            pd.DataFrame({"code": ["a1", "a2"], "species": "E", "year": np.nan}),
        )
        root_b = self._driams(
            tmp_path / "b",
            {"b1.txt": 2, "b2.txt": 3},
            pd.DataFrame({"code": ["b1", "b2"], "species": "E", "year": np.nan}),
        )
        sites = [DRIAMSLayout(root_a), DRIAMSLayout(root_b)]

        class TwoSites(DatasetLayout):
            def discover_metadata(self):
                return pd.concat([s.discover_metadata() for s in sites])

            def collect_spectrum_files(self, stage, year):
                return sorted(
                    f for s in sites for f in s.collect_spectrum_files("raw", None)
                )

            def detect_stage(self):
                return "raw"

        ds = DatasetLoader(TwoSites(), stage="raw", n_jobs=1).load()
        assert sorted(ds.X.index) == ["a1", "a2", "b1", "b2"]

    def test_subclass_rewriting_ids(self, tmp_path):
        root = self._driams(
            tmp_path / "upper",
            {"AB.txt": 0, "CD.txt": 1},
            pd.DataFrame({"code": ["ab", "cd"], "species": "E", "year": np.nan}),
        )

        class Upper(DRIAMSLayout):
            def discover_metadata(self):
                meta = super().discover_metadata()
                meta["ID"] = meta["ID"].str.upper()
                return meta

        ds = DatasetLoader(Upper(root), stage="raw", n_jobs=1).load()
        assert sorted(ds.X.index) == ["AB", "CD"]

    def test_partial_isolate_column_gives_no_missing_group(self, tmp_path):
        root = self._driams(
            tmp_path / "partial",
            {"2018/A_rep1.txt": 0, "2018/A_rep2.txt": 1, "2019/C.txt": 2},
            pd.DataFrame(
                {
                    "code": ["A_rep1", "A_rep2", "C"],
                    "species": "E",
                    "_isolate": ["A", "A", np.nan],
                    "year": ["2018", "2018", "2019"],
                }
            ),
        )
        ds = DatasetLoader(DRIAMSLayout(root), stage="raw", n_jobs=1).load()
        groups = pd.Series(ds.groups, index=ds.X.index)
        assert groups.to_dict() == {"A_rep1": "A", "A_rep2": "A", "C": "C"}

    @staticmethod
    def _marisma(root: Path, rows: list[tuple[str, str, str]]) -> Path:
        import shutil

        source = TestFilesOfTheirRows.BRUKER / "922145" / "0_G10"
        records = []
        for ident, folder, pos in rows:
            target = root / "2024" / folder
            if not target.exists():
                shutil.copytree(source, target)
            records.append(
                {
                    "Identifier": ident,
                    "Path": f"2024/{folder}",
                    "target_position": pos,
                    "Year": 2024,
                }
            )
        csv = root / "AMR.csv"
        pd.DataFrame(records).to_csv(csv, index=False)
        return csv

    def test_marisma_rows_naming_one_directory_raise(self, tmp_path):
        from maldiamrkit.data import MARISMaLayout

        root = tmp_path / "same_dir"
        csv = self._marisma(root, [("B", "B/0_B1", "0_B1"), ("Q", "B/0_B1", "0_B1")])
        with pytest.raises(ValueError, match="several metadata rows"):
            DatasetLoader(MARISMaLayout(root, csv), n_jobs=1).load()

    def test_marisma_keep_all_with_id_transform_suffixes_target_positions(
        self, tmp_path
    ):
        from maldiamrkit.data import MARISMaLayout

        root = tmp_path / "transform"
        csv = self._marisma(
            root,
            [("A", "A/0_A1", "0_A1"), ("A", "A/0_A2", "0_A2"), ("B", "B/0_B1", "0_B1")],
        )
        layout = MARISMaLayout(
            root, csv, duplicate_strategy="keep_all", id_transform=str
        )
        ds = DatasetLoader(layout, n_jobs=1).load()
        assert sorted(ds.X.index) == ["A_0_A1", "A_0_A2", "B"]
