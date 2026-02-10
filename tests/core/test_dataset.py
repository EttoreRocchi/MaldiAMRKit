"""Unit tests for MaldiSet class."""

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from maldiamrkit import MaldiSet, MaldiSpectrum
from maldiamrkit.filters import MetadataFilter, SpeciesFilter
from maldiamrkit.preprocessing import PreprocessingPipeline, get_bin_metadata
from maldiamrkit.preprocessing.binning import _uniform_edges

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binned_spectrum(sid: str = "in-memory", seed: int = 42) -> MaldiSpectrum:
    """Create a MaldiSpectrum with pre-set binned data.

    Bypasses the full preprocessing pipeline to avoid environment-specific
    numpy/coverage issues while still exercising MaldiSet logic.
    """
    rng = np.random.default_rng(seed)

    # Raw spectrum
    mz = np.linspace(2000, 20000, 18000)
    intensity = np.zeros_like(mz)
    for pos in [3000, 5000, 7500, 10000, 12500, 15000]:
        intensity += 1000 * np.exp(-0.5 * ((mz - pos) / 10.0) ** 2)
    intensity += rng.normal(0, 10.0, len(intensity))
    intensity = np.maximum(intensity, 0)
    raw_df = pd.DataFrame({"mass": mz, "intensity": intensity})

    spec = MaldiSpectrum(raw_df)
    spec.id = sid

    # Set preprocessed (use raw for simplicity - we're testing MaldiSet)
    spec._preprocessed = raw_df.copy()

    # Create binned data
    bin_width = 3
    n_bins = (20000 - 2000) // bin_width
    bin_mz = np.array([2000 + i * bin_width + bin_width / 2 for i in range(n_bins)])
    bin_intensity = rng.exponential(0.001, n_bins)
    spec._binned = pd.DataFrame({"mass": bin_mz, "intensity": bin_intensity})
    spec._bin_width = bin_width
    spec._bin_method = "uniform"

    # Create bin metadata
    edges = _uniform_edges(2000, 20000, bin_width)
    spec._bin_metadata = get_bin_metadata(edges)

    return spec


# ---------------------------------------------------------------------------
# TestMaldiSetInit
# ---------------------------------------------------------------------------


class TestMaldiSetInit:
    """Tests for MaldiSet initialization."""

    def test_init_with_spectra_and_meta(self):
        """Test basic initialization with antibiotics and species."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 4)]
        meta = pd.DataFrame(
            {
                "ID": ["1s", "2s", "3s"],
                "Drug": ["S", "R", "R"],
                "Species": ["taxon", "taxon", "taxon"],
            }
        )
        ds = MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )
        assert len(ds.spectra) == 3
        assert ds.antibiotics == ["Drug"]
        assert ds.antibiotic == "Drug"
        assert ds.species == "taxon"

    def test_init_no_aggregate_by(self):
        """Test initialization with aggregate_by=None."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 3)]
        meta = pd.DataFrame({"ID": ["1s", "2s"]})
        ds = MaldiSet(specs, meta)
        assert ds.antibiotics is None
        assert ds.antibiotic is None
        assert ds.species is None
        X = ds.X
        assert X.shape[0] == 2

    def test_init_antibiotics_as_list(self):
        """Test initialization with antibiotics as a list."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame(
            {"ID": ["1s"], "Drug1": ["S"], "Drug2": ["R"], "Species": ["taxon"]}
        )
        ds = MaldiSet(
            [spec],
            meta,
            aggregate_by={"antibiotics": ["Drug1", "Drug2"], "species": "taxon"},
        )
        assert ds.antibiotics == ["Drug1", "Drug2"]
        assert ds.antibiotic == "Drug1"

    def test_init_antibiotic_singular_key(self):
        """Test that 'antibiotic' (singular) key is recognized."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"], "Drug": ["S"]})
        ds = MaldiSet([spec], meta, aggregate_by={"antibiotic": "Drug"})
        assert ds.antibiotics == ["Drug"]

    def test_all_meta_columns_retained(self):
        """Test that all metadata columns are retained regardless of aggregate_by."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame(
            {"ID": ["1s"], "Drug": ["S"], "Species": ["taxon"], "batch": ["A"]}
        )
        ds = MaldiSet(
            [spec],
            meta,
            aggregate_by={"antibiotics": "Drug"},
        )
        assert "Species" in ds.meta.columns
        assert "batch" in ds.meta.columns
        assert "Drug" in ds.meta.columns

    def test_init_missing_columns_verbose(self, caplog):
        """Test that verbose mode logs missing columns."""
        import logging

        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        with caplog.at_level(logging.WARNING, logger="maldiamrkit.dataset"):
            MaldiSet(
                [spec],
                meta,
                aggregate_by={"antibiotics": "NonExistent"},
                verbose=True,
            )
        assert "not found in metadata" in caplog.text

    def test_init_verbose_logging(self, caplog):
        """Test that verbose mode logs dataset creation info."""
        import logging

        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"], "Drug": ["S"], "batch": ["A"]})
        with caplog.at_level(logging.INFO, logger="maldiamrkit.dataset"):
            MaldiSet(
                [spec],
                meta,
                aggregate_by={"antibiotics": "Drug"},
                verbose=True,
            )
        assert "Dataset created: 1 spectra" in caplog.text
        assert "Tracking antibiotics" in caplog.text

    def test_init_stores_binning_params(self):
        """Test that bin_width, bin_method, bin_kwargs are stored."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet(
            [spec],
            meta,
            bin_width=5,
            bin_method="logarithmic",
            bin_kwargs={"foo": "bar"},
        )
        assert ds.bin_width == 5
        assert ds.bin_method == "logarithmic"
        assert ds.bin_kwargs == {"foo": "bar"}


# ---------------------------------------------------------------------------
# TestMaldiSetFromDirectory
# ---------------------------------------------------------------------------


class TestMaldiSetFromDirectory:
    """Tests for MaldiSet.from_directory()."""

    def test_from_directory_loads_spectra(self, spectra_dir: Path, metadata_file: Path):
        """Test loading spectra from directory."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"},
            bin_width=3,
        )
        assert len(ds.spectra) > 0
        for spec in ds.spectra:
            assert spec._binned is not None

    def test_from_directory_respects_bin_width(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test that bin_width parameter is respected."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
            bin_width=5,
        )
        assert ds.bin_width == 5

    def test_from_directory_respects_bin_method(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test that bin_method parameter is respected."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
            bin_method="logarithmic",
        )
        assert ds.bin_method == "logarithmic"

    def test_from_directory_skips_files_not_in_metadata(
        self, spectra_dir: Path, metadata_file: Path, tmp_path: Path
    ):
        """Test that only spectra with IDs in metadata are loaded."""
        import shutil

        temp_dir = tmp_path / "spectra"
        temp_dir.mkdir()
        for f in sorted(spectra_dir.glob("*.txt"))[:3]:
            shutil.copy(f, temp_dir / f.name)
        (temp_dir / "NONEXISTENT_ID.txt").write_text("2000\t100\n2001\t200\n")

        ds = MaldiSet.from_directory(
            temp_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )
        loaded_ids = {s.id for s in ds.spectra}
        assert "NONEXISTENT_ID" not in loaded_ids

    def test_from_directory_no_aggregate_by(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test from_directory with aggregate_by=None loads all matching spectra."""
        ds = MaldiSet.from_directory(spectra_dir, metadata_file)
        assert len(ds.spectra) > 0
        assert ds.antibiotics is None
        assert ds.species is None
        X = ds.X
        assert X.shape[0] > 0

    def test_from_directory_verbose(
        self, spectra_dir: Path, metadata_file: Path, caplog
    ):
        """Test from_directory verbose logging."""
        import logging

        with caplog.at_level(logging.INFO, logger="maldiamrkit.dataset"):
            MaldiSet.from_directory(
                spectra_dir,
                metadata_file,
                aggregate_by={"antibiotics": "Drug"},
                verbose=True,
            )
        assert "Loading" in caplog.text

    def test_from_directory_custom_pipeline(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test from_directory with a custom pipeline."""
        from maldiamrkit.preprocessing import (
            ClipNegatives,
            MzTrimmer,
            TICNormalizer,
        )

        pipe = PreprocessingPipeline(
            [
                ("clip", ClipNegatives()),
                ("trim", MzTrimmer(mz_min=2000, mz_max=20000)),
                ("normalize", TICNormalizer()),
            ]
        )
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
            pipeline=pipe,
        )
        assert len(ds.spectra) > 0


# ---------------------------------------------------------------------------
# TestMaldiSetProperties
# ---------------------------------------------------------------------------


class TestMaldiSetProperties:
    """Tests for MaldiSet properties (X, y, spectra_paths, bin_metadata)."""

    def _make_dataset(self, n=3, species="taxon"):
        """Create a small dataset for property tests."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, n + 1)]
        meta = pd.DataFrame(
            {
                "ID": [f"{i}s" for i in range(1, n + 1)],
                "Drug": ["S", "R", "R"][:n],
                "Species": [species] * n,
            }
        )
        return MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": species}
        )

    def test_X_returns_feature_matrix(self):
        """Test that X property returns a feature matrix."""
        ds = self._make_dataset()
        X = ds.X
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_X_filters_by_species(self):
        """Test X filters by species when specified."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 4)]
        meta = pd.DataFrame(
            {
                "ID": ["1s", "2s", "3s"],
                "Drug": ["S", "R", "R"],
                "Species": ["taxon", "other", "taxon"],
            }
        )
        ds = MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )
        X = ds.X
        assert X.shape[0] == 2  # only 2 spectra match "taxon"

    def test_X_filters_by_antibiotic_notna(self):
        """Test X filters out rows with NaN antibiotic values."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 4)]
        meta = pd.DataFrame(
            {
                "ID": ["1s", "2s", "3s"],
                "Drug": ["S", np.nan, "R"],
                "Species": ["taxon", "taxon", "taxon"],
            }
        )
        ds = MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )
        X = ds.X
        assert X.shape[0] == 2  # only 2 have non-NaN Drug

    def test_X_warns_for_missing_id(self):
        """Test X warns when spectrum ID is not in metadata."""
        specs = [
            _make_binned_spectrum("1s"),
            _make_binned_spectrum("unknown_id", seed=99),
        ]
        meta = pd.DataFrame({"ID": ["1s"], "Drug": ["S"], "Species": ["taxon"]})
        ds = MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )
        with pytest.warns(UserWarning, match="not found in metadata"):
            X = ds.X
        assert X.shape[0] == 1

    def test_X_raises_when_no_match(self):
        """Test X raises ValueError when no spectra match metadata."""
        specs = [_make_binned_spectrum("unknown")]
        meta = pd.DataFrame({"ID": ["other_id"], "Drug": ["S"], "Species": ["taxon"]})
        ds = MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )
        with pytest.raises(ValueError, match="No spectra matched metadata"):
            with pytest.warns(UserWarning):
                _ = ds.X

    def test_X_raises_species_filter_no_match(self):
        """Test X raises when species filter leaves no samples."""
        specs = [_make_binned_spectrum("1s")]
        meta = pd.DataFrame({"ID": ["1s"], "Drug": ["S"], "Species": ["other_species"]})
        ds = MaldiSet(
            specs,
            meta,
            aggregate_by={"antibiotics": "Drug", "species": "nonexistent"},
        )
        with pytest.raises(ValueError, match="No samples remaining"):
            _ = ds.X

    def test_X_lazy_bins_unbinned_spectra(self):
        """Test that X bins spectra on the fly if not already binned."""
        rng = np.random.default_rng(42)
        mz = np.linspace(2000, 20000, 18000)
        intensity = rng.exponential(100, len(mz))
        raw_df = pd.DataFrame({"mass": mz, "intensity": intensity})

        spec = MaldiSpectrum(raw_df)
        spec.id = "1s"
        # Only set preprocessed, not binned
        spec._preprocessed = raw_df.copy()

        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta, bin_width=3)
        X = ds.X
        assert X.shape[0] == 1
        assert X.shape[1] > 0

    def test_y_returns_labels(self):
        """Test that y property returns labels."""
        ds = self._make_dataset()
        y = ds.y
        assert isinstance(y, pd.DataFrame)
        assert "Drug" in y.columns
        assert len(y) == 3

    def test_y_without_antibiotics_raises(self):
        """Test that y raises when no antibiotics specified."""
        specs = [_make_binned_spectrum("1s")]
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet(specs, meta)
        with pytest.raises(ValueError, match="No antibiotics specified"):
            _ = ds.y

    def test_y_antibiotics_not_in_metadata_raises(self):
        """Test that y raises when specified antibiotics not in metadata."""
        specs = [_make_binned_spectrum("1s")]
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet(specs, meta, aggregate_by={"antibiotics": "NonExistent"})
        with pytest.raises(ValueError, match="None of the specified antibiotics"):
            _ = ds.y

    def test_spectra_paths_from_files(self, spectra_dir: Path, metadata_file: Path):
        """Test spectra_paths returns paths for file-loaded spectra."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )
        paths = ds.spectra_paths
        assert isinstance(paths, dict)
        for _sid, path in paths.items():
            assert isinstance(path, Path)
            assert path.exists()

    def test_spectra_paths_in_memory(self):
        """Test spectra_paths excludes in-memory spectra (path=None)."""
        specs = [_make_binned_spectrum("1s")]
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet(specs, meta)
        assert ds.spectra_paths == {}

    def test_bin_metadata_from_spectrum(self):
        """Test bin_metadata returns data from first spectrum."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        bm = ds.bin_metadata
        assert "bin_index" in bm.columns
        assert "bin_start" in bm.columns
        assert "bin_end" in bm.columns
        assert "bin_width" in bm.columns

    def test_bin_metadata_computed_when_not_on_spectrum(self):
        """Test bin_metadata computes from stored params when spectrum has none."""
        spec = _make_binned_spectrum("1s")
        spec._bin_metadata = None  # clear spectrum-level metadata
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta, bin_width=3)
        bm = ds.bin_metadata
        assert "bin_index" in bm.columns
        assert len(bm) > 0

    def test_bin_metadata_computed_no_spectra(self):
        """Test bin_metadata uses default range when no spectra."""
        meta = pd.DataFrame({"ID": []})
        ds = MaldiSet([], meta, bin_width=3)
        bm = ds.bin_metadata
        assert "bin_index" in bm.columns


# ---------------------------------------------------------------------------
# TestMaldiSetGetYSingle
# ---------------------------------------------------------------------------


class TestMaldiSetGetYSingle:
    """Tests for get_y_single method."""

    def test_get_y_single_with_name(self):
        """Test get_y_single with explicit antibiotic name."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"], "Drug": ["S"], "Species": ["taxon"]})
        ds = MaldiSet(
            [spec], meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )
        y = ds.get_y_single("Drug")
        assert isinstance(y, pd.Series)
        assert len(y) == 1

    def test_get_y_single_default_antibiotic(self):
        """Test get_y_single with default (first) antibiotic."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"], "Drug": ["R"], "Species": ["taxon"]})
        ds = MaldiSet(
            [spec], meta, aggregate_by={"antibiotic": "Drug", "species": "taxon"}
        )
        y = ds.get_y_single()
        assert isinstance(y, pd.Series)
        assert y.iloc[0] == "R"

    def test_get_y_single_no_antibiotic_raises(self):
        """Test get_y_single raises when no antibiotic set."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        with pytest.raises(ValueError, match="No antibiotic"):
            ds.get_y_single()

    def test_get_y_single_antibiotic_not_in_meta_raises(self):
        """Test get_y_single raises when antibiotic not in metadata columns."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        with pytest.raises(ValueError, match="not found in metadata"):
            ds.get_y_single("NonExistent")


# ---------------------------------------------------------------------------
# TestMaldiSetFilter
# ---------------------------------------------------------------------------


class TestMaldiSetFilter:
    """Tests for MaldiSet.filter() method."""

    def _make_dataset(self):
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 4)]
        meta = pd.DataFrame(
            {
                "ID": ["1s", "2s", "3s"],
                "Drug": ["S", "R", "R"],
                "Species": ["E. coli", "K. pneumoniae", "E. coli"],
                "batch": ["A", "B", "A"],
            }
        )
        return MaldiSet(
            specs,
            meta,
            aggregate_by={
                "antibiotics": "Drug",
                "species": "E. coli",
            },
        )

    def test_filter_by_species(self):
        """Test filtering by species returns subset."""
        ds = self._make_dataset()
        filtered = ds.filter(SpeciesFilter("E. coli"))
        assert len(filtered.spectra) == 2
        assert all(s.id in ("1s", "3s") for s in filtered.spectra)

    def test_filter_preserves_aggregate_by(self):
        """Test that filter preserves antibiotics/species settings."""
        ds = self._make_dataset()
        filtered = ds.filter(SpeciesFilter("E. coli"))
        assert filtered.antibiotics == ["Drug"]
        assert filtered.species == "E. coli"

    def test_filter_by_metadata(self):
        """Test filtering by arbitrary metadata column."""
        ds = self._make_dataset()
        filtered = ds.filter(MetadataFilter("batch", lambda v: v == "A"))
        assert len(filtered.spectra) == 2

    def test_filter_combined(self):
        """Test combined filter with & operator."""
        ds = self._make_dataset()
        f = SpeciesFilter("E. coli") & MetadataFilter("Drug", lambda v: v == "R")
        filtered = ds.filter(f)
        assert len(filtered.spectra) == 1
        assert filtered.spectra[0].id == "3s"

    def test_filter_empty_result(self):
        """Test filter that matches nothing."""
        ds = self._make_dataset()
        filtered = ds.filter(SpeciesFilter("Nonexistent"))
        assert len(filtered.spectra) == 0

    def test_filter_no_aggregate_by(self):
        """Test filter works without aggregate_by (all columns retained)."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 3)]
        meta = pd.DataFrame({"ID": ["1s", "2s"], "Species": ["E. coli", "Staph"]})
        ds = MaldiSet(specs, meta)
        filtered = ds.filter(SpeciesFilter("E. coli"))
        assert len(filtered.spectra) == 1


# ---------------------------------------------------------------------------
# TestMaldiSetExport
# ---------------------------------------------------------------------------


class TestMaldiSetExport:
    """Tests for to_csv, to_parquet, and __repr__."""

    def _make_dataset(self):
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 3)]
        meta = pd.DataFrame(
            {
                "ID": ["1s", "2s"],
                "Drug": ["S", "R"],
                "Species": ["taxon", "taxon"],
            }
        )
        return MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )

    def test_to_csv(self, tmp_path):
        """Test exporting feature matrix to CSV."""
        ds = self._make_dataset()
        path = tmp_path / "features.csv"
        ds.to_csv(path)
        assert path.exists()
        df = pd.read_csv(path, index_col=0)
        assert df.shape[0] == 2

    def test_to_parquet(self, tmp_path):
        """Test exporting feature matrix to Parquet."""
        ds = self._make_dataset()
        path = tmp_path / "features.parquet"
        ds.to_parquet(path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert df.shape[0] == 2

    def test_repr_with_species_and_antibiotics(self):
        """Test __repr__ with species and antibiotics."""
        ds = self._make_dataset()
        r = repr(ds)
        assert "MaldiSet" in r
        assert "n_spectra=2" in r
        assert "taxon" in r
        assert "Drug" in r

    def test_repr_no_antibiotics(self):
        """Test __repr__ without antibiotics."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        r = repr(ds)
        assert "n_spectra=1" in r
        assert "species='all'" in r


# ---------------------------------------------------------------------------
# TestSaveSpectra
# ---------------------------------------------------------------------------


class TestSaveSpectra:
    """Tests for MaldiSet.save_spectra."""

    def test_save_preprocessed_txt(self, tmp_path):
        """Test saving preprocessed spectra as TXT."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 3)]
        meta = pd.DataFrame({"ID": ["1s", "2s"]})
        ds = MaldiSet(specs, meta)
        out_dir = tmp_path / "preprocessed"
        ds.save_spectra(out_dir, stage="preprocessed", fmt="txt")
        assert out_dir.exists()
        txt_files = list(out_dir.glob("*.txt"))
        assert len(txt_files) == 2
        df = pd.read_csv(txt_files[0], sep="\t")
        assert list(df.columns) == ["mass", "intensity"]

    def test_save_binned_csv(self, tmp_path):
        """Test saving binned spectra as CSV."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 3)]
        meta = pd.DataFrame({"ID": ["1s", "2s"]})
        ds = MaldiSet(specs, meta)
        out_dir = tmp_path / "binned"
        ds.save_spectra(out_dir, stage="binned", fmt="csv")
        csv_files = list(out_dir.glob("*.csv"))
        assert len(csv_files) == 2
        df = pd.read_csv(csv_files[0])
        assert list(df.columns) == ["mass", "intensity"]

    def test_save_raw(self, tmp_path):
        """Test saving raw spectra."""
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 3)]
        meta = pd.DataFrame({"ID": ["1s", "2s"]})
        ds = MaldiSet(specs, meta)
        out_dir = tmp_path / "raw"
        ds.save_spectra(out_dir, stage="raw")
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.txt"))) == 2

    def test_save_creates_directory(self, tmp_path):
        """Test that output directory is created automatically."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        nested = tmp_path / "a" / "b" / "c"
        ds.save_spectra(nested, stage="raw")
        assert nested.exists()

    def test_save_invalid_stage(self, tmp_path):
        """Test that invalid stage raises ValueError."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        with pytest.raises(ValueError, match="Invalid stage"):
            ds.save_spectra(tmp_path, stage="unknown")

    def test_save_invalid_fmt(self, tmp_path):
        """Test that invalid fmt raises ValueError."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        with pytest.raises(ValueError, match="Invalid fmt"):
            ds.save_spectra(tmp_path, stage="raw", fmt="parquet")

    def test_save_skips_missing_stage(self, tmp_path, caplog):
        """Test save_spectra logs warning when stage data missing."""
        import logging

        spec = _make_binned_spectrum("1s")
        spec._preprocessed = None  # remove preprocessed data
        spec._binned = None  # remove binned data
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        with caplog.at_level(logging.WARNING, logger="maldiamrkit.dataset"):
            ds.save_spectra(tmp_path, stage="binned")
        assert (
            "skipped" in caplog.text.lower() or len(list(tmp_path.glob("*.txt"))) == 0
        )

    def test_save_verbose(self, tmp_path, caplog):
        """Test save_spectra verbose logging."""
        import logging

        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta, verbose=True)
        with caplog.at_level(logging.INFO, logger="maldiamrkit.dataset"):
            ds.save_spectra(tmp_path, stage="raw")
        assert "Saved" in caplog.text


# ---------------------------------------------------------------------------
# TestMaldiSetReproducibility
# ---------------------------------------------------------------------------


class TestMaldiSetReproducibility:
    """Tests for reproducibility."""

    def test_same_directory_same_output(self, spectra_dir: Path, metadata_file: Path):
        """Test that loading same directory produces same output."""
        ds1 = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )
        ds2 = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )
        pd.testing.assert_frame_equal(ds1.X.sort_index(), ds2.X.sort_index())


# ---------------------------------------------------------------------------
# TestMaldiSetPlot
# ---------------------------------------------------------------------------


class TestMaldiSetPlot:
    """Tests for plot_pseudogel method."""

    def _make_dataset(self):
        specs = [_make_binned_spectrum(f"{i}s", seed=i) for i in range(1, 5)]
        meta = pd.DataFrame(
            {
                "ID": ["1s", "2s", "3s", "4s"],
                "Drug": ["S", "R", "R", "S"],
                "Species": ["taxon", "taxon", "taxon", "taxon"],
            }
        )
        return MaldiSet(
            specs, meta, aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )

    def test_plot_pseudogel_basic(self):
        """Test basic pseudogel plot."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(show=False)
        assert fig is not None
        assert len(axes) == 2  # R and S groups
        plt.close(fig)

    def test_plot_pseudogel_single_region_tuple(self):
        """Test pseudogel with a single (min, max) region tuple."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(regions=(3000, 5000), show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_multiple_regions(self):
        """Test pseudogel with multiple region filters."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(regions=[(3000, 5000), (8000, 10000)], show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_no_log_scale(self):
        """Test pseudogel without log scale."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(log_scale=False, show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_no_sort(self):
        """Test pseudogel without intensity sorting."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(sort_by_intensity=False, show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_with_title(self):
        """Test pseudogel with custom title."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(title="My Title", show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_custom_figsize(self):
        """Test pseudogel with custom figure size."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(figsize=(12, 6), show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_vmin_vmax(self):
        """Test pseudogel with custom color scale."""
        ds = self._make_dataset()
        fig, axes = ds.plot_pseudogel(vmin=0.0, vmax=5.0, show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_no_antibiotic_raises(self):
        """Test pseudogel raises when no antibiotic defined."""
        spec = _make_binned_spectrum("1s")
        meta = pd.DataFrame({"ID": ["1s"]})
        ds = MaldiSet([spec], meta)
        with pytest.raises(ValueError, match="Antibiotic column not defined"):
            ds.plot_pseudogel(show=False)

    def test_plot_pseudogel_invalid_region_raises(self):
        """Test pseudogel raises for invalid region (min > max)."""
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="Invalid region"):
            ds.plot_pseudogel(regions=[(10000, 3000)], show=False)

    def test_plot_pseudogel_no_mz_in_region_raises(self):
        """Test pseudogel raises when region contains no m/z values."""
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="No m/z values found"):
            ds.plot_pseudogel(regions=[(0, 1)], show=False)

    def test_plot_pseudogel_show(self):
        """Test pseudogel with show=True (mocked)."""
        ds = self._make_dataset()
        with patch.object(plt, "show"):
            fig, axes = ds.plot_pseudogel(show=True)
        assert fig is not None
        plt.close(fig)
