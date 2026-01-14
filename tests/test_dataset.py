"""Unit tests for MaldiSet class."""
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from maldiamrkit import MaldiSet, MaldiSpectrum


class TestMaldiSetInit:
    """Tests for MaldiSet initialization."""

    def test_init_with_spectra_and_meta(
        self, synthetic_spectrum: pd.DataFrame, data_dir: Path
    ):
        """Test basic initialization."""
        # Create a few spectra
        specs = [
            MaldiSpectrum(synthetic_spectrum.copy()).bin(3) for _ in range(3)
        ]
        # Override IDs to match metadata pattern
        specs[0].id = "1s"
        specs[1].id = "2s"
        specs[2].id = "3s"

        meta = pd.DataFrame({
            "ID": ["1s", "2s", "3s"],
            "Drug": ["S", "R", "R"],
            "Species": ["taxon", "taxon", "taxon"],
        })

        ds = MaldiSet(
            specs, meta,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )

        assert len(ds.spectra) == 3
        assert ds.antibiotics == ["Drug"]
        assert ds.species == "taxon"

    def test_init_antibiotics_as_list(self, synthetic_spectrum: pd.DataFrame):
        """Test initialization with antibiotics as a list."""
        specs = [MaldiSpectrum(synthetic_spectrum.copy()).bin(3)]
        specs[0].id = "1s"

        meta = pd.DataFrame({
            "ID": ["1s"],
            "Drug1": ["S"],
            "Drug2": ["R"],
            "Species": ["taxon"],
        })

        ds = MaldiSet(
            specs, meta,
            aggregate_by={"antibiotics": ["Drug1", "Drug2"], "species": "taxon"}
        )

        assert ds.antibiotics == ["Drug1", "Drug2"]


class TestMaldiSetFromDirectory:
    """Tests for MaldiSet.from_directory()."""

    def test_from_directory_loads_spectra(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test loading spectra from directory."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"},
            bin_width=3,
        )

        assert len(ds.spectra) > 0
        # All spectra should be binned
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


class TestMaldiSetProperties:
    """Tests for MaldiSet properties."""

    def test_X_returns_feature_matrix(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test that X property returns a feature matrix."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"},
        )

        X = ds.X
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] > 0  # Has samples
        assert X.shape[1] > 0  # Has features

    def test_X_filters_by_species(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test that X filters by species when specified."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"},
        )

        # All samples should be filtered to 'taxon' species
        X = ds.X
        assert len(X) > 0

    def test_y_returns_labels(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test that y property returns labels."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"},
        )

        y = ds.y
        assert isinstance(y, pd.DataFrame)
        assert "Drug" in y.columns

    def test_y_without_antibiotics_raises(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test that y raises when no antibiotics specified."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={},  # No antibiotics
        )

        with pytest.raises(ValueError, match="No antibiotics specified"):
            _ = ds.y

    def test_get_y_single(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test get_y_single method."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"},
        )

        y = ds.get_y_single("Drug")
        assert isinstance(y, pd.Series)
        assert len(y) == len(ds.X)

    def test_bin_metadata_available(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test that bin_metadata is available."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )

        meta = ds.bin_metadata
        assert "bin_index" in meta.columns
        assert "bin_start" in meta.columns

    def test_spectra_paths(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test spectra_paths property."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )

        paths = ds.spectra_paths
        assert isinstance(paths, dict)
        for sid, path in paths.items():
            assert isinstance(path, Path)
            assert path.exists()


class TestMaldiSetReproducibility:
    """Tests for reproducibility."""

    def test_same_directory_same_output(
        self, spectra_dir: Path, metadata_file: Path
    ):
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

        # X should be identical
        pd.testing.assert_frame_equal(
            ds1.X.sort_index(),
            ds2.X.sort_index()
        )


class TestMaldiSetOther:
    """Tests for other property."""

    def test_other_property(self, synthetic_spectrum: pd.DataFrame):
        """Test other aggregation variables."""
        specs = [MaldiSpectrum(synthetic_spectrum.copy()).bin(3)]
        specs[0].id = "1s"

        meta = pd.DataFrame({
            "ID": ["1s"],
            "Drug": ["S"],
            "Species": ["taxon"],
            "Age": [30],
        })

        ds = MaldiSet(
            specs, meta,
            aggregate_by={
                "antibiotics": "Drug",
                "species": "taxon",
                "other": ["Age"]
            }
        )

        other = ds.other
        assert isinstance(other, pd.DataFrame)
        assert "Age" in other.columns

    def test_other_property_no_key_raises(self, synthetic_spectrum: pd.DataFrame):
        """Test other with no key raises ValueError."""
        specs = [MaldiSpectrum(synthetic_spectrum.copy()).bin(3)]
        specs[0].id = "1s"

        meta = pd.DataFrame({
            "ID": ["1s"],
            "Drug": ["S"],
            "Species": ["taxon"],
        })

        ds = MaldiSet(
            specs, meta,
            aggregate_by={"antibiotics": "Drug", "species": "taxon"}
        )

        with pytest.raises(ValueError, match="No additional"):
            _ = ds.other

    def test_other_property_missing_column_raises(
        self, synthetic_spectrum: pd.DataFrame
    ):
        """Test other with missing column raises ValueError."""
        specs = [MaldiSpectrum(synthetic_spectrum.copy()).bin(3)]
        specs[0].id = "1s"

        meta = pd.DataFrame({
            "ID": ["1s"],
            "Drug": ["S"],
            "Species": ["taxon"],
        })

        ds = MaldiSet(
            specs, meta,
            aggregate_by={
                "antibiotics": "Drug",
                "species": "taxon",
                "other": ["NonExistent"]
            }
        )

        with pytest.raises(ValueError, match="not found"):
            _ = ds.other


class TestMaldiSetGetYSingle:
    """Tests for get_y_single method edge cases."""

    def test_get_y_single_default_antibiotic(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test get_y_single with default antibiotic."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotic": "Drug", "species": "taxon"},
        )
        y = ds.get_y_single()  # Should use default antibiotic
        assert isinstance(y, pd.Series)

    def test_get_y_single_antibiotic_not_set(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test get_y_single raises when antibiotic not set."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={},  # No antibiotic set
        )
        with pytest.raises(ValueError, match="No antibiotic"):
            ds.get_y_single()


class TestMaldiSetPlot:
    """Tests for plot_pseudogel method."""

    def test_plot_pseudogel_basic(self, spectra_dir: Path, metadata_file: Path):
        """Test basic pseudogel plot."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )
        fig, axes = ds.plot_pseudogel(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_with_regions(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test pseudogel with region filtering."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )
        fig, axes = ds.plot_pseudogel(regions=[(3000, 5000), (8000, 10000)], show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pseudogel_no_log_scale(
        self, spectra_dir: Path, metadata_file: Path
    ):
        """Test pseudogel without log scale."""
        ds = MaldiSet.from_directory(
            spectra_dir,
            metadata_file,
            aggregate_by={"antibiotics": "Drug"},
        )
        fig, axes = ds.plot_pseudogel(log_scale=False, show=False)
        assert fig is not None
        plt.close(fig)
