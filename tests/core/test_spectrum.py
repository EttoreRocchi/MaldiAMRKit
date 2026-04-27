"""Unit tests for MaldiSpectrum class."""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maldiamrkit import MaldiSpectrum
from maldiamrkit.preprocessing import MzTrimmer, PreprocessingPipeline


class TestMaldiSpectrumInit:
    """Tests for MaldiSpectrum initialization."""

    def test_init_from_dataframe(self, synthetic_spectrum: pd.DataFrame):
        """Test initialization from a DataFrame."""
        spec = MaldiSpectrum(synthetic_spectrum)

        assert spec.id == "in-memory"
        assert spec.path is None
        assert len(spec.raw) == len(synthetic_spectrum)
        assert "mass" in spec.raw.columns
        assert "intensity" in spec.raw.columns

    def test_init_from_file(self, real_spectrum_path: Path):
        """Test initialization from a file path."""
        spec = MaldiSpectrum(real_spectrum_path)

        assert spec.id == real_spectrum_path.stem
        assert spec.path == real_spectrum_path
        assert len(spec.raw) > 0

    def test_init_from_string_path(self, real_spectrum_path: Path):
        """Test initialization from a string path."""
        spec = MaldiSpectrum(str(real_spectrum_path))

        assert spec.id == real_spectrum_path.stem
        assert spec.path == real_spectrum_path

    def test_init_with_custom_pipeline(self, synthetic_spectrum: pd.DataFrame):
        """Test initialization with custom preprocessing pipeline."""
        pipe = PreprocessingPipeline.default()
        pipe.steps = [
            (n, s) if not isinstance(s, MzTrimmer) else (n, MzTrimmer(3000, 15000))
            for n, s in pipe.steps
        ]
        spec = MaldiSpectrum(synthetic_spectrum, pipeline=pipe)

        assert spec.pipeline.mz_range == (3000, 15000)

    def test_init_invalid_source_raises(self):
        """Test that invalid source type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported source type"):
            MaldiSpectrum([1, 2, 3])

    def test_raw_returns_copy(self, synthetic_spectrum: pd.DataFrame):
        """Test that raw property returns a copy, not the original."""
        spec = MaldiSpectrum(synthetic_spectrum)
        raw1 = spec.raw
        raw2 = spec.raw

        assert raw1 is not raw2
        raw1.loc[0, "intensity"] = -999
        assert spec.raw.loc[0, "intensity"] != -999


class TestMaldiSpectrumValidation:
    """Tests for MaldiSpectrum DataFrame validation."""

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            MaldiSpectrum(pd.DataFrame())

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing"):
            MaldiSpectrum(pd.DataFrame({"x": [1]}))

    def test_non_numeric_mass_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            MaldiSpectrum(pd.DataFrame({"mass": ["a"], "intensity": [1]}))

    def test_non_numeric_intensity_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            MaldiSpectrum(pd.DataFrame({"mass": [1], "intensity": ["a"]}))


class TestMaldiSpectrumPreprocess:
    """Tests for MaldiSpectrum preprocessing."""

    def test_preprocess_normalizes(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessing normalizes intensities to sum to 1."""
        spec = MaldiSpectrum(synthetic_spectrum).preprocess()

        # Trimmed spectrum should sum to ~1
        assert np.isclose(spec.preprocessed["intensity"].sum(), 1.0, atol=1e-6)

    def test_preprocess_removes_negatives(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessing removes negative intensities."""
        df = synthetic_spectrum.copy()
        df.loc[0, "intensity"] = -100

        spec = MaldiSpectrum(df).preprocess()

        assert (spec.preprocessed["intensity"] >= 0).all()

    def test_preprocess_trims_range(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocessing trims to configured m/z range."""
        pipe = PreprocessingPipeline.default()
        pipe.steps = [
            (n, s) if not isinstance(s, MzTrimmer) else (n, MzTrimmer(3000, 15000))
            for n, s in pipe.steps
        ]
        spec = MaldiSpectrum(synthetic_spectrum, pipeline=pipe).preprocess()

        assert spec.preprocessed["mass"].min() >= 3000
        assert spec.preprocessed["mass"].max() <= 15000

    def test_preprocess_returns_self(self, synthetic_spectrum: pd.DataFrame):
        """Test that preprocess() returns self for chaining."""
        spec = MaldiSpectrum(synthetic_spectrum)
        result = spec.preprocess()

        assert result is spec

    def test_preprocessed_before_preprocess_raises(
        self, synthetic_spectrum: pd.DataFrame
    ):
        """Test that accessing preprocessed before calling preprocess raises."""
        spec = MaldiSpectrum(synthetic_spectrum)

        with pytest.raises(RuntimeError, match="Call .preprocess()"):
            _ = spec.preprocessed


class TestMaldiSpectrumBin:
    """Tests for MaldiSpectrum binning."""

    def test_bin_uniform(self, synthetic_spectrum: pd.DataFrame):
        """Test uniform binning."""
        spec = MaldiSpectrum(synthetic_spectrum).bin(bin_width=3, method="uniform")

        expected_bins = (20000 - 2000) // 3
        assert len(spec.binned) == expected_bins
        assert spec.bin_method == "uniform"
        assert spec.bin_width == 3

    def test_bin_auto_preprocesses(self, synthetic_spectrum: pd.DataFrame):
        """Test that bin() automatically calls preprocess()."""
        spec = MaldiSpectrum(synthetic_spectrum)
        assert spec._preprocessed is None

        spec.bin(bin_width=3)

        assert spec._preprocessed is not None

    def test_bin_returns_self(self, synthetic_spectrum: pd.DataFrame):
        """Test that bin() returns self for chaining."""
        spec = MaldiSpectrum(synthetic_spectrum)
        result = spec.bin(bin_width=3)

        assert result is spec

    def test_bin_preserves_total_intensity(self, synthetic_spectrum: pd.DataFrame):
        """Test that binning preserves total intensity."""
        spec = MaldiSpectrum(synthetic_spectrum).preprocess()
        original_sum = spec.preprocessed["intensity"].sum()

        spec.bin(bin_width=3)
        binned_sum = spec.binned["intensity"].sum()

        # Allow small numerical tolerance
        assert np.isclose(original_sum, binned_sum, rtol=0.01)

    @pytest.mark.parametrize("method", ["uniform", "proportional", "adaptive"])
    def test_bin_methods_valid(self, synthetic_spectrum: pd.DataFrame, method: str):
        """Test all binning methods work."""
        spec = MaldiSpectrum(synthetic_spectrum)

        if method == "adaptive":
            spec.bin(method=method, adaptive_min_width=1.0, adaptive_max_width=10.0)
        else:
            spec.bin(bin_width=3, method=method)

        assert spec._bin_method == method
        assert len(spec.binned) > 0

    def test_bin_custom_edges(self, synthetic_spectrum: pd.DataFrame):
        """Test binning with custom edges."""
        edges = [2000, 5000, 10000, 15000, 20000]
        spec = MaldiSpectrum(synthetic_spectrum).bin(
            method="custom", custom_edges=edges
        )

        assert len(spec.binned) == len(edges) - 1

    def test_binned_before_bin_raises(self, synthetic_spectrum: pd.DataFrame):
        """Test that accessing binned before calling bin raises."""
        spec = MaldiSpectrum(synthetic_spectrum)

        with pytest.raises(RuntimeError, match="Call .bin()"):
            _ = spec.binned

    def test_bin_metadata_available(self, synthetic_spectrum: pd.DataFrame):
        """Test that bin metadata is available after binning."""
        spec = MaldiSpectrum(synthetic_spectrum).bin(bin_width=3)

        meta = spec.bin_metadata
        assert "bin_index" in meta.columns
        assert "bin_start" in meta.columns
        assert "bin_end" in meta.columns
        assert "bin_width" in meta.columns

    def test_bin_metadata_before_bin_raises(self, synthetic_spectrum: pd.DataFrame):
        """Test that accessing bin_metadata before bin raises."""
        spec = MaldiSpectrum(synthetic_spectrum)

        with pytest.raises(RuntimeError, match="Call .bin()"):
            _ = spec.bin_metadata


class TestMaldiSpectrumChaining:
    """Tests for method chaining."""

    def test_chained_preprocess_bin(self, synthetic_spectrum: pd.DataFrame):
        """Test chaining preprocess and bin."""
        spec = MaldiSpectrum(synthetic_spectrum).preprocess().bin(3)

        assert isinstance(spec, MaldiSpectrum)
        assert spec._preprocessed is not None
        assert spec._binned is not None

    def test_chained_from_file(self, real_spectrum_path: Path):
        """Test chaining from file initialization."""
        spec = MaldiSpectrum(real_spectrum_path).preprocess().bin(3)

        assert spec.id == real_spectrum_path.stem
        assert spec._binned is not None


class TestMaldiSpectrumReproducibility:
    """Tests for reproducibility."""

    def test_same_input_same_output(self, synthetic_spectrum: pd.DataFrame):
        """Test that same input produces same output."""
        spec1 = MaldiSpectrum(synthetic_spectrum.copy()).preprocess().bin(3)
        spec2 = MaldiSpectrum(synthetic_spectrum.copy()).preprocess().bin(3)

        pd.testing.assert_frame_equal(spec1.binned, spec2.binned)


class TestMaldiSpectrumSave:
    """Tests for the save method."""

    def test_save_csv(self, synthetic_spectrum: pd.DataFrame, tmp_path: Path):
        """Test saving as CSV."""
        spec = MaldiSpectrum(synthetic_spectrum).preprocess().bin(3)
        out = tmp_path / "spec.csv"
        spec.save(out, stage="binned", fmt="csv")

        assert out.exists()
        df = pd.read_csv(out)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) > 0

    def test_save_txt(self, synthetic_spectrum: pd.DataFrame, tmp_path: Path):
        """Test saving as tab-separated TXT."""
        spec = MaldiSpectrum(synthetic_spectrum).preprocess()
        out = tmp_path / "spec.txt"
        spec.save(out, stage="preprocessed", fmt="txt")

        assert out.exists()
        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) > 0

    def test_save_invalid_fmt(self, synthetic_spectrum: pd.DataFrame, tmp_path: Path):
        """Test that invalid format raises ValueError."""
        spec = MaldiSpectrum(synthetic_spectrum)
        with pytest.raises(ValueError, match="Invalid fmt"):
            spec.save(tmp_path / "spec.dat", stage="raw", fmt="dat")


class TestMaldiSpectrumRepr:
    """Tests for __repr__."""

    def test_repr_raw(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        assert "raw" in repr(spec)
        assert "in-memory" in repr(spec)

    def test_repr_preprocessed(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        spec.preprocess()
        assert "preprocessed" in repr(spec)

    def test_repr_binned(self, synthetic_spectrum: pd.DataFrame):
        spec = MaldiSpectrum(synthetic_spectrum)
        spec.bin(3)
        assert "binned" in repr(spec)
        assert "preprocessed" in repr(spec)


class TestMaldiSpectrumVerbose:
    """Tests for verbose logging."""

    def test_preprocess_verbose(self, synthetic_spectrum: pd.DataFrame, caplog):
        spec = MaldiSpectrum(synthetic_spectrum, verbose=True)
        with caplog.at_level(logging.INFO):
            spec.preprocess()
        assert "Preprocessed spectrum" in caplog.text

    def test_bin_verbose(self, synthetic_spectrum: pd.DataFrame, caplog):
        spec = MaldiSpectrum(synthetic_spectrum, verbose=True)
        with caplog.at_level(logging.INFO):
            spec.bin(3)
        assert "Binned spectrum" in caplog.text


class TestMaldiSpectrumPlot:
    """Tests for the plot_spectrum function."""

    def test_plot_binned(self, synthetic_spectrum: pd.DataFrame):
        from maldiamrkit.visualization import plot_spectrum

        spec = MaldiSpectrum(synthetic_spectrum)
        spec.bin(3)
        fig, ax = plot_spectrum(spec, stage="binned", show=False)
        assert fig is not None and ax is not None
        plt.close("all")

    def test_plot_raw(self, synthetic_spectrum: pd.DataFrame):
        from maldiamrkit.visualization import plot_spectrum

        spec = MaldiSpectrum(synthetic_spectrum)
        fig, ax = plot_spectrum(spec, stage="raw", show=False)
        assert fig is not None and ax is not None
        plt.close("all")

    def test_plot_preprocessed(self, synthetic_spectrum: pd.DataFrame):
        from maldiamrkit.visualization import plot_spectrum

        spec = MaldiSpectrum(synthetic_spectrum)
        spec.preprocess()
        fig, ax = plot_spectrum(spec, stage="preprocessed", show=False)
        assert fig is not None and ax is not None
        plt.close("all")

    def test_plot_with_ax(self, synthetic_spectrum: pd.DataFrame):
        from maldiamrkit.visualization import plot_spectrum

        spec = MaldiSpectrum(synthetic_spectrum)
        spec.bin(3)
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_spectrum(spec, ax=ax_ext, show=False)
        assert ax is ax_ext
        assert fig is fig_ext
        plt.close(fig)


class TestInferId:
    """Tests for the _infer_id helper covering Bruker directory branches."""

    def test_file_path_returns_stem(self, tmp_path):
        """Verify non-directory path returns file stem."""
        f = tmp_path / "spectrum_1.txt"
        f.write_text("1000 100\n")
        from maldiamrkit.spectrum import _infer_id

        assert _infer_id(f) == "spectrum_1"

    def test_dir_no_acqus_returns_name(self, tmp_path):
        """Verify directory without acqus returns dir name."""
        d = tmp_path / "some_dir"
        d.mkdir()
        from maldiamrkit.spectrum import _infer_id

        assert _infer_id(d) == "some_dir"

    def test_bruker_depth3_single_target(self, tmp_path):
        """Verify depth-3 Bruker dir returns name_target."""
        # Structure: identifier / target_pos / 1 / 1SLin / acqus
        identifier = tmp_path / "specimen_A"
        acqus_dir = identifier / "0_A1" / "1" / "1SLin"
        acqus_dir.mkdir(parents=True)
        (acqus_dir / "acqus").write_text("##TITLE= Test\n##END=\n")
        from maldiamrkit.spectrum import _infer_id

        result = _infer_id(identifier)
        assert result == "specimen_A_0_A1"

    def test_bruker_depth3_multiple_targets_warns(self, tmp_path, caplog):
        """Verify warning when multiple target positions found."""
        identifier = tmp_path / "specimen_A"
        for target in ("0_A1", "0_A2"):
            acqus_dir = identifier / target / "1" / "1SLin"
            acqus_dir.mkdir(parents=True)
            (acqus_dir / "acqus").write_text("##TITLE= Test\n##END=\n")
        from maldiamrkit.spectrum import _infer_id

        with caplog.at_level(logging.WARNING):
            result = _infer_id(identifier)
        assert "specimen_A" in result
        assert "Multiple target positions" in caplog.text
