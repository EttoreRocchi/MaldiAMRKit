"""Tests for mzML/mzXML file reading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maldiamrkit.io import read_spectrum

DATA_DIR = Path(__file__).parent / "data"

_has_pyteomics = True
try:
    import pyteomics  # noqa: F401
except ImportError:
    _has_pyteomics = False


def _make_mock_spectrum(mz: np.ndarray, intensity: np.ndarray) -> dict:
    """Create a mock spectrum dict matching pyteomics output."""
    return {"m/z array": mz, "intensity array": intensity}


def _make_mock_mzml_module(spectra: list[dict]) -> MagicMock:
    """Create a mock pyteomics.mzml module."""
    mock_module = MagicMock()
    mock_reader = MagicMock()
    mock_reader.__enter__ = MagicMock(return_value=iter(spectra))
    mock_reader.__exit__ = MagicMock(return_value=False)
    mock_module.MzML.return_value = mock_reader
    return mock_module


def _make_mock_mzxml_module(spectra: list[dict]) -> MagicMock:
    """Create a mock pyteomics.mzxml module."""
    mock_module = MagicMock()
    mock_reader = MagicMock()
    mock_reader.__enter__ = MagicMock(return_value=iter(spectra))
    mock_reader.__exit__ = MagicMock(return_value=False)
    mock_module.MzXML.return_value = mock_reader
    return mock_module


class TestReadMzML:
    """Tests for mzML file reading via read_spectrum()."""

    def test_dispatch_by_extension(self, tmp_path: Path):
        """read_spectrum dispatches to _read_mzml for .mzML files."""
        mz = np.array([2000.0, 3000.0, 4000.0])
        intensity = np.array([100.0, 200.0, 300.0])
        mock_spec = _make_mock_spectrum(mz, intensity)
        mock_mzml = _make_mock_mzml_module([mock_spec])

        fake_file = tmp_path / "test.mzML"
        fake_file.write_text("")

        with patch(
            "maldiamrkit.io.readers._import_pyteomics_mzml",
            return_value=mock_mzml,
        ):
            df = read_spectrum(fake_file)

        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 3
        np.testing.assert_array_equal(df["mass"].values, mz)
        np.testing.assert_array_equal(df["intensity"].values, intensity)

    def test_lowercase_extension(self, tmp_path: Path):
        """read_spectrum handles .mzml (lowercase) extension."""
        mz = np.array([2000.0, 3000.0])
        intensity = np.array([100.0, 200.0])
        mock_spec = _make_mock_spectrum(mz, intensity)
        mock_mzml = _make_mock_mzml_module([mock_spec])

        fake_file = tmp_path / "test.mzml"
        fake_file.write_text("")

        with patch(
            "maldiamrkit.io.readers._import_pyteomics_mzml",
            return_value=mock_mzml,
        ):
            df = read_spectrum(fake_file)

        assert len(df) == 2

    def test_missing_pyteomics_raises(self, tmp_path: Path):
        """ImportError with install instructions when pyteomics missing."""
        fake_file = tmp_path / "test.mzML"
        fake_file.write_text("")

        with patch(
            "maldiamrkit.io.readers._import_pyteomics_mzml",
            side_effect=ImportError(
                "Reading mzML files requires the 'pyteomics' package. "
                "Install it with: pip install maldiamrkit[formats]"
            ),
        ):
            with pytest.raises(ImportError, match="maldiamrkit\\[formats\\]"):
                read_spectrum(fake_file)

    def test_empty_file_raises(self, tmp_path: Path):
        """ValueError when mzML file contains no spectra."""
        mock_mzml = _make_mock_mzml_module([])

        fake_file = tmp_path / "empty.mzML"
        fake_file.write_text("")

        with patch(
            "maldiamrkit.io.readers._import_pyteomics_mzml",
            return_value=mock_mzml,
        ):
            with pytest.raises(ValueError, match="No spectra found"):
                read_spectrum(fake_file)


class TestReadMzXML:
    """Tests for mzXML file reading via read_spectrum()."""

    def test_dispatch_by_extension(self, tmp_path: Path):
        """read_spectrum dispatches to _read_mzxml for .mzXML files."""
        mz = np.array([2000.0, 3000.0, 4000.0])
        intensity = np.array([100.0, 200.0, 300.0])
        mock_spec = _make_mock_spectrum(mz, intensity)
        mock_mzxml = _make_mock_mzxml_module([mock_spec])

        fake_file = tmp_path / "test.mzXML"
        fake_file.write_text("")

        with patch(
            "maldiamrkit.io.readers._import_pyteomics_mzxml",
            return_value=mock_mzxml,
        ):
            df = read_spectrum(fake_file)

        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 3
        np.testing.assert_array_equal(df["mass"].values, mz)
        np.testing.assert_array_equal(df["intensity"].values, intensity)

    def test_lowercase_extension(self, tmp_path: Path):
        """read_spectrum handles .mzxml (lowercase) extension."""
        mz = np.array([5000.0, 6000.0])
        intensity = np.array([50.0, 60.0])
        mock_spec = _make_mock_spectrum(mz, intensity)
        mock_mzxml = _make_mock_mzxml_module([mock_spec])

        fake_file = tmp_path / "test.mzxml"
        fake_file.write_text("")

        with patch(
            "maldiamrkit.io.readers._import_pyteomics_mzxml",
            return_value=mock_mzxml,
        ):
            df = read_spectrum(fake_file)

        assert len(df) == 2


class TestTextFormatUnchanged:
    """Ensure text-based formats still work after dispatch changes."""

    def test_txt_file(self, tmp_path: Path):
        f = tmp_path / "spec.txt"
        f.write_text("2000\t100\n2001\t200\n2002\t300\n")
        df = read_spectrum(f)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 3

    def test_csv_file(self, tmp_path: Path):
        f = tmp_path / "spec.csv"
        f.write_text("2000,100\n2001,200\n2002,300\n")
        df = read_spectrum(f)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 3

    def test_comment_header(self, tmp_path: Path):
        f = tmp_path / "spec.txt"
        f.write_text("# mass intensity\n2000 100\n2001 200\n")
        df = read_spectrum(f)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 2


@pytest.mark.skipif(not _has_pyteomics, reason="pyteomics not installed")
class TestMzMLIntegration:
    """End-to-end tests using the real mzML fixture file."""

    EXPECTED_MZ = np.array([2000.0, 5000.0, 8000.0, 12000.0, 18000.0])
    EXPECTED_INTENSITY = np.array([100.0, 500.0, 1200.0, 300.0, 50.0])

    def test_read_sample_mzml(self):
        df = read_spectrum(DATA_DIR / "sample.mzML")
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 5
        np.testing.assert_allclose(df["mass"].values, self.EXPECTED_MZ)
        np.testing.assert_allclose(df["intensity"].values, self.EXPECTED_INTENSITY)

    def test_maldi_spectrum_from_mzml(self):
        from maldiamrkit import MaldiSpectrum

        spec = MaldiSpectrum(DATA_DIR / "sample.mzML")
        assert spec.id == "sample"
        assert len(spec.raw) == 5
        np.testing.assert_allclose(spec.raw["mass"].values, self.EXPECTED_MZ)
