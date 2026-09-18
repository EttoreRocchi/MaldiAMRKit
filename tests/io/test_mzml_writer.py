"""Tests for mzML writing."""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.io import read_spectrum
from maldiamrkit.io.writers import _write_mzml

_has_pyteomics = True
try:
    import pyteomics  # noqa: F401
except ImportError:
    _has_pyteomics = False

_has_lxml = True
try:
    from lxml import etree
except ImportError:
    _has_lxml = False

_MZML_XSD_URL = (
    "https://raw.githubusercontent.com/HUPO-PSI/mzML/"
    "33340a8cdb7ef441e290272d5895d17cc258c5e7/schema/schema_1.1/mzML1.1.0.xsd"
)

SPECTRUM = pd.DataFrame(
    {
        "mass": [2000.0, 5000.123456789, 8000.0, 12000.0, 18000.0],
        "intensity": [100.0, 500.5, 1e-9, 300.0, 0.0],
    }
)


@pytest.fixture(scope="module")
def mzml_schema():
    """Official mzML 1.1.0 schema, downloaded from the HUPO-PSI repository."""
    try:
        with urllib.request.urlopen(_MZML_XSD_URL, timeout=10) as response:
            return etree.XMLSchema(etree.fromstring(response.read()))
    except OSError as exc:
        pytest.skip(f"Could not download the mzML schema: {exc}")


def _read_stdlib(path: Path) -> pd.DataFrame:
    with patch("maldiamrkit.io.readers._pyteomics", return_value=None):
        return read_spectrum(path)


class TestWriteMzML:
    """Tests for the mzML writer."""

    def test_round_trip(self, tmp_path: Path):
        """Test that written arrays are read back exactly."""
        f = tmp_path / "spec.mzML"
        _write_mzml(SPECTRUM, f)
        pd.testing.assert_frame_equal(_read_stdlib(f), SPECTRUM)

    def test_gzipped(self, tmp_path: Path):
        """Test that a .gz suffix writes a gzipped file."""
        f = tmp_path / "spec.mzML.gz"
        _write_mzml(SPECTRUM, f)
        with gzip.open(f, "rb") as fh:
            assert fh.read(5) == b"<?xml"
        pd.testing.assert_frame_equal(_read_stdlib(f), SPECTRUM)

    def test_string_masses(self, tmp_path: Path):
        """Test that bin labels stored as strings are written as floats."""
        df = SPECTRUM.assign(mass=SPECTRUM["mass"].astype(str))
        f = tmp_path / "binned.mzML"
        _write_mzml(df, f)
        pd.testing.assert_frame_equal(_read_stdlib(f), SPECTRUM)

    def test_single_spectrum_metadata(self, tmp_path: Path):
        """Test the spectrum id and array length written to the file."""
        f = tmp_path / "spec.mzML"
        _write_mzml(SPECTRUM, f)
        text = f.read_text()
        assert 'id="scan=1"' in text
        assert f'defaultArrayLength="{len(SPECTRUM)}"' in text

    @pytest.mark.skipif(not _has_lxml, reason="lxml not installed")
    def test_valid_against_schema(self, tmp_path: Path, mzml_schema):
        """Test that the written file validates against the mzML 1.1.0 schema."""
        f = tmp_path / "spec.mzML"
        _write_mzml(SPECTRUM, f)
        assert mzml_schema.validate(etree.parse(str(f))), mzml_schema.error_log

    @pytest.mark.skipif(not _has_pyteomics, reason="pyteomics not installed")
    def test_readable_by_pyteomics(self, tmp_path: Path):
        """Test that pyteomics reads the written file."""
        f = tmp_path / "spec.mzML"
        _write_mzml(SPECTRUM, f)
        df = read_spectrum(f)
        np.testing.assert_array_equal(df["mass"], SPECTRUM["mass"])
        np.testing.assert_array_equal(df["intensity"], SPECTRUM["intensity"])
