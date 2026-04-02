"""Tests for Bruker fid/1r binary reading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from maldiamrkit.io.readers import (
    _find_bruker_acqus,
    _parse_acqus,
    _read_bruker,
    _tof_to_mass,
    read_spectrum,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BRUKER_SAMPLE = DATA_DIR / "bruker_sample" / "922145" / "0_G10"


@pytest.fixture()
def synthetic_acqus(tmp_path: Path) -> Path:
    """Create a minimal acqus file with known calibration params."""
    content = (
        "##TITLE= Test\n"
        "##$TD= 100\n"
        "##$DELAY= 19000\n"
        "##$DW= 2\n"
        "##$ML1= 5000000.0\n"
        "##$ML2= 400.0\n"
        "##$ML3= 0\n"
        "##$BYTORDA= 0\n"
    )
    acqus = tmp_path / "acqus"
    acqus.write_text(content)
    return acqus


@pytest.fixture()
def synthetic_bruker_dir(tmp_path: Path, synthetic_acqus: Path) -> Path:
    """Create a minimal Bruker directory with acqus + fid + 1r."""
    td = 100
    # Write fid
    fid = np.arange(td, dtype=np.int32)
    fid.tofile(tmp_path / "fid")
    # Write 1r in pdata/1/
    pdata = tmp_path / "pdata" / "1"
    pdata.mkdir(parents=True)
    (fid * 2).tofile(pdata / "1r")
    return tmp_path


class TestParseAcqus:
    """Tests for JCAMP-DX acqus parsing."""

    def test_parses_all_required_params(self, synthetic_acqus):
        params = _parse_acqus(synthetic_acqus)
        assert params["TD"] == 100
        assert params["DELAY"] == 19000
        assert params["DW"] == 2.0
        assert params["ML1"] == 5000000.0
        assert params["ML2"] == 400.0
        assert params["ML3"] == 0.0
        assert params["BYTORDA"] == 0

    def test_missing_param_raises(self, tmp_path):
        acqus = tmp_path / "acqus"
        acqus.write_text("##$TD= 100\n##$DELAY= 19000\n")
        with pytest.raises(ValueError, match="Missing required"):
            _parse_acqus(acqus)


class TestTofToMass:
    """Tests for TOF-to-mass calibration."""

    def test_linear_calibration(self):
        """ML3=0 should use the linear path."""
        tof = np.array([19000.0, 19100.0, 19200.0])
        mass = _tof_to_mass(5000000.0, 400.0, 0.0, tof)
        assert mass.shape == (3,)
        assert np.all(mass > 0)
        # Mass should increase with TOF
        assert np.all(np.diff(mass) > 0)

    def test_quadratic_calibration(self):
        """ML3 != 0 should use the quadratic path."""
        tof = np.array([19000.0, 19100.0, 19200.0])
        mass = _tof_to_mass(5000000.0, 400.0, -0.01, tof)
        assert mass.shape == (3,)
        assert np.all(mass > 0)


class TestFindBrukerAcqus:
    """Tests for acqus file discovery."""

    def test_finds_direct(self, synthetic_bruker_dir):
        result = _find_bruker_acqus(synthetic_bruker_dir)
        assert result is not None
        assert result.name == "acqus"

    def test_finds_nested(self, tmp_path):
        nested = tmp_path / "1" / "1SLin"
        nested.mkdir(parents=True)
        (nested / "acqus").write_text("##$TD= 100\n")
        result = _find_bruker_acqus(tmp_path)
        assert result is not None
        assert result == nested / "acqus"

    def test_returns_none_if_missing(self, tmp_path):
        assert _find_bruker_acqus(tmp_path) is None


class TestReadBruker:
    """Tests for reading Bruker binary data."""

    def test_reads_fid(self, synthetic_bruker_dir):
        df = _read_bruker(synthetic_bruker_dir, source="fid")
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 100
        assert df["mass"].iloc[0] > 0
        assert (df["intensity"] >= 0).all()

    def test_reads_1r(self, synthetic_bruker_dir):
        df = _read_bruker(synthetic_bruker_dir, source="1r")
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 100

    def test_fid_and_1r_differ(self, synthetic_bruker_dir):
        fid = _read_bruker(synthetic_bruker_dir, source="fid")
        one_r = _read_bruker(synthetic_bruker_dir, source="1r")
        # Same mass axis
        np.testing.assert_array_equal(fid["mass"], one_r["mass"])
        # Different intensities (1r = fid * 2 in our fixture)
        assert not np.array_equal(fid["intensity"], one_r["intensity"])

    def test_invalid_source_raises(self, synthetic_bruker_dir):
        with pytest.raises(ValueError, match="Unknown source"):
            _read_bruker(synthetic_bruker_dir, source="invalid")

    def test_missing_acqus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No 'acqus'"):
            _read_bruker(tmp_path)


class TestReadSpectrumBruker:
    """Tests for read_spectrum() Bruker directory dispatch."""

    def test_dispatches_to_bruker_for_directory(self, synthetic_bruker_dir):
        df = read_spectrum(synthetic_bruker_dir)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 100

    def test_bruker_source_parameter(self, synthetic_bruker_dir):
        df = read_spectrum(synthetic_bruker_dir, bruker_source="fid")
        assert len(df) == 100

    def test_text_files_still_work(self):
        """Ensure existing text dispatch is unchanged."""
        df = read_spectrum(DATA_DIR / "1s.txt")
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) > 0


@pytest.mark.skipif(
    not BRUKER_SAMPLE.is_dir(),
    reason="Real Bruker sample data not available",
)
class TestRealBrukerSample:
    """Integration tests with the real Bruker sample data."""

    def test_read_real_1r(self):
        df = read_spectrum(BRUKER_SAMPLE)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) > 10000
        assert df["mass"].iloc[0] > 1000
        assert df["mass"].iloc[-1] < 25000

    def test_read_real_fid(self):
        df = read_spectrum(BRUKER_SAMPLE, bruker_source="fid")
        assert len(df) > 10000

    def test_maldi_spectrum_from_bruker(self):
        from maldiamrkit import MaldiSpectrum

        spec = MaldiSpectrum(BRUKER_SAMPLE)
        assert spec.id == "922145_0_G10"
        assert spec.raw.shape[0] > 10000
