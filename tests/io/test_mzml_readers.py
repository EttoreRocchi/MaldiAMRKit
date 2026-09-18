"""Tests for mzML / mzXML reading via ``read_spectrum``.

The built-in (standard-library) parser is exercised on every run. The
pyteomics code path is exercised only when pyteomics is installed.
"""

from __future__ import annotations

import base64
import gzip
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maldiamrkit.io import read_spectrum
from maldiamrkit.io.readers import _read_mzml_stdlib, _read_mzxml_stdlib

DATA_DIR = Path(__file__).parent / "data"

_has_pyteomics = True
try:
    import pyteomics  # noqa: F401
except ImportError:
    _has_pyteomics = False

_has_pynumpress = True
try:
    import pynumpress  # noqa: F401
except ImportError:
    _has_pynumpress = False

MZ = np.array([2000.0, 5000.0, 8000.0, 12000.0, 18000.0])
INTENSITY = np.array([100.0, 500.0, 1200.0, 300.0, 50.0])


def _b64(values: np.ndarray, fmt: str, compress: bool) -> str:
    raw = struct.pack(f"{fmt[0]}{len(values)}{fmt[1]}", *values)
    if compress:
        raw = zlib.compress(raw)
    return base64.b64encode(raw).decode()


def _mzml_text(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    precision: int = 64,
    compress: bool = False,
    n_spectra: int = 1,
) -> str:
    prec_cv = (
        ("MS:1000523", "64-bit float")
        if precision == 64
        else ("MS:1000521", "32-bit float")
    )
    comp_cv = (
        ("MS:1000574", "zlib compression")
        if compress
        else ("MS:1000576", "no compression")
    )
    fmt = "<d" if precision == 64 else "<f"

    def array(acc: str, name: str, values: np.ndarray) -> str:
        return f"""
          <binaryDataArray encodedLength="0">
            <cvParam cvRef="MS" accession="{acc}" name="{name}"/>
            <cvParam cvRef="MS" accession="{prec_cv[0]}" name="{prec_cv[1]}"/>
            <cvParam cvRef="MS" accession="{comp_cv[0]}" name="{comp_cv[1]}"/>
            <binary>{_b64(values, fmt, compress)}</binary>
          </binaryDataArray>"""

    spectra = "".join(
        f"""
        <spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="{len(mz)}">
          <binaryDataArrayList count="2">{array("MS:1000514", "m/z array", mz + i)}{array("MS:1000515", "intensity array", intensity)}
          </binaryDataArrayList>
        </spectrum>"""
        for i in range(n_spectra)
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
  <mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
    <run id="run">
      <spectrumList count="{n_spectra}">{spectra}
      </spectrumList>
    </run>
  </mzML>
</indexedmzML>
"""


def _mzxml_text(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    precision: int = 32,
    compress: bool = False,
    byte_order: str = "network",
    n_scans: int = 1,
) -> str:
    fmt = (">" if byte_order == "network" else "<") + ("d" if precision == 64 else "f")
    comp_attr = (
        ' compressionType="zlib" compressedLen="0"'
        if compress
        else ' compressionType="none"'
    )

    def peaks(i: int) -> str:
        interleaved = np.empty(2 * len(mz))
        interleaved[0::2] = mz + i
        interleaved[1::2] = intensity
        return _b64(interleaved, fmt, compress)

    scans = "".join(
        f"""
    <scan num="{i + 1}" msLevel="1" peaksCount="{len(mz)}">
      <peaks precision="{precision}" byteOrder="{byte_order}" contentType="m/z-int"{comp_attr}>{peaks(i)}</peaks>
    </scan>"""
        for i in range(n_scans)
    )
    return f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2">
  <msRun scanCount="{n_scans}">{scans}
  </msRun>
</mzXML>
"""


def _numpress_mzml_text(mz: np.ndarray, intensity: np.ndarray) -> str:
    """Build an mzML with numpress-compressed arrays (linear m/z, slof intensity)."""
    import pynumpress

    encoded = {
        "MS:1000514": (
            "m/z array",
            ("MS:1002312", "MS-Numpress linear prediction compression"),
            pynumpress.encode_linear(mz, 100000.0),
        ),
        "MS:1000515": (
            "intensity array",
            ("MS:1002314", "MS-Numpress short logged float compression"),
            pynumpress.encode_slof(
                intensity, pynumpress.optimal_slof_fixed_point(intensity)
            ),
        ),
    }
    arrays = "".join(
        f"""
          <binaryDataArray encodedLength="0">
            <cvParam cvRef="MS" accession="{acc}" name="{name}"/>
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>
            <cvParam cvRef="MS" accession="{comp[0]}" name="{comp[1]}"/>
            <binary>{base64.b64encode(bytes(data)).decode()}</binary>
          </binaryDataArray>"""
        for acc, (name, comp, data) in encoded.items()
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run id="run">
    <spectrumList count="1">
      <spectrum index="0" id="scan=1" defaultArrayLength="{len(mz)}">
        <binaryDataArrayList count="2">{arrays}
        </binaryDataArrayList>
      </spectrum>
    </spectrumList>
  </run>
</mzML>
"""


def _assert_expected(df, mz=MZ, intensity=INTENSITY, rtol=0.0):
    assert list(df.columns) == ["mass", "intensity"]
    assert len(df) == len(mz)
    np.testing.assert_allclose(df["mass"].to_numpy(), mz, rtol=rtol)
    np.testing.assert_allclose(df["intensity"].to_numpy(), intensity, rtol=rtol)


class TestStdlibMzML:
    def test_fixture_file(self):
        _assert_expected(_read_mzml_stdlib(DATA_DIR / "sample.mzML", gzipped=False))

    @pytest.mark.parametrize("precision", [32, 64])
    @pytest.mark.parametrize("compress", [False, True])
    def test_precision_and_compression(
        self, tmp_path: Path, precision: int, compress: bool
    ):
        f = tmp_path / "spec.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY, precision=precision, compress=compress))
        _assert_expected(_read_mzml_stdlib(f, gzipped=False), rtol=1e-6)

    def test_gzipped(self, tmp_path: Path):
        f = tmp_path / "spec.mzML.gz"
        with gzip.open(f, "wt") as fh:
            fh.write(_mzml_text(MZ, INTENSITY))
        _assert_expected(_read_mzml_stdlib(f, gzipped=True))

    def test_first_spectrum_of_many(self, tmp_path: Path):
        f = tmp_path / "multi.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY, n_spectra=3))
        _assert_expected(_read_mzml_stdlib(f, gzipped=False))

    @pytest.mark.parametrize("spectrum", [2, "scan=3"])
    def test_select_spectrum(self, tmp_path: Path, spectrum: int | str):
        f = tmp_path / "multi.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY, n_spectra=3))
        _assert_expected(
            _read_mzml_stdlib(f, gzipped=False, spectrum=spectrum), mz=MZ + 2
        )

    @pytest.mark.parametrize("spectrum", [3, "scan=4"])
    def test_selected_spectrum_not_found_raises(
        self, tmp_path: Path, spectrum: int | str
    ):
        f = tmp_path / "multi.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY, n_spectra=3))
        with pytest.raises(ValueError, match=r"not found .*\(3 spectra\)"):
            _read_mzml_stdlib(f, gzipped=False, spectrum=spectrum)

    def test_no_spectra_raises(self, tmp_path: Path):
        f = tmp_path / "empty.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY, n_spectra=0))
        with pytest.raises(ValueError, match="No spectra found"):
            _read_mzml_stdlib(f, gzipped=False)

    def test_numpress_requires_pyteomics(self, tmp_path: Path):
        text = _mzml_text(MZ, INTENSITY).replace(
            'accession="MS:1000576" name="no compression"',
            'accession="MS:1002312" name="MS-Numpress linear prediction compression"',
        )
        f = tmp_path / "numpress.mzML"
        f.write_text(text)
        with pytest.raises(ValueError, match="pyteomics"):
            _read_mzml_stdlib(f, gzipped=False)

    def test_length_mismatch_raises(self, tmp_path: Path):
        text = _mzml_text(MZ, INTENSITY)
        bad = _b64(INTENSITY[:3], "<d", False)
        good = _b64(INTENSITY, "<d", False)
        f = tmp_path / "bad.mzML"
        f.write_text(text.replace(good, bad))
        with pytest.raises(ValueError, match="differ in length"):
            _read_mzml_stdlib(f, gzipped=False)


class TestStdlibMzXML:
    @pytest.mark.parametrize("precision", [32, 64])
    @pytest.mark.parametrize("compress", [False, True])
    @pytest.mark.parametrize("byte_order", ["network", "little"])
    def test_variants(
        self, tmp_path: Path, precision: int, compress: bool, byte_order: str
    ):
        f = tmp_path / "spec.mzXML"
        f.write_text(
            _mzxml_text(
                MZ,
                INTENSITY,
                precision=precision,
                compress=compress,
                byte_order=byte_order,
            )
        )
        _assert_expected(_read_mzxml_stdlib(f, gzipped=False), rtol=1e-6)

    def test_gzipped(self, tmp_path: Path):
        f = tmp_path / "spec.mzXML.gz"
        with gzip.open(f, "wt") as fh:
            fh.write(_mzxml_text(MZ, INTENSITY))
        _assert_expected(_read_mzxml_stdlib(f, gzipped=True), rtol=1e-6)

    @pytest.mark.parametrize("spectrum", [1, "2"])
    def test_select_scan(self, tmp_path: Path, spectrum: int | str):
        f = tmp_path / "multi.mzXML"
        f.write_text(_mzxml_text(MZ, INTENSITY, precision=64, n_scans=3))
        _assert_expected(
            _read_mzxml_stdlib(f, gzipped=False, spectrum=spectrum), mz=MZ + 1
        )

    def test_selected_scan_not_found_raises(self, tmp_path: Path):
        f = tmp_path / "multi.mzXML"
        f.write_text(_mzxml_text(MZ, INTENSITY, n_scans=3))
        with pytest.raises(ValueError, match="not found"):
            _read_mzxml_stdlib(f, gzipped=False, spectrum="scan=1")

    def test_no_scan_raises(self, tmp_path: Path):
        f = tmp_path / "empty.mzXML"
        f.write_text(
            '<?xml version="1.0"?><mzXML><msRun scanCount="0"></msRun></mzXML>'
        )
        with pytest.raises(ValueError, match="No spectra found"):
            _read_mzxml_stdlib(f, gzipped=False)

    def test_unsupported_content_type_raises(self, tmp_path: Path):
        f = tmp_path / "bad.mzXML"
        f.write_text(
            _mzxml_text(MZ, INTENSITY).replace(
                'contentType="m/z-int"', 'contentType="m/z ruler"'
            )
        )
        with pytest.raises(ValueError, match="contentType"):
            _read_mzxml_stdlib(f, gzipped=False)


class TestDispatchWithoutPyteomics:
    @pytest.mark.parametrize(
        "name", ["spec.mzML", "spec.mzml", "spec.MZML", "spec.mzML.gz"]
    )
    def test_mzml_extensions(self, tmp_path: Path, name: str):
        f = tmp_path / name
        if name.endswith(".gz"):
            with gzip.open(f, "wt") as fh:
                fh.write(_mzml_text(MZ, INTENSITY))
        else:
            f.write_text(_mzml_text(MZ, INTENSITY))
        with patch("maldiamrkit.io.readers._pyteomics", return_value=None):
            _assert_expected(read_spectrum(f))

    @pytest.mark.parametrize("name", ["spec.mzXML", "spec.mzxml", "spec.mzXML.gz"])
    def test_mzxml_extensions(self, tmp_path: Path, name: str):
        f = tmp_path / name
        if name.endswith(".gz"):
            with gzip.open(f, "wt") as fh:
                fh.write(_mzxml_text(MZ, INTENSITY, precision=64))
        else:
            f.write_text(_mzxml_text(MZ, INTENSITY, precision=64))
        with patch("maldiamrkit.io.readers._pyteomics", return_value=None):
            _assert_expected(read_spectrum(f))

    def test_spectrum_is_forwarded(self, tmp_path: Path):
        mzml = tmp_path / "multi.mzML"
        mzml.write_text(_mzml_text(MZ, INTENSITY, n_spectra=3))
        mzxml = tmp_path / "multi.mzXML"
        mzxml.write_text(_mzxml_text(MZ, INTENSITY, precision=64, n_scans=3))
        with patch("maldiamrkit.io.readers._pyteomics", return_value=None):
            _assert_expected(read_spectrum(mzml, spectrum="scan=2"), mz=MZ + 1)
            _assert_expected(read_spectrum(mzxml, spectrum=2), mz=MZ + 2)

    def test_negative_spectrum_raises(self, tmp_path: Path):
        f = tmp_path / "spec.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY))
        with pytest.raises(ValueError, match="Invalid spectrum"):
            read_spectrum(f, spectrum=-1)

    def test_maldispectrum_accepts_mzml(self, tmp_path: Path):
        from maldiamrkit import MaldiSpectrum

        f = tmp_path / "isolate01.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY))
        with patch("maldiamrkit.io.readers._pyteomics", return_value=None):
            spec = MaldiSpectrum(f)
            _assert_expected(spec.raw)
        assert spec.id == "isolate01"


class TestDispatchWithMockedPyteomics:
    """The pyteomics path is preferred when the module can be imported."""

    @staticmethod
    def _mock_module(cls_name: str, spectra: list[dict]) -> MagicMock:
        module = MagicMock()
        reader = MagicMock()
        reader.__enter__ = MagicMock(return_value=iter(spectra))
        reader.__exit__ = MagicMock(return_value=False)
        getattr(module, cls_name).return_value = reader
        return module

    def test_mzml_uses_pyteomics(self, tmp_path: Path):
        module = self._mock_module(
            "MzML", [{"m/z array": MZ, "intensity array": INTENSITY}]
        )
        f = tmp_path / "spec.mzML"
        f.write_text("")
        with patch("maldiamrkit.io.readers._pyteomics", return_value=module):
            _assert_expected(read_spectrum(f))
        module.MzML.assert_called_once()

    def test_mzxml_uses_pyteomics(self, tmp_path: Path):
        module = self._mock_module(
            "MzXML", [{"m/z array": MZ, "intensity array": INTENSITY}]
        )
        f = tmp_path / "spec.mzXML"
        f.write_text("")
        with patch("maldiamrkit.io.readers._pyteomics", return_value=module):
            _assert_expected(read_spectrum(f))
        module.MzXML.assert_called_once()

    def test_empty_pyteomics_reader_raises(self, tmp_path: Path):
        module = self._mock_module("MzML", [])
        f = tmp_path / "empty.mzML"
        f.write_text("")
        with patch("maldiamrkit.io.readers._pyteomics", return_value=module):
            with pytest.raises(ValueError, match="No spectra found"):
                read_spectrum(f)

    @pytest.mark.parametrize(
        ("cls_name", "name", "id_key", "native_ids"),
        [
            ("MzML", "spec.mzML", "id", ["scan=1", "scan=2", "scan=3"]),
            ("MzXML", "spec.mzXML", "num", ["1", "2", "3"]),
        ],
    )
    def test_select_spectrum_by_native_id(
        self,
        tmp_path: Path,
        cls_name: str,
        name: str,
        id_key: str,
        native_ids: list[str],
    ):
        spectra = [
            {id_key: native_id, "m/z array": MZ + i, "intensity array": INTENSITY}
            for i, native_id in enumerate(native_ids)
        ]
        module = self._mock_module(cls_name, spectra)
        f = tmp_path / name
        f.write_text("")
        with patch("maldiamrkit.io.readers._pyteomics", return_value=module):
            _assert_expected(read_spectrum(f, spectrum=native_ids[1]), mz=MZ + 1)


class TestTextFormatsUnchanged:
    def test_txt_file(self, tmp_path: Path):
        f = tmp_path / "spec.txt"
        f.write_text("2000\t100\n2001\t200\n2002\t300\n")
        df = read_spectrum(f)
        assert list(df.columns) == ["mass", "intensity"]
        assert len(df) == 3

    def test_csv_file(self, tmp_path: Path):
        f = tmp_path / "spec.csv"
        f.write_text("2000,100\n2001,200\n2002,300\n")
        assert len(read_spectrum(f)) == 3


@pytest.mark.skipif(not _has_pyteomics, reason="pyteomics not installed")
class TestPyteomicsIntegration:
    def test_fixture_file(self):
        _assert_expected(read_spectrum(DATA_DIR / "sample.mzML"))

    @pytest.mark.parametrize("compress", [False, True])
    def test_generated_mzml(self, tmp_path: Path, compress: bool):
        f = tmp_path / "spec.mzML"
        f.write_text(_mzml_text(MZ, INTENSITY, compress=compress))
        _assert_expected(read_spectrum(f))

    def test_generated_mzxml(self, tmp_path: Path):
        f = tmp_path / "spec.mzXML"
        f.write_text(_mzxml_text(MZ, INTENSITY, precision=64, compress=True))
        _assert_expected(read_spectrum(f))

    def test_select_spectrum(self, tmp_path: Path):
        mzml = tmp_path / "multi.mzML"
        mzml.write_text(_mzml_text(MZ, INTENSITY, n_spectra=3))
        mzxml = tmp_path / "multi.mzXML"
        mzxml.write_text(_mzxml_text(MZ, INTENSITY, precision=64, n_scans=3))
        _assert_expected(read_spectrum(mzml, spectrum="scan=3"), mz=MZ + 2)
        _assert_expected(read_spectrum(mzxml, spectrum="2"), mz=MZ + 1)
        with pytest.raises(ValueError, match="not found"):
            read_spectrum(mzml, spectrum=3)

    @pytest.mark.skipif(not _has_pynumpress, reason="pynumpress not installed")
    def test_numpress_arrays(self, tmp_path: Path):
        f = tmp_path / "numpress.mzML"
        f.write_text(_numpress_mzml_text(MZ, INTENSITY))
        _assert_expected(read_spectrum(f), rtol=1e-3)
        with pytest.raises(ValueError, match="pyteomics"):
            _read_mzml_stdlib(f, gzipped=False)

    def test_gzipped_mzml(self, tmp_path: Path):
        f = tmp_path / "spec.mzML.gz"
        with gzip.open(f, "wt") as fh:
            fh.write(_mzml_text(MZ, INTENSITY))
        _assert_expected(read_spectrum(f))
