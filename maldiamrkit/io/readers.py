"""File reading utilities for MALDI-TOF spectrum data."""

from __future__ import annotations

import base64
import csv
import itertools
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sniff_delimiter(path: str | Path, sample_lines: int = 10) -> str:
    """
    Detect the delimiter used in a text file.

    Parameters
    ----------
    path : str or Path
        Path to the file to analyze.
    sample_lines : int, default=10
        Number of lines to sample for delimiter detection.

    Returns
    -------
    str
        Detected delimiter character.

    Raises
    ------
    csv.Error
        If the file is empty and the delimiter cannot be detected.
    """
    with open(path, "r", newline="") as f:
        lines = list(itertools.islice(f, sample_lines))
    if not lines:
        raise csv.Error("File is empty, cannot detect delimiter")
    dialect = csv.Sniffer().sniff("".join(lines), delimiters=",;\t ")
    return dialect.delimiter


_BRUKER_REQUIRED_PARAMS = {"TD", "DELAY", "DW", "ML1", "ML2", "ML3", "BYTORDA"}

_ACQUS_PARAM_TYPES: dict[str, type] = {
    "TD": int,
    "DELAY": int,
    "DW": float,
    "ML1": float,
    "ML2": float,
    "ML3": float,
    "BYTORDA": int,
}


def _parse_acqus(acqus_path: Path) -> dict[str, int | float]:
    """
    Parse a Bruker JCAMP-DX ``acqus`` file for calibration parameters.

    Parameters
    ----------
    acqus_path : Path
        Path to the ``acqus`` file.

    Returns
    -------
    dict[str, int | float]
        Dictionary with keys ``TD``, ``DELAY``, ``DW``, ``ML1``,
        ``ML2``, ``ML3``, ``BYTORDA``.

    Raises
    ------
    ValueError
        If any required parameter is missing from the file.
    """
    params: dict[str, int | float] = {}
    with open(acqus_path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("##$"):
                continue
            try:
                key_part, value_part = line.split("= ", 1)
            except ValueError:
                continue
            key = key_part[3:]  # strip '##$'
            if key in _ACQUS_PARAM_TYPES:
                params[key] = _ACQUS_PARAM_TYPES[key](value_part)

    missing = _BRUKER_REQUIRED_PARAMS - params.keys()
    if missing:
        raise ValueError(
            f"Missing required parameters in {acqus_path}: {sorted(missing)}"
        )
    return params


def _tof_to_mass(ml1: float, ml2: float, ml3: float, tof: np.ndarray) -> np.ndarray:
    """
    Convert time-of-flight values to m/z using Bruker calibration.

    Implements the quadratic TOF calibration equation used by Bruker
    flexAnalysis (reference: MARISMa ``SpectrumObject.tof2mass``).

    The calibration solves ``a*x^2 + b*x + c = 0`` where ``a = ML3``,
    ``b = sqrt(1e12 / ML1)``, ``c = ML2 - TOF``, and
    ``m/z = ((-b + sqrt(b^2 - 4ac)) / (2a))^2``.  When ``ML3 = 0``
    (linear calibration), this degenerates to ``m/z = (c / b)^2``.
    The ``-b + sqrt(...)`` root is selected to match the MARISMa
    reference implementation; this yields physical (positive) m/z for
    the typical case where ``ML3 >= 0`` and ``ML1 > 0``.

    Parameters
    ----------
    ml1, ml2, ml3 : float
        Bruker calibration constants from the ``acqus`` file.
    tof : np.ndarray
        Time-of-flight array.

    Returns
    -------
    np.ndarray
        Mass-to-charge (m/z) array.

    Raises
    ------
    ValueError
        If ``ml1`` is not positive, or if the quadratic discriminant
        is negative (invalid calibration constants).
    """
    if ml1 <= 0:
        raise ValueError(
            f"Bruker calibration constant ML1 must be positive, got {ml1}. "
            f"Check the acqus file for corrupt or missing calibration data."
        )

    a = ml3
    b = np.sqrt(1e12 / ml1)
    c = ml2 - tof

    if a == 0:
        return (c * c) / (b * b)

    discriminant = b * b - 4 * a * c
    if np.any(discriminant < 0):
        n_neg = np.sum(discriminant < 0)
        raise ValueError(
            f"Bruker calibration produced negative discriminant for "
            f"{n_neg}/{len(tof)} TOF values (ML1={ml1}, ML2={ml2}, "
            f"ML3={ml3}). This indicates invalid calibration constants."
        )

    return ((-b + np.sqrt(discriminant)) / (2 * a)) ** 2


def _read_bruker_binary(path: Path, byte_order: int, n_points: int) -> np.ndarray:
    """
    Read a Bruker binary file (``fid`` or ``1r``) as an int32 array.

    Parameters
    ----------
    path : Path
        Path to the binary file.
    byte_order : int
        Byte order flag from ``BYTORDA``: 0 = little-endian,
        1 = big-endian.
    n_points : int
        Expected number of data points (``TD``).

    Returns
    -------
    np.ndarray
        Intensity array with negatives clipped to zero.
    """
    dtype = np.dtype("<i4") if byte_order == 0 else np.dtype(">i4")
    data = np.fromfile(path, dtype=dtype)
    data = data[:n_points]
    data = data.astype(np.float64)
    data[data < 0] = 0
    return data


def _find_bruker_acqus(path: Path) -> Path | None:
    """
    Find the ``acqus`` file within a Bruker directory.

    Searches progressively deeper: ``path/acqus``, ``path/*/acqus``,
    ``path/*/*/acqus``.

    Parameters
    ----------
    path : Path
        Root directory to search.

    Returns
    -------
    Path or None
        Path to the ``acqus`` file, or ``None`` if not found.
    """
    direct = path / "acqus"
    if direct.is_file():
        return direct
    for depth in ("*", "*/*", "*/*/*"):
        matches = sorted(path.glob(f"{depth}/acqus"))
        if matches:
            return matches[0]
    return None


def _read_bruker(path: Path, *, source: str = "1r") -> pd.DataFrame:
    """
    Read a Bruker MALDI-TOF spectrum directory.

    Parameters
    ----------
    path : Path
        Path to a Bruker data directory (containing ``acqus`` and
        ``fid``/``pdata`` either directly or in a subdirectory).
    source : str, default="1r"
        Which binary to read: ``"1r"`` for the processed spectrum
        from ``pdata/1/1r``, or ``"fid"`` for the raw free induction
        decay.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['mass', 'intensity']``.

    Raises
    ------
    FileNotFoundError
        If the ``acqus`` file or the requested binary cannot be found.
    ValueError
        If required calibration parameters are missing.
    """
    acqus_path = _find_bruker_acqus(path)
    if acqus_path is None:
        raise FileNotFoundError(
            f"No 'acqus' file found in {path} or its subdirectories"
        )
    acqus_dir = acqus_path.parent
    params = _parse_acqus(acqus_path)

    td = int(params["TD"])
    delay = int(params["DELAY"])
    dw = float(params["DW"])
    ml1 = float(params["ML1"])
    ml2 = float(params["ML2"])
    ml3 = float(params["ML3"])
    byte_order = int(params["BYTORDA"])

    tof = delay + np.arange(td) * dw
    mass = _tof_to_mass(ml1, ml2, ml3, tof)

    if source == "1r":
        binary_path = acqus_dir / "pdata" / "1" / "1r"
    elif source == "fid":
        binary_path = acqus_dir / "fid"
    else:
        raise ValueError(f"Unknown source '{source}', expected '1r' or 'fid'")

    if not binary_path.is_file():
        raise FileNotFoundError(f"Binary file not found: {binary_path}")

    intensity = _read_bruker_binary(binary_path, byte_order, td)

    return pd.DataFrame({"mass": mass, "intensity": intensity})


_MZML_EXTENSIONS = {".mzml"}
_MZXML_EXTENSIONS = {".mzxml"}

_CV_MZ_ARRAY = "MS:1000514"
_CV_INTENSITY_ARRAY = "MS:1000515"
_CV_FLOAT32 = "MS:1000521"
_CV_FLOAT64 = "MS:1000523"
_CV_ZLIB = "MS:1000574"
_CV_NUMPRESS = frozenset(
    {"MS:1002312", "MS:1002313", "MS:1002314", "MS:1002746", "MS:1002747", "MS:1002748"}
)


def _format_suffixes(path: Path) -> tuple[str, bool]:
    """Return ``(format_extension, gzipped)`` for ``path``, lower-cased."""
    suffixes = [s.lower() for s in path.suffixes]
    gzipped = bool(suffixes) and suffixes[-1] == ".gz"
    if gzipped:
        suffixes = suffixes[:-1]
    return (suffixes[-1] if suffixes else ""), gzipped


def _open_binary(path: Path, gzipped: bool):
    if gzipped:
        import gzip

        return gzip.open(path, "rb")
    return open(path, "rb")


def _local_name(tag: str) -> str:
    """Strip the XML namespace from an element tag."""
    return tag.rsplit("}", 1)[-1]


def _decode_binary(text: str, *, dtype: np.dtype, compressed: bool) -> np.ndarray:
    raw = base64.b64decode("".join(text.split()))
    if compressed:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=dtype).astype(np.float64)


def _is_selected(spectrum: int | str, position: int, native_id: str | None) -> bool:
    """Whether the spectrum at ``position`` with ``native_id`` is the requested one."""
    if isinstance(spectrum, str):
        return native_id == spectrum
    return position == spectrum


def _spectrum_not_found(
    path: Path, fmt: str, spectrum: int | str, n_spectra: int
) -> ValueError:
    if n_spectra == 0:
        return ValueError(f"No spectra found in {fmt} file: {path}")
    return ValueError(
        f"Spectrum {spectrum!r} not found in {fmt} file: {path} ({n_spectra} spectra)"
    )


def _pyteomics(module_name: str):
    """Return ``pyteomics.<module_name>``, or ``None`` when pyteomics is absent."""
    try:
        import importlib

        return importlib.import_module(f"pyteomics.{module_name}")
    except ImportError:
        return None


def _select_spectrum_pyteomics(
    reader, path: Path, fmt: str, *, spectrum: int | str, id_key: str
) -> pd.DataFrame:
    """Return the selected spectrum yielded by a pyteomics reader as a DataFrame."""
    n_spectra = 0
    with reader as spectra:
        for entry in spectra:
            n_spectra += 1
            if not _is_selected(spectrum, n_spectra - 1, entry.get(id_key)):
                continue
            mz = np.asarray(entry["m/z array"], dtype=np.float64)
            intensity = np.asarray(entry["intensity array"], dtype=np.float64)
            return pd.DataFrame({"mass": mz, "intensity": intensity})
    raise _spectrum_not_found(path, fmt, spectrum, n_spectra)


def _read_mzml_stdlib(
    path: Path, gzipped: bool, *, spectrum: int | str = 0
) -> pd.DataFrame:
    """Dependency-free mzML reader (zlib or uncompressed arrays)."""
    import xml.etree.ElementTree as ET

    n_spectra = 0
    with _open_binary(path, gzipped) as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            if _local_name(elem.tag) != "spectrum":
                continue
            n_spectra += 1
            if not _is_selected(spectrum, n_spectra - 1, elem.get("id")):
                elem.clear()
                continue
            arrays: dict[str, np.ndarray] = {}
            for bda in elem.iter():
                if _local_name(bda.tag) != "binaryDataArray":
                    continue
                accessions = {
                    c.get("accession")
                    for c in bda.iter()
                    if _local_name(c.tag) == "cvParam"
                }
                binary = next(
                    (c for c in bda.iter() if _local_name(c.tag) == "binary"), None
                )
                if binary is None:
                    continue
                if accessions & _CV_NUMPRESS:
                    raise ValueError(
                        f"{path}: numpress-compressed mzML arrays require pyteomics "
                        "(pip install maldiamrkit[formats])"
                    )
                if _CV_MZ_ARRAY in accessions:
                    key = "mass"
                elif _CV_INTENSITY_ARRAY in accessions:
                    key = "intensity"
                else:
                    continue
                dtype = (
                    np.dtype("<f4") if _CV_FLOAT32 in accessions else np.dtype("<f8")
                )
                arrays[key] = _decode_binary(
                    binary.text or "", dtype=dtype, compressed=_CV_ZLIB in accessions
                )
            if "mass" not in arrays or "intensity" not in arrays:
                raise ValueError(
                    f"{path}: spectrum {spectrum!r} has no m/z and intensity arrays"
                )
            if len(arrays["mass"]) != len(arrays["intensity"]):
                raise ValueError(f"{path}: m/z and intensity arrays differ in length")
            return pd.DataFrame(
                {"mass": arrays["mass"], "intensity": arrays["intensity"]}
            )
    raise _spectrum_not_found(path, "mzML", spectrum, n_spectra)


def _read_mzxml_stdlib(
    path: Path, gzipped: bool, *, spectrum: int | str = 0
) -> pd.DataFrame:
    """Dependency-free mzXML reader (zlib or uncompressed peaks)."""
    import xml.etree.ElementTree as ET

    n_spectra = 0
    with _open_binary(path, gzipped) as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            if _local_name(elem.tag) != "scan":
                continue
            n_spectra += 1
            if not _is_selected(spectrum, n_spectra - 1, elem.get("num")):
                elem.clear()
                continue
            peaks = next(
                (c for c in elem.iter() if _local_name(c.tag) == "peaks"), None
            )
            if peaks is None or not (peaks.text or "").strip():
                raise ValueError(f"{path}: spectrum {spectrum!r} has no peaks")
            content = peaks.get("contentType") or peaks.get("pairOrder") or "m/z-int"
            if content not in ("m/z-int", "mz-int"):
                raise ValueError(
                    f"{path}: unsupported mzXML peaks contentType {content!r}"
                )
            precision = int(peaks.get("precision", "32"))
            byte_order = ">" if peaks.get("byteOrder", "network") == "network" else "<"
            compressed = (peaks.get("compressionType") or "none").lower() == "zlib"
            dtype = np.dtype(byte_order + ("f8" if precision == 64 else "f4"))
            data = _decode_binary(peaks.text, dtype=dtype, compressed=compressed)
            if data.size % 2:
                raise ValueError(f"{path}: odd number of values in mzXML peaks array")
            return pd.DataFrame({"mass": data[0::2], "intensity": data[1::2]})
    raise _spectrum_not_found(path, "mzXML", spectrum, n_spectra)


def _read_mzml(
    path: Path, *, gzipped: bool = False, spectrum: int | str = 0
) -> pd.DataFrame:
    """Read one spectrum of an mzML file (pyteomics if available)."""
    mzml = _pyteomics("mzml")
    if mzml is None:
        return _read_mzml_stdlib(path, gzipped, spectrum=spectrum)
    fh = _open_binary(path, gzipped)
    try:
        return _select_spectrum_pyteomics(
            mzml.MzML(fh, use_index=False),
            path,
            "mzML",
            spectrum=spectrum,
            id_key="id",
        )
    finally:
        fh.close()


def _read_mzxml(
    path: Path, *, gzipped: bool = False, spectrum: int | str = 0
) -> pd.DataFrame:
    """Read one scan of an mzXML file (pyteomics if available)."""
    mzxml = _pyteomics("mzxml")
    if mzxml is None:
        return _read_mzxml_stdlib(path, gzipped, spectrum=spectrum)
    fh = _open_binary(path, gzipped)
    try:
        return _select_spectrum_pyteomics(
            mzxml.MzXML(fh, use_index=False),
            path,
            "mzXML",
            spectrum=spectrum,
            id_key="num",
        )
    finally:
        fh.close()


def read_spectrum(
    path: str | Path, *, bruker_source: str = "1r", spectrum: int | str = 0
) -> pd.DataFrame:
    """
    Read a raw spectrum file into a DataFrame.

    Supports text-based formats (``.txt``, ``.csv``, ``.tsv``) with
    automatic delimiter detection, the open ``.mzML`` and ``.mzXML``
    formats (optionally gzipped, e.g. ``.mzML.gz``), and Bruker binary
    directories (containing ``fid``/``acqus`` files).  Unrecognised
    extensions are treated as text files with automatic delimiter
    detection.

    mzML/mzXML files are parsed with a built-in reader (zlib or
    uncompressed 32/64-bit arrays).  If ``pyteomics`` is installed
    (``pip install maldiamrkit[formats]``) it is used instead and
    additionally handles numpress-compressed arrays.  The first
    spectrum of the file is returned unless ``spectrum`` selects another.

    Parameters
    ----------
    path : str or Path
        Path to the spectrum file, or to a Bruker data directory.
    bruker_source : str, default="1r"
        For Bruker directories, which binary to read: ``"1r"``
        (processed) or ``"fid"`` (raw).
    spectrum : int or str, default=0
        For mzML/mzXML files, which spectrum to read: its 0-based position
        in the file, or its native id (the mzML spectrum ``id``, e.g.
        ``"scan=1"``, or the mzXML scan ``num``, e.g. ``"1"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['mass', 'intensity']``.

    Raises
    ------
    ValueError
        If ``spectrum`` is a negative position, or if the requested
        spectrum is not found in an mzML/mzXML file.

    Examples
    --------
    >>> from maldiamrkit.io import read_spectrum
    >>> df = read_spectrum("spectrum.txt")
    >>> df.head()
       mass  intensity
    0  2000       1234
    1  2001       1456

    Read a Bruker directory:

    >>> df = read_spectrum("path/to/bruker_dir")
    >>> df = read_spectrum("path/to/bruker_dir", bruker_source="fid")

    Read an open-format file:

    >>> df = read_spectrum("spectrum.mzML")
    >>> df = read_spectrum("spectrum.mzXML.gz")
    >>> df = read_spectrum("run.mzML", spectrum="scan=2")
    """
    if isinstance(spectrum, int) and spectrum < 0:
        raise ValueError(
            f"Invalid spectrum {spectrum}. Use a non-negative position or a native id."
        )
    path = Path(path)

    if path.is_dir():
        return _read_bruker(path, source=bruker_source)

    fmt, gzipped = _format_suffixes(path)
    if fmt in _MZML_EXTENSIONS:
        return _read_mzml(path, gzipped=gzipped, spectrum=spectrum)
    if fmt in _MZXML_EXTENSIONS:
        return _read_mzxml(path, gzipped=gzipped, spectrum=spectrum)

    # Default: text-based format (txt, csv, tsv, or any other extension)
    try:
        delim = sniff_delimiter(path)
    except csv.Error:
        delim = r"\s+"

    df = pd.read_csv(
        path, sep=delim, comment="#", header=None, names=["mass", "intensity"]
    )

    df = _coerce_numeric(df)

    # If sniffed delimiter produced no valid rows, retry with whitespace regex.
    if df.empty and delim != r"\s+":
        df = pd.read_csv(
            path, sep=r"\s+", comment="#", header=None, names=["mass", "intensity"]
        )
        df = _coerce_numeric(df)

    if df.empty:
        raise ValueError(f"No valid numeric data in {path}")

    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mass/intensity to numeric and drop unparseable rows."""
    df = df.copy()
    df["mass"] = pd.to_numeric(df["mass"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    return df.dropna(subset=["mass", "intensity"]).reset_index(drop=True)
