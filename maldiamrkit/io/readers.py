"""File reading utilities for MALDI-TOF spectrum data."""

from __future__ import annotations

import csv
import itertools
import logging
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


def read_spectrum(path: str | Path, *, bruker_source: str = "1r") -> pd.DataFrame:
    """
    Read a raw spectrum file into a DataFrame.

    Supports text-based formats (``.txt``, ``.csv``, ``.tsv``) with
    automatic delimiter detection and Bruker binary directories
    (containing ``fid``/``acqus`` files).  Unrecognised extensions
    are treated as text files with automatic delimiter detection.

    Parameters
    ----------
    path : str or Path
        Path to the spectrum file, or to a Bruker data directory.
    bruker_source : str, default="1r"
        For Bruker directories, which binary to read: ``"1r"``
        (processed) or ``"fid"`` (raw).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['mass', 'intensity']``.

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
    """
    path = Path(path)

    if path.is_dir():
        return _read_bruker(path, source=bruker_source)

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
