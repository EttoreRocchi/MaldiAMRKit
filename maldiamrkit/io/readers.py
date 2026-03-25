"""File reading utilities for MALDI-TOF spectrum data."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


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


def _require_pyteomics(format_name: str) -> None:
    """Raise an informative error if pyteomics is not installed."""
    try:
        import pyteomics  # noqa: F401
    except ImportError:
        raise ImportError(
            f"Reading {format_name} files requires the 'pyteomics' package. "
            "Install it with: pip install maldiamrkit[formats]"
        ) from None


def _import_pyteomics_mzml():
    """Lazily import pyteomics.mzml."""
    _require_pyteomics("mzML")
    from pyteomics import mzml

    return mzml


def _import_pyteomics_mzxml():
    """Lazily import pyteomics.mzxml."""
    _require_pyteomics("mzXML")
    from pyteomics import mzxml

    return mzxml


def _read_mzml(path: str | Path) -> pd.DataFrame:
    """
    Read a spectrum from an mzML file.

    Uses the first spectrum in the file.  For MALDI-TOF data this is
    typically the only scan.

    Parameters
    ----------
    path : str or Path
        Path to the ``.mzML`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['mass', 'intensity']``.

    Raises
    ------
    ImportError
        If ``pyteomics`` is not installed.
    ValueError
        If the file contains no spectra.
    """
    mzml = _import_pyteomics_mzml()

    with mzml.MzML(str(path)) as reader:
        for spectrum in reader:
            mz = np.asarray(spectrum["m/z array"], dtype=np.float64)
            intensity = np.asarray(spectrum["intensity array"], dtype=np.float64)
            return pd.DataFrame({"mass": mz, "intensity": intensity})

    raise ValueError(f"No spectra found in mzML file: {path}")


def _read_mzxml(path: str | Path) -> pd.DataFrame:
    """
    Read a spectrum from an mzXML file.

    Uses the first spectrum in the file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.mzXML`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['mass', 'intensity']``.

    Raises
    ------
    ImportError
        If ``pyteomics`` is not installed.
    ValueError
        If the file contains no spectra.
    """
    mzxml = _import_pyteomics_mzxml()

    with mzxml.MzXML(str(path)) as reader:
        for spectrum in reader:
            mz = np.asarray(spectrum["m/z array"], dtype=np.float64)
            intensity = np.asarray(spectrum["intensity array"], dtype=np.float64)
            return pd.DataFrame({"mass": mz, "intensity": intensity})

    raise ValueError(f"No spectra found in mzXML file: {path}")


_TEXT_EXTENSIONS = {".txt", ".csv", ".tsv"}
_MZML_EXTENSIONS = {".mzml"}
_MZXML_EXTENSIONS = {".mzxml"}


def read_spectrum(path: str | Path) -> pd.DataFrame:
    """
    Read a raw spectrum file into a DataFrame.

    Supports text-based formats (``.txt``, ``.csv``, ``.tsv``) with
    automatic delimiter detection, as well as ``mzML`` and ``mzXML``
    files (requires the ``pyteomics`` package - install with
    ``pip install maldiamrkit[formats]``).  Unrecognised extensions
    are treated as text files with automatic delimiter detection.

    Parameters
    ----------
    path : str or Path
        Path to the spectrum file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['mass', 'intensity']``.

    Raises
    ------
    ImportError
        If an mzML/mzXML file is passed but ``pyteomics`` is not
        installed.

    Examples
    --------
    >>> from maldiamrkit.io import read_spectrum
    >>> df = read_spectrum("spectrum.txt")
    >>> df.head()
       mass  intensity
    0  2000       1234
    1  2001       1456
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in _MZML_EXTENSIONS:
        return _read_mzml(path)

    if suffix in _MZXML_EXTENSIONS:
        return _read_mzxml(path)

    # Default: text-based format (txt, csv, tsv, or any other extension)
    try:
        delim = sniff_delimiter(path)
    except csv.Error:
        delim = r"\s+"

    df = pd.read_csv(
        path, sep=delim, comment="#", header=None, names=["mass", "intensity"]
    )

    return df
