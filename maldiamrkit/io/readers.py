"""File reading utilities for MALDI-TOF spectrum data."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

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
    """
    with open(path, "r", newline="") as f:
        lines = list(itertools.islice(f, sample_lines))
    if not lines:
        raise csv.Error("File is empty, cannot detect delimiter")
    dialect = csv.Sniffer().sniff("".join(lines), delimiters=",;\t ")
    return dialect.delimiter


def read_spectrum(path: str | Path) -> pd.DataFrame:
    """
    Read a raw spectrum file into a DataFrame.

    Reads txt/csv files with two columns (mass and intensity) and
    automatically detects the delimiter.

    Parameters
    ----------
    path : str or Path
        Path to the spectrum file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['mass', 'intensity'].

    Examples
    --------
    >>> from maldiamrkit.io import read_spectrum
    >>> df = read_spectrum("spectrum.txt")
    >>> df.head()
       mass  intensity
    0  2000       1234
    1  2001       1456
    """
    try:
        delim = sniff_delimiter(path)
    except csv.Error:
        delim = r"\s+"

    df = pd.read_csv(
        path, sep=delim, comment="#", header=None, names=["mass", "intensity"]
    )

    return df
