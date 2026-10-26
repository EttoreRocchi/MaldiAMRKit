"""Metadata CSV reading shared by the input and dataset layouts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _detect_id_column(meta: pd.DataFrame) -> str:
    """Auto-detect the ID column: ``'code'`` > ``'ID'`` > first column."""
    for candidate in ("code", "ID"):
        if candidate in meta.columns:
            return candidate
    return meta.columns[0]


def _read_metadata_csv(
    path: str | Path, id_column: str | None = None, **kwargs
) -> pd.DataFrame:
    """Read a metadata CSV with its ID columns as text.

    Reading the IDs as text keeps numeric and zero-padded IDs (``"007"``)
    exactly as written, so they match the spectrum file names. This holds
    for the ID column and for the isolate columns that built datasets
    record (``_original_id``, ``_isolate``). When ``id_column`` is None,
    the column is chosen with :func:`_detect_id_column`; extra keyword
    arguments go to :func:`pandas.read_csv`.
    """
    header = pd.read_csv(path, nrows=0)
    column = id_column if id_column is not None else _detect_id_column(header)
    text = [c for c in (column, "_original_id", "_isolate") if c in header.columns]
    return pd.read_csv(path, dtype=dict.fromkeys(text, str) or None, **kwargs)
