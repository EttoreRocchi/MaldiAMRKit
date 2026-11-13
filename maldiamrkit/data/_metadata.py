"""Metadata CSV reading and path resolution shared by the layouts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .duplicates import _merge_spectrum_rows


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


def _read_bruker_metadata(
    path: str | Path,
    id_column: str,
    path_column: str,
    target_position_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read the metadata CSV of a Bruker tree, one row per spectrum.

    The ID column is renamed to ``"ID"`` and read as text, and the rows
    of one spectrum are merged with :func:`_merge_spectrum_rows`, whose
    counts are returned with the metadata.
    ``low_memory=False`` keeps pandas from flagging as mixed the columns
    that hold S/I/R calls next to MIC values (as in MARISMa's AMR.csv).
    """
    meta = _read_metadata_csv(path, id_column, low_memory=False)
    if id_column not in meta.columns:
        raise ValueError(
            f"ID column '{id_column}' not in metadata. Available: {list(meta.columns)}"
        )
    if id_column != "ID":
        meta = meta.rename(columns={id_column: "ID"})
    meta["ID"] = meta["ID"].astype(str)
    return _merge_spectrum_rows(meta, path_column, target_position_column)


def _resolve_metadata_path(root_dir: Path, raw_path: str) -> Path:
    """Resolve a metadata path against *root_dir*, stripping overlapping prefixes.

    Metadata may store absolute-looking paths (e.g. ``/MARISMa/2024/...``)
    whose leading segments duplicate part of *root_dir*.  This helper tries
    the literal join first, then progressively strips leading components
    (up to 3) until the resolved path exists on disk.
    """
    rel = raw_path.lstrip("/")
    candidate = root_dir / rel
    if candidate.exists():
        return candidate

    parts = Path(rel).parts
    for n in range(1, min(len(parts), 4)):
        if parts[n - 1].lower() == root_dir.name.lower():
            trimmed = Path(*parts[n:]) if n < len(parts) else Path()
            candidate = root_dir / trimmed
            if candidate.exists():
                return candidate

    return root_dir / rel
