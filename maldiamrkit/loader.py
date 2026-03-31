"""Load DRIAMS-formatted datasets into MaldiSet objects."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from .dataset import MaldiSet
from .spectrum import MaldiSpectrum

logger = logging.getLogger(__name__)


_STAGE_PRIORITY = re.compile(r"^binned_\d+$")


def _detect_stage(driams_dir: Path) -> str:
    """Auto-detect the best available processing stage folder.

    Priority: binned_* > preprocessed > raw.

    Parameters
    ----------
    driams_dir : Path
        Root of the DRIAMS dataset directory.

    Returns
    -------
    str
        Name of the selected stage folder.

    Raises
    ------
    FileNotFoundError
        If no recognised stage folder exists.
    """
    subdirs = {d.name for d in driams_dir.iterdir() if d.is_dir()}

    # Prefer binned folders (pick first alphabetically for determinism)
    binned = sorted(d for d in subdirs if _STAGE_PRIORITY.match(d))
    if binned:
        return binned[0]
    if "preprocessed" in subdirs:
        return "preprocessed"
    if "raw" in subdirs:
        return "raw"
    raise FileNotFoundError(
        f"No recognised stage folder (binned_*, preprocessed, raw) "
        f"found in {driams_dir}. Found: {sorted(subdirs)}"
    )


def _detect_id_column(meta: pd.DataFrame) -> str:
    """Auto-detect the spectrum identifier column in metadata.

    Checks for ``'code'`` (DRIAMS convention), then ``'ID'``,
    then falls back to the first column.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame.

    Returns
    -------
    str
        Name of the identifier column.
    """
    for candidate in ("code", "ID"):
        if candidate in meta.columns:
            return candidate
    return meta.columns[0]


def _discover_metadata(
    id_dir: Path,
    year: str | int | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load and merge metadata CSV(s) from the ``id/`` directory.

    Parameters
    ----------
    id_dir : Path
        The ``id/`` directory inside the DRIAMS dataset.
    year : str, int, or None
        If set, load only this year's metadata.

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        Merged metadata and the list of discovered year strings
        (empty list if the dataset has no year subfolders).

    Raises
    ------
    FileNotFoundError
        If no metadata CSV files are found.
    """
    if year is not None:
        year_str = str(year)
        csv_path = id_dir / year_str / f"{year_str}_clean.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        return pd.read_csv(csv_path), [year_str]

    # Try year-based layout: id/{year}/{year}_clean.csv
    year_dirs = sorted(
        d
        for d in id_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 4
    )

    if year_dirs:
        frames = []
        years = []
        for yd in year_dirs:
            csv_path = yd / f"{yd.name}_clean.csv"
            if csv_path.exists():
                frames.append(pd.read_csv(csv_path))
                years.append(yd.name)
        if not frames:
            raise FileNotFoundError(
                f"Year subdirectories found in {id_dir} but no "
                f"*_clean.csv metadata files inside them."
            )
        return pd.concat(frames, ignore_index=True), years

    # Flat layout: id/{name}_clean.csv
    csv_files = sorted(id_dir.glob("*_clean.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0]), []

    # Last resort: any CSV in id/
    csv_files = sorted(id_dir.glob("*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0]), []

    raise FileNotFoundError(f"No metadata CSV files found in {id_dir}")


def _collect_spectrum_files(
    stage_dir: Path,
    years: list[str],
    year: str | int | None,
) -> list[Path]:
    """Collect spectrum files from a stage folder.

    Parameters
    ----------
    stage_dir : Path
        Path to the stage folder (e.g. ``driams_dir / "binned_6000"``).
    years : list of str
        Year strings discovered from metadata.
    year : str, int, or None
        If set, restrict to this year only.

    Returns
    -------
    list of Path
        Sorted list of spectrum file paths.
    """
    if year is not None:
        subdir = stage_dir / str(year)
        return sorted(subdir.glob("*.txt")) if subdir.is_dir() else []

    if years:
        files: list[Path] = []
        for y in years:
            subdir = stage_dir / y
            if subdir.is_dir():
                files.extend(subdir.glob("*.txt"))
        return sorted(files)

    # Flat layout
    return sorted(stage_dir.glob("*.txt"))


def _load_single(path: Path) -> MaldiSpectrum:
    """Load a single spectrum without preprocessing or binning."""
    return MaldiSpectrum(path)


def load_driams_dataset(
    driams_dir: str | Path,
    *,
    stage: str | None = None,
    year: str | int | None = None,
    id_column: str | None = None,
    aggregate_by: dict[str, str | list[str]] | None = None,
    n_jobs: int = -1,
) -> MaldiSet:
    """Load a DRIAMS-formatted dataset directory into a :class:`MaldiSet`.

    This is the inverse of :func:`build_driams_dataset`.  It reads
    spectra from a chosen processing stage and metadata from the
    ``id/`` directory, returning a ready-to-use dataset object.

    Parameters
    ----------
    driams_dir : str or Path
        Root directory of the DRIAMS dataset (the ``output_dir`` that
        was passed to :func:`build_driams_dataset`).
    stage : str or None
        Processing stage folder to load from, e.g. ``"raw"``,
        ``"preprocessed"``, or ``"binned_6000"``.  If ``None``,
        auto-detects by preferring binned > preprocessed > raw.
    year : str, int, or None
        Load only spectra from this year.  If ``None``, all available
        years are loaded.
    id_column : str or None
        Column name for specimen identifiers in the metadata CSV.
        If ``None``, auto-detects by trying ``"code"`` then ``"ID"``
        then the first column.
    aggregate_by : dict, optional
        Passed through to :class:`MaldiSet`; see its documentation.
    n_jobs : int, default=-1
        Number of parallel jobs for loading spectra.

    Returns
    -------
    MaldiSet
        Dataset with loaded spectra and metadata.

    Raises
    ------
    FileNotFoundError
        If the directory, stage folder, or metadata cannot be found.

    Examples
    --------
    >>> from maldiamrkit import load_driams_dataset
    >>> ds = load_driams_dataset("output/my_dataset")
    >>> ds.X.shape
    (100, 6000)

    >>> ds = load_driams_dataset(
    ...     "output/my_dataset",
    ...     stage="preprocessed",
    ...     year=2024,
    ...     aggregate_by=dict(antibiotics="Ceftriaxone"),
    ... )
    """
    driams_dir = Path(driams_dir)
    if not driams_dir.is_dir():
        raise FileNotFoundError(f"DRIAMS directory not found: {driams_dir}")

    # 1. Resolve stage
    stage_name = stage if stage is not None else _detect_stage(driams_dir)
    stage_dir = driams_dir / stage_name
    if not stage_dir.is_dir():
        raise FileNotFoundError(
            f"Stage folder '{stage_name}' not found in {driams_dir}"
        )

    # 2. Load metadata
    id_dir = driams_dir / "id"
    if not id_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory 'id/' not found in {driams_dir}")

    meta, years = _discover_metadata(id_dir, year)

    # 3. Detect / apply ID column
    detected_col = id_column if id_column is not None else _detect_id_column(meta)
    if detected_col != "ID":
        meta = meta.rename(columns={detected_col: "ID"})
    meta["ID"] = meta["ID"].astype(str)

    # 4. Collect spectrum files
    spectrum_files = _collect_spectrum_files(stage_dir, years, year)
    if not spectrum_files:
        raise FileNotFoundError(f"No .txt spectrum files found in {stage_dir}")

    # 5. Filter to files matching metadata IDs
    meta_ids = set(meta["ID"])
    matched_files = [f for f in spectrum_files if f.stem in meta_ids]

    n_total = len(spectrum_files)
    n_matched = len(matched_files)
    if n_matched == 0:
        raise ValueError(
            f"No spectrum files matched metadata IDs. "
            f"Found {n_total} files and {len(meta_ids)} metadata IDs."
        )
    if n_matched < n_total:
        logger.info(
            "Loading %d/%d spectra (others not in metadata)",
            n_matched,
            n_total,
        )

    # 6. Load spectra in parallel
    spectra = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_load_single)(p) for p in matched_files
    )

    # 7. Build MaldiSet
    return MaldiSet(
        spectra,
        meta,
        aggregate_by=aggregate_by,
    )
