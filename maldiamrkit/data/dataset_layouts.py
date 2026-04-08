"""Dataset layout adapters for navigating and loading datasets.

DatasetLayouts describe **how to navigate a dataset** for loading.
They are consumed by :class:`DatasetLoader` to discover metadata and
spectrum files from different dataset structures.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..io.readers import _find_bruker_acqus
from .duplicates import DuplicateStrategy, apply_metadata_strategy

logger = logging.getLogger(__name__)

_STAGE_PRIORITY = re.compile(r"^binned_\d+$")


class DatasetLayout(ABC):
    """Abstract adapter for navigating and loading from a dataset."""

    @abstractmethod
    def discover_metadata(self) -> pd.DataFrame:
        """Load metadata, return DataFrame with ``'ID'`` column."""

    @abstractmethod
    def collect_spectrum_files(
        self,
        stage: str | None,
        year: str | int | None,
    ) -> list[Path]:
        """Return paths to spectrum files for the given stage/year."""

    @abstractmethod
    def detect_stage(self) -> str:
        """Auto-detect best available processing stage."""


class DRIAMSLayout(DatasetLayout):
    """Navigate a DRIAMS-like dataset structure.

    Works with both the output of :class:`DatasetBuilder` and the
    original DRIAMS-A/B/C/D datasets.

    Parameters
    ----------
    dataset_dir : str or Path
        Root of the dataset.
    id_column : str or None
        Metadata column for spectrum IDs.  ``None`` triggers
        auto-detection (``'code'`` > ``'ID'`` > first column).
    species_column : str or None
        Metadata column for species names.  ``None`` triggers
        auto-detection (case-insensitive match for ``'species'``).
        The column is renamed to ``'Species'`` for downstream use.
    year : str, int, or None
        Restrict to a single year.
    metadata_dir : str, default="id"
        Subdirectory name containing metadata CSV files.
    metadata_suffix : str, default="_clean.csv"
        Filename suffix for metadata CSV files.
    spectrum_ext : str, default=".txt"
        File extension for spectrum files (including the dot).
    duplicate_strategy : str or DuplicateStrategy, default ``"first"``
        How to handle duplicate spectrum IDs (e.g. the same sample
        appearing in multiple year subdirectories):

        * ``"first"``  -- keep the first occurrence (default).
        * ``"last"``   -- keep the last occurrence.
        * ``"drop"``   -- remove all duplicates.
        * ``"keep_all"`` -- keep every replicate with ``_repN`` suffixes.
        * ``"average"`` -- tag replicates for downstream averaging.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        id_column: str | None = None,
        species_column: str | None = None,
        year: str | int | None = None,
        metadata_dir: str = "id",
        metadata_suffix: str = "_clean.csv",
        spectrum_ext: str = ".txt",
        duplicate_strategy: str | DuplicateStrategy = DuplicateStrategy.first,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.id_column = id_column
        self.species_column = species_column
        self.year = str(year) if year is not None else None
        self.metadata_dir = metadata_dir
        self.metadata_suffix = metadata_suffix
        self.spectrum_ext = spectrum_ext
        self.duplicate_strategy = DuplicateStrategy(duplicate_strategy)

    def discover_metadata(self) -> pd.DataFrame:
        """Load metadata CSV(s) from the metadata directory."""
        id_dir = self.dataset_dir / self.metadata_dir
        if not id_dir.is_dir():
            raise FileNotFoundError(
                f"Metadata directory '{self.metadata_dir}/' not found "
                f"in {self.dataset_dir}"
            )

        meta, _ = _discover_driams_metadata(id_dir, self.year, self.metadata_suffix)
        col = self.id_column or _detect_id_column(meta)
        if col != "ID":
            meta = meta.rename(columns={col: "ID"})
        meta["ID"] = meta["ID"].astype(str)
        meta = apply_metadata_strategy(meta, self.duplicate_strategy)

        species_col = self.species_column or _detect_species_column(meta)
        if species_col is not None and species_col != "Species":
            meta = meta.rename(columns={species_col: "Species"})

        return meta

    def collect_spectrum_files(
        self,
        stage: str | None,
        year: str | int | None,
    ) -> list[Path]:
        """Glob spectrum files from the stage directory."""
        stage_name = stage or self.detect_stage()
        stage_dir = self.dataset_dir / stage_name
        if not stage_dir.is_dir():
            raise FileNotFoundError(
                f"Stage folder '{stage_name}' not found in {self.dataset_dir}"
            )

        ext_glob = f"*{self.spectrum_ext}"
        yr = str(year) if year is not None else self.year

        if yr is not None:
            subdir = stage_dir / yr
            return sorted(subdir.glob(ext_glob)) if subdir.is_dir() else []

        # Try year subfolders
        year_dirs = sorted(
            d
            for d in stage_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 4
        )
        if year_dirs:
            files: list[Path] = []
            for yd in year_dirs:
                files.extend(yd.glob(ext_glob))
            return sorted(files)

        return sorted(stage_dir.glob(ext_glob))

    def detect_stage(self) -> str:
        """Auto-detect: binned_* > preprocessed > raw."""
        subdirs = {d.name for d in self.dataset_dir.iterdir() if d.is_dir()}
        binned = sorted(d for d in subdirs if _STAGE_PRIORITY.match(d))
        if binned:
            return binned[0]
        if "preprocessed" in subdirs:
            return "preprocessed"
        if "raw" in subdirs:
            return "raw"
        raise FileNotFoundError(
            f"No recognised stage folder in {self.dataset_dir}. "
            f"Found: {sorted(subdirs)}"
        )


class MARISMaLayout(DatasetLayout):
    """Navigate a dataset of raw Bruker spectra organised in a tree.

    Load spectra directly from Bruker binary files without requiring
    a build step.  The metadata CSV must contain a column with
    relative paths pointing to the Bruker data directories.

    Parameters
    ----------
    root_dir : str or Path
        Root directory of the dataset.
    metadata_csv : str or Path
        Path to the metadata CSV.
    id_column : str, default="Identifier"
        Column for specimen identifier.
    path_column : str, default="Path"
        Column with relative path to the Bruker directory.
    target_position_column : str, default="target_position"
        Column for the plate target position.
    duplicate_strategy : str or DuplicateStrategy, default ``"first"``
        How to handle duplicate specimen identifiers (e.g. the same
        sample measured at multiple MALDI target positions):

        * ``"first"``  -- keep the first occurrence (default).
        * ``"last"``   -- keep the last occurrence.
        * ``"drop"``   -- remove all duplicates.
        * ``"keep_all"`` -- keep every replicate, appending the
          target-position value to the ID
          (``{identifier}_{target_position}``).
        * ``"average"`` -- tag replicates for downstream averaging
          (adds ``_original_id`` column).
    year : str, int, or None
        Restrict to a single year.
    """

    def __init__(
        self,
        root_dir: str | Path,
        metadata_csv: str | Path,
        *,
        id_column: str = "Identifier",
        path_column: str = "Path",
        target_position_column: str = "target_position",
        duplicate_strategy: str | DuplicateStrategy = DuplicateStrategy.first,
        year: str | int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.metadata_csv = Path(metadata_csv)
        self.id_column = id_column
        self.path_column = path_column
        self.target_position_column = target_position_column
        self.duplicate_strategy = DuplicateStrategy(duplicate_strategy)
        self.year = str(year) if year is not None else None

    def discover_metadata(self) -> pd.DataFrame:
        """Read metadata CSV and normalise the ID column."""
        meta = pd.read_csv(self.metadata_csv)
        if self.id_column not in meta.columns:
            raise ValueError(
                f"ID column '{self.id_column}' not in metadata. "
                f"Available: {list(meta.columns)}"
            )
        if self.id_column != "ID":
            meta = meta.rename(columns={self.id_column: "ID"})
        meta["ID"] = meta["ID"].astype(str)

        meta = apply_metadata_strategy(
            meta,
            self.duplicate_strategy,
            suffix_col=self.target_position_column,
        )

        if self.year is not None and "Year" in meta.columns:
            meta = meta[meta["Year"].astype(str) == self.year]

        return meta.reset_index(drop=True)

    def collect_spectrum_files(
        self,
        stage: str | None,
        year: str | int | None,
    ) -> list[Path]:
        """Resolve Bruker directories from metadata Path column.

        The ``stage`` parameter is ignored (only raw Bruker available).
        """
        meta = self.discover_metadata()
        yr = str(year) if year is not None else self.year

        if yr is not None and "Year" in meta.columns:
            meta = meta[meta["Year"].astype(str) == yr]

        paths: list[Path] = []
        for _, row in meta.iterrows():
            bruker_dir = _resolve_metadata_path(
                self.root_dir, str(row[self.path_column])
            )
            if bruker_dir is None or not bruker_dir.is_dir():
                continue
            acqus = _find_bruker_acqus(bruker_dir)
            if acqus is not None:
                paths.append(bruker_dir)

        return sorted(paths)

    def detect_stage(self) -> str:
        """Return ``'raw'`` as the only available stage."""
        return "raw"


def _resolve_metadata_path(root_dir: Path, raw_path: str) -> Path | None:
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


def _detect_id_column(meta: pd.DataFrame) -> str:
    """Auto-detect the ID column: ``'code'`` > ``'ID'`` > first column."""
    for candidate in ("code", "ID"):
        if candidate in meta.columns:
            return candidate
    return meta.columns[0]


def _detect_species_column(meta: pd.DataFrame) -> str | None:
    """Auto-detect a species column via case-insensitive match."""
    if "Species" in meta.columns:
        return "Species"
    for col in meta.columns:
        if col.lower() == "species":
            return col
    return None


def _discover_driams_metadata(
    id_dir: Path,
    year: str | None,
    metadata_suffix: str = "_clean.csv",
) -> tuple[pd.DataFrame, list[str]]:
    """Load and merge metadata CSV(s) from a metadata directory."""
    if year is not None:
        csv_path = id_dir / year / f"{year}{metadata_suffix}"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        return pd.read_csv(csv_path), [year]

    year_dirs = sorted(
        d
        for d in id_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 4
    )

    if year_dirs:
        return _load_year_metadata(id_dir, year_dirs, metadata_suffix)

    return _load_flat_metadata(id_dir, metadata_suffix)


def _load_year_metadata(
    id_dir: Path, year_dirs: list[Path], suffix: str
) -> tuple[pd.DataFrame, list[str]]:
    """Load metadata from year-based subdirectories."""
    frames = []
    years = []
    for yd in year_dirs:
        csv_path = yd / f"{yd.name}{suffix}"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path, low_memory=False))
            years.append(yd.name)
    if not frames:
        raise FileNotFoundError(
            f"Year subdirectories found in {id_dir} but no *{suffix} files inside them."
        )
    return pd.concat(frames, ignore_index=True), years


def _load_flat_metadata(id_dir: Path, suffix: str) -> tuple[pd.DataFrame, list[str]]:
    """Load metadata from flat CSV files (no year structure)."""
    csv_files = sorted(id_dir.glob(f"*{suffix}"))
    if csv_files:
        return pd.read_csv(csv_files[0]), []

    csv_files = sorted(id_dir.glob("*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0]), []

    raise FileNotFoundError(f"No metadata CSV files found in {id_dir}")
