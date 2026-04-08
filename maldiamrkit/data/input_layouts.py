"""Input layout adapters for discovering spectra and metadata.

InputLayouts describe **how to read source data** for building.
They are consumed by :class:`DatasetBuilder` to discover spectrum
files/directories and metadata from different directory structures.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..io.readers import _find_bruker_acqus
from .duplicates import DuplicateStrategy, apply_metadata_strategy

logger = logging.getLogger(__name__)


class InputLayout(ABC):
    """Abstract adapter for discovering spectra and metadata."""

    @abstractmethod
    def discover_spectra(self) -> list[Path]:
        """Return paths to all spectrum sources (files or directories)."""

    @abstractmethod
    def discover_metadata(self) -> pd.DataFrame:
        """Return metadata DataFrame with an ``'ID'`` column."""

    @abstractmethod
    def get_id(self, spectrum_path: Path) -> str:
        """Extract the spectrum identifier from a path."""

    @abstractmethod
    def get_year(self, spectrum_id: str) -> str | None:
        """Return the year for a spectrum, or ``None``."""


def _extract_year(value: object) -> str:
    """Extract a four-digit year string from various input types.

    Supports date strings, datetime objects, and integers/floats.
    """
    if hasattr(value, "year"):
        return str(value.year)
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, str):
        part = value.strip().split("-")[0].split("/")[0]
        if part.isdigit() and len(part) == 4:
            return part
    raise ValueError(f"Cannot extract year from {value!r}")


class FlatLayout(InputLayout):
    """Flat directory of pre-exported text spectrum files + metadata CSV.

    Suitable for datasets where spectra are already exported as text files.

    Parameters
    ----------
    spectra_dir : str or Path
        Directory containing spectrum text files (flat or with year
        subfolders).
    metadata_csv : str or Path
        CSV with an ID column, species, and antibiotic columns.
    id_column : str, default="ID"
        Column name for the spectrum identifier in the metadata.
    year_column : str or None
        Column to extract year from, or ``None`` for flat layout.
    """

    def __init__(
        self,
        spectra_dir: str | Path,
        metadata_csv: str | Path,
        *,
        id_column: str = "ID",
        year_column: str | None = None,
    ) -> None:
        self.spectra_dir = Path(spectra_dir)
        self.metadata_csv = Path(metadata_csv)
        self.id_column = id_column
        self.year_column = year_column
        self._year_map: dict[str, str] | None = None

    def discover_spectra(self) -> list[Path]:
        """Glob for ``.txt`` files, flat or with year subfolders."""
        files = sorted(self.spectra_dir.glob("*.txt"))
        if not files:
            files = sorted(self.spectra_dir.glob("*/*.txt"))
        if not files:
            raise ValueError(f"No .txt spectrum files found in {self.spectra_dir}")
        return files

    def discover_metadata(self) -> pd.DataFrame:
        """Read metadata CSV and normalise the ID column."""
        meta = pd.read_csv(self.metadata_csv)
        if self.id_column not in meta.columns:
            raise ValueError(
                f"ID column '{self.id_column}' not found in metadata. "
                f"Available: {list(meta.columns)}"
            )
        if self.id_column != "ID":
            meta = meta.rename(columns={self.id_column: "ID"})
        meta["ID"] = meta["ID"].astype(str)

        if self.year_column is not None:
            if self.year_column not in meta.columns:
                raise ValueError(
                    f"year_column '{self.year_column}' not found in metadata."
                )
            self._year_map = dict(
                zip(
                    meta["ID"],
                    meta[self.year_column].apply(_extract_year),
                    strict=True,
                )
            )
        return meta

    def get_id(self, spectrum_path: Path) -> str:
        """Filename stem is the spectrum ID."""
        return spectrum_path.stem

    def get_year(self, spectrum_id: str) -> str | None:
        """Year from the metadata column, or ``None``."""
        if self._year_map is None:
            return None
        return self._year_map.get(spectrum_id)


class BrukerTreeLayout(InputLayout):
    """Hierarchical directory tree containing raw Bruker binary data.

    Suitable for datasets where spectra are stored as Bruker ``fid``/``acqus``
    binaries in a hierarchical directory tree.  The metadata CSV must
    contain a column with relative paths pointing to the Bruker data
    directories.

    Parameters
    ----------
    root_dir : str or Path
        Root directory of the dataset.
    metadata_csv : str or Path
        Metadata CSV with columns for identifier, path to Bruker
        data, and (optionally) year and target position.
    id_column : str, default="Identifier"
        Column for specimen identifier.
    year_column : str, default="Year"
        Column for year.
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
    validate : bool, default=True
        If ``True``, skip empty spectra (all-zero ``fid``) and warn
        on duplicate spectra (SHA256 hash matching).
    """

    def __init__(
        self,
        root_dir: str | Path,
        metadata_csv: str | Path,
        *,
        id_column: str = "Identifier",
        year_column: str = "Year",
        path_column: str = "Path",
        target_position_column: str = "target_position",
        duplicate_strategy: str | DuplicateStrategy = DuplicateStrategy.first,
        validate: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.metadata_csv = Path(metadata_csv)
        self.id_column = id_column
        self.year_column = year_column
        self.path_column = path_column
        self.target_position_column = target_position_column
        self.duplicate_strategy = DuplicateStrategy(duplicate_strategy)
        self.validate = validate
        self._year_map: dict[str, str] = {}
        self._id_to_path: dict[str, Path] = {}

    def discover_spectra(self) -> list[Path]:
        """Resolve Bruker directories from metadata paths.

        Applies :attr:`duplicate_strategy` to handle specimens that
        appear at multiple target positions.  Optionally validates
        for empty and duplicate spectra.
        """
        meta = self._read_raw_metadata()
        meta = apply_metadata_strategy(
            meta,
            self.duplicate_strategy,
            suffix_col=self.target_position_column,
        )

        paths: list[Path] = []
        seen_hashes: dict[str, str] = {}

        for _, row in meta.iterrows():
            rel_path = str(row[self.path_column]).lstrip("/")
            bruker_dir = self.root_dir / rel_path
            if not bruker_dir.is_dir():
                logger.warning("Directory not found: %s", bruker_dir)
                continue

            acqus = _find_bruker_acqus(bruker_dir)
            if acqus is None:
                logger.warning("No acqus in %s", bruker_dir)
                continue

            if self.validate and not self._validate_bruker_fid(
                acqus.parent, str(row["ID"]), seen_hashes
            ):
                continue

            self._id_to_path[str(row["ID"])] = bruker_dir
            self._year_map[str(row["ID"])] = str(row[self.year_column])
            paths.append(bruker_dir)

        if not paths:
            raise ValueError("No valid Bruker spectra found.")
        return paths

    def _validate_bruker_fid(
        self,
        acqus_dir: Path,
        row_id: str,
        seen_hashes: dict[str, str],
    ) -> bool:
        """Check a Bruker fid file for emptiness and duplicates.

        Returns ``True`` if the spectrum should be kept.
        """
        fid_path = acqus_dir / "fid"
        if not fid_path.is_file():
            return True
        content = fid_path.read_bytes()
        if all(b == 0 for b in content):
            logger.warning("Skipping empty spectrum: %s", row_id)
            return False
        h = hashlib.sha256(content).hexdigest()
        if h in seen_hashes:
            logger.warning("Duplicate spectrum: %s matches %s", row_id, seen_hashes[h])
        seen_hashes[h] = row_id
        return True

    def discover_metadata(self) -> pd.DataFrame:
        """Read metadata CSV, normalise ID column."""
        meta = self._read_raw_metadata()
        meta = apply_metadata_strategy(
            meta,
            self.duplicate_strategy,
            suffix_col=self.target_position_column,
        )
        if self._id_to_path:
            meta = meta[meta["ID"].isin(self._id_to_path.keys())]
        return meta.reset_index(drop=True)

    def get_id(self, spectrum_path: Path) -> str:
        """Look up ID from the path mapping built during discovery."""
        for sid, p in self._id_to_path.items():
            if p == spectrum_path:
                return sid
        return spectrum_path.parent.parent.parent.name

    def get_year(self, spectrum_id: str) -> str | None:
        """Year from the metadata."""
        return self._year_map.get(spectrum_id)

    def _read_raw_metadata(self) -> pd.DataFrame:
        """Read and normalise the raw metadata CSV."""
        meta = pd.read_csv(self.metadata_csv)
        if self.id_column not in meta.columns:
            raise ValueError(
                f"ID column '{self.id_column}' not in metadata. "
                f"Available: {list(meta.columns)}"
            )
        if self.id_column != "ID":
            meta = meta.rename(columns={self.id_column: "ID"})
        meta["ID"] = meta["ID"].astype(str)
        return meta
