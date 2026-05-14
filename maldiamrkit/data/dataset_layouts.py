"""Dataset layout adapters for navigating and loading datasets.

DatasetLayouts describe **how to navigate a dataset** for loading.
They are consumed by :class:`DatasetLoader` to discover metadata and
spectrum files from different dataset structures.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..io.readers import _find_bruker_acqus
from .duplicates import DuplicateStrategy, apply_metadata_strategy
from .site_info import read_site_info

if TYPE_CHECKING:
    from ..spectrum import MaldiSpectrum

logger = logging.getLogger(__name__)


class _Sentinel:
    """Sentinel for ``DRIAMSLayout`` kwargs.

    A kwarg defaulted to :data:`_AUTO` means: "fill from ``site_info.json``
    if present, otherwise use the library default".  Explicit kwargs
    always win.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<auto>"


_AUTO: _Sentinel = _Sentinel()

# Pattern used by :class:`DRIAMSLayout` to detect files that look like
# technical replicates of an underlying sample (``UUID_MALDI1``,
# ``UUID_MALDI2``, ...). Consulted only by the replicate-leakage
# warning in :meth:`DRIAMSLayout.discover_metadata`; ``id_transform``
# itself is user-supplied and doesn't rely on this pattern.
_DRIAMS_MALDI_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_MALDI\d+$")


def _warn_on_likely_replicates(ids: pd.Series) -> None:
    """Emit a one-shot warning when DRIAMS IDs look like shared-sample replicates.

    DRIAMS encodes technical replicates with an ``_MALDI<N>`` suffix
    (``UUID_MALDI1``, ``UUID_MALDI2``). Two such files share an
    underlying biological sample. The default
    ``duplicate_strategy="first"`` dedupes on the raw ID and keeps
    both. Cross-validation on the resulting feature matrix then leaks
    replicates across folds unless callers use a group-aware
    splitter.

    Callers who set ``id_transform`` (even to ``str``, a no-op) opt
    out of this warning - they've acknowledged the issue.
    """
    stems = ids.astype(str).str.extract(_DRIAMS_MALDI_SUFFIX_RE.pattern, expand=False)
    # ``stems`` is NaN for IDs that don't match the ``_MALDI<N>``
    # suffix; restrict to the matched rows.
    matched = stems.dropna()
    if matched.empty:
        return
    n_rows_affected = int(matched.duplicated(keep=False).sum())
    if n_rows_affected == 0:
        return
    logger.warning(
        "DRIAMSLayout: detected %d rows whose IDs share an underlying sample "
        "UUID after stripping the _MALDI<N> suffix (%d distinct samples). "
        "These are kept as distinct rows by the default duplicate_strategy, "
        "which causes replicate-leakage across folds under shuffled CV. "
        "Pass id_transform=lambda s: re.sub(r'_MALDI\\d+$', '', s) to collapse "
        "replicates at load time (or id_transform=str to silence this warning "
        "if the per-replicate semantics are intentional).",
        n_rows_affected,
        matched.nunique(),
    )


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

    def postprocess_spectrum(
        self,
        spec: MaldiSpectrum,
        *,
        stage: str | None = None,
    ) -> MaldiSpectrum:
        """Apply dataset-specific fix-ups to a freshly-loaded spectrum.

        Default is a no-op.  Layouts whose on-disk format deviates from
        the ``(mass, intensity)`` convention assumed by
        :func:`~maldiamrkit.io.read_spectrum` can override this
        to reshape the spectrum.  Called by :class:`DatasetLoader` after
        each file is loaded.
        """
        return spec


class DRIAMSLayout(DatasetLayout):
    r"""Navigate a DRIAMS-like dataset structure.

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
    id_transform : callable, optional
        Function mapping raw ``ID`` strings to a canonical *sample*
        identifier. When set, duplicates are detected on the
        transformed identifier rather than the raw one -- so
        technical-replicate files that share an underlying sample
        (e.g. DRIAMS ``UUID_MALDI1`` / ``UUID_MALDI2``) are
        recognized as duplicates by ``duplicate_strategy``. The raw
        ``ID`` column is preserved for spectrum-file matching; only
        deduplication uses the transformed key. Typical DRIAMS usage::

            import re
            DRIAMSLayout(
                ...,
                id_transform=lambda s: re.sub(r"_MALDI\\d+$", "", s),
                duplicate_strategy="first",   # or "average"
            )

        Leaving this at ``None`` preserves the legacy behaviour (each
        replicate counted as a distinct row). A one-time warning is
        emitted when ``_MALDI<N>``-suffixed IDs are detected and
        ``id_transform`` is ``None``, pointing at this kwarg; the
        warning can be silenced by passing ``id_transform=str`` if
        the per-replicate semantics are intentional.
    mz_min : float, default=2000.0
        Lower m/z edge to assign to bin index 0 when a ``binned_N/`` stage
        is loaded.  Only consulted by :meth:`postprocess_spectrum`.
    mz_max : float, default=19997.0
        Upper m/z edge assigned to bin index ``N-1``.
    normalize_tic : bool, default=False
        When ``True``, re-apply a TIC normalization
        (``intensity <- intensity / sum(intensity)``) to every loaded
        spectrum in :meth:`postprocess_spectrum`.  Useful because the
        published DRIAMS / MS-UMG ``binned_6000/`` files do not sum
        to 1.0 on disk (empirically ~1.29 and ~1.36 respectively),
        despite the DRIAMS preprocessing script calling
        ``calibrateIntensity(method="TIC")`` before trimming -- the
        cause is somewhere in the upstream pipeline (MALDIquant version
        or an implicit scaling step) and has not been reproduced here.
        Enabling this kwarg gives sum=1.0 per spectrum, aligning DRIAMS
        / MS-UMG with flat-text datasets whose preprocessing pipeline
        already produces TIC=1.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        id_column: str | None | _Sentinel = _AUTO,
        species_column: str | None = None,
        year: str | int | None = None,
        metadata_dir: str | _Sentinel = _AUTO,
        metadata_suffix: str | _Sentinel = _AUTO,
        spectrum_ext: str | _Sentinel = _AUTO,
        duplicate_strategy: str | DuplicateStrategy = DuplicateStrategy.first,
        id_transform: Callable[[str], str] | None = None,
        mz_min: float | _Sentinel = _AUTO,
        mz_max: float | _Sentinel = _AUTO,
        normalize_tic: bool = False,
    ) -> None:
        """Initialise the layout.

        Several kwargs accept the sentinel :data:`_AUTO` as their default.
        When ``_AUTO``, the value is filled from ``site_info.json`` at the
        dataset root (if present) and otherwise falls back to the
        library-level default.  Explicit kwargs always win.  Fields with
        per-call semantics (``year``, ``species_column``, ``id_transform``,
        ``duplicate_strategy``, ``normalize_tic``) stay user-controlled
        and are never read from the manifest.
        """
        self.dataset_dir = Path(dataset_dir)

        # Try to read the manifest; absent manifest = pure library defaults.
        # `read_site_info` is tolerant: an unreadable manifest raises a
        # clear ValueError, while a missing manifest returns None.
        site_info = read_site_info(self.dataset_dir, missing_ok=True)

        def _resolve(
            user_value,
            manifest_attr: str | None,
            library_default,
        ):
            """Resolve a kwarg per precedence: user > manifest > default."""
            if not isinstance(user_value, _Sentinel):
                return user_value
            if site_info is not None and manifest_attr is not None:
                return getattr(site_info, manifest_attr, library_default)
            return library_default

        self.id_column = _resolve(id_column, "id_column", None)
        self.metadata_dir = _resolve(metadata_dir, "metadata_dir", "id")
        self.metadata_suffix = _resolve(
            metadata_suffix, "metadata_suffix", "_clean.csv"
        )
        self.spectrum_ext = _resolve(spectrum_ext, "spectrum_ext", ".txt")

        # `mz_range` is stored as a tuple in the manifest; split into the
        # two scalar kwargs here.
        manifest_mz = site_info.mz_range if site_info is not None else None
        self.mz_min = float(
            _resolve(
                mz_min, None, manifest_mz[0] if manifest_mz is not None else 2000.0
            )
        )
        self.mz_max = float(
            _resolve(
                mz_max, None, manifest_mz[1] if manifest_mz is not None else 19997.0
            )
        )

        # Per-call / load-only kwargs - never sourced from the manifest.
        self.species_column = species_column
        self.year = str(year) if year is not None else None
        self.duplicate_strategy = DuplicateStrategy(duplicate_strategy)
        self.id_transform = id_transform
        self.normalize_tic = bool(normalize_tic)

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

        if self.id_transform is not None:
            # Deduplicate on the canonical identifier so technical
            # replicates of the same underlying sample collapse to a
            # single row. Raw ``ID`` is preserved so the loader's
            # spectrum-file matching still works on the per-replicate
            # filename stem.
            meta["_canonical_id"] = meta["ID"].map(self.id_transform)
            meta = apply_metadata_strategy(
                meta, self.duplicate_strategy, id_col="_canonical_id"
            )
            meta = meta.drop(columns="_canonical_id")
        else:
            _warn_on_likely_replicates(meta["ID"])
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

    def postprocess_spectrum(
        self,
        spec: MaldiSpectrum,
        *,
        stage: str | None = None,
    ) -> MaldiSpectrum:
        """Rewrite ``binned_N/`` spectra from bin_index to real m/z.

        DRIAMS (and MS-UMG) ``binned_6000/*.txt`` files store
        ``bin_index binned_intensity`` rather than ``(mass, intensity)``.
        Without conversion, every downstream m/z-aware API
        (``SpectrumQuality.noise_region``, ``MzTrimmer``,
        ``plot_spectrum`` axes, m/z-range filters) would operate in
        [0, N) instead of [mz_min, mz_max].

        When ``stage`` matches ``binned_N`` and the loaded spectrum's
        ``mass`` column looks like contiguous integers ``0..N-1``, the
        spectrum is rewritten:

        - ``mass`` becomes ``mz_min + i * (mz_max - mz_min) / (N - 1)``,
        - the spectrum is marked as pre-binned (``_binned`` populated),
          so ``MaldiSet`` does not re-bin already-binned data,
        - ``_bin_metadata`` is filled in consistently.

        Idempotent: a second call on already-converted data is a no-op
        (mass is no longer integer 0..N-1).

        When ``self.normalize_tic`` is ``True``, the intensities are
        additionally rescaled so that each spectrum sums to 1.
        """
        if stage is None or not _STAGE_PRIORITY.match(stage):
            return spec
        spec = _convert_bin_index_spectrum(spec, mz_min=self.mz_min, mz_max=self.mz_max)
        if self.normalize_tic:
            spec = _apply_tic_normalization(spec)
        return spec


def _convert_bin_index_spectrum(
    spec: MaldiSpectrum,
    *,
    mz_min: float,
    mz_max: float,
) -> MaldiSpectrum:
    """Convert a spectrum whose ``mass`` column is 0..N-1 bin indices.

    See :meth:`DRIAMSLayout.postprocess_spectrum`.  Separate module-level
    function for ease of testing.
    """
    from ..preprocessing.binning import get_bin_metadata

    raw = getattr(spec, "_raw", None)
    if raw is None or raw.empty:
        return spec
    n = len(raw)
    if n < 2:
        return spec

    mass = raw["mass"].to_numpy(dtype=float)
    # Fast-path: mass already looks like real m/z (well above any plausible
    # bin index). Keeps the hook idempotent.
    if mass[0] > mz_min / 2:
        return spec
    # Must look like 0, 1, 2, ..., N-1.
    if not np.allclose(mass, np.arange(n, dtype=float)):
        return spec

    step = (mz_max - mz_min) / (n - 1)
    mz = mz_min + mass * step
    new_df = pd.DataFrame(
        {"mass": mz, "intensity": raw["intensity"].to_numpy(dtype=float)}
    )
    edges = mz_min - step / 2 + np.arange(n + 1, dtype=float) * step
    bin_meta = get_bin_metadata(edges)

    spec._raw = new_df
    spec._binned = new_df.copy()
    spec._bin_width = step
    spec._bin_method = "uniform"
    spec._bin_metadata = bin_meta
    return spec


def _apply_tic_normalization(spec: MaldiSpectrum) -> MaldiSpectrum:
    """Rescale ``spec`` so that its intensity column sums to 1.

    No-op on spectra with zero or non-finite total intensity.  Writes
    back to both ``_raw`` and ``_binned`` (when present) so every
    downstream accessor sees the normalized data.
    """
    raw = getattr(spec, "_raw", None)
    if raw is None or raw.empty:
        return spec
    intensity = raw["intensity"].to_numpy(dtype=float)
    total = float(intensity.sum())
    if not np.isfinite(total) or total <= 0.0:
        return spec
    normalized = intensity / total
    spec._raw = pd.DataFrame(
        {"mass": raw["mass"].to_numpy(dtype=float), "intensity": normalized}
    )
    if getattr(spec, "_binned", None) is not None:
        spec._binned = spec._raw.copy()
    return spec


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
    id_transform : callable, optional
        Function mapping raw ``ID`` strings to a canonical *sample*
        identifier. When set, duplicates are detected on the
        transformed identifier rather than the raw one, so technical
        replicates that encode the underlying sample in a filename
        suffix / prefix pattern collapse under ``duplicate_strategy``.
        The raw ``ID`` column is preserved for downstream matching;
        only deduplication uses the transformed key. Leave at ``None``
        for the legacy behaviour (each replicate counted as a distinct
        row).
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
        id_transform: Callable[[str], str] | None = None,
        year: str | int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.metadata_csv = Path(metadata_csv)
        self.id_column = id_column
        self.path_column = path_column
        self.target_position_column = target_position_column
        self.duplicate_strategy = DuplicateStrategy(duplicate_strategy)
        self.id_transform = id_transform
        self.year = str(year) if year is not None else None

    def discover_metadata(self) -> pd.DataFrame:
        """Read metadata CSV and normalise the ID column."""
        # low_memory=False: the MARISMa AMR.csv has ~160 antibiotic columns
        # mixing "R"/"S"/"I" strings with MIC values like "<=8" / ">16" and
        # bare numbers, which pandas' chunked type inference flags as mixed.
        meta = pd.read_csv(self.metadata_csv, low_memory=False)
        if self.id_column not in meta.columns:
            raise ValueError(
                f"ID column '{self.id_column}' not in metadata. "
                f"Available: {list(meta.columns)}"
            )
        if self.id_column != "ID":
            meta = meta.rename(columns={self.id_column: "ID"})
        meta["ID"] = meta["ID"].astype(str)

        if self.id_transform is not None:
            # Deduplicate on the canonical identifier; preserve raw
            # ``ID`` for downstream path resolution.
            meta["_canonical_id"] = meta["ID"].map(self.id_transform)
            meta = apply_metadata_strategy(
                meta,
                self.duplicate_strategy,
                id_col="_canonical_id",
                suffix_col=self.target_position_column,
            )
            meta = meta.drop(columns="_canonical_id")
        else:
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
        return pd.read_csv(csv_path, low_memory=False), [year]

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
        return pd.read_csv(csv_files[0], low_memory=False), []

    csv_files = sorted(id_dir.glob("*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0], low_memory=False), []

    raise FileNotFoundError(f"No metadata CSV files found in {id_dir}")
