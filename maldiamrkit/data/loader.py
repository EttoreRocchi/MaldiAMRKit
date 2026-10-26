"""Load datasets into MaldiSet objects.

The :class:`DatasetLoader` uses a :class:`DatasetLayout` to navigate
different dataset structures and load spectra into a
:class:`~maldiamrkit.dataset.MaldiSet`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..dataset import MaldiSet
from ..spectrum import MaldiSpectrum, _infer_id
from .dataset_layouts import DatasetLayout

logger = logging.getLogger(__name__)


def _load_single(path: Path) -> MaldiSpectrum:
    """Load a single spectrum without preprocessing or binning."""
    return MaldiSpectrum(path)


class DatasetLoader:
    """Load a dataset into a :class:`~maldiamrkit.dataset.MaldiSet`.

    Parameters
    ----------
    layout : DatasetLayout
        Dataset navigation adapter (e.g. :class:`DRIAMSLayout` or
        :class:`MARISMaLayout`).
    stage : str or None
        Processing stage to load.  ``None`` triggers auto-detection
        via the layout.
    n_jobs : int, default=-1
        Number of parallel workers for spectrum loading.
    verbose : bool, default=False
        If True, show tqdm progress bars during spectrum loading and
        pass ``verbose`` through to :class:`~maldiamrkit.dataset.MaldiSet`.

    Examples
    --------
    >>> from maldiamrkit.data import DatasetLoader, DRIAMSLayout
    >>> layout = DRIAMSLayout("output/my_dataset")
    >>> loader = DatasetLoader(layout)
    >>> ds = loader.load(aggregate_by=dict(antibiotics="Ceftriaxone"))
    """

    def __init__(
        self,
        layout: DatasetLayout,
        *,
        stage: str | None = None,
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> None:
        self.layout = layout
        self.stage = stage
        self.n_jobs = n_jobs
        self.verbose = verbose

    def load(
        self,
        aggregate_by: dict[str, str | list[str]] | None = None,
    ) -> MaldiSet:
        """Load the dataset.

        Parameters
        ----------
        aggregate_by : dict, optional
            Passed through to :class:`~maldiamrkit.dataset.MaldiSet`.

        Returns
        -------
        MaldiSet
            Dataset with loaded spectra and metadata.
        """
        # 1. Resolve stage
        stage_name = (
            self.stage if self.stage is not None else self.layout.detect_stage()
        )

        # 2. Load metadata
        meta = self.layout.discover_metadata()

        # 3. Pre-filter metadata by aggregate_by criteria
        n_meta_before = len(meta)
        meta = self._prefilter_metadata(meta, aggregate_by)
        if len(meta) < n_meta_before:
            logger.info(
                "Pre-filtered metadata: %d -> %d rows",
                n_meta_before,
                len(meta),
            )

        # 4. Collect spectrum files
        year = getattr(self.layout, "year", None)
        spectrum_files = self.layout.collect_spectrum_files(stage_name, year)
        if not spectrum_files:
            raise FileNotFoundError(f"No spectrum files found for stage '{stage_name}'")

        # 5. Match spectrum files to metadata rows (by the file each row
        # names, when the layout records it, otherwise by ID)
        n_meta_ids = meta["ID"].nunique()
        if "_file_id" in meta.columns:
            meta, matched_files = _match_row_files(
                meta,
                spectrum_files,
                self.layout._file_key,
                getattr(self.layout, "target_position_column", None),
            )
        else:
            meta_ids = set(meta["ID"].astype(str))
            matched_files = [f for f in spectrum_files if _infer_id(f) in meta_ids]

        if not matched_files:
            raise ValueError(
                f"No spectrum files matched metadata IDs. "
                f"Found {len(spectrum_files)} files and "
                f"{n_meta_ids} metadata IDs."
            )

        n_total = len(spectrum_files)
        n_matched = len(matched_files)
        if n_matched < n_total:
            logger.info(
                "Loading %d/%d spectra (others not in metadata)",
                n_matched,
                n_total,
            )

        # 6. Load spectra
        if self.verbose and self.n_jobs == 1:
            spectra = [
                _load_single(p)
                for p in tqdm(matched_files, desc="Loading spectra", unit="file")
            ]
        else:
            spectra = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_load_single)(p) for p in matched_files
            )

        # 6a. Dataset-specific post-processing (e.g. DRIAMS binned_N/ files
        # store bin_index, not m/z, and need rewriting into real m/z).
        spectra = [
            self.layout.postprocess_spectrum(s, stage=stage_name) for s in spectra
        ]
        # Spectra of the "preprocessed" stage are already preprocessed:
        # mark them so that binning does not preprocess them again.
        if stage_name == "preprocessed":
            for s in spectra:
                s._preprocessed = s._raw.copy()

        if "_file_id" in meta.columns:
            # Each matched row was given its own file: name the spectra
            # after the (possibly renamed) rows.
            for s, sid in zip(spectra, meta["ID"], strict=True):
                s.id = sid
            meta = meta.drop(columns=["_file_id"]).drop(
                columns="_year", errors="ignore"
            )

        # Rows renamed for duplicates keep their isolate in ``_isolate``.
        isolate_column = self.layout.isolate_column
        if "_isolate" in meta.columns and isolate_column is None:
            isolates = meta["_isolate"].fillna(meta["ID"]).astype(str)
            pattern = self.layout.replicate_pattern
            if pattern is not None:
                isolates = isolates.str.replace(pattern, "", regex=True)
            meta["_isolate"] = isolates
            isolate_column = "_isolate"

        # 6b. Average replicates when the layout used strategy="average"
        if "_original_id" in meta.columns:
            spectra, meta = self._average_replicates(spectra, meta)

        # 7. Build MaldiSet
        return MaldiSet(
            spectra,
            meta,
            aggregate_by=aggregate_by,
            verbose=self.verbose,
            isolate_pattern=self.layout.replicate_pattern,
            isolate_column=isolate_column,
        )

    @staticmethod
    def _prefilter_metadata(
        meta: pd.DataFrame,
        aggregate_by: dict[str, str | list[str]] | None,
    ) -> pd.DataFrame:
        """Filter metadata rows before spectrum loading to reduce I/O."""
        if not aggregate_by:
            return meta

        species_val = aggregate_by.get("species")
        if species_val and "Species" in meta.columns:
            meta = meta[meta["Species"] == species_val]

        antibiotics_val = aggregate_by.get("antibiotics")
        if antibiotics_val is not None:
            if isinstance(antibiotics_val, str):
                antibiotics_val = [antibiotics_val]
            available = [ab for ab in antibiotics_val if ab in meta.columns]
            if available:
                meta = meta[meta[available].notna().any(axis=1)]

        return meta

    @staticmethod
    def _average_replicates(
        spectra: list[MaldiSpectrum],
        meta: pd.DataFrame,
    ) -> tuple[list[MaldiSpectrum], pd.DataFrame]:
        """Average replicate spectra that share an ``_original_id``.

        Spectra are matched to metadata rows by ID. Each isolate keeps its
        first metadata row; a single replicate is kept as it is (under the
        isolate's ID), several are averaged (see :func:`_average_members`).
        Isolates without any loaded spectrum are skipped.

        Returns the deduplicated spectra list and metadata DataFrame.
        """
        id_to_spec = {str(s.id): s for s in spectra}
        meta = meta.assign(_original_id=meta["_original_id"].astype(str))
        groups = meta.groupby("_original_id", sort=False)["ID"].agg(list)

        averaged_spectra: list[MaldiSpectrum] = []
        kept_ids: list[str] = []
        for orig_id, member_ids in groups.items():
            members = [id_to_spec[str(m)] for m in member_ids if str(m) in id_to_spec]
            if not members:
                continue
            spec = members[0] if len(members) == 1 else _average_members(members)
            spec.id = str(orig_id)
            averaged_spectra.append(spec)
            kept_ids.append(orig_id)

        meta = meta.drop_duplicates("_original_id", keep="first")
        meta = meta[meta["_original_id"].isin(kept_ids)].copy()
        meta["ID"] = meta["_original_id"]
        meta = meta.drop(columns=["_original_id"]).reset_index(drop=True)

        logger.info(
            "Averaged replicates: %d spectra -> %d unique IDs.",
            len(spectra),
            len(averaged_spectra),
        )

        return averaged_spectra, meta


def _average_members(members: list[MaldiSpectrum]) -> MaldiSpectrum:
    """Average the replicate spectra of one isolate.

    Binned replicates (as loaded from a binned stage) must share one grid:
    they are averaged bin by bin and the result stays binned on that grid.
    Other replicates are interpolated onto a common m/z grid and averaged;
    the result stays preprocessed when every replicate was.
    """
    if all(m.is_binned for m in members):
        first = members[0]
        for m in members[1:]:
            same_grid = np.array_equal(
                m._raw["mass"].to_numpy(), first._raw["mass"].to_numpy()
            ) and m._binned["mass"].equals(first._binned["mass"])
            if not same_grid:
                raise ValueError(
                    f"Replicates {first.id!r} and {m.id!r} are binned on "
                    "different m/z grids and cannot be averaged."
                )
        intensity = np.mean(
            np.vstack([m._raw["intensity"].to_numpy() for m in members]), axis=0
        )
        avg = MaldiSpectrum(first._raw.assign(intensity=intensity))
        avg._binned = first._binned.assign(intensity=intensity)
        avg._bin_width = first._bin_width
        avg._bin_method = first._bin_method
        avg._bin_metadata = first._bin_metadata
        return avg

    mz_min = max(m._raw["mass"].min() for m in members)
    mz_max = min(m._raw["mass"].max() for m in members)
    step = min(np.median(np.diff(m._raw["mass"].to_numpy())) for m in members)
    common_mz = np.arange(mz_min, mz_max, step)
    intensities = np.vstack(
        [
            np.interp(common_mz, m._raw["mass"].to_numpy(), m._raw["intensity"])
            for m in members
        ]
    )
    avg = MaldiSpectrum(
        pd.DataFrame({"mass": common_mz, "intensity": intensities.mean(axis=0)})
    )
    if all(m.is_preprocessed for m in members):
        avg._preprocessed = avg._raw.copy()
    return avg


def _match_row_files(
    meta: pd.DataFrame,
    spectrum_files: list[Path],
    file_key: Callable[[Path], tuple[str | None, str]],
    suffix_col: str | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Match each metadata row to the file it names.

    Files are keyed by ``file_key`` (year folder and file ID), and each
    row by its ``_file_id`` (the file name or directory it gives, or its ID
    when missing) and, when recorded, its ``_year``. A row with a year
    matches the file of that year folder, or a file outside year folders,
    or else the only unclaimed file with its ID in another year folder
    (with a warning); a row without a year matches the one file with its
    ID. Rows without a file are dropped. The matched rows follow the order
    of their files; those that still share an ID (the same replicate in
    several years) are renamed with ``suffix_col`` (``_repN`` without it),
    keeping their ID in ``_isolate``.

    Returns the matched rows and their files, in the same order.

    Raises
    ------
    ValueError
        If one file matches several rows, or a row without a year matches
        files of several year folders.

    Warns
    -----
    UserWarning
        If rows were matched to a file of another year folder.
    """
    from .duplicates import _suffix_duplicates

    by_key: dict[tuple[str | None, str], Path] = {}
    by_id: dict[str, list[Path]] = {}
    for f in dict.fromkeys(spectrum_files):
        year, fid = file_key(f)
        by_key[(year, fid)] = f
        by_id.setdefault(fid, []).append(f)

    ids = meta["ID"].astype(str).tolist()
    file_ids = meta["_file_id"].fillna(meta["ID"]).astype(str).tolist()
    years = meta["_year"].tolist() if "_year" in meta.columns else [None] * len(meta)
    matched: dict[int, Path] = {}
    owner: dict[Path, str] = {}

    def claim(pos: int, f: Path) -> None:
        if f in owner:
            raise ValueError(
                f"Spectrum file {f} matches several metadata rows "
                f"({owner[f]!r} and {ids[pos]!r}); remove the duplicate rows."
            )
        owner[f] = ids[pos]
        matched[pos] = f

    pending: list[int] = []
    for pos, (year, fid) in enumerate(zip(years, file_ids, strict=True)):
        if year is not None and not pd.isna(year):
            f = by_key.get((str(year), fid), by_key.get((None, fid)))
            if f is None:
                pending.append(pos)
                continue
        else:
            candidates = by_id.get(fid, [])
            if len(candidates) > 1:
                raise ValueError(
                    f"Spectrum files {[str(c) for c in candidates]} share the "
                    f"ID {fid!r} and the metadata gives no year to tell them "
                    "apart."
                )
            if not candidates:
                continue
            f = candidates[0]
        claim(pos, f)

    moved: list[str] = []
    for pos in pending:
        candidates = [f for f in by_id.get(file_ids[pos], []) if f not in owner]
        if len(candidates) == 1:
            claim(pos, candidates[0])
            moved.append(ids[pos])
    if moved:
        warnings.warn(
            f"{len(moved)} metadata row(s) have no spectrum file in their year "
            "folder and were matched to the only file with their ID in another "
            f"year folder: {moved[:5]}",
            UserWarning,
            stacklevel=3,
        )

    order = sorted(matched, key=lambda pos: str(matched[pos]))
    rows = meta.iloc[order]
    if rows["ID"].duplicated().any():
        if "_isolate" not in rows.columns:
            rows = rows.assign(_isolate=rows["ID"])
        rows = _suffix_duplicates(rows, "ID", suffix_col)
    return rows, [matched[pos] for pos in order]
