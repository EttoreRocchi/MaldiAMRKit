"""Load datasets into MaldiSet objects.

The :class:`DatasetLoader` uses a :class:`DatasetLayout` to navigate
different dataset structures and load spectra into a
:class:`~maldiamrkit.dataset.MaldiSet`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..dataset import MaldiSet
from ..spectrum import MaldiSpectrum
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

        # 5. Match spectrum files to metadata IDs
        meta_ids = set(meta["ID"].astype(str))

        matched_files: list[Path] = []
        for f in spectrum_files:
            fid = f.name if f.is_dir() else f.stem
            if fid in meta_ids:
                matched_files.append(f)

        if not matched_files:
            raise ValueError(
                f"No spectrum files matched metadata IDs. "
                f"Found {len(spectrum_files)} files and "
                f"{len(meta_ids)} metadata IDs."
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

        # 7. Build MaldiSet
        return MaldiSet(
            spectra,
            meta,
            aggregate_by=aggregate_by,
            verbose=self.verbose,
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
