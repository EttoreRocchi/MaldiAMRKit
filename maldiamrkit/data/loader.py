"""Load datasets into MaldiSet objects.

The :class:`DatasetLoader` uses a :class:`DatasetLayout` to navigate
different dataset structures and load spectra into a
:class:`~maldiamrkit.dataset.MaldiSet`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from joblib import Parallel, delayed

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
    ) -> None:
        self.layout = layout
        self.stage = stage
        self.n_jobs = n_jobs

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

        # 3. Collect spectrum files
        year = getattr(self.layout, "year", None)
        spectrum_files = self.layout.collect_spectrum_files(stage_name, year)
        if not spectrum_files:
            raise FileNotFoundError(f"No spectrum files found for stage '{stage_name}'")

        # 4. Match spectrum files to metadata IDs
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

        # 5. Load spectra in parallel
        spectra = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_load_single)(p) for p in matched_files
        )

        # 6. Build MaldiSet
        return MaldiSet(
            spectra,
            meta,
            aggregate_by=aggregate_by,
        )
