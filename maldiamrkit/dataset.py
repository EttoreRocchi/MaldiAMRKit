"""Multi-spectrum dataset handling."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .filters import SpectrumFilter
from .preprocessing.binning import _uniform_edges, get_bin_metadata
from .preprocessing.preprocessing_pipeline import PreprocessingPipeline
from .spectrum import MaldiSpectrum

logger = logging.getLogger(__name__)


def _load_single_spectrum(
    path: Path,
    pipeline: PreprocessingPipeline | None,
    bin_width: int,
    bin_method: str,
    bin_kwargs: dict,
) -> MaldiSpectrum:
    """Load and process a single spectrum (helper for parallel loading)."""
    return MaldiSpectrum(path, pipeline=pipeline).bin(
        bin_width, method=bin_method, **bin_kwargs
    )


class MaldiSet:
    """
    A collection of MALDI-TOF spectra with metadata.

    Provides methods for loading multiple spectra from a directory,
    filtering by metadata, and generating feature matrices for ML.

    Parameters
    ----------
    spectra : list of MaldiSpectrum
        List of spectrum objects.
    meta : pd.DataFrame
        Metadata DataFrame with 'ID' column matching spectrum IDs.
    aggregate_by : dict, optional
        Dictionary specifying aggregation columns:

        - 'antibiotics': str or list of antibiotic column names
        - 'species': str, species value to filter by
          (metadata must have a column named 'Species')

        All metadata columns are retained regardless of ``aggregate_by``.
        If None, all spectra are included without antibiotic/species filtering.
    bin_width : int, default=3
        Bin width for spectra.
    bin_method : str, default='uniform'
        Binning method. One of 'uniform', 'proportional', 'adaptive', 'custom'.
    bin_kwargs : dict, optional
        Additional keyword arguments for binning (e.g., custom_edges, adaptive_min_width).
    verbose : bool, default=False
        If True, print progress messages.

    Attributes
    ----------
    spectra : list of MaldiSpectrum
        The spectrum objects.
    antibiotics : list of str or None
        Antibiotic column names.
    species : str or None
        Species value to filter by.
    meta : pd.DataFrame
        Metadata indexed by ID (all columns retained).

    Examples
    --------
    >>> ds = MaldiSet.from_directory(
    ...     "spectra/", "meta.csv",
    ...     aggregate_by=dict(
    ...         antibiotics=["Ceftriaxone", "Ceftazidime"],
    ...         species="Escherichia coli",
    ...     )
    ... )
    >>> ds.X.shape, ds.y.shape
    """

    def __init__(
        self,
        spectra: list[MaldiSpectrum],
        meta: pd.DataFrame,
        *,
        aggregate_by: dict[str, str | list[str]] | None = None,
        bin_width: int = 3,
        bin_method: str = "uniform",
        bin_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> None:
        self.spectra = spectra

        self.antibiotics, self.species = self._parse_aggregate_by(aggregate_by)

        # Validate that aggregate_by columns exist
        required_columns: list[str] = []
        if self.antibiotics:
            required_columns.extend(self.antibiotics)
        if self.species:
            required_columns.append("Species")

        missing_columns = [col for col in required_columns if col not in meta.columns]
        if missing_columns and verbose:
            logger.warning("Columns %s not found in metadata", missing_columns)

        self.meta = meta.set_index("ID")
        self.meta_cols = self.meta.columns.tolist()

        self.bin_width = bin_width
        self.bin_method = bin_method
        self.bin_kwargs = bin_kwargs
        self._bin_metadata: pd.DataFrame | None = None
        self._X_cache: pd.DataFrame | None = None

        self.verbose = verbose
        if verbose:
            logger.info("Dataset created: %d spectra", len(self.spectra))
            if self.antibiotics:
                logger.info("Tracking antibiotics: %s", self.antibiotics)

    @staticmethod
    def _parse_aggregate_by(
        aggregate_by: dict[str, str | list[str]] | None,
    ) -> tuple[list[str] | None, str | None]:
        """Parse the aggregate_by dict into antibiotics list and species."""
        aggregate_by = aggregate_by or {}
        antibiotics_val = aggregate_by.get("antibiotics")
        if isinstance(antibiotics_val, str):
            antibiotics = [antibiotics_val]
        elif isinstance(antibiotics_val, list):
            antibiotics = antibiotics_val
        else:
            antibiotics = None
        species = aggregate_by.get("species")
        return antibiotics, species

    @classmethod
    def from_directory(
        cls,
        spectra_dir: str | Path,
        meta_file: str | Path,
        *,
        aggregate_by: dict[str, str | list[str]] | None = None,
        pipeline: PreprocessingPipeline | None = None,
        bin_width: int = 3,
        bin_method: str = "uniform",
        bin_kwargs: dict | None = None,
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> MaldiSet:
        """
        Load spectra from a directory and metadata from a CSV file.

        Only spectrum files whose filename stem matches an ID in the
        metadata are loaded, avoiding unnecessary I/O and preprocessing.

        Parameters
        ----------
        spectra_dir : str or Path
            Directory containing spectrum .txt files.
        meta_file : str or Path
            Path to CSV file with metadata.
        aggregate_by : dict, optional
            Dictionary specifying aggregation columns:

            - 'antibiotics': str or list of antibiotic column names
            - 'species': str, species value to filter by
              (metadata must have a column named 'Species')

            All metadata columns are retained regardless of ``aggregate_by``.
            If None, all spectra matching metadata are loaded without
            antibiotic/species filtering.
        pipeline : PreprocessingPipeline, optional
            Preprocessing pipeline. If None, uses the default pipeline.
        bin_width : int, default=3
            Bin width for spectra.
        bin_method : str, default='uniform'
            Binning method. One of 'uniform', 'proportional', 'adaptive', 'custom'.
        bin_kwargs : dict, optional
            Additional keyword arguments for binning.
        n_jobs : int, default=-1
            Number of parallel jobs for loading spectra.
            Use -1 for all available cores, 1 for sequential processing.
        verbose : bool, default=False
            If True, print progress messages.

        Returns
        -------
        MaldiSet
            Dataset with loaded spectra and metadata.

        Notes
        -----
        Files are sorted alphabetically before loading to ensure reproducibility
        across runs with different parallelization settings.
        """
        spectra_dir = Path(spectra_dir)
        _bin_kwargs = bin_kwargs or {}

        meta = pd.read_csv(meta_file)
        meta_ids = set(meta["ID"].astype(str))

        # Sort file list for reproducibility, filter to metadata IDs
        all_files = sorted(spectra_dir.glob("*.txt"))
        spectrum_files = [f for f in all_files if f.stem in meta_ids]

        n_skipped = len(all_files) - len(spectrum_files)
        if verbose:
            logger.info(
                "Loading %d spectra (%d skipped, not in metadata) with n_jobs=%d",
                len(spectrum_files),
                n_skipped,
                n_jobs,
            )

        # Use parallel loading with joblib
        specs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_load_single_spectrum)(
                p, pipeline, bin_width, bin_method, _bin_kwargs
            )
            for p in spectrum_files
        )

        return cls(
            specs,
            meta,
            aggregate_by=aggregate_by,
            bin_width=bin_width,
            bin_method=bin_method,
            bin_kwargs=bin_kwargs,
            verbose=verbose,
        )

    @property
    def spectra_paths(self) -> dict[str, Path]:
        """
        Return mapping from spectrum ID to file path.

        Returns
        -------
        dict
            Dictionary mapping spectrum IDs to their file paths.
            Only includes spectra that were loaded from files.
        """
        return {s.id: s.path for s in self.spectra if s.path is not None}

    @property
    def bin_metadata(self) -> pd.DataFrame:
        """
        Return bin metadata with bin boundaries and widths.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: bin_index, bin_start, bin_end, bin_width.

        Notes
        -----
        If spectra have been binned, returns metadata from the first spectrum.
        Otherwise, computes metadata based on stored binning parameters.
        """
        # Try to get from first spectrum
        for spec in self.spectra:
            if spec.has_bin_metadata:
                return spec.bin_metadata

        # Compute from stored parameters if no spectrum has metadata
        if self._bin_metadata is None:
            if self.spectra:
                mz_min, mz_max = self.spectra[0].pipeline.mz_range
            else:
                mz_min, mz_max = 2000, 20000
            edges = _uniform_edges(mz_min, mz_max, self.bin_width)
            self._bin_metadata = get_bin_metadata(edges)

        return self._bin_metadata.copy()

    @property
    def X(self) -> pd.DataFrame:
        """
        Return feature matrix (n_samples, n_features).

        Returns
        -------
        pd.DataFrame
            Feature matrix with samples as rows and m/z bins as columns.
            Filtered to configured subset (antibiotics, species).

        Raises
        ------
        ValueError
            If no spectra match metadata IDs, or if no samples remain
            after filtering by species.
        """
        if self._X_cache is not None:
            return self._X_cache.copy()
        df = self._build_feature_dataframe()
        df = self._apply_subset_filters(df)
        result = df.drop(columns=self.meta_cols)
        self._X_cache = result
        return result.copy()

    def _build_feature_dataframe(self) -> pd.DataFrame:
        """Build raw feature DataFrame from spectra, joined with metadata."""
        bin_kwargs = self.bin_kwargs or {}
        ids: list[str] = []
        arrays: list[np.ndarray] = []
        columns: pd.Index | None = None

        for s in self.spectra:
            sid = s.id
            if sid not in self.meta.index:
                warnings.warn(
                    f"Spectrum ID '{sid}' not found in metadata - skipped.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            binned = (
                s.binned
                if s.is_binned
                else s.bin(self.bin_width, method=self.bin_method, **bin_kwargs).binned
            )
            series = binned.set_index("mass")["intensity"]
            ids.append(sid)
            arrays.append(series.values)
            if columns is None:
                columns = series.index

        if not ids:
            raise ValueError(
                "No spectra matched metadata IDs. "
                "Check that spectrum file names match the 'ID' column in metadata."
            )

        df = pd.DataFrame(np.array(arrays), index=ids, columns=columns)
        if not self.meta.columns.empty:
            df = df.join(self.meta, how="left")
        return df

    def _apply_subset_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter feature DataFrame by antibiotics and species."""
        if self.antibiotics:
            antibiotic_mask = pd.Series(False, index=df.index)
            for antibiotic in self.antibiotics:
                if antibiotic in df.columns:
                    antibiotic_mask |= df[antibiotic].notna()
            df = df[antibiotic_mask]

        if self.species:
            df = df[df["Species"] == self.species]
            if len(df) == 0:
                raise ValueError(
                    f"No samples remaining after filtering by species='{self.species}'"
                )
        return df

    @property
    def y(self) -> pd.DataFrame:
        """
        Return label matrix for all specified antibiotics.

        Returns
        -------
        pd.DataFrame
            Label matrix with one column per antibiotic.

        Raises
        ------
        ValueError
            If no antibiotics specified or none found in metadata.
        """
        if not self.antibiotics:
            raise ValueError("No antibiotics specified for classification labels")

        available_antibiotics = [
            ab for ab in self.antibiotics if ab in self.meta.columns
        ]
        if not available_antibiotics:
            raise ValueError(
                f"None of the specified antibiotics {self.antibiotics} found in metadata"
            )

        return self.meta.loc[self.X.index, available_antibiotics]

    def filter(self, *filters: SpectrumFilter) -> MaldiSet:
        """Return a new MaldiSet keeping only samples that pass all filters.

        Filters are applied to the metadata rows (indexed by spectrum ID).
        Multiple filters can be combined with logical operators.

        Parameters
        ----------
        *filters : SpectrumFilter
            One or more filter objects. Use ``&``, ``|``, ``~`` to compose
            complex predicates before passing them in.

        Returns
        -------
        MaldiSet
            A new dataset containing only the matching spectra.

        Examples
        --------
        >>> from maldiamrkit.filters import SpeciesFilter, QualityFilter
        >>> ds.filter(SpeciesFilter("Escherichia coli"))
        >>> ds.filter(SpeciesFilter("E. coli") & QualityFilter(min_snr=5.0))
        """
        keep_ids = set()
        for sid in self.meta.index:
            row = self.meta.loc[sid]
            if all(f(row) for f in filters):
                keep_ids.add(sid)

        kept_spectra = [s for s in self.spectra if s.id in keep_ids]

        # Rebuild metadata with original columns (un-indexed)
        new_meta = self.meta.loc[self.meta.index.isin(keep_ids)].reset_index()

        aggregate_by: dict[str, str | list[str]] = {}
        if self.antibiotics:
            aggregate_by["antibiotics"] = self.antibiotics
        if self.species:
            aggregate_by["species"] = self.species
        return MaldiSet(
            kept_spectra,
            new_meta,
            aggregate_by=aggregate_by or None,
            bin_width=self.bin_width,
            bin_method=self.bin_method,
            bin_kwargs=self.bin_kwargs,
            verbose=self.verbose,
        )

    def get_y_single(self, antibiotic: str | None = None) -> pd.Series:
        """
        Return labels for a single antibiotic.

        Parameters
        ----------
        antibiotic : str, optional
            Antibiotic column name. If None, uses the first antibiotic.

        Returns
        -------
        pd.Series
            Classification labels.

        Raises
        ------
        ValueError
            If antibiotic not specified or not found.
        """
        if antibiotic is None:
            antibiotic = self.antibiotics[0] if self.antibiotics else None
        if antibiotic is None:
            raise ValueError("No antibiotic specified")
        if antibiotic not in self.meta.columns:
            raise ValueError(f"Antibiotic '{antibiotic}' not found in metadata")

        return self.meta.loc[self.X.index, antibiotic]

    def to_csv(self, path: str | Path) -> None:
        """Export the feature matrix to CSV.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        self.X.to_csv(path)

    def to_parquet(self, path: str | Path) -> None:
        """Export the feature matrix to Parquet.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        self.X.to_parquet(path)

    def save_spectra(
        self,
        output_dir: str | Path,
        *,
        stage: str = "preprocessed",
        fmt: str = "txt",
    ) -> None:
        """Save individual spectra to a directory.

        Parameters
        ----------
        output_dir : str or Path
            Directory where spectra will be saved. Created if it does not
            exist.
        stage : str, default="preprocessed"
            Which processing stage to save. One of ``"raw"``,
            ``"preprocessed"``, ``"binned"``.
        fmt : str, default="txt"
            Output format. ``"csv"`` for comma-separated, ``"txt"`` for
            tab-separated.

        Raises
        ------
        ValueError
            If ``stage`` or ``fmt`` is invalid.

        Examples
        --------
        >>> data = MaldiSet.from_directory("spectra/", "metadata.csv")
        >>> data.save_spectra("processed/", stage="preprocessed", fmt="txt")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = fmt
        saved = 0
        for spec in self.spectra:
            try:
                spec.save(output_dir / f"{spec.id}.{ext}", stage=stage, fmt=fmt)
                saved += 1
            except RuntimeError:
                logger.warning("Spectrum %s has no '%s' data - skipped", spec.id, stage)
        if self.verbose:
            logger.info("Saved %d spectra to %s", saved, output_dir)

    def __repr__(self) -> str:
        n = len(self.spectra)
        antibiotics = self.antibiotics or []
        species = self.species or "all"
        return (
            f"MaldiSet(n_spectra={n}, species={species!r}, antibiotics={antibiotics!r})"
        )
