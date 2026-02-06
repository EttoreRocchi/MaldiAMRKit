"""Multi-spectrum dataset handling."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..preprocessing.binning import _uniform_edges, get_bin_metadata
from .config import PreprocessingSettings
from .spectrum import MaldiSpectrum


def _load_single_spectrum(
    path: Path,
    cfg: PreprocessingSettings | None,
    bin_width: int,
    bin_method: str,
    bin_kwargs: dict,
) -> MaldiSpectrum:
    """Load and process a single spectrum (helper for parallel loading)."""
    return MaldiSpectrum(path, cfg=cfg).bin(bin_width, method=bin_method, **bin_kwargs)


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
        - 'antibiotics' or 'antibiotic': str or list of antibiotic column names
        - 'species': str, species column name
        - 'other': str or list of additional column names
        If None, all spectra are included without antibiotic/species filtering.
    bin_width : int, default=3
        Bin width for spectra.
    bin_method : str, default='uniform'
        Binning method. One of 'uniform', 'logarithmic', 'adaptive', 'custom'.
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
        Species column name.
    meta : pd.DataFrame
        Filtered metadata indexed by ID.

    Examples
    --------
    >>> ds = MaldiSet.from_directory(
    ...     "spectra/", "meta.csv",
    ...     aggregate_by=dict(
    ...         antibiotics=["Ceftriaxone", "Ceftazidime"],
    ...         species="Escherichia coli",
    ...         other="batch_id"
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
    ) -> MaldiSet:
        self.spectra = spectra

        aggregate_by = aggregate_by or {}

        antibiotics = aggregate_by.get("antibiotics") or aggregate_by.get("antibiotic")
        if isinstance(antibiotics, str):
            self.antibiotics = [antibiotics]
        elif isinstance(antibiotics, list):
            self.antibiotics = antibiotics
        else:
            self.antibiotics = None

        self.antibiotic = self.antibiotics[0] if self.antibiotics else None

        self.species = aggregate_by.get("species")

        other_key = aggregate_by.get("other")
        if isinstance(other_key, str):
            self.other_key = [other_key]
        elif isinstance(other_key, list):
            self.other_key = other_key
        else:
            self.other_key = None

        columns_to_keep = ["ID"]
        if self.antibiotics:
            columns_to_keep.extend(self.antibiotics)
        if self.species:
            columns_to_keep.append("Species")
        if self.other_key:
            columns_to_keep.extend(self.other_key)
        columns_to_keep = list(dict.fromkeys(columns_to_keep))

        available_columns = [col for col in columns_to_keep if col in meta.columns]
        missing_columns = [col for col in columns_to_keep if col not in meta.columns]
        if missing_columns and verbose:
            print(f"WARNING: Columns {missing_columns} not found in metadata")

        self.meta = meta[available_columns].set_index("ID")
        self.meta_cols = self.meta.columns.tolist()

        self.bin_width = bin_width
        self.bin_method = bin_method
        self.bin_kwargs = bin_kwargs
        self._bin_metadata: pd.DataFrame | None = None

        self.verbose = verbose
        if verbose:
            print(f"INFO: Dataset created: {len(self.spectra)} spectra")
            if self.antibiotics:
                print(f"INFO: Tracking antibiotics: {self.antibiotics}")
            if self.other_key:
                print(f"INFO: Additional aggregation by: {self.other_key}")

    @classmethod
    def from_directory(
        cls,
        spectra_dir: str | Path,
        meta_file: str | Path,
        *,
        aggregate_by: dict[str, str | list[str]] | None = None,
        cfg: PreprocessingSettings | None = None,
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
            - 'antibiotics' or 'antibiotic': str or list of antibiotic column names
            - 'species': str, species column name
            - 'other': str or list of additional column names
            If None, all spectra matching metadata are loaded without
            antibiotic/species filtering.
        cfg : PreprocessingSettings, optional
            Preprocessing configuration.
        bin_width : int, default=3
            Bin width for spectra.
        bin_method : str, default='uniform'
            Binning method. One of 'uniform', 'logarithmic', 'adaptive', 'custom'.
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
            print(
                f"INFO: Loading {len(spectrum_files)} spectra "
                f"({n_skipped} skipped, not in metadata) with n_jobs={n_jobs}"
            )

        # Use parallel loading with joblib
        specs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_load_single_spectrum)(p, cfg, bin_width, bin_method, _bin_kwargs)
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
            if spec._bin_metadata is not None:
                return spec._bin_metadata.copy()

        # Compute from stored parameters if no spectrum has metadata
        if self._bin_metadata is None:
            cfg = self.spectra[0].cfg if self.spectra else PreprocessingSettings()
            edges = _uniform_edges(cfg, self.bin_width)
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
        """
        bin_kwargs = self.bin_kwargs or {}
        rows = []
        for s in self.spectra:
            sid = s.id
            if sid not in self.meta.index:
                warnings.warn(
                    f"Spectrum ID '{sid}' not found in metadata - skipped.",
                    UserWarning,
                )
                continue
            row = (
                (
                    s.binned
                    if s._binned is not None
                    else s.bin(
                        self.bin_width, method=self.bin_method, **bin_kwargs
                    ).binned
                )
                .set_index("mass")["intensity"]
                .rename(sid)
            )
            rows.append(row)

        if not rows:
            raise ValueError(
                "No spectra matched metadata IDs. "
                "Check that spectrum file names match the 'ID' column in metadata."
            )

        df = pd.concat(rows, axis=1).T

        df = df.join(self.meta, how="left")

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

        to_drop = self.meta_cols
        return df.drop(columns=to_drop)

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

    @property
    def other(self) -> pd.Series:
        """
        Return additional aggregation variables.

        Returns
        -------
        pd.Series or pd.DataFrame
            Additional metadata columns.

        Raises
        ------
        ValueError
            If no other_key specified or column not found.
        """
        if not self.other_key:
            raise ValueError("No additional aggregation key specified")

        for o_k in self.other_key:
            if o_k not in self.meta.columns:
                raise ValueError(f"Column '{o_k}' not found in metadata")

        return self.meta.loc[self.X.index, self.other_key]

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
            antibiotic = self.antibiotic
        if antibiotic is None:
            raise ValueError("No antibiotic specified")
        if antibiotic not in self.meta.columns:
            raise ValueError(f"Antibiotic '{antibiotic}' not found in metadata")

        return self.meta.loc[self.X.index, antibiotic]

    def plot_pseudogel(
        self,
        *,
        antibiotic: str | None = None,
        regions: tuple[float, float] | list[tuple[float, float]] | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        figsize: tuple[int, int] | None = None,
        log_scale: bool = True,
        sort_by_intensity: bool = True,
        title: str | None = None,
        show: bool = True,
    ):
        """
        Display a pseudogel heatmap of the spectra.

        Creates one subplot for each unique value of the antibiotic column.

        Parameters
        ----------
        antibiotic : str, optional
            Target column to group by. Defaults to first antibiotic.
        regions : tuple or list of tuples, optional
            m/z region(s) to display. None shows all.
        cmap : str, default="inferno"
            Matplotlib colormap name.
        vmin, vmax : float, optional
            Color scale limits.
        figsize : tuple, optional
            Figure size. Auto-calculated if None.
        log_scale : bool, default=True
            Apply log1p to intensities.
        sort_by_intensity : bool, default=True
            Sort samples by average intensity.
        title : str, optional
            Figure title.
        show : bool, default=True
            If True, call plt.show().

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : ndarray of Axes
            The subplot axes.
        """
        if antibiotic is None:
            antibiotic = self.antibiotic
        if antibiotic is None:
            raise ValueError("Antibiotic column not defined.")

        X = self.X.copy()
        y = self.get_y_single(antibiotic)

        # Region filtering
        if regions is not None:
            # Normalize to list of tuples
            if isinstance(regions, tuple) and len(regions) == 2:
                regions = [regions]

            # X with regions separated by blank columns
            mz_values = X.columns.astype(float)
            region_dfs = []

            for min_mz, max_mz in regions:
                if min_mz > max_mz:
                    raise ValueError(
                        f"Invalid region: min_mz ({min_mz}) > max_mz ({max_mz})"
                    )

                mask = (mz_values >= min_mz) & (mz_values <= max_mz)
                if not mask.any():
                    raise ValueError(
                        f"No m/z values found in region ({min_mz}, {max_mz})"
                    )

                region_dfs.append(X.iloc[:, mask])

                # Add blank separator column except after last region
                if len(region_dfs) < len(regions):
                    blank_col = pd.DataFrame(
                        np.nan, index=X.index, columns=[f"_blank_{len(region_dfs)}"]
                    )
                    region_dfs.append(blank_col)

            X = pd.concat(region_dfs, axis=1)

        groups = y.groupby(y).groups
        n_groups = len(groups)
        if figsize is None:
            figsize = (6.0, 2.5 * n_groups)

        fig, axes = plt.subplots(
            n_groups, 1, figsize=figsize, sharex=True, constrained_layout=True
        )
        if n_groups == 1:
            axes = np.asarray([axes])

        # Set colormap to handle NaN values (for region separators)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="white", alpha=1.0)

        for ax, (label, idx) in zip(
            axes, sorted(groups.items(), key=lambda t: str(t[0]))
        ):
            M = X.loc[idx].to_numpy()
            if sort_by_intensity:
                order = np.argsort(np.nanmean(M, axis=1))[::-1]
                M = M[order]
            if log_scale:
                M = np.log1p(M)

            im = ax.imshow(
                M,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap_obj,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_ylabel(
                f"{label}\n(n={M.shape[0]})", rotation=0, ha="right", va="center"
            )
            ax.set_yticks([])

        # Set x-axis ticks and labels
        n_ticks = min(10, X.shape[1])
        xticks = np.linspace(0, X.shape[1] - 1, n_ticks, dtype=int)

        # Skip blank separator columns in labels
        xticklabels = []
        for i in xticks:
            col_name = str(X.columns[i])
            if col_name.startswith("_blank_"):
                xticklabels.append("")
            else:
                xticklabels.append(col_name)

        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels(xticklabels, rotation=90)
        axes[-1].set_xlabel("m/z (binned)")

        cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.01)
        cbar.set_label("Log(intensity + 1)" if log_scale else "intensity")

        if title:
            fig.suptitle(title, y=1.02)

        if show:
            plt.show()

        return fig, axes
