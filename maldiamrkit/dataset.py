from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .spectrum import MaldiSpectrum
from .config import PreprocessingSettings


class MaldiSet:
    """
    A collection of spectra with metadata.

    Example
    -------
    >>> ds = MaldiSet.from_directory(
                "spectra/", "meta.csv",
                aggregate_by=dict(
                    antibiotics=["Ceftriaxone", "Ceftazidime"],
                    species="Escherichia coli",
                    other="batch_id"
                )
        )
    >>> ds.X.shape, ds.y.shape, ds.other.shape
    """

    def __init__(
            self,
            spectra: list[MaldiSpectrum],
            meta: pd.DataFrame,
            *,
            aggregate_by: dict[str, str | list[str]],
            bin_width: int = 3,
            verbose: bool = False,
        ) -> MaldiSet:
        self.spectra = spectra

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

        self.verbose = verbose      
        if verbose:
            print(f"INFO: Dataset created: {len(self.spectra)} spectra")
            if self.antibiotics:
                print(f"INFO: Tracking antibiotics: {self.antibiotics}")
            if self.other_key:
                print(f"INFO: Additional aggregation by: {self.other_key}")

    @classmethod
    def from_directory(
            self,
            spectra_dir: str | Path,
            meta_file: str | Path,
            *,
            aggregate_by: dict[str, str | list[str]],
            cfg: PreprocessingSettings | None = None,
            bin_width: int = 3,
            verbose: bool = False,
        ) -> MaldiSet:
        spectra_dir = Path(spectra_dir)
        specs = [MaldiSpectrum(p, cfg=cfg).bin(bin_width) for p in spectra_dir.glob("*.txt")]
        meta = pd.read_csv(meta_file)
        return self(specs, meta, aggregate_by=aggregate_by, bin_width=bin_width, verbose=verbose)

    @property
    def X(self) -> pd.DataFrame:
        """
        Return matrix (n_samples, n_features) limited to the configured subset.
        """
        rows = []
        for s in self.spectra:
            sid = s.id
            if sid not in self.meta.index and self.verbose:
                print(f"WARNING: ID {sid} missing in metadata - skipped.")
                continue
            row = (s.binned if s._binned is not None else s.bin(self.bin_width).binned) \
                    .set_index("mass")["intensity"].rename(sid)
            rows.append(row)

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

        to_drop = self.meta_cols
        return df.drop(columns=to_drop)

    @property
    def y(self) -> pd.DataFrame:
        """
        Return the classification/label matrix for all specified antibiotics.
        Returns a DataFrame with one column per antibiotic.
        """
        if not self.antibiotics:
            raise ValueError("No antibiotics specified for classification labels")
        
        available_antibiotics = [ab for ab in self.antibiotics if ab in self.meta.columns]
        if not available_antibiotics:
            raise ValueError(f"None of the specified antibiotics {self.antibiotics} found in metadata")
        
        return self.meta.loc[self.X.index, available_antibiotics]

    @property
    def other(self) -> pd.Series:
        """
        Return the additional aggregation variable if specified.
        """
        if not self.other_key:
            raise ValueError("No additional aggregation key specified")
        
        for o_k in self.other_key:
            if o_k not in self.meta.columns:
                raise ValueError(f"Column '{o_k}' not found in metadata")
            
        return self.meta.loc[self.X.index, self.other_key]

    def get_y_single(self, antibiotic: str | None = None) -> pd.Series:
        """
        Return the classification/label vector for a single antibiotic.
        
        Parameters
        ----------
        antibiotic : str | None
            Name of the antibiotic column. If None, uses the first antibiotic.
            
        Returns
        -------
        pd.Series
            Classification labels for the specified antibiotic.
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
        Displays a pseudogel heatmap of the spectra, with one subplot
        for each unique value of the antibiotic column.

        Parameters
        ----------
        antibiotic : str | None
            Name of the target column to use (default: self.antibiotic).
        regions : tuple or list of tuples, optional
            Specific m/z region(s) to display:
            - None: show all regions (default)
            - (min_mz, max_mz): show single region
            - [(min1, max1), (min2, max2), ...]: show multiple regions
        cmap : str
            Matplotlib colormap to use (default: "inferno").
        vmin, vmax : float | None
            Color scale limits. Use None for automatic scaling.
        figsize : (int, int) | None
            Figure size. If None, it is automatically set based on the number of subplots.
        log_scale : bool
            Apply log1p to intensity values to emphasize weaker signals.
        sort_by_intensity : bool
            Sort samples by average intensity before plotting.
        title : str | None
            Title of the overall figure.
        show : bool
            If True, calls plt.show() at the end of the method.

        Returns
        -------
        fig, axes : matplotlib.figure.Figure, ndarray[Axes]
            Matplotlib figure and axes objects, useful for further customization.
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
                    raise ValueError(f"Invalid region: min_mz ({min_mz}) > max_mz ({max_mz})")

                mask = (mz_values >= min_mz) & (mz_values <= max_mz)
                if not mask.any():
                    raise ValueError(f"No m/z values found in region ({min_mz}, {max_mz})")

                region_dfs.append(X.iloc[:, mask])

                # Add blank separator column except after last region
                if len(region_dfs) < len(regions):
                    blank_col = pd.DataFrame(
                        np.nan, 
                        index=X.index, 
                        columns=[f"_blank_{len(region_dfs)}"]
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
        cmap_obj.set_bad(color='white', alpha=1.0)

        for ax, (label, idx) in zip(axes, sorted(groups.items(), key=lambda t: str(t[0]))):
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
            ax.set_ylabel(f"{label}\n(n={M.shape[0]})", rotation=0, ha="right", va="center")
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
