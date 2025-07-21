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
                aggregate_by=dict(antibiotic="Ceftriaxone")
        )
    >>> ds.X.shape, ds.y.value_counts()
    """

    def __init__(
            self,
            spectra: list[MaldiSpectrum],
            meta: pd.DataFrame,
            *,
            aggregate_by: dict[str, str],
            bin_width: int = 3,
            verbose: bool = False,
        ) -> MaldiSet:
        self.spectra = spectra
        self.meta = meta.set_index("ID")

        self.antibiotic = aggregate_by.get("antibiotic")
        self.species = aggregate_by.get("species")
        self.bin_width = bin_width

        self.verbose = verbose      
        if verbose:
            print(f"INFO: Dataset created: {len(self.spectra)} spectra")

    @classmethod
    def from_directory(
            self,
            spectra_dir: str | Path,
            meta_file: str | Path,
            *,
            aggregate_by: dict[str, str],
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
        if self.antibiotic:
            df = df[df[self.antibiotic].notna()]
        if self.species:
            df = df[df["Species"] == self.species]

        return df.select_dtypes("number")

    @property
    def y(self) -> pd.Series:
        """Return the classification/label vector (antibiotic resistance)."""
        return self.meta.loc[self.X.index, self.antibiotic]

    def plot_pseudogel(
        self,
        *,
        antibiotic: str | None = None,
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
            raise ValueError(
                "Antibiotic column not defined. "
            )

        X = self.X
        y = self.y

        groups = y.groupby(y).groups
        n_groups = len(groups)
        if figsize is None:
            figsize = (6.0, 2.5 * n_groups)

        fig, axes = plt.subplots(
            n_groups, 1, figsize=figsize, sharex=True, constrained_layout=True
        )
        if n_groups == 1:
            axes = np.asarray([axes])

        for ax, (label, idx) in zip(axes, sorted(groups.items(), key=lambda t: str(t[0]))):
            M = X.loc[idx].to_numpy()
            if sort_by_intensity:
                order = np.argsort(M.mean(axis=1))[::-1]
                M = M[order]
            if log_scale:
                M = np.log1p(M)

            im = ax.imshow(
                M,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_ylabel(f"{label}\n(n={M.shape[0]})", rotation=0, ha="right", va="center")
            ax.set_yticks([])

        xticks = np.linspace(0, X.shape[1] - 1, 6, dtype=int)
        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels([f"{m}" for m in X.columns[xticks]])
        axes[-1].set_xlabel("m/z (binned)")

        cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.01)
        cbar.set_label("Log(intensity + 1)" if log_scale else "intensity")

        if title:
            fig.suptitle(title, y=1.02)

        if show:
            plt.show()

        return fig, axes
