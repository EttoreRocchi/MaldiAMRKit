from __future__ import annotations
from pathlib import Path
import pandas as pd

from .spectrum import MaldiSpectrum
from .config import PreprocessingConfig


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
            cfg: PreprocessingConfig | None = None,
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
