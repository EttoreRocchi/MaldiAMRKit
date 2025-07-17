from __future__ import annotations
from pathlib import Path
import pandas as pd

from .config import PreprocessingConfig
from .preprocessing import preprocess, bin_spectrum
from .io import read_spectrum


class MaldiSpectrum:
    """
    A single MALDI-TOF spectrum.

    Workflow
    --------
    >>> spec = MaldiSpectrum("raw/abc.txt")
    >>> spec.preprocess()
    >>> spec.bin(3)
    """

    def __init__(
            self,
            source: str | Path | pd.DataFrame,
            *,
            cfg: PreprocessingConfig | None = None,
            verbose: bool = False,
        ) -> MaldiSpectrum:
        self.cfg = cfg or PreprocessingConfig()
        self._raw: pd.DataFrame
        self._preprocessed: pd.DataFrame | None = None
        self._binned: pd.DataFrame | None = None
        self.verbose = verbose

        if isinstance(source, (str, Path)):
            self.path = Path(source)
            self._raw = read_spectrum(self.path)
            self.id = self.path.stem
        elif isinstance(source, pd.DataFrame):
            self.path = None
            self._raw = source.copy()
            self.id = "in-memory"
        else:
            raise TypeError("Unsupported source type for MaldiSpectrum")

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()
    
    @property
    def bin_width(self) -> int | float | None:
        return self._bin_width

    @property
    def preprocessed(self) -> pd.DataFrame:
        if self._preprocessed is None:
            raise RuntimeError("Call .preprocess() before accessing this property.")
        return self._preprocessed.copy()

    @property
    def binned(self) -> pd.DataFrame:
        if self._binned is None:
            raise RuntimeError("Call .bin() before accessing this property.")
        return self._binned.copy()

    def preprocess(self, **override) -> MaldiSpectrum:
        """
        Run baseline correction, smoothing, normalisation, trimming.
        Optionally override parameters from the current `PreprocessingConfig`
        with `**override` *kwargs*.
        """
        cfg = self.cfg if not override else self.cfg.__class__(**{**self.cfg.as_dict(), **override})
        self._preprocessed = preprocess(self._raw, cfg)
        if self.verbose:
            print(f"INFO: Preprocessed spectrum {self.id}")
        return self

    def bin(self, bin_width: int | float) -> MaldiSpectrum:
        """
        Binning. If `bin_width` is None we skip binning.
        """
        self._bin_width = bin_width

        if self._preprocessed is None:
            self.preprocess()

        self._binned = bin_spectrum(self._preprocessed, self.cfg, self._bin_width)
        if self.verbose:
            print(F"INFO: Binned spectrum {self.id} (w={self._bin_width})")
        return self

    def plot(self, binned: bool = True, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        _ax = ax or plt.subplots(figsize=(10, 4))[1]
        data = self.binned if binned else (self.preprocessed if self._preprocessed is not None else self.raw)
        if binned:
            sns.barplot(data=data, x="mass", y="intensity", ax=_ax, **kwargs)
        else:
            _ax.plot(data.mass, data.intensity, **kwargs)
        _ax.set(
            title=f"{self.id}{' (binned)' if binned else ''}",
            xlabel="m/z", ylabel="intensity", xticks=[],
            ylim=[0,(data.intensity.max())*1.05]
        )
        return _ax
