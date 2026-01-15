"""Single MALDI-TOF spectrum handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io.readers import read_spectrum
from ..preprocessing.binning import bin_spectrum
from ..preprocessing.pipeline import preprocess
from .config import PreprocessingSettings


class MaldiSpectrum:
    """
    A single MALDI-TOF spectrum.

    Provides methods for loading, preprocessing, binning, and visualizing
    individual mass spectra.

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Source of the spectrum data. Can be a file path or a DataFrame
        with columns 'mass' and 'intensity'.
    cfg : PreprocessingSettings, optional
        Configuration for preprocessing. If None, uses default settings.
    verbose : bool, default=False
        If True, print progress messages.

    Attributes
    ----------
    path : Path or None
        Path to the source file, if loaded from file.
    id : str
        Identifier for the spectrum (filename stem or 'in-memory').
    cfg : PreprocessingSettings
        Preprocessing configuration.

    Examples
    --------
    >>> spec = MaldiSpectrum("raw/abc.txt")
    >>> spec.preprocess()
    >>> spec.bin(3)
    >>> spec.plot()
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        *,
        cfg: PreprocessingSettings | None = None,
        verbose: bool = False,
    ) -> MaldiSpectrum:
        self.cfg = cfg or PreprocessingSettings()
        self._raw: pd.DataFrame
        self._preprocessed: pd.DataFrame | None = None
        self._binned: pd.DataFrame | None = None
        self._bin_width: int | float | None = None
        self._bin_method: str | None = None
        self._bin_metadata: pd.DataFrame | None = None
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
        """Return a copy of the raw spectrum data."""
        return self._raw.copy()

    @property
    def bin_width(self) -> int | float | None:
        """Return the bin width used for binning, or None if not binned."""
        return self._bin_width

    @property
    def bin_method(self) -> str | None:
        """Return the binning method used, or None if not binned."""
        return self._bin_method

    @property
    def bin_metadata(self) -> pd.DataFrame:
        """
        Return bin metadata with bin boundaries and widths.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: bin_index, bin_start, bin_end, bin_width.

        Raises
        ------
        RuntimeError
            If bin() has not been called.
        """
        if self._bin_metadata is None:
            raise RuntimeError("Call .bin() before accessing this property.")
        return self._bin_metadata.copy()

    @property
    def preprocessed(self) -> pd.DataFrame:
        """
        Return the preprocessed spectrum.

        Raises
        ------
        RuntimeError
            If preprocess() has not been called.
        """
        if self._preprocessed is None:
            raise RuntimeError("Call .preprocess() before accessing this property.")
        return self._preprocessed.copy()

    @property
    def binned(self) -> pd.DataFrame:
        """
        Return the binned spectrum.

        Raises
        ------
        RuntimeError
            If bin() has not been called.
        """
        if self._binned is None:
            raise RuntimeError("Call .bin() before accessing this property.")
        return self._binned.copy()

    def preprocess(self, **override) -> MaldiSpectrum:
        """
        Run preprocessing pipeline on the raw spectrum.

        Applies baseline correction, smoothing, normalization, and trimming.

        Parameters
        ----------
        **override : dict
            Override parameters from the current PreprocessingSettings.

        Returns
        -------
        MaldiSpectrum
            Self, for method chaining.
        """
        cfg = (
            self.cfg
            if not override
            else self.cfg.__class__(**{**self.cfg.as_dict(), **override})
        )
        self._preprocessed = preprocess(self._raw, cfg)
        if self.verbose:
            print(f"INFO: Preprocessed spectrum {self.id}")
        return self

    def bin(
        self,
        bin_width: int | float = 3,
        method: str = "uniform",
        custom_edges: np.ndarray | list | None = None,
        **kwargs,
    ) -> MaldiSpectrum:
        """
        Bin the spectrum into m/z intervals.

        Automatically calls preprocess() if not already done.
        Supports multiple binning strategies.

        Parameters
        ----------
        bin_width : int or float, default=3
            Width of each bin in Daltons. For 'uniform', this is the fixed width.
            For 'logarithmic', this is the reference width at mz_start.
            Ignored for 'adaptive' and 'custom' methods.
        method : str, default='uniform'
            Binning method. One of 'uniform', 'logarithmic', 'adaptive', 'custom'.
        custom_edges : array-like, optional
            User-provided bin edges. Required if method='custom'.
        **kwargs : dict
            Additional parameters for specific methods:
            - adaptive_min_width : float, default=1.0
            - adaptive_max_width : float, default=10.0

        Returns
        -------
        MaldiSpectrum
            Self, for method chaining.

        Examples
        --------
        >>> spec.bin(3)  # uniform binning
        >>> spec.bin(3, method='logarithmic')
        >>> spec.bin(method='adaptive', adaptive_min_width=1.0, adaptive_max_width=10.0)
        >>> spec.bin(method='custom', custom_edges=[2000, 5000, 10000, 20000])
        """
        self._bin_width = bin_width
        self._bin_method = method

        if self._preprocessed is None:
            self.preprocess()

        self._binned, self._bin_metadata = bin_spectrum(
            self._preprocessed,
            self.cfg,
            bin_width=bin_width,
            method=method,
            custom_edges=custom_edges,
            **kwargs,
        )
        if self.verbose:
            print(f"INFO: Binned spectrum {self.id} (method={method}, w={bin_width})")
        return self

    def plot(self, binned: bool = True, ax=None, **kwargs):
        """
        Plot the spectrum.

        Parameters
        ----------
        binned : bool, default=True
            If True, plot the binned spectrum. Otherwise, plot preprocessed
            or raw spectrum.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        _ax = ax or plt.subplots(figsize=(10, 4))[1]
        data = (
            self.binned
            if binned
            else (self.preprocessed if self._preprocessed is not None else self.raw)
        )
        if binned:
            sns.barplot(data=data, x="mass", y="intensity", ax=_ax, **kwargs)
        else:
            _ax.plot(data.mass, data.intensity, **kwargs)
        _ax.set(
            title=f"{self.id}{' (binned)' if binned else ''}",
            xlabel="m/z",
            ylabel="intensity",
            xticks=[],
            ylim=[0, (data.intensity.max()) * 1.05],
        )
        return _ax
