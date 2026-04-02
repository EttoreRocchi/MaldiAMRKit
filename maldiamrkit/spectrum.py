"""Single MALDI-TOF spectrum handling."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .io.readers import _find_bruker_acqus, read_spectrum
from .preprocessing.binning import bin_spectrum
from .preprocessing.pipeline import preprocess
from .preprocessing.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


def _infer_id(path: Path) -> str:
    """Infer a spectrum identifier from a file or directory path.

    For files, use the stem (filename without extension).
    For Bruker directories, combine the identifier and target position
    extracted from the relative path to the ``acqus`` file:

    - ``{identifier}/{target_pos}/1/1SLin/acqus`` (depth 2 from
      ``{target_pos}``) -> ``{identifier}_{target_pos}``
    - ``{identifier}/.../{target_pos}/1/1SLin/acqus`` (depth 3 from
      ``{identifier}``) -> ``{identifier}_{target_pos}``
    - ``acqus`` directly in path -> ``path.name``
    """
    if not path.is_dir():
        return path.stem
    acqus = _find_bruker_acqus(path)
    if acqus is None:
        return path.name
    rel_parts = acqus.parent.relative_to(path).parts
    depth = len(rel_parts)
    if depth == 2:
        # Path is at {identifier}/{target_pos} level
        return f"{path.parent.name}_{path.name}"
    if depth == 3:
        # Path is at {identifier} level; first rel part is target_pos
        all_acqus = sorted(path.glob("*/*/*/acqus"))
        if len(all_acqus) > 1:
            targets = [a.parent.relative_to(path).parts[0] for a in all_acqus]
            logger.warning(
                "Multiple target positions found for %s: %s. Using '%s'.",
                path.name,
                targets,
                rel_parts[0],
            )
        return f"{path.name}_{rel_parts[0]}"
    return path.name


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
    pipeline : PreprocessingPipeline, optional
        Preprocessing pipeline. If None, uses the default pipeline.
    verbose : bool, default=False
        If True, print progress messages.

    Attributes
    ----------
    path : Path or None
        Path to the source file, if loaded from file.
    id : str
        Identifier for the spectrum (filename stem or 'in-memory').
    pipeline : PreprocessingPipeline
        Preprocessing pipeline.

    Raises
    ------
    ValueError
        If the source DataFrame is empty or missing required columns
        ('mass', 'intensity').
    TypeError
        If the 'mass' or 'intensity' columns are not numeric, or if
        ``source`` is not a supported type.

    Examples
    --------
    >>> spec = MaldiSpectrum("raw/abc.txt")
    >>> spec.preprocess()
    >>> spec.bin(3)
    >>> from maldiamrkit.visualization import plot_spectrum
    >>> plot_spectrum(spec)
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        *,
        pipeline: PreprocessingPipeline | None = None,
        verbose: bool = False,
    ) -> None:
        self.pipeline = pipeline or PreprocessingPipeline.default()
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
            self.id = _infer_id(self.path)
        elif isinstance(source, pd.DataFrame):
            if source.empty:
                raise ValueError("Cannot create MaldiSpectrum from an empty DataFrame.")
            missing = {"mass", "intensity"} - set(source.columns)
            if missing:
                raise ValueError(
                    f"DataFrame missing required columns: {missing}. "
                    f"Expected 'mass' and 'intensity'."
                )
            if not pd.api.types.is_numeric_dtype(source["mass"]):
                raise TypeError("Column 'mass' must be numeric.")
            if not pd.api.types.is_numeric_dtype(source["intensity"]):
                raise TypeError("Column 'intensity' must be numeric.")
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

    def preprocess(self) -> MaldiSpectrum:
        """
        Run preprocessing pipeline on the raw spectrum.

        Returns
        -------
        MaldiSpectrum
            Self, for method chaining.
        """
        self._preprocessed = preprocess(self._raw, self.pipeline)
        if self.verbose:
            logger.info("Preprocessed spectrum %s", self.id)
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
            For 'proportional', this is the reference width at mz_min.
            Ignored for 'adaptive' and 'custom' methods.
        method : str, default='uniform'
            Binning method. One of 'uniform', 'proportional', 'adaptive', 'custom'.
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
        >>> spec.bin(3, method='proportional')
        >>> spec.bin(method='adaptive', adaptive_min_width=1.0, adaptive_max_width=10.0)
        >>> spec.bin(method='custom', custom_edges=[2000, 5000, 10000, 20000])
        """
        self._bin_width = bin_width
        self._bin_method = method

        if self._preprocessed is None:
            self.preprocess()

        mz_min, mz_max = self.pipeline.mz_range

        self._binned, self._bin_metadata = bin_spectrum(
            self._preprocessed,
            mz_min=mz_min,
            mz_max=mz_max,
            bin_width=bin_width,
            method=method,
            custom_edges=custom_edges,
            **kwargs,
        )
        if self.verbose:
            logger.info(
                "Binned spectrum %s (method=%s, w=%s)", self.id, method, bin_width
            )
        return self

    def save(
        self, path: str | Path, *, stage: str = "binned", fmt: str = "csv"
    ) -> None:
        """Save spectrum data to a file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        stage : str, default="binned"
            Which processing stage to save. One of ``"raw"``,
            ``"preprocessed"``, ``"binned"``.
        fmt : str, default="csv"
            Output format. ``"csv"`` for comma-separated, ``"txt"`` for
            tab-separated.

        Raises
        ------
        ValueError
            If ``stage`` is not one of 'raw', 'preprocessed', or 'binned',
            or if ``fmt`` is not one of 'csv' or 'txt'.
        RuntimeError
            If the requested stage has not been computed yet.
        """
        if stage == "raw":
            df = self.raw
        elif stage == "preprocessed":
            df = self.preprocessed
        elif stage == "binned":
            df = self.binned
        else:
            raise ValueError(
                f"Invalid stage '{stage}'. Use 'raw', 'preprocessed', or 'binned'."
            )
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "txt":
            df.to_csv(path, sep="\t", index=False)
        else:
            raise ValueError(f"Invalid fmt '{fmt}'. Use 'csv' or 'txt'.")

    def get_data(self, prefer: str = "preprocessed") -> pd.DataFrame:
        """Return spectrum data, preferring the requested processing stage.

        Parameters
        ----------
        prefer : str, default="preprocessed"
            Preferred stage: ``"preprocessed"`` or ``"binned"``.  Falls back
            to raw data if the requested stage has not been computed.

        Returns
        -------
        pd.DataFrame
            Copy of the spectrum data at the best available stage.
        """
        if prefer == "binned" and self._binned is not None:
            return self._binned.copy()
        if self._preprocessed is not None:
            return self._preprocessed.copy()
        return self._raw.copy()

    @property
    def is_binned(self) -> bool:
        """Whether the spectrum has been binned."""
        return self._binned is not None

    @property
    def is_preprocessed(self) -> bool:
        """Whether the spectrum has been preprocessed."""
        return self._preprocessed is not None

    @property
    def has_bin_metadata(self) -> bool:
        """Whether bin metadata is available (i.e. ``bin()`` has been called)."""
        return self._bin_metadata is not None

    def __repr__(self) -> str:
        status = []
        if self._preprocessed is not None:
            status.append("preprocessed")
        if self._binned is not None:
            n = len(self._binned)
            status.append(f"binned({n} bins)")
        state = ", ".join(status) if status else "raw"
        return f"MaldiSpectrum(id={self.id!r}, {state})"
