"""Raw spectra warping transformer operating at full m/z resolution."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from ..data.duplicates import DuplicateStrategy, apply_index_strategy
from ..detection.peak_detector import MaldiPeakDetector
from ..io.readers import read_spectrum
from ..preprocessing.binning import BinningMethod, bin_spectrum
from ..preprocessing.pipeline import preprocess
from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
from .strategies import (
    _ALIGNMENT_REGISTRY,
    _SHIFT_ESTIMATORS,
    AlignmentMethod,
    DTWStrategy,
    ShiftStrategy,
    _most_prominent,
)
from .warping import _renormalize


def create_raw_input(
    spectra_dir: str | Path,
    sample_ids: list[str] | None = None,
    file_extension: str = ".txt",
    duplicate_strategy: str | DuplicateStrategy = DuplicateStrategy.first,
) -> pd.DataFrame:
    """
    Create input DataFrame for RawWarping from a directory of spectrum files.

    This utility function creates a DataFrame suitable for use with RawWarping
    in sklearn pipelines. The DataFrame has sample IDs as index and file paths
    as values.

    Parameters
    ----------
    spectra_dir : str or Path
        Directory containing raw spectrum files.
    sample_ids : list of str, optional
        List of sample IDs. If None, discovers all files matching the extension
        in spectra_dir and uses filenames (without extension) as sample IDs.
    file_extension : str, default=".txt"
        File extension for spectrum files.
    duplicate_strategy : str or DuplicateStrategy, default ``"first"``
        How to handle duplicate sample IDs (e.g. the same sample
        appearing in multiple year subdirectories):

        * ``"first"``: keep the first occurrence (default).
        * ``"last"``: keep the last occurrence.
        * ``"drop"``: remove all duplicates.
        * ``"keep_all"``: keep every replicate with ``_repN`` suffixes.
        * ``"average"``: keep all replicates and add an
          ``_original_id`` column so that
          :meth:`RawWarping.transform` can group and average them.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - Index: sample IDs
        - Column "path": full paths to spectrum files

    Raises
    ------
    ValueError
        If no files with the specified extension are found in the directory.

    Examples
    --------
    >>> # Discover all .txt files in directory
    >>> X_raw = create_raw_input("spectra/")
    >>>
    >>> # Specify sample IDs explicitly
    >>> X_raw = create_raw_input("spectra/", sample_ids=["s1", "s2", "s3"])
    >>>
    >>> # Use in pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     ("warp", RawWarping(method="lowess")),
    ...     ("scaler", StandardScaler()),
    ... ])
    >>> X_binned = pipe.fit_transform(X_raw)
    """
    spectra_dir = Path(spectra_dir)

    if sample_ids is None:
        # Discover all spectrum files (recursively for year-subdirectory layouts)
        files = sorted(spectra_dir.rglob(f"*{file_extension}"))
        if not files:
            raise ValueError(
                f"No files with extension '{file_extension}' found in {spectra_dir}"
            )
        sample_ids = [f.stem for f in files]
        paths = [str(f) for f in files]
    else:
        # Build paths from sample IDs
        paths = [str(spectra_dir / f"{sid}{file_extension}") for sid in sample_ids]

    df = pd.DataFrame({"path": paths}, index=sample_ids)
    df = apply_index_strategy(df, DuplicateStrategy(duplicate_strategy))
    return df


class RawWarping(BaseEstimator, TransformerMixin):
    """
    Align MALDI-TOF spectra using raw (full resolution) data.

    Unlike Warping (which operates on binned data), RawWarping:
    - Loads original raw spectra from file paths
    - Performs warping at full m/z resolution
    - Outputs binned spectra for pipeline compatibility

    This approach provides more accurate alignment by avoiding binning
    artifacts during the warping process.

    Parameters
    ----------
    method : str, default="shift"
        Warping method:
        - "shift" : global m/z shift in Daltons, see ``shift_estimator``
        - "linear" : linear m/z transformation (mz' = a*mz + b)
        - "piecewise" : segment-wise m/z shifts with smoothing
        - "dtw" : dynamic time warping
        - "quadratic" : quadratic polynomial fit on matched peak pairs
        - "cubic" : cubic polynomial fit on matched peak pairs
        - "lowess" : LOWESS (Cleveland 1979) non-linear warping
    bin_width : float, default=3
        Width of output bins in Daltons.
    bin_method : str, default="uniform"
        Binning method. One of 'uniform', 'proportional', 'adaptive', 'custom'.
    bin_kwargs : dict, optional
        Additional keyword arguments for binning.
    max_shift_da : float, default=10.0
        Largest shift considered, in Daltons. It bounds the global shift
        (the total of both steps with ``prealign``) and the distance between
        matched peak pairs.
    n_segments : int, default=5
        Number of segments for piecewise warping.
    dtw_radius : int, default=10
        Radius constraint for DTW. DTW memory grows with the square of the
        number of reference points (about 10 GB per spectrum for a 0.5 Da
        grid over 2-20 kDa), so a coarser ``interp_step`` is needed to run
        it on full-resolution spectra.
    smooth_sigma : float, default=2.0
        Gaussian smoothing for piecewise transitions.
    lowess_frac : float, default=0.3
        LOWESS smoothing bandwidth (fraction of matched peaks used for
        each local fit). Applies when ``method="lowess"``.
    lowess_it : int, default=3
        Number of LOWESS robustness iterations. Applies when
        ``method="lowess"``.
    reference : str or int, default="median"
        Reference selection: "median" or int index.
    pipeline : PreprocessingPipeline, optional
        Settings for preprocessing raw spectra.
    peak_detector : MaldiPeakDetector, optional
        Peak detector used to find peaks in spectra. If None, a default
        detector is created with binary=True and prominence=1e-5.
    min_reference_peaks : int, default=5
        Minimum peaks expected in reference.
    interp_step : float, default=0.5
        Step size in Daltons for the common m/z grid used when
        computing a median reference spectrum.
    n_jobs : int, default=1
        Number of parallel jobs for transform. Use -1 for all available
        cores, 1 for sequential processing.
    shift_estimator : {"xcorr", "peaks"}, default="xcorr"
        How the global shift is estimated. ``"xcorr"`` resamples the
        spectrum onto a uniform grid at the reference spacing and takes the
        lag (within ``max_shift_da``) that maximises the Pearson correlation
        with the reference. ``"peaks"`` takes the median offset between
        mutually nearest sample and reference peaks within ``max_shift_da``.
        Either shift is applied only if it raises the correlation with the
        reference by ``min_gain``.
    min_gain : float, default=0.05
        Minimum relative gain in correlation with the reference (as a
        fraction of the correlation it improves on) required to apply a
        shift and to keep the "dtw" result instead of the unwarped spectrum.
        With ``prealign``, each step is checked against its own reference.
    min_support : float, default=0.6
        Share of the matched peak pairs that a non-rigid warp must bring
        within ``max_shift_da / 3`` Daltons of their reference partner to be
        used instead of the global shift.
    n_peaks : int or None, default=200
        Number of most prominent detected peaks, in each spectrum and in
        the reference, used to match peaks (by the "peaks" shift estimator
        and the non-rigid methods). ``None`` uses every detected peak.
    prealign : bool, default=True
        Shift the m/z axis of each spectrum first and fit the alignment to
        the median of the shifted training spectra; :meth:`transform`
        applies the same two steps at full resolution before binning, and
        the global shift of both steps together stays within
        ``max_shift_da``. This keeps a median reference sharp when the
        spectra come from different calibrations (for instance several
        instruments).
    renormalize : bool, default=False
        Divide each aligned, binned spectrum by its total intensity, so
        that it sums to 1 (spectra with no intensity are left unchanged).

    Attributes
    ----------
    ref_mz_ : np.ndarray
        Reference spectrum m/z values (after fit).
    ref_intensity_ : np.ndarray
        Reference spectrum intensities (after fit).
    ref_peaks_mz_ : np.ndarray
        m/z positions of the ``n_peaks`` most prominent peaks of the
        reference spectrum.
    prealign_mz_, prealign_intensity_, prealign_peaks_mz_ : np.ndarray or None
        Reference of the global-shift step applied first (and its peaks),
        with ``prealign=True``; otherwise None. ``ref_mz_`` and
        ``ref_intensity_`` then hold the reference of the shifted spectra.
    output_columns_ : pd.Index
        Column names for output DataFrame (m/z bin centers).
    pipeline_ : PreprocessingPipeline
        Preprocessing configuration used.

    Examples
    --------
    >>> from maldiamrkit.alignment import RawWarping, create_raw_input
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> # Create input DataFrame from directory
    >>> X_raw = create_raw_input("spectra/")
    >>>
    >>> # Use in sklearn pipeline
    >>> pipe = Pipeline([
    ...     ("warp", RawWarping(method="lowess", bin_width=3)),
    ...     ("scaler", StandardScaler()),
    ...     ("clf", RandomForestClassifier())
    ... ])
    >>> pipe.fit(X_raw, y)

    Notes
    -----
    Input DataFrame X must have:
    - Index: sample IDs
    - Column "path": paths to raw spectrum files

    Use `create_raw_input()` to easily create this DataFrame from a directory.

    The non-rigid methods fit their warp on the ``n_peaks`` most prominent
    peaks, matched around the global shift (after the shift step of
    ``prealign``), and use it only if the warped m/z axis is strictly
    increasing (otherwise the global shift is used, with a warning) and it
    is supported by at least ``min_support`` of the matched pairs. Apart
    from "dtw", every method moves the m/z values and leaves the
    intensities unchanged, so binning preserves the total intensity within
    the m/z range; "dtw" does not, because it averages the sample points
    matched to each reference point. ``renormalize=True`` makes every
    output spectrum sum to 1 whatever the method.

    Only the reference spectrum is learned in :meth:`fit`; :meth:`transform`
    aligns each spectrum on its own (replicates of one sample are averaged
    after alignment), so its result does not depend on the other spectra
    transformed with it.
    """

    def __init__(
        self,
        method: str | AlignmentMethod = AlignmentMethod.shift,
        bin_width: float = 3,
        bin_method: str | BinningMethod = BinningMethod.uniform,
        bin_kwargs: dict | None = None,
        max_shift_da: float = 10.0,
        n_segments: int = 5,
        dtw_radius: int = 10,
        smooth_sigma: float = 2.0,
        lowess_frac: float = 0.3,
        lowess_it: int = 3,
        reference: str | int = "median",
        pipeline: PreprocessingPipeline | None = None,
        peak_detector: MaldiPeakDetector | None = None,
        min_reference_peaks: int = 5,
        interp_step: float = 0.5,
        n_jobs: int = 1,
        shift_estimator: str = "xcorr",
        min_gain: float = 0.05,
        min_support: float = 0.6,
        n_peaks: int | None = 200,
        prealign: bool = True,
        renormalize: bool = False,
    ) -> None:
        self.method = AlignmentMethod(method)
        self.bin_width = bin_width
        self.bin_method = BinningMethod(bin_method)
        self.bin_kwargs = bin_kwargs
        self.max_shift_da = max_shift_da
        self.n_segments = n_segments
        self.dtw_radius = dtw_radius
        self.smooth_sigma = smooth_sigma
        self.lowess_frac = lowess_frac
        self.lowess_it = lowess_it
        self.reference = reference
        self.interp_step = interp_step
        self.pipeline = pipeline
        self.peak_detector = peak_detector or MaldiPeakDetector(
            binary=True, prominence=1e-5
        )
        self.min_reference_peaks = min_reference_peaks
        self.n_jobs = n_jobs
        self.shift_estimator = shift_estimator
        self.min_gain = min_gain
        self.min_support = min_support
        self.n_peaks = n_peaks
        self.prealign = prealign
        self.renormalize = renormalize

    def _load_raw_spectrum(self, path: str) -> pd.DataFrame:
        """Load and preprocess a raw spectrum from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Spectrum file not found: {path}. "
                f"Ensure paths in input DataFrame are correct."
            )
        raw = read_spectrum(path)
        return preprocess(raw, self.pipeline_)

    def _detect_peaks_mz(self, mz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """m/z positions of the ``n_peaks`` most prominent detected peaks."""
        # Create temporary single-row DataFrame for peak detection
        spec_df = pd.DataFrame([intensity], columns=mz)
        peaks_df = self.peak_detector.transform(spec_df)
        peaks = np.flatnonzero(peaks_df.iloc[0].to_numpy() != 0)
        return mz[_most_prominent(intensity, peaks, self.n_peaks)]

    def _compute_raw_reference(
        self, paths: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reference spectrum from raw data.

        Parameters
        ----------
        paths : list[str]
            Paths to raw spectrum files.

        Returns
        -------
        ref_mz : np.ndarray
            Reference m/z values.
        ref_intensity : np.ndarray
            Reference intensity values.
        ref_peaks_mz : np.ndarray
            Peak m/z positions in reference.

        Notes
        -----
        For ``reference='median'``, all spectra are loaded into memory.
        Memory usage is O(N * M) where N is the number of spectra and M
        is the number of m/z points per spectrum.
        """
        if isinstance(self.reference, int):
            # Use specific spectrum as reference
            if self.reference < 0 or self.reference >= len(paths):
                raise ValueError(
                    f"Reference index {self.reference} out of bounds "
                    f"for {len(paths)} samples"
                )
            ref_df = self._load_raw_spectrum(paths[self.reference])
            ref_mz = ref_df["mass"].to_numpy()
            ref_intensity = ref_df["intensity"].to_numpy()

        elif self.reference == "median":
            # Compute median spectrum on a common m/z grid.
            # First pass: determine the common m/z range.
            all_mz = []
            for path in paths:
                spec_df = self._load_raw_spectrum(path)
                all_mz.append(spec_df["mass"].to_numpy())

            mz_min = max(mz.min() for mz in all_mz)
            mz_max = min(mz.max() for mz in all_mz)
            common_mz = np.arange(mz_min, mz_max, self.interp_step)

            # Second pass: interpolate onto common grid incrementally
            # to avoid holding all raw DataFrames in memory.
            # Memory usage: O(N * len(common_mz)) float64.
            intensities = np.empty((len(paths), len(common_mz)))
            for i, path in enumerate(paths):
                spec_df = self._load_raw_spectrum(path)
                intensities[i] = np.interp(
                    common_mz,
                    spec_df["mass"].to_numpy(),
                    spec_df["intensity"].to_numpy(),
                )

            ref_mz = common_mz
            ref_intensity = np.median(intensities, axis=0)
        else:
            raise ValueError(
                f"Unsupported reference: {self.reference}. Use 'median' or int."
            )

        # Detect peaks in reference
        ref_peaks_mz = self._detect_peaks_mz(ref_mz, ref_intensity)

        return ref_mz, ref_intensity, ref_peaks_mz

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit by computing the reference spectrum from raw data.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with sample IDs as index and a "path" column
            containing paths to raw spectrum files. Use `create_raw_input()`
            to easily create this DataFrame.
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : RawWarping
            Fitted transformer.

        Raises
        ------
        ValueError
            If the input DataFrame is empty, lacks a 'path' column,
            or uses an unknown warping method.
        """
        if X.empty:
            raise ValueError("Input DataFrame X is empty")

        if "path" not in X.columns:
            raise ValueError(
                "Input DataFrame must have a 'path' column with file paths. "
                "Use create_raw_input() to create the input DataFrame."
            )

        if self.shift_estimator not in _SHIFT_ESTIMATORS:
            raise ValueError(
                f"shift_estimator must be one of {_SHIFT_ESTIMATORS}, "
                f"got {self.shift_estimator!r}"
            )
        if self.min_gain < 0:
            raise ValueError(f"min_gain must be >= 0, got {self.min_gain}")
        if not (0.0 <= self.min_support <= 1.0):
            raise ValueError(f"min_support must be in [0, 1], got {self.min_support}")
        if self.n_peaks is not None and self.n_peaks < 1:
            raise ValueError(f"n_peaks must be >= 1 or None, got {self.n_peaks}")

        # Store preprocessing config
        self.pipeline_ = self.pipeline or PreprocessingPipeline.default()

        # Get paths from DataFrame
        paths = X["path"].tolist()

        # Compute reference from raw spectra
        self.ref_mz_, self.ref_intensity_, self.ref_peaks_mz_ = (
            self._compute_raw_reference(paths)
        )
        self.prealign_mz_ = self.prealign_intensity_ = self.prealign_peaks_mz_ = None
        if self._prealigns():
            self.prealign_mz_ = self.ref_mz_
            self.prealign_intensity_ = self.ref_intensity_
            self.prealign_peaks_mz_ = self.ref_peaks_mz_
            if self.reference == "median":
                self.ref_intensity_ = self._shifted_median(paths)
                self.ref_peaks_mz_ = self._detect_peaks_mz(
                    self.ref_mz_, self.ref_intensity_
                )

        # Validate reference quality
        n_peaks = len(self.ref_peaks_mz_)
        if n_peaks < self.min_reference_peaks:
            warnings.warn(
                f"Reference spectrum has only {n_peaks} peaks. "
                f"Expected at least {self.min_reference_peaks}. "
                f"Alignment quality may be poor.",
                UserWarning,
                stacklevel=2,
            )

        # Determine output columns by binning a sample spectrum
        sample_binned = self._bin_warped(self.ref_mz_, self.ref_intensity_)
        self.output_columns_ = sample_binned.set_index("mass").index.astype(str)

        return self

    def _get_strategy(self):
        """Build strategy instance from current parameters."""
        cls = _ALIGNMENT_REGISTRY[self.method]
        shared = dict(
            max_shift=self.max_shift_da,
            estimator=self.shift_estimator,
            min_gain=self.min_gain,
        )
        warp = dict(shared, min_support=self.min_support)
        if self.method == "shift":
            return cls(**shared)
        elif self.method == "linear":
            return cls(**warp)
        elif self.method == "piecewise":
            return cls(
                n_segments=self.n_segments, smooth_sigma=self.smooth_sigma, **warp
            )
        elif self.method == "dtw":
            return cls(dtw_radius=self.dtw_radius, min_gain=self.min_gain)
        elif self.method == "quadratic":
            return cls(degree=2, **warp)
        elif self.method == "cubic":
            return cls(degree=3, **warp)
        elif self.method == "lowess":
            return cls(frac=self.lowess_frac, it=self.lowess_it, **warp)
        return cls()

    def _prealigns(self) -> bool:
        """Whether a global-shift step runs before the alignment."""
        return bool(self.prealign)

    def _prealign_shift(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray | None = None,
    ) -> float:
        """Global shift (in Daltons) of a spectrum to the prealign reference."""
        if peaks_mz is None and self.shift_estimator == "peaks":
            peaks_mz = self._detect_peaks_mz(mz, intensity)
        strategy = ShiftStrategy(self.max_shift_da, self.shift_estimator, self.min_gain)
        return strategy.estimate_raw(
            mz,
            intensity,
            peaks_mz,
            self.prealign_peaks_mz_,
            self.prealign_mz_,
            self.prealign_intensity_,
        )

    def _shifted_median(self, paths: list[str]) -> np.ndarray:
        """Median on ``ref_mz_`` of the training spectra after their shift."""
        rows = np.empty((len(paths), len(self.ref_mz_)))
        for i, path in enumerate(paths):
            spec_df = self._load_raw_spectrum(path)
            mz = spec_df["mass"].to_numpy()
            intensity = spec_df["intensity"].to_numpy()
            shift = self._prealign_shift(mz, intensity)
            rows[i] = np.interp(self.ref_mz_, mz + shift, intensity)
        return np.median(rows, axis=0)

    def _bin_warped(self, mz: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
        """Bin warped raw spectrum to output grid."""
        warped_df = pd.DataFrame({"mass": mz, "intensity": intensity})
        bin_kwargs = self.bin_kwargs or {}
        mz_min, mz_max = self.pipeline_.mz_range
        binned, _ = bin_spectrum(
            warped_df,
            mz_min=mz_min,
            mz_max=mz_max,
            bin_width=self.bin_width,
            method=self.bin_method,
            **bin_kwargs,
        )
        return binned

    def _process_single_sample(self, path: str) -> np.ndarray:
        """Process a single sample: load, warp, and bin."""
        spec_df = self._load_raw_spectrum(path)
        mz = spec_df["mass"].to_numpy()
        intensity = spec_df["intensity"].to_numpy()

        peaks_mz = self._detect_peaks_mz(mz, intensity)
        applied = 0.0
        if self.prealign_mz_ is not None:
            applied = self._prealign_shift(mz, intensity, peaks_mz)
            mz, peaks_mz = mz + applied, peaks_mz + applied

        strategy = self._get_strategy()
        budget = {}
        if not isinstance(strategy, DTWStrategy):
            budget["shift_bounds"] = (
                -self.max_shift_da - applied,
                self.max_shift_da - applied,
            )
        warped_mz, warped_int = strategy.align_raw(
            mz,
            intensity,
            peaks_mz,
            self.ref_peaks_mz_,
            self.ref_mz_,
            self.ref_intensity_,
            **budget,
        )

        binned = self._bin_warped(warped_mz, warped_int)
        return binned.set_index("mass")["intensity"].to_numpy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform spectra by loading raw data, warping, and binning.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with sample IDs as index and a "path" column
            containing paths to raw spectrum files.

        Returns
        -------
        X_aligned : pd.DataFrame
            Aligned and binned spectra with sample IDs as index and
            m/z bin centers as columns.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted.
        ValueError
            If the input DataFrame lacks a 'path' column.
        """
        if not hasattr(self, "ref_mz_"):
            raise RuntimeError("RawWarping must be fitted before transform")

        if "path" not in X.columns:
            raise ValueError(
                "Input DataFrame must have a 'path' column with file paths."
            )

        # Get paths from DataFrame
        paths = X["path"].tolist()

        # Use parallel processing with joblib
        # "loky" backend works well for mixed I/O + CPU workloads
        aligned_rows = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._process_single_sample)(path) for path in paths
        )

        # Combine into DataFrame
        aligned = np.vstack(aligned_rows)
        if self.renormalize:
            _renormalize(aligned)
        result = pd.DataFrame(aligned, index=X.index, columns=self.output_columns_)

        # Average replicates when create_raw_input used strategy="average"
        if "_original_id" in X.columns:
            original_ids = X["_original_id"]
            result["_original_id"] = original_ids.values
            averaged = result.groupby("_original_id").mean()
            averaged.index.name = None
            result = averaged

        return result

    def get_alignment_quality(
        self, X: pd.DataFrame, X_aligned: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Compute alignment quality metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with "path" column.
        X_aligned : pd.DataFrame, optional
            Aligned spectra. If None, will compute via transform.

        Returns
        -------
        pd.DataFrame
            Quality metrics per sample with columns:
            - correlation_before: Pearson correlation with reference (before)
            - correlation_after: Pearson correlation with reference (after)
            - improvement: correlation_after - correlation_before

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted.
        """
        if not hasattr(self, "ref_mz_"):
            raise RuntimeError("RawWarping must be fitted first")

        if X_aligned is None:
            X_aligned = self.transform(X)

        # Bin reference for comparison
        ref_binned = self._bin_warped(self.ref_mz_, self.ref_intensity_)
        ref_vec = ref_binned.set_index("mass")["intensity"].to_numpy()

        metrics = []
        paths = X["path"].tolist()
        for sample_id, path in zip(X.index, paths, strict=True):
            # Load and bin original spectrum for comparison
            spec_df = self._load_raw_spectrum(path)
            original_binned = self._bin_warped(
                spec_df["mass"].to_numpy(), spec_df["intensity"].to_numpy()
            )
            original = original_binned.set_index("mass")["intensity"].to_numpy()
            aligned = X_aligned.loc[sample_id].to_numpy()

            # Ensure same length
            min_len = min(len(original), len(aligned), len(ref_vec))
            original = original[:min_len]
            aligned = aligned[:min_len]
            ref = ref_vec[:min_len]

            # Correlation with reference (NaN when a signal has zero variance)
            corr_before = np.corrcoef(original, ref)[0, 1]
            corr_after = np.corrcoef(aligned, ref)[0, 1]

            if np.isnan(corr_before) or np.isnan(corr_after):
                warnings.warn(
                    f"Correlation undefined for sample {sample_id} "
                    f"(constant signal); defaulting to 0.0",
                    UserWarning,
                    stacklevel=2,
                )
                corr_before = 0.0 if np.isnan(corr_before) else corr_before
                corr_after = 0.0 if np.isnan(corr_after) else corr_after

            metrics.append(
                {
                    "correlation_before": corr_before,
                    "correlation_after": corr_after,
                    "improvement": corr_after - corr_before,
                }
            )

        return pd.DataFrame(metrics, index=X.index)
