"""Raw spectra warping transformer operating at full m/z resolution."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from ..detection.peak_detector import MaldiPeakDetector
from ..io.readers import read_spectrum
from ..preprocessing.binning import bin_spectrum
from ..preprocessing.pipeline import preprocess
from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
from .strategies import ALIGNMENT_REGISTRY


def create_raw_input(
    spectra_dir: str | Path,
    sample_ids: list[str] | None = None,
    file_extension: str = ".txt",
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
    ...     ("warp", RawWarping(method="piecewise")),
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

    return pd.DataFrame({"path": paths}, index=sample_ids)


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
        - "shift" : global m/z shift in Daltons
        - "linear" : linear m/z transformation (mz' = a*mz + b)
        - "piecewise" : segment-wise m/z shifts with smoothing
        - "dtw" : dynamic time warping
    bin_width : float, default=3
        Width of output bins in Daltons.
    bin_method : str, default="uniform"
        Binning method. One of 'uniform', 'proportional', 'adaptive', 'custom'.
    bin_kwargs : dict, optional
        Additional keyword arguments for binning.
    max_shift_da : float, default=50.0
        Maximum allowed shift in Daltons.
    n_segments : int, default=5
        Number of segments for piecewise warping.
    dtw_radius : int, default=10
        Radius constraint for DTW.
    smooth_sigma : float, default=2.0
        Gaussian smoothing for piecewise transitions.
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

    Attributes
    ----------
    ref_mz_ : np.ndarray
        Reference spectrum m/z values (after fit).
    ref_intensity_ : np.ndarray
        Reference spectrum intensities (after fit).
    ref_peaks_mz_ : np.ndarray
        Peak m/z positions in reference spectrum.
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
    ...     ("warp", RawWarping(method="piecewise", bin_width=3)),
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
    """

    def __init__(
        self,
        method: str = "shift",
        bin_width: float = 3,
        bin_method: str = "uniform",
        bin_kwargs: dict | None = None,
        max_shift_da: float = 50.0,
        n_segments: int = 5,
        dtw_radius: int = 10,
        smooth_sigma: float = 2.0,
        reference: str | int = "median",
        pipeline: PreprocessingPipeline | None = None,
        peak_detector: MaldiPeakDetector | None = None,
        min_reference_peaks: int = 5,
        interp_step: float = 0.5,
        n_jobs: int = 1,
    ) -> None:
        self.method = method
        self.bin_width = bin_width
        self.bin_method = bin_method
        self.bin_kwargs = bin_kwargs
        self.max_shift_da = max_shift_da
        self.n_segments = n_segments
        self.dtw_radius = dtw_radius
        self.smooth_sigma = smooth_sigma
        self.reference = reference
        self.interp_step = interp_step
        self.pipeline = pipeline
        self.peak_detector = peak_detector or MaldiPeakDetector(
            binary=True, prominence=1e-5
        )
        self.min_reference_peaks = min_reference_peaks
        self.n_jobs = n_jobs

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
        """Detect peaks and return their m/z positions using the peak detector."""
        # Create temporary single-row DataFrame for peak detection
        spec_df = pd.DataFrame([intensity], columns=mz)
        peaks_df = self.peak_detector.transform(spec_df)
        # Get m/z positions where peaks were detected (non-zero values)
        peak_mask = peaks_df.iloc[0].to_numpy() != 0
        return mz[peak_mask]

    def _compute_raw_reference(
        self, paths: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reference spectrum from raw data.

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

        # Validate method
        if self.method not in ALIGNMENT_REGISTRY:
            raise ValueError(
                f"Unknown method: {self.method}. Use: {', '.join(ALIGNMENT_REGISTRY)}"
            )

        # Store preprocessing config
        self.pipeline_ = self.pipeline or PreprocessingPipeline.default()

        # Get paths from DataFrame
        paths = X["path"].tolist()

        # Compute reference from raw spectra
        self.ref_mz_, self.ref_intensity_, self.ref_peaks_mz_ = (
            self._compute_raw_reference(paths)
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
        cls = ALIGNMENT_REGISTRY[self.method]
        if self.method == "shift":
            return cls(max_shift=self.max_shift_da)
        elif self.method == "linear":
            return cls(max_shift=self.max_shift_da)
        elif self.method == "piecewise":
            return cls(
                n_segments=self.n_segments,
                smooth_sigma=self.smooth_sigma,
                max_shift=self.max_shift_da,
            )
        elif self.method == "dtw":
            return cls(dtw_radius=self.dtw_radius)
        return cls()

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

        strategy = self._get_strategy()
        warped_mz, warped_int = strategy.align_raw(
            mz,
            intensity,
            peaks_mz,
            self.ref_peaks_mz_,
            self.ref_mz_,
            self.ref_intensity_,
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
        result = pd.DataFrame(
            np.vstack(aligned_rows), index=X.index, columns=self.output_columns_
        )

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
