"""Raw spectra warping transformer operating at full m/z resolution."""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed

from ..core.config import PreprocessingSettings
from ..preprocessing.pipeline import preprocess
from ..preprocessing.binning import bin_spectrum
from ..io.readers import read_spectrum
from ..detection.peak_detector import MaldiPeakDetector

from fastdtw import fastdtw


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
        # Discover all spectrum files
        files = sorted(spectra_dir.glob(f"*{file_extension}"))
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
        Binning method. One of 'uniform', 'logarithmic', 'adaptive', 'custom'.
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
    preprocessing_cfg : PreprocessingSettings, optional
        Settings for preprocessing raw spectra.
    peak_detector : MaldiPeakDetector, optional
        Peak detector used to find peaks in spectra. If None, a default
        detector is created with binary=True and prominence=1e-5.
    min_reference_peaks : int, default=5
        Minimum peaks expected in reference.
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
    preprocessing_cfg_ : PreprocessingSettings
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
        preprocessing_cfg: PreprocessingSettings | None = None,
        peak_detector: MaldiPeakDetector | None = None,
        min_reference_peaks: int = 5,
        n_jobs: int = 1,
    ) -> RawWarping:
        self.method = method
        self.bin_width = bin_width
        self.bin_method = bin_method
        self.bin_kwargs = bin_kwargs
        self.max_shift_da = max_shift_da
        self.n_segments = n_segments
        self.dtw_radius = dtw_radius
        self.smooth_sigma = smooth_sigma
        self.reference = reference
        self.preprocessing_cfg = preprocessing_cfg
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
        return preprocess(raw, self.preprocessing_cfg_)

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
        """
        if isinstance(self.reference, int):
            # Use specific spectrum as reference
            if self.reference < 0 or self.reference >= len(paths):
                raise ValueError(
                    f"Reference index {self.reference} out of bounds "
                    f"for {len(paths)} samples"
                )
            ref_df = self._load_raw_spectrum(paths[self.reference])
            ref_mz = ref_df['mass'].to_numpy()
            ref_intensity = ref_df['intensity'].to_numpy()

        elif self.reference == "median":
            # Compute median spectrum on a common m/z grid
            all_mz = []
            all_specs = []

            for path in paths:
                spec_df = self._load_raw_spectrum(path)
                all_mz.append(spec_df['mass'].to_numpy())
                all_specs.append(spec_df)

            # Create common grid based on min/max m/z
            mz_min = max(mz.min() for mz in all_mz)
            mz_max = min(mz.max() for mz in all_mz)
            # Use fine step for interpolation
            common_mz = np.arange(mz_min, mz_max, 0.5)

            # Interpolate all spectra onto common grid
            intensities = []
            for spec_df in all_specs:
                interp_int = np.interp(
                    common_mz,
                    spec_df['mass'].to_numpy(),
                    spec_df['intensity'].to_numpy()
                )
                intensities.append(interp_int)

            # Compute median
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
        """
        if X.empty:
            raise ValueError("Input DataFrame X is empty")

        if "path" not in X.columns:
            raise ValueError(
                "Input DataFrame must have a 'path' column with file paths. "
                "Use create_raw_input() to create the input DataFrame."
            )

        # Validate method
        if self.method not in ["shift", "linear", "piecewise", "dtw"]:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Use: shift, linear, piecewise, dtw"
            )

        # Store preprocessing config
        self.preprocessing_cfg_ = self.preprocessing_cfg or PreprocessingSettings()

        # Get paths from DataFrame
        paths = X["path"].tolist()

        # Compute reference from raw spectra
        self.ref_mz_, self.ref_intensity_, self.ref_peaks_mz_ = \
            self._compute_raw_reference(paths)

        # Validate reference quality
        n_peaks = len(self.ref_peaks_mz_)
        if n_peaks < self.min_reference_peaks:
            warnings.warn(
                f"Reference spectrum has only {n_peaks} peaks. "
                f"Expected at least {self.min_reference_peaks}. "
                f"Alignment quality may be poor.",
                UserWarning
            )

        # Determine output columns by binning a sample spectrum
        sample_binned = self._bin_warped(self.ref_mz_, self.ref_intensity_)
        self.output_columns_ = sample_binned.set_index('mass').index.astype(str)

        return self

    def _shift_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply global m/z shift based on peak matching."""
        if len(peaks_mz) == 0 or len(self.ref_peaks_mz_) == 0:
            return mz, intensity

        # Match peaks to nearest reference peaks
        shifts = []
        for p in peaks_mz:
            nearest_idx = np.argmin(np.abs(self.ref_peaks_mz_ - p))
            nearest = self.ref_peaks_mz_[nearest_idx]
            shifts.append(nearest - p)

        shift_da = np.median(shifts) if shifts else 0.0
        shift_da = np.clip(shift_da, -self.max_shift_da, self.max_shift_da)

        return mz + shift_da, intensity

    def _linear_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply linear m/z transformation: mz' = a*mz + b."""
        if len(peaks_mz) < 2 or len(self.ref_peaks_mz_) < 2:
            # Fall back to shift
            return self._shift_raw(mz, intensity, peaks_mz)

        # Match peaks to nearest reference peaks
        sample_peaks = []
        ref_peaks = []
        for p in peaks_mz:
            nearest_idx = np.argmin(np.abs(self.ref_peaks_mz_ - p))
            nearest = self.ref_peaks_mz_[nearest_idx]
            sample_peaks.append(p)
            ref_peaks.append(nearest)

        sample_peaks = np.array(sample_peaks)
        ref_peaks = np.array(ref_peaks)

        # Fit linear transformation: ref = a * sample + b
        A = np.vstack([sample_peaks, np.ones_like(sample_peaks)]).T
        a, b = np.linalg.lstsq(A, ref_peaks, rcond=None)[0]

        # Apply transformation
        new_mz = a * mz + b

        return new_mz, intensity

    def _piecewise_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peaks_mz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply piecewise m/z transformation with smoothing."""
        if len(peaks_mz) == 0 or len(self.ref_peaks_mz_) == 0:
            return mz, intensity

        # Match peaks to reference
        sample_peaks = []
        ref_peaks = []
        for p in peaks_mz:
            nearest_idx = np.argmin(np.abs(self.ref_peaks_mz_ - p))
            nearest = self.ref_peaks_mz_[nearest_idx]
            sample_peaks.append(p)
            ref_peaks.append(nearest)

        sample_peaks = np.array(sample_peaks)
        ref_peaks = np.array(ref_peaks)

        # Divide into segments
        quantiles = np.linspace(0, 1, self.n_segments + 1)
        boundaries = np.quantile(sample_peaks, quantiles)

        seg_x, seg_shift = [], []
        for q in range(self.n_segments):
            if q == self.n_segments - 1:
                mask = (
                    (sample_peaks >= boundaries[q]) &
                    (sample_peaks <= boundaries[q + 1])
                )
            else:
                mask = (
                    (sample_peaks >= boundaries[q]) &
                    (sample_peaks < boundaries[q + 1])
                )

            if mask.sum() > 0:
                seg_x.append(np.median(sample_peaks[mask]))
                seg_shift.append(np.median(ref_peaks[mask] - sample_peaks[mask]))

        if len(seg_x) == 0:
            return mz, intensity

        # Interpolate shifts across spectrum
        shift_interp = np.interp(
            mz, seg_x, seg_shift,
            left=seg_shift[0], right=seg_shift[-1]
        )

        # Apply Gaussian smoothing
        if self.smooth_sigma > 0:
            # Estimate appropriate sigma based on m/z spacing
            mz_spacing = np.median(np.diff(mz))
            sigma_points = int(self.smooth_sigma / mz_spacing)
            if sigma_points > 1:
                shift_interp = gaussian_filter1d(
                    shift_interp, sigma=sigma_points, mode='nearest'
                )

        new_mz = mz + shift_interp

        return new_mz, intensity

    def _dtw_raw(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply DTW alignment on raw spectra."""
        # Resample query spectrum to reference m/z grid
        query_intensity = np.interp(self.ref_mz_, mz, intensity)

        # Compute DTW
        distance, path = fastdtw(
            query_intensity, self.ref_intensity_,
            radius=self.dtw_radius,
            dist=lambda a, b: (a - b) ** 2
        )

        # Create aligned intensity by following warping path
        aligned_sum = np.zeros_like(self.ref_intensity_)
        aligned_count = np.zeros_like(self.ref_intensity_)

        for i, j in path:
            if 0 <= j < len(aligned_sum):
                aligned_sum[j] += query_intensity[i]
                aligned_count[j] += 1

        aligned_intensity = np.zeros_like(self.ref_intensity_)
        mask = aligned_count > 0
        aligned_intensity[mask] = aligned_sum[mask] / aligned_count[mask]

        return self.ref_mz_, aligned_intensity

    def _bin_warped(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> pd.DataFrame:
        """Bin warped raw spectrum to output grid."""
        warped_df = pd.DataFrame({'mass': mz, 'intensity': intensity})
        bin_kwargs = self.bin_kwargs or {}
        binned, _ = bin_spectrum(
            warped_df,
            self.preprocessing_cfg_,
            bin_width=self.bin_width,
            method=self.bin_method,
            **bin_kwargs
        )
        return binned

    def _process_single_sample(self, path: str) -> np.ndarray:
        """Process a single sample: load, warp, and bin."""
        # Load raw spectrum
        spec_df = self._load_raw_spectrum(path)
        mz = spec_df['mass'].to_numpy()
        intensity = spec_df['intensity'].to_numpy()

        # Detect peaks in raw spectrum
        peaks_mz = self._detect_peaks_mz(mz, intensity)

        # Apply warping method
        if self.method == "shift":
            warped_mz, warped_int = self._shift_raw(mz, intensity, peaks_mz)
        elif self.method == "linear":
            warped_mz, warped_int = self._linear_raw(mz, intensity, peaks_mz)
        elif self.method == "piecewise":
            warped_mz, warped_int = self._piecewise_raw(mz, intensity, peaks_mz)
        elif self.method == "dtw":
            warped_mz, warped_int = self._dtw_raw(mz, intensity)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Bin to output grid
        binned = self._bin_warped(warped_mz, warped_int)
        return binned.set_index('mass')['intensity'].to_numpy()

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
        """
        if not hasattr(self, 'ref_mz_'):
            raise RuntimeError("RawWarping must be fitted before transform")

        if "path" not in X.columns:
            raise ValueError(
                "Input DataFrame must have a 'path' column with file paths."
            )

        # Get paths from DataFrame
        paths = X["path"].tolist()

        # Use parallel processing with joblib
        # "loky" backend works well for mixed I/O + CPU workloads
        aligned_rows = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._process_single_sample)(path)
            for path in paths
        )

        # Combine into DataFrame
        result = pd.DataFrame(
            np.vstack(aligned_rows),
            index=X.index,
            columns=self.output_columns_
        )

        return result

    def get_alignment_quality(
        self,
        X: pd.DataFrame,
        X_aligned: pd.DataFrame | None = None
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
        """
        if not hasattr(self, 'ref_mz_'):
            raise RuntimeError("RawWarping must be fitted first")

        if X_aligned is None:
            X_aligned = self.transform(X)

        # Bin reference for comparison
        ref_binned = self._bin_warped(self.ref_mz_, self.ref_intensity_)
        ref_vec = ref_binned.set_index('mass')['intensity'].to_numpy()

        metrics = []
        paths = X["path"].tolist()
        for i, (sample_id, path) in enumerate(zip(X.index, paths)):
            # Load and bin original spectrum for comparison
            spec_df = self._load_raw_spectrum(path)
            original_binned = self._bin_warped(
                spec_df['mass'].to_numpy(),
                spec_df['intensity'].to_numpy()
            )
            original = original_binned.set_index('mass')['intensity'].to_numpy()
            aligned = X_aligned.loc[sample_id].to_numpy()

            # Ensure same length
            min_len = min(len(original), len(aligned), len(ref_vec))
            original = original[:min_len]
            aligned = aligned[:min_len]
            ref = ref_vec[:min_len]

            # Correlation with reference
            corr_before = np.corrcoef(original, ref)[0, 1]
            corr_after = np.corrcoef(aligned, ref)[0, 1]

            metrics.append({
                'correlation_before': corr_before,
                'correlation_after': corr_after,
                'improvement': corr_after - corr_before,
            })

        return pd.DataFrame(metrics, index=X.index)
