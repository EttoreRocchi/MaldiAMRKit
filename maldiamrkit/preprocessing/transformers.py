"""Individual preprocessing step transformers for MALDI-TOF spectra.

Each transformer is a callable that operates on a DataFrame with
``mass`` and ``intensity`` columns and returns a transformed DataFrame.

Examples
--------
>>> from maldiamrkit.preprocessing.transformers import (
...     ClipNegatives, SqrtTransform, SavitzkyGolaySmooth,
...     SNIPBaseline, MzTrimmer, TICNormalizer,
... )
>>> steps = [ClipNegatives(), SqrtTransform(), SavitzkyGolaySmooth()]
>>> df = steps[0](raw_df)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.ndimage import grey_opening, median_filter, uniform_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


class ClipNegatives:
    """Clip negative intensity values to zero."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply clipping to the spectrum."""
        df = df.copy()
        df["intensity"] = df["intensity"].clip(lower=0)
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "ClipNegatives"}

    def __repr__(self) -> str:
        return "ClipNegatives()"


class SqrtTransform:
    """Variance-stabilizing square root transformation."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply square-root transformation to the spectrum."""
        df = df.copy()
        if (df["intensity"] < 0).any():
            warnings.warn(
                "SqrtTransform received negative intensity values which "
                "will produce NaN. Apply ClipNegatives before SqrtTransform.",
                RuntimeWarning,
                stacklevel=2,
            )
        df["intensity"] = np.sqrt(df["intensity"])
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "SqrtTransform"}

    def __repr__(self) -> str:
        return "SqrtTransform()"


class LogTransform:
    """Log1p intensity transformation (alternative to sqrt)."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p transformation to the spectrum."""
        df = df.copy()
        df["intensity"] = np.log1p(df["intensity"])
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "LogTransform"}

    def __repr__(self) -> str:
        return "LogTransform()"


class SavitzkyGolaySmooth:
    """Savitzky-Golay smoothing filter.

    Parameters
    ----------
    window_length : int, default=21
        Length of the filter window. Must be a positive odd integer
        (per Savitzky & Golay 1964).
    polyorder : int, default=2
        Order of the polynomial used to fit the samples.
    """

    def __init__(self, window_length: int = 21, polyorder: int = 2):
        self.window_length = window_length
        self.polyorder = polyorder

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Savitzky-Golay smoothing.

        Raises
        ------
        ValueError
            If ``window_length`` exceeds the data length, or if
            ``window_length`` is not greater than ``polyorder``.
        """
        n = len(df)
        if self.window_length > n:
            raise ValueError(
                f"window_length ({self.window_length}) exceeds data length ({n})."
            )
        if self.window_length <= self.polyorder:
            raise ValueError(
                f"window_length ({self.window_length}) must be greater than "
                f"polyorder ({self.polyorder})."
            )
        df = df.copy()
        df["intensity"] = savgol_filter(
            df["intensity"],
            window_length=self.window_length,
            polyorder=self.polyorder,
        )
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {
            "name": "SavitzkyGolaySmooth",
            "window_length": self.window_length,
            "polyorder": self.polyorder,
        }

    def __repr__(self) -> str:
        return (
            f"SavitzkyGolaySmooth(window_length={self.window_length}, "
            f"polyorder={self.polyorder})"
        )


class SNIPBaseline:
    """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) baseline correction.

    Parameters
    ----------
    half_window : int, default=40
        Half-window size for the SNIP algorithm.
    """

    def __init__(self, half_window: int = 40):
        self.half_window = half_window

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SNIP baseline correction to the spectrum."""
        df = df.copy()
        bkg = Baseline(x_data=df["mass"]).snip(
            df["intensity"],
            max_half_window=self.half_window,
            decreasing=True,
            smooth_half_window=0,
        )[0]
        intensity = df["intensity"] - bkg
        intensity[intensity < 0] = 0
        df["intensity"] = intensity
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "SNIPBaseline", "half_window": self.half_window}

    def __repr__(self) -> str:
        return f"SNIPBaseline(half_window={self.half_window})"


class TopHatBaseline:
    """Morphological top-hat baseline subtraction.

    Estimates the baseline by morphological grey-level opening of the
    intensity trace (erosion followed by dilation), then subtracts it
    from the spectrum and clips negative values to zero.

    Parameters
    ----------
    half_window : int, default=100
        Half-width of the structuring element in bins. The full
        element size is ``2 * half_window + 1``. Must be a positive
        integer.

    Raises
    ------
    ValueError
        If ``half_window`` is not a positive integer or exceeds the
        data length.
    """

    def __init__(self, half_window: int = 100):
        self.half_window = half_window

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply top-hat baseline subtraction to the spectrum."""
        if self.half_window < 1:
            raise ValueError(
                f"half_window ({self.half_window}) must be a positive integer."
            )
        n = len(df)
        size = 2 * self.half_window + 1
        if size > n:
            raise ValueError(
                f"Structuring element size ({size}) exceeds data length ({n})."
            )
        df = df.copy()
        baseline = grey_opening(df["intensity"].to_numpy(), size=size)
        df["intensity"] = np.maximum(0.0, df["intensity"].to_numpy() - baseline)
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "TopHatBaseline", "half_window": self.half_window}

    def __repr__(self) -> str:
        return f"TopHatBaseline(half_window={self.half_window})"


class ConvexHullBaseline:
    """Parameter-free baseline from the lower convex hull of the spectrum.

    Computes the convex hull of the ``(mass, intensity)`` points,
    extracts the lower hull (vertices traversed in ascending mass
    with minimum intensity), linearly interpolates it onto the full
    m/z axis, and subtracts the resulting baseline from the spectrum.
    Negative residuals are clipped to zero.

    Notes
    -----
    Requires at least three distinct points to form a hull. For
    shorter inputs the baseline is taken as the per-point minimum
    of the first and last intensities (degenerate "flat" hull).
    """

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply convex-hull baseline subtraction to the spectrum."""
        df = df.copy()
        mz = df["mass"].to_numpy(dtype=float)
        intensity = df["intensity"].to_numpy(dtype=float)
        n = len(mz)

        if n < 3 or not np.all(np.isfinite(intensity)):
            df["intensity"] = np.maximum(
                0.0, intensity - np.minimum.accumulate(intensity)
            )
            return df

        points = np.column_stack([mz, intensity])
        try:
            hull = ConvexHull(points)
        except Exception:
            flat = float(np.nanmin(intensity))
            df["intensity"] = np.maximum(0.0, intensity - flat)
            return df

        verts = hull.vertices
        n_v = len(verts)
        mz_v = mz[verts]
        int_v = intensity[verts]

        left = int(np.lexsort((int_v, mz_v))[0])
        right = int(np.lexsort((-int_v, -mz_v))[0])

        lower = []
        i = left
        while True:
            lower.append(verts[i])
            if i == right:
                break
            i = (i + 1) % n_v

        lower_mz = mz[lower]
        lower_int = intensity[lower]
        if not np.all(np.diff(lower_mz) > 0):
            order = np.argsort(lower_mz)
            lower_mz = lower_mz[order]
            lower_int = lower_int[order]
            lower_mz, unique_idx = np.unique(lower_mz, return_index=True)
            lower_int = lower_int[unique_idx]

        baseline = np.interp(mz, lower_mz, lower_int)
        df["intensity"] = np.maximum(0.0, intensity - baseline)
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "ConvexHullBaseline"}

    def __repr__(self) -> str:
        return "ConvexHullBaseline()"


class MedianBaseline:
    """Rolling-median baseline subtraction.

    Estimates the baseline via a rolling median filter applied
    ``iterations`` times, then subtracts it from the spectrum and
    clips negative values to zero.

    Parameters
    ----------
    half_window : int, default=100
        Half-width of the median filter in bins. The full window
        size is ``2 * half_window + 1``. Must be a positive integer.
    iterations : int, default=1
        Number of times the median filter is applied. Must be a
        positive integer. Additional iterations further flatten
        broad features at the cost of compute time.

    Raises
    ------
    ValueError
        If ``half_window`` or ``iterations`` is not a positive
        integer, or if the filter window exceeds the data length.
    """

    def __init__(self, half_window: int = 100, iterations: int = 1):
        self.half_window = half_window
        self.iterations = iterations

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling-median baseline subtraction to the spectrum."""
        if self.half_window < 1:
            raise ValueError(
                f"half_window ({self.half_window}) must be a positive integer."
            )
        if self.iterations < 1:
            raise ValueError(
                f"iterations ({self.iterations}) must be a positive integer."
            )
        n = len(df)
        size = 2 * self.half_window + 1
        if size > n:
            raise ValueError(f"Median filter size ({size}) exceeds data length ({n}).")
        df = df.copy()
        baseline = df["intensity"].to_numpy(dtype=float)
        for _ in range(self.iterations):
            baseline = median_filter(baseline, size=size, mode="reflect")
        df["intensity"] = np.maximum(
            0.0, df["intensity"].to_numpy(dtype=float) - baseline
        )
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {
            "name": "MedianBaseline",
            "half_window": self.half_window,
            "iterations": self.iterations,
        }

    def __repr__(self) -> str:
        return (
            f"MedianBaseline(half_window={self.half_window}, "
            f"iterations={self.iterations})"
        )


class MovingAverageSmooth:
    """Moving-average smoothing filter.

    Applies a uniform (boxcar) moving average of length
    ``window_length`` using reflective boundary handling.

    Parameters
    ----------
    window_length : int, default=5
        Length of the smoothing window. Must be an odd integer
        greater than or equal to 3.

    Raises
    ------
    ValueError
        If ``window_length`` is not an odd integer ``>= 3``, or if
        it exceeds the data length.
    """

    def __init__(self, window_length: int = 5):
        self.window_length = window_length

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply moving-average smoothing to the spectrum."""
        if self.window_length < 3 or self.window_length % 2 == 0:
            raise ValueError(
                f"window_length ({self.window_length}) must be an odd integer >= 3."
            )
        n = len(df)
        if self.window_length > n:
            raise ValueError(
                f"window_length ({self.window_length}) exceeds data length ({n})."
            )
        df = df.copy()
        df["intensity"] = uniform_filter1d(
            df["intensity"].to_numpy(dtype=float),
            size=self.window_length,
            mode="reflect",
        )
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "MovingAverageSmooth", "window_length": self.window_length}

    def __repr__(self) -> str:
        return f"MovingAverageSmooth(window_length={self.window_length})"


class MzTrimmer:
    """Trim spectrum to a specified m/z range.

    Parameters
    ----------
    mz_min : int, default=2000
        Lower m/z bound in Daltons.
    mz_max : int, default=20000
        Upper m/z bound in Daltons.

    Raises
    ------
    ValueError
        If ``mz_min`` is greater than or equal to ``mz_max``.
    """

    def __init__(self, mz_min: int = 2000, mz_max: int = 20000):
        if mz_min >= mz_max:
            raise ValueError(f"mz_min ({mz_min}) must be less than mz_max ({mz_max}).")
        self.mz_min = mz_min
        self.mz_max = mz_max

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply m/z trimming to the spectrum."""
        return df[df["mass"].between(self.mz_min, self.mz_max)].reset_index(drop=True)

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {
            "name": "MzTrimmer",
            "mz_min": self.mz_min,
            "mz_max": self.mz_max,
        }

    def __repr__(self) -> str:
        return f"MzTrimmer(mz_min={self.mz_min}, mz_max={self.mz_max})"


class TICNormalizer:
    """Total Ion Current normalization (intensities sum to 1)."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply TIC normalization to the spectrum."""
        df = df.copy()
        total = df["intensity"].sum()
        if total > 0:
            df["intensity"] = df["intensity"] / total
        else:
            warnings.warn(
                "TIC total is zero - spectrum has no signal. "
                "This may indicate a failed acquisition.",
                UserWarning,
                stacklevel=2,
            )
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "TICNormalizer"}

    def __repr__(self) -> str:
        return "TICNormalizer()"


class MedianNormalizer:
    """Normalize intensities by median value."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply median normalization to the spectrum."""
        df = df.copy()
        median = df["intensity"].median()
        if median > 0:
            df["intensity"] = df["intensity"] / median
        else:
            warnings.warn(
                "Median intensity is zero - spectrum has no signal. "
                "This may indicate a failed acquisition.",
                UserWarning,
                stacklevel=2,
            )
        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {"name": "MedianNormalizer"}

    def __repr__(self) -> str:
        return "MedianNormalizer()"


class PQNNormalizer:
    """Probabilistic Quotient Normalization.

    First normalizes by TIC, then divides by the median of the quotient
    spectrum (sample / reference). If no reference is provided, the
    reference is the median spectrum across the dataset.

    Parameters
    ----------
    reference : np.ndarray, list, or None, default=None
        Reference intensity vector. If None, uses TIC normalization only
        (the full PQN requires a reference from the dataset). Lists are
        converted to arrays internally.
    """

    def __init__(self, reference: np.ndarray | list | None = None):
        if isinstance(reference, list):
            reference = np.asarray(reference)
        self.reference = reference

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PQN normalization to the spectrum."""
        df = df.copy()
        # Step 1: TIC normalization
        total = df["intensity"].sum()
        if total > 0:
            df["intensity"] = df["intensity"] / total

        # Step 2: quotient normalization if reference is available
        if self.reference is not None:
            if len(self.reference) != len(df):
                raise ValueError(
                    f"PQNNormalizer reference length ({len(self.reference)}) differs "
                    f"from input length ({len(df)}). This produces invalid "
                    f"normalization due to misaligned m/z positions. Ensure the "
                    f"reference was computed on spectra with the same m/z grid."
                )
            ref = self.reference
            mask = ref > 0
            if mask.any():
                quotients = df.loc[mask, "intensity"].values / ref[mask]
                median_quotient = np.median(quotients)
                if median_quotient > 0:
                    df["intensity"] = df["intensity"] / median_quotient

        return df

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        d: dict = {"name": "PQNNormalizer"}
        if self.reference is not None:
            d["reference"] = self.reference.tolist()
        return d

    def __repr__(self) -> str:
        return "PQNNormalizer()"


class MzMultiTrimmer:
    """Keep only specific m/z ranges from the spectrum.

    Parameters
    ----------
    mz_ranges : list of tuple[float, float]
        List of (mz_min, mz_max) ranges to keep.

    Raises
    ------
    ValueError
        If ``mz_ranges`` is empty.
    """

    def __init__(self, mz_ranges: list[tuple[float, float]]):
        if not mz_ranges:
            raise ValueError("mz_ranges must not be empty.")
        self.mz_ranges = mz_ranges

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply m/z range subsetting to the spectrum."""
        masks = [df["mass"].between(lo, hi) for lo, hi in self.mz_ranges]
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m
        return df[combined_mask].reset_index(drop=True)

    def to_dict(self) -> dict:
        """Serialize transformer to a dictionary."""
        return {
            "name": "MzMultiTrimmer",
            "mz_ranges": self.mz_ranges,
        }

    def __repr__(self) -> str:
        return f"MzMultiTrimmer(mz_ranges={self.mz_ranges})"


# Registry mapping transformer names to classes (for deserialization)
TRANSFORMER_REGISTRY: dict[str, type] = {
    "ClipNegatives": ClipNegatives,
    "SqrtTransform": SqrtTransform,
    "LogTransform": LogTransform,
    "SavitzkyGolaySmooth": SavitzkyGolaySmooth,
    "MovingAverageSmooth": MovingAverageSmooth,
    "SNIPBaseline": SNIPBaseline,
    "TopHatBaseline": TopHatBaseline,
    "ConvexHullBaseline": ConvexHullBaseline,
    "MedianBaseline": MedianBaseline,
    "MzTrimmer": MzTrimmer,
    "TICNormalizer": TICNormalizer,
    "MedianNormalizer": MedianNormalizer,
    "PQNNormalizer": PQNNormalizer,
    "MzMultiTrimmer": MzMultiTrimmer,
}
