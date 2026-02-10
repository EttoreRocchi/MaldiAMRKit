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

import warnings

import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import savgol_filter


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
    window_length : int, default=20
        Length of the filter window. Must be a positive odd integer.
    polyorder : int, default=2
        Order of the polynomial used to fit the samples.
    """

    def __init__(self, window_length: int = 20, polyorder: int = 2):
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
    reference : np.ndarray or None, default=None
        Reference intensity vector. If None, uses TIC normalization only
        (the full PQN requires a reference from the dataset).
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
                warnings.warn(
                    f"PQNNormalizer reference length ({len(self.reference)}) differs "
                    f"from input length ({len(df)}). This may produce invalid "
                    f"normalization due to misaligned m/z positions.",
                    UserWarning,
                    stacklevel=2,
                )
            ref = self.reference[: len(df)]
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
    "SNIPBaseline": SNIPBaseline,
    "MzTrimmer": MzTrimmer,
    "TICNormalizer": TICNormalizer,
    "MedianNormalizer": MedianNormalizer,
    "PQNNormalizer": PQNNormalizer,
    "MzMultiTrimmer": MzMultiTrimmer,
}
