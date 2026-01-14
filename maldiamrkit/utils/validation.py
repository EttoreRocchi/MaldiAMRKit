"""Input validation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_spectrum_input(X: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Validate and normalize spectrum input.

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Input spectrum/spectra data.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with spectra as rows.

    Raises
    ------
    ValueError
        If input is empty or has invalid structure.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame().T

    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            f"Expected DataFrame or Series, got {type(X).__name__}"
        )

    if X.empty:
        raise ValueError("Input DataFrame is empty")

    return X


def validate_mz_range(
    mz_values: np.ndarray,
    min_mz: float | None = None,
    max_mz: float | None = None
) -> bool:
    """
    Validate that m/z values are within expected range.

    Parameters
    ----------
    mz_values : np.ndarray
        Array of m/z values.
    min_mz : float, optional
        Minimum expected m/z value.
    max_mz : float, optional
        Maximum expected m/z value.

    Returns
    -------
    bool
        True if all values are within range.

    Raises
    ------
    ValueError
        If values are outside expected range.
    """
    if min_mz is not None and np.min(mz_values) < min_mz:
        raise ValueError(
            f"m/z values below minimum: {np.min(mz_values)} < {min_mz}"
        )
    if max_mz is not None and np.max(mz_values) > max_mz:
        raise ValueError(
            f"m/z values above maximum: {np.max(mz_values)} > {max_mz}"
        )

    return True
