"""Spectral replicate merging for MALDI-TOF spectra.

Clinical workflows often acquire multiple replicates per isolate.
This module provides functions to merge replicates into a single
consensus spectrum and detect outlier replicates.

Examples
--------
>>> from maldiamrkit import MaldiSpectrum
>>> from maldiamrkit.preprocessing.merging import (
...     merge_replicates, detect_outlier_replicates,
... )
>>> reps = [MaldiSpectrum(f"rep{i}.txt") for i in range(3)]
>>> merged = merge_replicates(reps, method="mean")
>>> keep = detect_outlier_replicates(reps)
>>> clean = [s for s, ok in zip(reps, keep) if ok]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from maldiamrkit.spectrum import MaldiSpectrum


def _to_common_grid(
    spectra: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate spectra onto a common m/z grid.

    Parameters
    ----------
    spectra : list of pd.DataFrame
        Each with ``mass`` and ``intensity`` columns.

    Returns
    -------
    common_mz : np.ndarray
        Sorted union of all m/z values.
    matrix : np.ndarray
        Intensity matrix of shape ``(n_spectra, len(common_mz))``.
    """
    common_mz = np.unique(np.concatenate([s["mass"].values for s in spectra]))

    matrix = np.empty((len(spectra), len(common_mz)))
    for i, s in enumerate(spectra):
        matrix[i] = np.interp(common_mz, s["mass"].values, s["intensity"].values)

    return common_mz, matrix


def merge_replicates(
    spectra: list[MaldiSpectrum],
    method: str = "mean",
    weights: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Merge replicate spectra into a single consensus spectrum.

    Parameters
    ----------
    spectra : list of MaldiSpectrum
        Replicate spectra to merge.
    method : str, default="mean"
        Merging strategy:

        - ``"mean"``: arithmetic mean (or weighted mean if ``weights``
          is provided).
        - ``"median"``: element-wise median (``weights`` is ignored).
    weights : array-like of float, optional
        Per-replicate weights for the ``"mean"`` method (e.g. SNR
        values).  Ignored when ``method="median"``.  Must have the
        same length as ``spectra``.

    Returns
    -------
    pd.DataFrame
        Merged spectrum with ``mass`` and ``intensity`` columns.

    Raises
    ------
    ValueError
        If *spectra* is empty, *method* is invalid, or *weights*
        length does not match *spectra*.
    """
    if not spectra:
        raise ValueError("spectra must not be empty.")

    valid_methods = ("mean", "median")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}.")

    dfs = [s.raw for s in spectra]

    if len(dfs) == 1:
        return dfs[0].copy()

    common_mz, matrix = _to_common_grid(dfs)

    if method == "mean":
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if len(w) != len(spectra):
                raise ValueError(
                    f"weights length ({len(w)}) must match "
                    f"spectra length ({len(spectra)})."
                )
            merged = np.average(matrix, axis=0, weights=w)
        else:
            merged = np.mean(matrix, axis=0)
    else:  # median
        merged = np.median(matrix, axis=0)

    return pd.DataFrame({"mass": common_mz, "intensity": merged})


def detect_outlier_replicates(
    spectra: list[MaldiSpectrum],
    threshold: float = 3.0,
) -> np.ndarray:
    """Identify outlier replicates using correlation with the median spectrum.

    Computes the Pearson correlation of each replicate against the
    element-wise median spectrum.  Replicates whose correlation falls
    below ``median(corrs) - threshold * MAD(corrs)`` are flagged as
    outliers.

    Parameters
    ----------
    spectra : list of MaldiSpectrum
        Replicate spectra.
    threshold : float, default=3.0
        Number of MAD units below the median correlation to flag a
        replicate as an outlier.

    Returns
    -------
    np.ndarray
        Boolean array of length ``len(spectra)``.  ``True`` means the
        replicate is kept; ``False`` means it is an outlier.

    Raises
    ------
    ValueError
        If *spectra* has fewer than 3 elements (need at least 3 to
        estimate spread).
    """
    if len(spectra) < 3:
        raise ValueError(
            f"Need at least 3 replicates for outlier detection, got {len(spectra)}."
        )

    dfs = [s.raw for s in spectra]
    common_mz, matrix = _to_common_grid(dfs)
    median_spectrum = np.median(matrix, axis=0)

    corrs = np.array([np.corrcoef(row, median_spectrum)[0, 1] for row in matrix])

    med_corr = np.median(corrs)
    mad_corr = np.median(np.abs(corrs - med_corr))

    if mad_corr == 0:
        # All correlations identical - no outliers
        return np.ones(len(spectra), dtype=bool)

    cutoff = med_corr - threshold * 1.4826 * mad_corr
    return corrs >= cutoff
