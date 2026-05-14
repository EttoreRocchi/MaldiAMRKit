"""Pairwise spectral distance matrix computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .metrics import METRIC_REGISTRY, SpectralMetric, spectral_distance

if TYPE_CHECKING:
    from maldiamrkit.spectrum import MaldiSpectrum

_BINNED_METRICS = frozenset({"cosine", "spectral_contrast_angle", "pearson"})


def pairwise_distances(
    spectra: list[MaldiSpectrum] | pd.DataFrame,
    metric: str | SpectralMetric = SpectralMetric.wasserstein,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute an *n x n* symmetric distance matrix.

    Parameters
    ----------
    spectra : list[MaldiSpectrum] or DataFrame
        If a :class:`~pandas.DataFrame` (binned feature matrix, rows are
        samples), row vectors are used.  If a list of
        :class:`~maldiamrkit.spectrum.MaldiSpectrum`, raw/preprocessed data
        is used.
    metric : str or SpectralMetric, default="wasserstein"
        Key in :data:`~maldiamrkit.similarity.METRIC_REGISTRY`.
    n_jobs : int, default=1
        Number of parallel jobs for pairwise computation.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape ``(n, n)`` with zeros on the
        diagonal.

    Raises
    ------
    ValueError
        If *metric* is not in the registry.
    """
    metric = SpectralMetric(metric)

    # Fast path: binned metric on DataFrame input.
    if isinstance(spectra, pd.DataFrame) and metric in _BINNED_METRICS:
        return _pairwise_binned(spectra, metric)

    # General path: compute upper triangle with joblib parallelization.
    n = len(spectra) if isinstance(spectra, list) else len(spectra)
    return _pairwise_general(spectra, metric, n, n_jobs)


def _pairwise_binned(X: pd.DataFrame, metric: str) -> np.ndarray:
    """Fast path using sklearn for binned feature matrices."""
    from sklearn.metrics import pairwise_distances as sklearn_pd

    metric_fn = METRIC_REGISTRY[metric]
    arr = X.values

    def _metric(a: np.ndarray, b: np.ndarray) -> float:
        return metric_fn(a, b)

    D = sklearn_pd(arr, metric=_metric)
    np.fill_diagonal(D, 0.0)
    return D


def _pairwise_general(
    spectra: list | pd.DataFrame,
    metric: str,
    n: int,
    n_jobs: int,
) -> np.ndarray:
    """General path: upper-triangle computation with joblib."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if isinstance(spectra, pd.DataFrame):
        rows = [spectra.iloc[i].values for i in range(n)]
    else:
        rows = spectra

    distances = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(spectral_distance)(rows[i], rows[j], metric=metric) for i, j in pairs
    )

    D = np.zeros((n, n), dtype=np.float64)
    for (i, j), d in zip(pairs, distances, strict=True):
        D[i, j] = d
        D[j, i] = d
    return D
