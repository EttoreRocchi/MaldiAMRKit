"""Pairwise spectral distance matrix computation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from .metrics import SpectralMetric, _resolve_metric

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
        One of the values of :class:`~maldiamrkit.similarity.SpectralMetric`,
        or a custom name registered with
        :func:`~maldiamrkit.similarity.register_spectral_metric`.
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

    Notes
    -----
    The metric function is resolved in the calling process and handed to the
    workers, so custom metrics registered here work at any ``n_jobs`` (worker
    processes do not inherit the registry itself). Pairs are dispatched to
    the workers in blocks, so the per-task overhead is paid once per block
    of pairs rather than once per pair.

    Choose ``n_jobs`` by how expensive the metric is per pair. Cheap vector
    metrics ('cosine', 'pearson', 'spectral_contrast_angle') gain nothing from
    process parallelism. Expensive metrics ('wasserstein', 'dtw') repay the
    overhead substantially.
    """
    key, metric_fn = _resolve_metric(metric)

    # Fast path: binned metric on DataFrame input.
    if isinstance(spectra, pd.DataFrame) and key in _BINNED_METRICS:
        return _pairwise_binned(spectra, metric_fn)

    # General path: compute upper triangle with joblib parallelization.
    n = len(spectra)
    return _pairwise_general(spectra, metric_fn, n, n_jobs)


def _pairwise_binned(X: pd.DataFrame, metric_fn: Callable) -> np.ndarray:
    """Fast path using sklearn for binned feature matrices."""
    from sklearn.metrics import pairwise_distances as sklearn_pd

    D = sklearn_pd(X.values, metric=metric_fn)
    np.fill_diagonal(D, 0.0)
    return D


def _block_distances(
    metric_fn: Callable,
    rows_a: list,
    offset_a: int,
    rows_b: list | None,
    offset_b: int,
) -> list[tuple[int, int, float]]:
    """Distances for one block pair as ``(i, j, d)`` triples (runs in a worker).

    ``rows_b is None`` marks a diagonal block: the pairs are the upper
    triangle within ``rows_a``.
    """
    out = []
    if rows_b is None:
        for a, row_a in enumerate(rows_a):
            for b in range(a + 1, len(rows_a)):
                out.append((offset_a + a, offset_a + b, metric_fn(row_a, rows_a[b])))
    else:
        for a, row_a in enumerate(rows_a):
            for b, row_b in enumerate(rows_b):
                out.append((offset_a + a, offset_b + b, metric_fn(row_a, row_b)))
    return out


def _pairwise_general(
    spectra: list | pd.DataFrame,
    metric_fn: Callable,
    n: int,
    n_jobs: int,
) -> np.ndarray:
    """General path: upper-triangle computation with joblib.

    ``metric_fn`` is the already-resolved distance callable: passing the
    function (rather than its registry name) is what lets a metric registered
    in this process run inside joblib worker processes.

    Parallel work is dispatched as a blocked decomposition: the indices are
    split into contiguous blocks and each task computes every pair between
    two blocks, so a spectrum is serialised to the workers once per block
    pair it appears in (a few dozen times at most) instead of once per pair.
    """
    if isinstance(spectra, pd.DataFrame):
        rows = list(spectra.to_numpy())
    else:
        rows = list(spectra)

    D = np.zeros((n, n), dtype=np.float64)
    n_workers = effective_n_jobs(n_jobs)

    if n_workers == 1 or n < 3:
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = metric_fn(rows[i], rows[j])
        return D

    # ~3 tasks per worker: B blocks give B * (B + 1) / 2 block pairs.
    n_blocks = min(n, max(2, math.isqrt(6 * n_workers) + 1))
    blocks = np.array_split(np.arange(n), n_blocks)
    tasks = [(bi, bj) for bi in range(n_blocks) for bj in range(bi, n_blocks)]

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_block_distances)(
            metric_fn,
            [rows[i] for i in blocks[bi]],
            int(blocks[bi][0]),
            None if bi == bj else [rows[j] for j in blocks[bj]],
            int(blocks[bj][0]) if bi != bj else 0,
        )
        for bi, bj in tasks
    )

    for triples in results:
        for i, j, d in triples:
            D[i, j] = D[j, i] = d
    return D
