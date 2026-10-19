"""Pairwise spectral distance matrix computation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from .metrics import SpectralMetric, _resolve_metric, extract_mz_intensity

if TYPE_CHECKING:
    from maldiamrkit.spectrum import MaldiSpectrum


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
        Number of worker processes for metrics evaluated pair by pair.
        Unused when the metric has a vectorized form (the built-in
        'cosine', 'pearson', and 'spectral_contrast_angle', or a custom
        metric registered with ``pairwise=``) and the spectra are a
        DataFrame or share one m/z grid.

    Returns
    -------
    np.ndarray
        Symmetric float64 distance matrix of shape ``(n, n)`` with zeros on
        the diagonal.

    Raises
    ------
    ValueError
        If *metric* is not in the registry, if a built-in vectorized metric
        receives NaN or infinite intensities, or if a custom ``pairwise``
        form returns an array of the wrong shape.

    Notes
    -----
    A metric with a vectorized form computes the whole matrix in one call,
    in the calling process, when the spectra are a DataFrame or share one
    m/z grid. This is much faster than evaluating the pairs one at a time.
    The linear algebra may run on several BLAS threads, so cap them (e.g.
    with ``threadpoolctl``) when calling this from inside your own process
    pool. Peak memory is a few ``(n, n)`` float64 arrays.

    Every other case is evaluated pair by pair over the upper triangle. The
    metric function is resolved in the calling process and handed to the
    workers, so custom metrics registered here work at any ``n_jobs``
    (worker processes do not inherit the registry itself). Pairs are
    dispatched to the workers in blocks, so the per-task overhead is paid
    once per block of pairs rather than once per pair.

    Choose ``n_jobs`` by how long the computation takes sequentially.
    Starting the worker processes costs up to several seconds (more for
    metrics whose dependencies are slow to load, such as 'dtw'), and much
    less while the workers of a previous call are still alive, so
    ``n_jobs=1`` is fastest for short runs such as a cheap metric on a few
    hundred spectra. Long runs, from expensive metrics ('wasserstein',
    'dtw') or from many spectra, repay the start-up substantially. For cheap
    metrics on a machine with many cores, a moderate ``n_jobs`` can be
    faster than ``n_jobs=-1``, whose start-up grows with the number of
    workers.
    """
    key, entry = _resolve_metric(metric)
    n = len(spectra)
    if n < 2:
        return np.zeros((n, n))

    if entry.pairwise is not None:
        X = _stack_intensities(spectra)
        if X is not None:
            return _pairwise_vectorized(X, entry.pairwise, key)

    return _pairwise_general(spectra, entry.fn, n, n_jobs)


def _stack_intensities(spectra: list | pd.DataFrame) -> np.ndarray | None:
    """Stack the spectra into one float64 intensity matrix, if possible.

    A DataFrame is used as is. Any other sequence is stacked only when every
    intensity vector is 1-D with the same non-zero length and the m/z axes
    are either all absent (binned vectors) or all identical. Otherwise
    ``None`` is returned and the spectra are evaluated pair by pair.
    """
    if isinstance(spectra, pd.DataFrame):
        if spectra.shape[1] == 0:
            return None
        return spectra.to_numpy(dtype=np.float64)

    extracted = [extract_mz_intensity(spec) for spec in spectra]
    intensities = [intensity for _, intensity in extracted]
    shapes = {intensity.shape for intensity in intensities}
    if len(shapes) != 1 or intensities[0].ndim != 1 or intensities[0].size == 0:
        return None

    mzs = [mz for mz, _ in extracted]
    if any(mz is None for mz in mzs):
        if not all(mz is None for mz in mzs):
            return None
    elif not all(np.array_equal(mz, mzs[0]) for mz in mzs[1:]):
        return None
    return np.asarray(np.vstack(intensities), dtype=np.float64)


def _pairwise_vectorized(
    X: np.ndarray,
    pairwise_fn: Callable,
    key: str,
) -> np.ndarray:
    """Whole-matrix path: one ``pairwise_fn(X, X)`` call, then the contract.

    ``X`` is handed over read-only, so a custom form cannot modify the
    caller's data. The result is copied unless it is a writeable array of
    its own, mirrored from its upper triangle when not exactly symmetric
    (as on the per-pair path, which computes ``i < j`` only), and given a
    zero diagonal.
    """
    n = X.shape[0]
    X = X.view()
    X.flags.writeable = False

    D = np.asarray(pairwise_fn(X, X), dtype=np.float64)
    if D.shape != (n, n):
        raise ValueError(
            f"The pairwise form of metric {key!r} returned an array of shape "
            f"{D.shape}; expected {(n, n)}."
        )
    if not (D.flags.writeable and D.flags.owndata):
        D = D.copy()
    if not np.array_equal(D, D.T):
        upper = np.triu(D, 1)
        D = upper + upper.T
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
