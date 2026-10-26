"""Banded Sakoe-Chiba DTW path, computed on the band only.

The path is the one :func:`tslearn.metrics.dtw_path` returns for two series
of equal length with ``global_constraint="sakoe_chiba"``: same float64 cost,
same tie order when walking back. Memory is ``O(n * (2 * radius + 1))``
instead of tslearn's ``O(n ** 2)``.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _banded_path(s1, s2, radius):
    """Forward DTW path ``(path_i, path_j)`` of two float64 series of length n."""
    n = s1.shape[0]
    r = min(radius, n - 1)
    w = 2 * r + 1
    # cost[i, k] is the accumulated cost of cell (i, j) with j = i - r + k.
    cost = np.full((n, w), np.inf)
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n - 1, i + r)
        for j in range(lo, hi + 1):
            k = j - i + r
            if i == 0 and j == 0:
                best = 0.0
            else:
                up = cost[i - 1, k + 1] if i > 0 and k + 1 < w else np.inf
                left = cost[i, k - 1] if j > 0 and k > 0 else np.inf
                diag = cost[i - 1, k] if i > 0 and j > 0 else np.inf
                best = min(up, left, diag)
            diff = s1[i] - s2[j]
            dist = diff * diff
            cost[i, k] = dist + best

    path_i = np.empty(2 * n, dtype=np.int64)
    path_j = np.empty(2 * n, dtype=np.int64)
    i = n - 1
    j = n - 1
    m = 0
    path_i[0] = i
    path_j[0] = j
    while i != 0 or j != 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Same order as tslearn: diagonal, then up, then left.
            k = j - i + r
            diag = cost[i - 1, k] if 0 <= k < w else np.inf
            up = cost[i - 1, k + 1] if 0 <= k + 1 < w else np.inf
            left = cost[i, k - 1] if 0 <= k - 1 < w else np.inf
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        m += 1
        path_i[m] = i
        path_j[m] = j
    return path_i[: m + 1][::-1].copy(), path_j[: m + 1][::-1].copy()


def _dtw_path(
    query: np.ndarray, reference: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Banded DTW path of two series of equal length.

    Both series are cast to contiguous float64 (as tslearn does), so the
    kernel is compiled once whatever the input dtype.
    """
    return _banded_path(
        np.ascontiguousarray(query, dtype=np.float64),
        np.ascontiguousarray(reference, dtype=np.float64),
        int(radius),
    )
