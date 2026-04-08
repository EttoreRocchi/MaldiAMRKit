"""Clustering algorithms and evaluation for precomputed distance matrices."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class ClusteringMethod(str, Enum):
    """Supported clustering algorithms for :func:`cluster_spectra`.

    Attributes
    ----------
    hierarchical : str
        Agglomerative hierarchical clustering.
    hdbscan : str
        HDBSCAN density-based clustering.
    kmedoids : str
        K-medoids (PAM) clustering.
    """

    hierarchical = "hierarchical"
    hdbscan = "hdbscan"
    kmedoids = "kmedoids"


class KMedoidsInit(str, Enum):
    """Initialization strategy for :func:`kmedoids_clustering`.

    Attributes
    ----------
    build : str
        Deterministic BUILD phase of PAM.
    random : str
        Random medoid selection.
    """

    build = "build"
    random = "random"


def hierarchical_clustering(
    distance_matrix: np.ndarray,
    method: str = "average",
    **kwargs,
) -> np.ndarray:
    """Agglomerative hierarchical clustering on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Symmetric pairwise distance matrix.
    method : str, default="average"
        Linkage method forwarded to
        :func:`scipy.cluster.hierarchy.linkage`.
    **kwargs
        Extra keyword arguments for :func:`~scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    ndarray of shape (n - 1, 4)
        Linkage matrix.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(distance_matrix)
    return linkage(condensed, method=method, **kwargs)


def hdbscan_clustering(
    distance_matrix: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """HDBSCAN clustering on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Symmetric pairwise distance matrix.
    eps : float, default=0.5
        Cluster selection epsilon passed to
        ``cluster_selection_epsilon``.
    min_samples : int, default=5
        Minimum number of samples in a neighbourhood.

    Returns
    -------
    ndarray of shape (n,)
        Cluster labels (``-1`` for noise points).
    """
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        metric="precomputed",
        min_samples=min_samples,
        cluster_selection_epsilon=eps,
    )
    return clusterer.fit_predict(distance_matrix)


def _build_medoids(
    D: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """BUILD phase of PAM: greedy medoid initialization.

    Parameters
    ----------
    D : ndarray of shape (n, n)
        Distance matrix.
    n_clusters : int
        Number of medoids to select.

    Returns
    -------
    ndarray of shape (n_clusters,)
        Indices of selected medoids.
    """
    n = D.shape[0]
    # First medoid: point that minimises total distance to all others.
    medoids = [int(np.argmin(D.sum(axis=1)))]

    # Current cost of assigning each point to the nearest selected medoid.
    nearest_dist = D[medoids[0]].copy()

    for _ in range(1, n_clusters):
        # For each candidate, the gain is the total reduction in distance.
        gains = np.zeros(n)
        for c in range(n):
            if c in medoids:
                continue
            gains[c] = np.maximum(nearest_dist - D[c], 0.0).sum()
        new_medoid = int(np.argmax(gains))
        medoids.append(new_medoid)
        nearest_dist = np.minimum(nearest_dist, D[new_medoid])

    return np.array(medoids)


def _swap_medoids(
    D: np.ndarray,
    medoids: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SWAP phase of PAM: iterative medoid refinement.

    Returns
    -------
    medoids : ndarray
        Final medoid indices.
    labels : ndarray
        Cluster assignments.
    """
    n = D.shape[0]
    medoids = medoids.copy()

    all_idx = np.arange(n)
    medoid_set = set(medoids.tolist())

    for _ in range(max_iter):
        # Assign to nearest medoid.
        labels = np.argmin(D[:, medoids], axis=1)
        current_cost = float(np.sum(D[all_idx, medoids[labels]]))

        best_swap = None
        best_cost = current_cost

        for m_idx in range(len(medoids)):
            for candidate in range(n):
                if candidate in medoid_set:
                    continue
                new_medoids = medoids.copy()
                new_medoids[m_idx] = candidate
                new_labels = np.argmin(D[:, new_medoids], axis=1)
                cost = float(np.sum(D[all_idx, new_medoids[new_labels]]))
                if cost < best_cost:
                    best_cost = cost
                    best_swap = (m_idx, candidate)

        if best_swap is None:
            break
        medoid_set.discard(int(medoids[best_swap[0]]))
        medoids[best_swap[0]] = best_swap[1]
        medoid_set.add(best_swap[1])

    labels = np.argmin(D[:, medoids], axis=1)
    return medoids, labels


def kmedoids_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int = 3,
    max_iter: int = 300,
    random_state: int | None = None,
    init: str | KMedoidsInit = KMedoidsInit.build,
) -> np.ndarray:
    """K-medoids clustering using the PAM algorithm.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Symmetric pairwise distance matrix.
    n_clusters : int, default=3
        Number of clusters.
    max_iter : int, default=300
        Maximum SWAP iterations.
    random_state : int or None, default=None
        Random seed (used only when ``init="random"``).
    init : str or KMedoidsInit, default="build"
        Medoid initialization strategy.  ``"build"`` uses the
        deterministic BUILD phase of PAM; ``"random"`` selects
        initial medoids uniformly at random.

    Returns
    -------
    ndarray of shape (n,)
        Cluster labels.

    Raises
    ------
    ValueError
        If *init* is not ``"build"`` or ``"random"``.
    """
    init = KMedoidsInit(init)

    n = distance_matrix.shape[0]

    if init == "build":
        medoids = _build_medoids(distance_matrix, n_clusters)
    else:
        rng = np.random.default_rng(random_state)
        medoids = rng.choice(n, n_clusters, replace=False)

    _, labels = _swap_medoids(distance_matrix, medoids, max_iter)
    return labels


def silhouette_scores(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Silhouette score for a clustering on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Symmetric pairwise distance matrix.
    labels : ndarray of shape (n,)
        Cluster assignments.

    Returns
    -------
    float
        Mean silhouette coefficient in ``[-1, 1]``.
    """
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def cluster_metadata_concordance(
    labels: np.ndarray,
    metadata: pd.Series,
) -> dict[str, float]:
    """Evaluate clustering agreement with known metadata labels.

    Parameters
    ----------
    labels : ndarray of shape (n,)
        Cluster assignments.
    metadata : Series of shape (n,)
        Ground-truth categorical labels.

    Returns
    -------
    dict[str, float]
        ``{"adjusted_rand_index": float, "normalized_mutual_info": float}``.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    return {
        "adjusted_rand_index": float(adjusted_rand_score(metadata, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(metadata, labels)),
    }


def cluster_spectra(
    distance_matrix: np.ndarray,
    method: str | ClusteringMethod = ClusteringMethod.hierarchical,
    n_clusters: int | None = None,
    threshold: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Cluster spectra from a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Symmetric pairwise distance matrix.
    method : {"hierarchical", "hdbscan", "kmedoids"}, default="hierarchical"
        Clustering algorithm.
    n_clusters : int or None, default=None
        Number of clusters.  Required for ``"kmedoids"`` and one of
        ``n_clusters`` / ``threshold`` for ``"hierarchical"``.
    threshold : float or None, default=None
        Distance threshold for cutting the dendrogram (hierarchical only).
    **kwargs
        Extra keyword arguments forwarded to the underlying function:

        - **hierarchical**: ``method`` (linkage method, default ``"average"``)
          and any extra keyword arguments accepted by
          :func:`scipy.cluster.hierarchy.linkage`.
        - **hdbscan**: ``eps`` (cluster selection epsilon, default ``0.5``),
          ``min_samples`` (default ``5``).
        - **kmedoids**: ``max_iter`` (default ``300``),
          ``random_state``, ``init`` (``"build"`` or ``"random"``).

    Returns
    -------
    ndarray of shape (n,)
        Cluster labels.

    Raises
    ------
    ValueError
        If *method* is unknown, or required parameters are missing /
        conflicting.
    """
    method = ClusteringMethod(method)

    if method == "hierarchical":
        if (n_clusters is None) == (threshold is None):
            raise ValueError(
                "Exactly one of 'n_clusters' or 'threshold' must be "
                "provided for hierarchical clustering."
            )
        from scipy.cluster.hierarchy import fcluster

        linkage_matrix = hierarchical_clustering(distance_matrix, **kwargs)
        if n_clusters is not None:
            return fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        return fcluster(linkage_matrix, t=threshold, criterion="distance")

    if method == "hdbscan":
        return hdbscan_clustering(distance_matrix, **kwargs)

    # method == "kmedoids"
    if n_clusters is None:
        raise ValueError("'n_clusters' is required for kmedoids clustering.")
    return kmedoids_clustering(distance_matrix, n_clusters=n_clusters, **kwargs)
