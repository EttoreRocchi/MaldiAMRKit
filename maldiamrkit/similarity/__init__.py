"""Spectral similarity metrics, pairwise distances, and clustering."""

from .clustering import (
    ClusteringMethod,
    KMedoidsInit,
    cluster_metadata_concordance,
    cluster_spectra,
    hdbscan_clustering,
    hierarchical_clustering,
    kmedoids_clustering,
    silhouette_scores,
)
from .metrics import METRIC_REGISTRY, SpectralMetric, spectral_distance
from .pairwise import pairwise_distances
from .plots import plot_dendrogram, plot_distance_heatmap

__all__ = [
    "ClusteringMethod",
    "KMedoidsInit",
    "METRIC_REGISTRY",
    "SpectralMetric",
    "cluster_metadata_concordance",
    "cluster_spectra",
    "hdbscan_clustering",
    "hierarchical_clustering",
    "kmedoids_clustering",
    "pairwise_distances",
    "plot_dendrogram",
    "plot_distance_heatmap",
    "silhouette_scores",
    "spectral_distance",
]
