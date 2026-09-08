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
from .metrics import (
    SpectralMetric,
    extract_mz_intensity,
    list_spectral_metrics,
    register_spectral_metric,
    spectral_distance,
    unregister_spectral_metric,
)
from .pairwise import pairwise_distances
from .plots import plot_dendrogram, plot_distance_heatmap

__all__ = [
    "ClusteringMethod",
    "KMedoidsInit",
    "SpectralMetric",
    "cluster_metadata_concordance",
    "cluster_spectra",
    "extract_mz_intensity",
    "hdbscan_clustering",
    "hierarchical_clustering",
    "kmedoids_clustering",
    "list_spectral_metrics",
    "pairwise_distances",
    "plot_dendrogram",
    "plot_distance_heatmap",
    "register_spectral_metric",
    "silhouette_scores",
    "spectral_distance",
    "unregister_spectral_metric",
]
