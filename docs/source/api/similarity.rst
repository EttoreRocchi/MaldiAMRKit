Similarity Module
=================

Spectral distance metrics, pairwise distance matrix computation,
clustering algorithms, and visualizations for spectral similarity
analysis.

Metrics
-------

.. autofunction:: maldiamrkit.similarity.spectral_distance

.. autoclass:: maldiamrkit.similarity.SpectralMetric
   :members:
   :undoc-members:
   :show-inheritance:

.. py:data:: maldiamrkit.similarity.METRIC_REGISTRY

   Dictionary mapping metric names to callable distance functions.
   See :class:`SpectralMetric` for the built-in keys.

Pairwise Distances
------------------

.. autofunction:: maldiamrkit.similarity.pairwise_distances

Clustering
----------

.. autofunction:: maldiamrkit.similarity.cluster_spectra

.. autofunction:: maldiamrkit.similarity.hierarchical_clustering

.. autofunction:: maldiamrkit.similarity.hdbscan_clustering

.. autofunction:: maldiamrkit.similarity.kmedoids_clustering

.. autofunction:: maldiamrkit.similarity.silhouette_scores

.. autofunction:: maldiamrkit.similarity.cluster_metadata_concordance

.. autoclass:: maldiamrkit.similarity.ClusteringMethod
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.similarity.KMedoidsInit
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. autofunction:: maldiamrkit.similarity.plot_distance_heatmap

.. autofunction:: maldiamrkit.similarity.plot_dendrogram

Example
-------

.. code-block:: python

    from maldiamrkit.similarity import (
        pairwise_distances,
        cluster_spectra,
        plot_distance_heatmap,
        plot_dendrogram,
        hierarchical_clustering,
    )

    # Compute pairwise distance matrix
    D = pairwise_distances(spectra, metric="cosine", n_jobs=-1)

    # Visualize distances
    plot_distance_heatmap(D, labels=sample_ids)

    # Cluster spectra
    labels = cluster_spectra(D, method="hierarchical", n_clusters=3)

    # Plot dendrogram
    linkage = hierarchical_clustering(D)
    plot_dendrogram(linkage, labels=sample_ids)
