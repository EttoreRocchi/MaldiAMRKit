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

Custom Metrics
~~~~~~~~~~~~~~

Register your own distance function to use it anywhere a metric name is
accepted, including :func:`~maldiamrkit.similarity.pairwise_distances` and
:class:`~maldiamrkit.drift.DriftMonitor`.

.. autofunction:: maldiamrkit.similarity.register_spectral_metric

.. autofunction:: maldiamrkit.similarity.unregister_spectral_metric

.. autofunction:: maldiamrkit.similarity.list_spectral_metrics

.. autofunction:: maldiamrkit.similarity.extract_mz_intensity

.. code-block:: python

    import numpy as np
    from maldiamrkit.similarity import (
        extract_mz_intensity,
        pairwise_distances,
        register_spectral_metric,
        spectral_distance,
    )

    def manhattan(spec_a, spec_b):
        _, a = extract_mz_intensity(spec_a)
        _, b = extract_mz_intensity(spec_b)
        return float(np.abs(np.asarray(a) - np.asarray(b)).sum())

    register_spectral_metric("manhattan", manhattan)

    spectral_distance(X.iloc[0], X.iloc[1], metric="manhattan")
    D = pairwise_distances(X, metric="manhattan", n_jobs=-1)

The built-in metrics are protected: registering over one of them requires
``override=True``, and unregistering an overridden built-in restores the
default implementation. The built-in names themselves can never be removed.

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
