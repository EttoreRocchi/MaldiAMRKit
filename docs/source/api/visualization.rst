Visualization Module
====================

Standalone plotting functions for spectra, datasets, peaks, and alignment.

All plotting functions use lazy ``matplotlib`` imports, so ``matplotlib``
is only required when a plot function is actually called.

Exploratory Plots
-----------------

Dimensionality reduction scatter plots for exploring datasets, colored by
metadata columns such as species or resistance phenotype. PCA and t-SNE
are available with no extra dependencies; UMAP requires
``pip install maldiamrkit[batch]``.

.. autofunction:: maldiamrkit.visualization.plot_pca

.. autofunction:: maldiamrkit.visualization.plot_tsne

.. autofunction:: maldiamrkit.visualization.plot_umap

Spectrum Plots
--------------

.. autofunction:: maldiamrkit.visualization.plot_spectrum

.. autofunction:: maldiamrkit.visualization.plot_pseudogel

Peak Plots
----------

.. autofunction:: maldiamrkit.visualization.plot_peaks

Alignment Plots
---------------

.. autofunction:: maldiamrkit.visualization.plot_alignment
