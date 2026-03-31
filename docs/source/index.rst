:html_theme.sidebar_secondary.remove: true

.. image:: _static/maldiamrkit.png
   :align: center
   :width: 280px
   :class: only-light

.. image:: _static/maldiamrkit.png
   :align: center
   :width: 280px
   :class: only-dark

.. rst-class:: hero-section

MaldiAMRKit Documentation
=========================

A Python toolkit for MALDI-TOF mass spectrometry preprocessing for
antimicrobial resistance (AMR) prediction. Scikit-learn compatible transformers
for seamless integration into machine learning pipelines.

.. container:: sd-d-flex-row sd-flex-justify-content-center sd-gap-2 sd-mb-4

   .. button-link:: installation.html
      :color: primary
      :shadow:

      Get Started

   .. button-link:: api/index.html
      :color: primary
      :outline:
      :shadow:

      API Reference

   .. button-link:: quickstart.html
      :color: primary
      :outline:
      :shadow:

      Quickstart Guide

----

Key Features
------------

.. grid:: 2 2 3 3
   :gutter: 3
   :class-container: feature-grid

   .. grid-item-card:: Preprocessing Pipeline
      :link: quickstart.html#custom-preprocessing-pipeline
      :link-type: url

      Composable transformers (smoothing, baseline, trimming, normalization),
      multiple binning strategies, and peak detection. Serializable to JSON/YAML.

   .. grid-item-card:: Sklearn Pipelines
      :link: quickstart.html#building-ml-pipelines
      :link-type: url

      Scikit-learn compatible transformers. Drop into any ``Pipeline``,
      ``cross_val_score``, or ``GridSearchCV`` workflow.

   .. grid-item-card:: Spectral Alignment
      :link: api/alignment.html
      :link-type: url

      Shift, linear, piecewise, and DTW warping for both binned and
      raw full-resolution spectra.

   .. grid-item-card:: AMR Evaluation
      :link: api/evaluation.html
      :link-type: url

      VME, ME, sensitivity, specificity, and classification reports
      following EUCAST/CLSI conventions. Species-drug stratified and
      case-based splitting to prevent data leakage.

   .. grid-item-card:: DRIAMS Dataset Builder
      :link: quickstart.html#building-driams-like-datasets
      :link-type: url

      Build and load DRIAMS-like dataset directories from raw spectra
      and metadata with year-based subfolders and custom processing handlers.

   .. grid-item-card:: Composable Filters
      :link: api/core.html#filters
      :link-type: url

      ``SpeciesFilter``, ``DrugFilter``, ``QualityFilter``, ``MetadataFilter``
      combinable with ``&``, ``|``, ``~`` operators.

   .. grid-item-card:: Exploratory Plots
      :link: api/visualization.html#exploratory-plots
      :link-type: url

      PCA, t-SNE, and UMAP scatter plots colored by species, resistance
      phenotype, or any metadata column.

   .. grid-item-card:: CLI & Export
      :link: quickstart.html#command-line-interface
      :link-type: url

      ``maldiamrkit preprocess``, ``maldiamrkit quality``, and
      ``maldiamrkit build-driams`` for batch processing. Export to CSV/TXT.

   .. grid-item-card:: Batch Correction
      :link: quickstart.html#batch-effect-correction
      :link-type: url

      Multi-site and multi-instrument correction via
      `combatlearn <https://github.com/EttoreRocchi/combatlearn>`_.

----

Quick Example
-------------

.. code-block:: python

   from maldiamrkit import MaldiSpectrum, MaldiSet
   from maldiamrkit.alignment import Warping
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier

   # Load dataset (with parallel loading)
   data = MaldiSet.from_directory(
       "spectra/", "metadata.csv",
       aggregate_by=dict(antibiotics="Ceftriaxone"),
       n_jobs=-1  # Use all cores
   )

   # Create pipeline (with parallel warping)
   pipe = Pipeline([
       ("warp", Warping(method="shift", n_jobs=-1)),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier())
   ])

   # Train and evaluate
   pipe.fit(data.X, data.get_y_single())

----

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   api/index
   tutorials/index
   contributing
   papers
   changelog
