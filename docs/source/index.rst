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

   .. grid-item-card:: Peak Detection
      :link: api/detection.html
      :link-type: url

      Local maxima and persistent homology methods with parallel
      processing support.

   .. grid-item-card:: Multiple Binning Strategies
      :link: api/preprocessing.html#binning
      :link-type: url

      Uniform, logarithmic, adaptive, and custom bin edges for
      domain-specific analysis.

   .. grid-item-card:: Preprocessing Pipeline
      :link: quickstart.html#custom-preprocessing-pipeline
      :link-type: url

      Composable pipeline of transformers (smoothing, baseline, trimming,
      normalization). Serializable to JSON/YAML.

   .. grid-item-card:: Composable Filters
      :link: api/core.html#filters
      :link-type: url

      ``SpeciesFilter``, ``DrugFilter``, ``QualityFilter``, ``MetadataFilter``
      combinable with ``&``, ``|``, ``~`` operators.

   .. grid-item-card:: AMR Evaluation
      :link: api/evaluation.html
      :link-type: url

      VME, ME, sensitivity, specificity, and classification reports
      following EUCAST/CLSI conventions.

   .. grid-item-card:: Stratified Splitting
      :link: api/evaluation.html#splitting-utilities
      :link-type: url

      Species-drug stratified and case-based (patient-grouped) splits
      to prevent data leakage.

   .. grid-item-card:: CLI & Export
      :link: quickstart.html#command-line-interface
      :link-type: url

      ``maldiamrkit preprocess`` and ``maldiamrkit quality`` for batch
      processing. Export to CSV/Parquet.

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
   changelog
