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

      Installation

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
      following EUCAST conventions. Species-drug stratified, case-based,
      and group-aware (replicate-safe) splitting to prevent data leakage.

   .. grid-item-card:: MIC & Susceptibility
      :link: api/susceptibility.html
      :link-type: url

      ``MICEncoder`` and ``BreakpointTable`` turn raw MIC strings into
      log2(MIC) regression targets and S/I/R categories, with bundled
      EUCAST clinical breakpoints (v1.0-v16.0).

   .. grid-item-card:: Differential Analysis
      :link: api/differential.html
      :link-type: url

      Per-bin resistant-vs-susceptible testing with multiple-testing
      correction, fold change, and effect size. Volcano, Manhattan, and
      multi-drug comparison plots.

   .. grid-item-card:: Drift Monitoring
      :link: api/drift.html
      :link-type: url

      ``DriftMonitor`` tracks temporal drift via reference similarity,
      PCA centroid trajectory, and top-peak stability, with ready-made
      trajectory plots.

   .. grid-item-card:: DRIAMS Dataset Builder
      :link: quickstart.html#building-driams-like-datasets
      :link-type: url

      Build and load DRIAMS-like dataset directories from raw spectra and
      metadata, with year-based subfolders, custom processing handlers, and
      technical-replicate collapsing.

   .. grid-item-card:: Composable Filters
      :link: api/core.html#filters
      :link-type: url

      ``SpeciesFilter``, ``DrugFilter``, ``QualityFilter``, ``MetadataFilter``
      combinable with ``&``, ``|``, ``~`` operators.

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
   :caption: Get Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api/index
   cli

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Resources

   tutorials/index
   contributing
   papers
   changelog
