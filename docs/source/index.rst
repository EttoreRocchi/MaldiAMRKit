MaldiAMRKit Documentation
=========================

A Python toolkit for MALDI-TOF mass spectrometry preprocessing for
antimicrobial resistance (AMR) prediction.

MaldiAMRKit provides scikit-learn compatible transformers for preprocessing
MALDI-TOF mass spectra, enabling seamless integration into machine learning
pipelines.

Features
--------

- **Scikit-learn compatible transformers** - Use directly in sklearn pipelines
- **Multiple peak detection methods** - Local maxima and persistent homology
- **Spectral alignment** - Shift, linear, piecewise, and DTW warping
- **Raw spectra warping** - Full m/z resolution alignment before binning
- **Parallel processing** - Multi-core support via ``n_jobs`` parameter
- **Multiple binning strategies** - Uniform, logarithmic, adaptive, and custom binning
- **Quality metrics** - SNR estimation, spectrum quality assessment, and alignment quality

Quick Start
-----------

.. code-block:: python

   from maldiamrkit import MaldiSpectrum, MaldiSet, Warping
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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   tutorials/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
