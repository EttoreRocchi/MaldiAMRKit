Tutorials
=========

Interactive tutorials demonstrating MaldiAMRKit usage.

.. toctree::
   :maxdepth: 2

Jupyter Notebooks
-----------------

The following Jupyter notebooks are available in the ``notebooks/`` directory:

- **01_quick_start.ipynb** - Loading, preprocessing, binning, and quality assessment
- **02_peak_detection.ipynb** - Local maxima and persistent homology methods
- **03_alignment.ipynb** - Warping methods and alignment quality

To view the notebooks, navigate to the ``notebooks/`` directory and start Jupyter:

.. code-block:: bash

   cd notebooks
   jupyter notebook

Example Workflows
-----------------

Basic Classification Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldiamrkit import MaldiSet, Warping, MaldiPeakDetector
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score, StratifiedKFold

   # Load data
   data = MaldiSet.from_directory(
       "spectra/", "meta.csv",
       aggregate_by=dict(antibiotics="Ceftriaxone")
   )

   # Build pipeline
   pipe = Pipeline([
       ("peaks", MaldiPeakDetector(binary=False, prominence=0.005)),
       ("warp", Warping(method="piecewise")),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
   ])

   # Cross-validation
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   scores = cross_val_score(
       pipe, data.X, data.get_y_single(),
       cv=cv, scoring="accuracy"
   )
   print(f"CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

Raw Spectra Alignment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldiamrkit import RawWarping

   # Use raw warping for better alignment
   warper = RawWarping(
       spectra_dir="spectra/",
       method="piecewise",
       bin_width=3,
       max_shift_da=10.0
   )

   # Fit and transform
   warper.fit(data.X)
   X_aligned = warper.transform(data.X)

   # Check quality
   quality = warper.get_alignment_quality(data.X, X_aligned)
   print(f"Mean improvement: {quality['improvement'].mean():.4f}")
