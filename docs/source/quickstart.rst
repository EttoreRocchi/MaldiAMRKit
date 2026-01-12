Quickstart Guide
================

This guide will help you get started with MaldiAMRKit for MALDI-TOF
mass spectrometry preprocessing.

Loading a Single Spectrum
-------------------------

Load and preprocess a single spectrum:

.. code-block:: python

   from maldiamrkit import MaldiSpectrum

   # Load spectrum from file
   spec = MaldiSpectrum("path/to/spectrum.txt")

   # Preprocess and bin
   spec.preprocess().bin(bin_width=3)

   # Visualize
   spec.plot()

Loading a Dataset
-----------------

Load multiple spectra with metadata:

.. code-block:: python

   from maldiamrkit import MaldiSet

   # Load from directory
   data = MaldiSet.from_directory(
       spectra_dir="spectra/",
       meta_file="metadata.csv",
       aggregate_by=dict(
           antibiotics=["Ceftriaxone", "Ceftazidime"],
           species="Escherichia coli"
       )
   )

   # Access feature matrix and labels
   X, y = data.X, data.y

   # Visualize as pseudogel
   data.plot_pseudogel(antibiotic="Ceftriaxone")

Building ML Pipelines
---------------------

Create scikit-learn compatible pipelines:

.. code-block:: python

   from maldiamrkit import Warping, MaldiPeakDetector
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score

   # Define pipeline
   pipe = Pipeline([
       ("peaks", MaldiPeakDetector(method="local", prominence=0.01)),
       ("warp", Warping(method="shift")),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier(n_estimators=100))
   ])

   # Cross-validation
   scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
   print(f"Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

Using Raw Spectra Warping
-------------------------

For higher precision alignment, use RawWarping:

.. code-block:: python

   from maldiamrkit import RawWarping

   # Create pipeline with raw warping
   pipe = Pipeline([
       ("warp", RawWarping(
           spectra_dir="spectra/",
           method="piecewise",
           bin_width=3
       )),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier())
   ])

   pipe.fit(X, y)

Quality Assessment
------------------

Evaluate preprocessing quality:

.. code-block:: python

   from maldiamrkit import estimate_snr, Warping

   # Estimate signal-to-noise ratio
   spec = MaldiSpectrum("spectrum.txt").preprocess()
   snr = estimate_snr(spec.preprocessed)
   print(f"SNR: {snr:.1f}")

   # Evaluate alignment quality
   warper = Warping(method="shift")
   warper.fit(X)
   X_aligned = warper.transform(X)

   quality = warper.get_alignment_quality(X, X_aligned)
   print(quality.mean())
