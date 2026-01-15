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

   # Load from directory (with parallel loading)
   data = MaldiSet.from_directory(
       spectra_dir="spectra/",
       meta_file="metadata.csv",
       aggregate_by=dict(
           antibiotics=["Ceftriaxone", "Ceftazidime"],
           species="Escherichia coli"
       ),
       n_jobs=-1  # Use all cores
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

   # Define pipeline (with parallel processing)
   pipe = Pipeline([
       ("peaks", MaldiPeakDetector(method="local", prominence=0.01, n_jobs=-1)),
       ("warp", Warping(method="shift", n_jobs=-1)),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier(n_estimators=100))
   ])

   # Cross-validation
   scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
   print(f"Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

Using Raw Spectra Warping
-------------------------

For higher precision alignment, use RawWarping with ``create_raw_input()``:

.. code-block:: python

   from maldiamrkit import RawWarping, create_raw_input
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier

   # Create input DataFrame from directory (discovers all .txt files)
   X_raw = create_raw_input("spectra/")

   # Create pipeline with raw warping
   pipe = Pipeline([
       ("warp", RawWarping(method="piecewise", bin_width=3, n_jobs=-1)),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier())
   ])

   pipe.fit(X_raw, y)

Quality Assessment
------------------

Evaluate preprocessing quality with comprehensive QC metrics:

.. code-block:: python

   from maldiamrkit import MaldiSpectrum, SpectrumQuality, estimate_snr, Warping

   # Load and preprocess spectrum
   spec = MaldiSpectrum("spectrum.txt").preprocess()

   # Quick SNR estimation
   snr = estimate_snr(spec.preprocessed)
   print(f"SNR: {snr:.1f}")

   # Comprehensive quality assessment
   qc = SpectrumQuality()
   report = qc.assess(spec.preprocessed)
   print(f"Peak count: {report.peak_count}")
   print(f"Total ion count: {report.total_ion_count:.2e}")
   print(f"Baseline fraction: {report.baseline_fraction:.2%}")

   # Evaluate alignment quality
   warper = Warping(method="shift", n_jobs=-1)
   warper.fit(X)
   X_aligned = warper.transform(X)

   quality = warper.get_alignment_quality(X, X_aligned)
   print(quality.mean())
