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
   from maldiamrkit.visualization import plot_spectrum
   plot_spectrum(spec)

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
   from maldiamrkit.visualization import plot_pseudogel
   plot_pseudogel(data, antibiotic="Ceftriaxone")

Building ML Pipelines
---------------------

Create scikit-learn compatible pipelines:

.. code-block:: python

   from maldiamrkit.alignment import Warping
   from maldiamrkit.detection import MaldiPeakDetector
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

   from maldiamrkit.alignment import RawWarping, create_raw_input
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

Replicate Merging
-----------------

Merge multiple spectral replicates into a single consensus spectrum:

.. code-block:: python

   from maldiamrkit import MaldiSpectrum
   from maldiamrkit.preprocessing import merge_replicates, detect_outlier_replicates

   # Load replicates as MaldiSpectrum objects
   spectra = [MaldiSpectrum(f"data/isolate_rep{i}.txt") for i in range(1, 4)]
   keep = detect_outlier_replicates(spectra)
   clean = [s for s, k in zip(spectra, keep) if k]
   merged = merge_replicates(clean, method="mean")

Quality Assessment
------------------

Evaluate preprocessing quality with comprehensive QC metrics:

.. code-block:: python

   from maldiamrkit import MaldiSpectrum
   from maldiamrkit.preprocessing import SpectrumQuality, estimate_snr
   from maldiamrkit.alignment import Warping

   # Load and preprocess spectrum
   spec = MaldiSpectrum("spectrum.txt").preprocess()

   # Quick SNR estimation
   snr = estimate_snr(spec)
   print(f"SNR: {snr:.1f}")

   # Comprehensive quality assessment
   qc = SpectrumQuality()
   report = qc.assess(spec)
   print(f"Peak count: {report.peak_count}")
   print(f"Total ion count: {report.total_ion_count:.2e}")
   print(f"Baseline fraction: {report.baseline_fraction:.2%}")

   # Evaluate alignment quality
   warper = Warping(method="shift", n_jobs=-1)
   warper.fit(X)
   X_aligned = warper.transform(X)

   quality = warper.get_alignment_quality(X, X_aligned)
   print(quality.mean())

Custom Preprocessing Pipeline
------------------------------

Build and serialize custom preprocessing pipelines:

.. code-block:: python

   from maldiamrkit import MaldiSpectrum
   from maldiamrkit.preprocessing import (
       PreprocessingPipeline,
       ClipNegatives, LogTransform, SavitzkyGolaySmooth,
       SNIPBaseline, MzTrimmer, TICNormalizer,
   )

   # Build custom pipeline
   pipe = PreprocessingPipeline([
       ("clip", ClipNegatives()),
       ("log", LogTransform()),
       ("smooth", SavitzkyGolaySmooth(window_length=15)),
       ("baseline", SNIPBaseline(half_window=30)),
       ("trim", MzTrimmer(mz_min=2000, mz_max=20000)),
       ("norm", TICNormalizer()),
   ])

   # Use with MaldiSpectrum
   spec = MaldiSpectrum("spectrum.txt", pipeline=pipe)
   spec.preprocess().bin(3)

   # Save for reproducibility
   pipe.to_json("my_pipeline.json")

Using Composable Filters
-------------------------

Select subsets of a dataset using composable filter predicates:

.. code-block:: python

   from maldiamrkit import MaldiSet
   from maldiamrkit.filters import SpeciesFilter, DrugFilter, QualityFilter

   data = MaldiSet.from_directory(
       "spectra/", "metadata.csv",
       aggregate_by=dict(antibiotics="Drug")
   )

   # Filter by species and quality
   f = SpeciesFilter("Escherichia coli") & QualityFilter(min_snr=5.0)
   filtered = data.filter(f)

   # Filter by antibiotic resistance status
   f = SpeciesFilter("Escherichia coli") & DrugFilter("Ceftriaxone", status="R")
   resistant_ecoli = data.filter(f)

Evaluation Metrics
------------------

Compute AMR-specific metrics following EUCAST/CLSI conventions:

.. code-block:: python

   from maldiamrkit.evaluation import (
       amr_classification_report, vme_scorer,
       LabelEncoder,
   )
   from sklearn.model_selection import cross_val_score

   # Encode labels
   enc = LabelEncoder()
   y_binary = enc.fit_transform(y_raw)

   # Full classification report
   report = amr_classification_report(y_true, y_pred)
   print(f"VME: {report['vme']:.3f}, ME: {report['me']:.3f}")

   # Use VME scorer in cross-validation
   scores = cross_val_score(pipe, X, y, cv=5, scoring=vme_scorer)

Stratified Splitting
--------------------

Prevent data leakage with species-aware splits:

.. code-block:: python

   from maldiamrkit.evaluation import stratified_species_drug_split, SpeciesDrugStratifiedKFold

   # Single split
   X_train, X_test, y_train, y_test = stratified_species_drug_split(
       X, y, species=species_labels, test_size=0.2, random_state=42
   )

   # Cross-validation
   cv = SpeciesDrugStratifiedKFold(n_splits=5)
   for train_idx, test_idx in cv.split(X, y, species=species_labels):
       pass

Building DRIAMS-like Datasets
-----------------------------

Create a standardised DRIAMS-like dataset directory from raw spectra and a
metadata CSV:

.. code-block:: python

   from maldiamrkit.data import DatasetBuilder, FlatLayout, ProcessingHandler

   # Basic: produces raw/, preprocessed/, binned_6000/, id/
   layout = FlatLayout("spectra/", "metadata.csv")
   report = DatasetBuilder(layout, "output/my_dataset").build()
   print(f"Processed {report.succeeded}/{report.total} spectra")

With year-based subfolders (requires a date or year column in the metadata):

.. code-block:: python

   layout = FlatLayout("spectra/", "metadata.csv", year_column="acquisition_date")
   report = DatasetBuilder(layout, "output/my_dataset").build()

Build from Bruker binary data (e.g. MARISMa-style directory tree):

.. code-block:: python

   from maldiamrkit.data import BrukerTreeLayout

   layout = BrukerTreeLayout("path/to/MARISMa", "path/to/AMR.csv")
   report = DatasetBuilder(layout, "output/my_dataset").build()

Add extra processing variants alongside the defaults using
``ProcessingHandler``:

.. code-block:: python

   from maldiamrkit.preprocessing import PreprocessingPipeline

   sqrt_pipeline = PreprocessingPipeline.from_yaml("sqrt_pipeline.yaml")

   layout = FlatLayout("spectra/", "metadata.csv", year_column="acquisition_date")
   report = DatasetBuilder(
       layout, "output/my_dataset",
       extra_handlers=[
           ProcessingHandler("preprocessed_sqrt", "preprocessed",
                             pipeline=sqrt_pipeline),
           ProcessingHandler("binned_3000", "binned", bin_width=6),
       ],
   ).build()

The output structure follows the DRIAMS convention:

.. code-block:: text

   my_dataset/
   ├── raw/{year}/
   ├── preprocessed/{year}/
   ├── preprocessed_sqrt/{year}/
   ├── binned_6000/{year}/
   ├── binned_3000/{year}/
   └── id/{year}/{year}_clean.csv

Loading Datasets
----------------

Load an existing dataset (built with ``DatasetBuilder`` or downloaded
from the public DRIAMS repository):

.. code-block:: python

   from maldiamrkit.data import DatasetLoader, DRIAMSLayout

   # Auto-detect stage (prefers binned), load all years
   ds = DatasetLoader(DRIAMSLayout("output/my_dataset")).load()
   print(ds.X.shape)

   # Load a specific stage and year
   ds = DatasetLoader(
       DRIAMSLayout("output/my_dataset", year=2015),
       stage="preprocessed",
   ).load(aggregate_by=dict(antibiotics="Ceftriaxone"))

Load directly from a MARISMa-style Bruker tree (no build step needed):

.. code-block:: python

   from maldiamrkit.data import DatasetLoader, MARISMaLayout

   ds = DatasetLoader(
       MARISMaLayout("path/to/MARISMa", "path/to/AMR.csv", year=2024),
   ).load()

Exporting Spectra
-----------------

Save individual spectra (raw, preprocessed, or binned) to files:

.. code-block:: python

   # Save preprocessed spectra as TXT files
   data.save_spectra("output/preprocessed/", stage="preprocessed", fmt="txt")

   # Save binned spectra as CSV files
   data.save_spectra("output/binned/", stage="binned", fmt="csv")

Batch Effect Correction
-----------------------

For multi-site or multi-instrument data, use
`combatlearn <https://github.com/EttoreRocchi/combatlearn>`_
(``pip install maldiamrkit[batch]``):

.. code-block:: python

   from combatlearn import Combat

   combat = Combat(method="fortin")
   combat.fit(X, y=batch_labels)
   X_corrected = combat.transform(X, y=batch_labels)

See the `combatlearn documentation <https://combatlearn.readthedocs.io/>`_ for
full usage and the :doc:`exploration tutorial </tutorials/notebooks/05_exploration>`
for a worked example.

Command-Line Interface
----------------------

MaldiAMRKit provides CLI commands for batch processing:

- ``maldiamrkit preprocess`` - build a CSV feature matrix from raw spectra
- ``maldiamrkit quality`` - generate quality reports
- ``maldiamrkit build`` - construct a DRIAMS-like dataset directory

.. code-block:: bash

   maldiamrkit preprocess --input-dir data/ --output features.csv

See the :doc:`CLI Reference <cli>` for the full command documentation and
examples.
