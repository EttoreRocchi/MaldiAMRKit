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
- **04_evaluation.ipynb** - AMR metrics, label encoding, and stratified splitting

To view the notebooks, navigate to the ``notebooks/`` directory and start Jupyter:

.. code-block:: bash

   cd notebooks
   jupyter notebook

Example Workflows
-----------------

Basic Classification Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldiamrkit import MaldiSet
   from maldiamrkit.alignment import Warping
   from maldiamrkit.detection import MaldiPeakDetector
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

   from maldiamrkit.alignment import RawWarping, create_raw_input

   # Build input DataFrame from spectrum directory
   X_raw = create_raw_input("spectra/")

   # Use raw warping for better alignment
   warper = RawWarping(method="piecewise", bin_width=3, max_shift_da=10.0)

   # Fit and transform
   warper.fit(X_raw)
   X_aligned = warper.transform(X_raw)

Building DRIAMS-like Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldiamrkit import build_driams_dataset, ProcessingHandler
   from maldiamrkit.preprocessing import PreprocessingPipeline

   # Build a DRIAMS-like dataset with year-based subfolders
   report = build_driams_dataset(
       spectra_dir="spectra/",
       metadata_csv="metadata.csv",
       output_dir="output/my_dataset",
       year_column="acquisition_date",
   )
   print(f"Processed {report.succeeded}/{report.total} spectra")
   print(f"Folders: {report.folders_created}")

   # Add extra processing variants
   sqrt_pipe = PreprocessingPipeline.from_yaml("sqrt_pipeline.yaml")
   report = build_driams_dataset(
       "spectra/", "metadata.csv", "output/my_dataset",
       year_column="acquisition_date",
       extra_handlers=[
           ProcessingHandler("preprocessed_sqrt", "preprocessed",
                             pipeline=sqrt_pipe),
           ProcessingHandler("binned_3000", "binned", bin_width=6),
       ],
   )
