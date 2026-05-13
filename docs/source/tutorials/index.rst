Tutorials
=========

Interactive tutorials demonstrating MaldiAMRKit usage.

.. toctree::
   :maxdepth: 2

   notebooks/01_quick_start
   notebooks/02_peak_detection
   notebooks/03_alignment
   notebooks/04_evaluation
   notebooks/05_exploration
   notebooks/06_differential_analysis
   notebooks/07_drift_monitoring

Datasets
--------

Notebooks ``01``--``03`` run on the small example dataset bundled with the
repository under ``data/``. Notebooks ``04``--``07`` require more samples
and use the real **MALDI-Kleb-AI** archive (Rocchi *et al.*, 2026;
`Zenodo DOI 10.5281/zenodo.17405072 <https://zenodo.org/records/17405072>`_)
via the :file:`notebooks/_demo.py` helper. The helper caches the 370 MB
tarball under ``~/.cache/maldiamrkit/`` (or ``$MALDIAMRKIT_CACHE_DIR``)
on first use, and by default restricts the dataset to the **Rome
sub-cohort** (single acquisition centre, ~470 spectra) so that the
demonstrations do not require batch-effect correction.

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

Building Datasets
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldiamrkit.data import DatasetBuilder, FlatLayout, ProcessingHandler
   from maldiamrkit.preprocessing import PreprocessingPipeline

   # Build a dataset with year-based subfolders
   layout = FlatLayout("spectra/", "metadata.csv", year_column="acquisition_date")
   report = DatasetBuilder(layout, "output/my_dataset").build()
   print(f"Processed {report.succeeded}/{report.total} spectra")
   print(f"Folders: {report.folders_created}")

   # Add extra processing variants
   sqrt_pipe = PreprocessingPipeline.from_yaml("sqrt_pipeline.yaml")
   report = DatasetBuilder(
       layout, "output/my_dataset",
       extra_handlers=[
           ProcessingHandler("preprocessed_sqrt", "preprocessed",
                             pipeline=sqrt_pipe),
           ProcessingHandler("binned_3000", "binned", bin_width=6),
       ],
   ).build()

   # Build from Bruker binary data (e.g. MARISMa)
   from maldiamrkit.data import BrukerTreeLayout

   bruker_layout = BrukerTreeLayout("path/to/MARISMa", "path/to/AMR.csv")
   report = DatasetBuilder(bruker_layout, "output/marisma_built").build()
