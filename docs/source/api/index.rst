API Reference
=============

Complete reference for all public classes and functions in MaldiAMRKit,
organized by module.

.. toctree::
   :maxdepth: 2
   :hidden:

   core
   preprocessing
   alignment
   detection
   evaluation
   susceptibility
   similarity
   differential
   drift
   builder
   visualization
   io

Core Data Structures
--------------------

.. autosummary::
   :nosignatures:

   maldiamrkit.MaldiSpectrum
   maldiamrkit.MaldiSet

Filters
-------

.. autosummary::
   :nosignatures:

   maldiamrkit.filters.SpectrumFilter
   maldiamrkit.filters.SpeciesFilter
   maldiamrkit.filters.DrugFilter
   maldiamrkit.filters.QualityFilter
   maldiamrkit.filters.MetadataFilter

Preprocessing
-------------

.. autosummary::
   :nosignatures:

   maldiamrkit.preprocessing.PreprocessingPipeline
   maldiamrkit.preprocessing.preprocess
   maldiamrkit.preprocessing.bin_spectrum
   maldiamrkit.preprocessing.get_bin_metadata
   maldiamrkit.preprocessing.BinningMethod
   maldiamrkit.preprocessing.estimate_snr
   maldiamrkit.preprocessing.SpectrumQuality
   maldiamrkit.preprocessing.SpectrumQualityReport
   maldiamrkit.preprocessing.SignalMethod
   maldiamrkit.preprocessing.merge_replicates
   maldiamrkit.preprocessing.detect_outlier_replicates
   maldiamrkit.preprocessing.MergingMethod

Transformers
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.preprocessing.ClipNegatives
   maldiamrkit.preprocessing.SqrtTransform
   maldiamrkit.preprocessing.LogTransform
   maldiamrkit.preprocessing.SavitzkyGolaySmooth
   maldiamrkit.preprocessing.MovingAverageSmooth
   maldiamrkit.preprocessing.SNIPBaseline
   maldiamrkit.preprocessing.TopHatBaseline
   maldiamrkit.preprocessing.ConvexHullBaseline
   maldiamrkit.preprocessing.MedianBaseline
   maldiamrkit.preprocessing.MzTrimmer
   maldiamrkit.preprocessing.TICNormalizer
   maldiamrkit.preprocessing.MedianNormalizer
   maldiamrkit.preprocessing.PQNNormalizer
   maldiamrkit.preprocessing.MzMultiTrimmer

Alignment
---------

.. autosummary::
   :nosignatures:

   maldiamrkit.alignment.Warping
   maldiamrkit.alignment.RawWarping
   maldiamrkit.alignment.create_raw_input
   maldiamrkit.alignment.AlignmentStrategy
   maldiamrkit.alignment.AlignmentMethod

Peak Detection
--------------

.. autosummary::
   :nosignatures:

   maldiamrkit.detection.MaldiPeakDetector
   maldiamrkit.detection.PeakMethod

Evaluation
----------

Metrics
~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.evaluation.very_major_error_rate
   maldiamrkit.evaluation.major_error_rate
   maldiamrkit.evaluation.sensitivity_score
   maldiamrkit.evaluation.specificity_score
   maldiamrkit.evaluation.categorical_agreement

Reports
~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.evaluation.amr_classification_report
   maldiamrkit.evaluation.mic_regression_report
   maldiamrkit.evaluation.amr_multilabel_report
   maldiamrkit.evaluation.vme_me_curve

Scorers
~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.evaluation.vme_scorer
   maldiamrkit.evaluation.me_scorer

Splitting
~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.evaluation.stratified_species_drug_split
   maldiamrkit.evaluation.case_based_split
   maldiamrkit.evaluation.SpeciesDrugStratifiedKFold
   maldiamrkit.evaluation.CaseGroupedKFold

Susceptibility
--------------

MIC encoding, clinical breakpoint tables, and R/I/S label encoding.

.. autosummary::
   :nosignatures:

   maldiamrkit.susceptibility.MICEncoder
   maldiamrkit.susceptibility.BreakpointTable
   maldiamrkit.susceptibility.BreakpointResult
   maldiamrkit.susceptibility.LabelEncoder
   maldiamrkit.susceptibility.IntermediateHandling

Similarity
----------

.. autosummary::
   :nosignatures:

   maldiamrkit.similarity.spectral_distance
   maldiamrkit.similarity.SpectralMetric
   maldiamrkit.similarity.pairwise_distances
   maldiamrkit.similarity.cluster_spectra
   maldiamrkit.similarity.hierarchical_clustering
   maldiamrkit.similarity.hdbscan_clustering
   maldiamrkit.similarity.kmedoids_clustering
   maldiamrkit.similarity.silhouette_scores
   maldiamrkit.similarity.cluster_metadata_concordance
   maldiamrkit.similarity.ClusteringMethod
   maldiamrkit.similarity.KMedoidsInit
   maldiamrkit.similarity.plot_distance_heatmap
   maldiamrkit.similarity.plot_dendrogram

Differential Analysis
---------------------

Analysis
~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.differential.DifferentialAnalysis
   maldiamrkit.differential.StatisticalTest
   maldiamrkit.differential.CorrectionMethod

Plots
~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.differential.plot_volcano
   maldiamrkit.differential.plot_manhattan
   maldiamrkit.differential.plot_drug_comparison
   maldiamrkit.differential.DrugComparisonKind

Drift Monitoring
----------------

Monitor
~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.drift.DriftMonitor

Plots
~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.drift.plot_reference_drift
   maldiamrkit.drift.plot_pca_drift
   maldiamrkit.drift.plot_peak_stability
   maldiamrkit.drift.plot_effect_size_drift

Visualization
-------------

.. autosummary::
   :nosignatures:

   maldiamrkit.visualization.plot_spectrum
   maldiamrkit.visualization.plot_pseudogel
   maldiamrkit.visualization.plot_peaks
   maldiamrkit.visualization.plot_alignment
   maldiamrkit.visualization.plot_pca
   maldiamrkit.visualization.plot_tsne
   maldiamrkit.visualization.plot_umap

Dataset Building & Loading
--------------------------

Builder
~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.data.DatasetBuilder
   maldiamrkit.data.ProcessingHandler
   maldiamrkit.data.BuildReport

Input Layouts
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.data.InputLayout
   maldiamrkit.data.FlatLayout
   maldiamrkit.data.BrukerTreeLayout

Loader
~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.data.DatasetLoader

Dataset Layouts
~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.data.DatasetLayout
   maldiamrkit.data.DRIAMSLayout
   maldiamrkit.data.MARISMaLayout

Duplicate Handling
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.data.DuplicateStrategy

I/O
---

.. autosummary::
   :nosignatures:

   maldiamrkit.io.read_spectrum
   maldiamrkit.io.parse_mic_column
