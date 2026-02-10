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
   io

Core Data Structures
--------------------

.. autosummary::
   :nosignatures:

   maldiamrkit.MaldiSpectrum
   maldiamrkit.MaldiSet
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
   maldiamrkit.preprocessing.estimate_snr
   maldiamrkit.preprocessing.merge_replicates
   maldiamrkit.preprocessing.detect_outlier_replicates

Alignment & Detection
---------------------

.. autosummary::
   :nosignatures:

   maldiamrkit.alignment.Warping
   maldiamrkit.alignment.RawWarping
   maldiamrkit.alignment.create_raw_input
   maldiamrkit.detection.MaldiPeakDetector

Evaluation
----------

.. autosummary::
   :nosignatures:

   maldiamrkit.evaluation.very_major_error_rate
   maldiamrkit.evaluation.major_error_rate
   maldiamrkit.evaluation.sensitivity_score
   maldiamrkit.evaluation.specificity_score
   maldiamrkit.evaluation.categorical_agreement
   maldiamrkit.evaluation.amr_classification_report
   maldiamrkit.evaluation.vme_me_curve
   maldiamrkit.evaluation.LabelEncoder
   maldiamrkit.evaluation.stratified_species_drug_split
   maldiamrkit.evaluation.case_based_split
   maldiamrkit.evaluation.SpeciesDrugStratifiedKFold
   maldiamrkit.evaluation.CaseGroupedKFold

I/O
---

.. autosummary::
   :nosignatures:

   maldiamrkit.io.read_spectrum
