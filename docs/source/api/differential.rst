Differential Module
===================

Per-bin differential peak testing between resistant (R) and susceptible
(S) groups, with multiple-testing correction, log2 fold change,
Cohen's d effect size, and AMR-aware visualizations.

Analysis
--------

.. autoclass:: maldiamrkit.differential.DifferentialAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.differential.StatisticalTest
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.differential.CorrectionMethod
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. autofunction:: maldiamrkit.differential.plot_volcano

.. autofunction:: maldiamrkit.differential.plot_manhattan

.. autofunction:: maldiamrkit.differential.plot_drug_comparison

.. autoclass:: maldiamrkit.differential.DrugComparisonKind
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

    from maldiamrkit.differential import (
        DifferentialAnalysis,
        plot_volcano,
        plot_manhattan,
        plot_drug_comparison,
    )

    # Per-drug analysis: run Mann-Whitney + FDR-BH across all m/z bins
    analysis = DifferentialAnalysis.from_maldi_set(
        maldi_set, antibiotic="Ceftriaxone"
    ).run(test="mann_whitney", correction="fdr_bh")

    # On small datasets, narrow the hypothesis set before correction:
    from maldiamrkit.detection import MaldiPeakDetector
    analysis = DifferentialAnalysis.from_maldi_set(
        maldi_set, antibiotic="Ceftriaxone"
    ).run(
        mz_ranges=[(2000, 5000), (9000, 12000)],
        peak_detector=MaldiPeakDetector(prominence=1e-4),
    )

    # Inspect the top 20 peaks by adjusted p-value
    analysis.top_peaks(n=20)

    # Significance filter: |log2FC| >= 1 and adjusted p-value <= 0.05
    analysis.significant_peaks(fc_threshold=1.0, p_threshold=0.05)

    # Volcano and Manhattan visualizations
    plot_volcano(analysis.results, fc_threshold=1.0, p_threshold=0.05)
    plot_manhattan(analysis.results, p_threshold=0.05)

    # Multi-drug comparison: which peaks are shared / unique across drugs?
    comparison = DifferentialAnalysis.compare_drugs({
        "Ceftriaxone": analysis_cro,
        "Ceftazidime": analysis_caz,
        "Meropenem":   analysis_mem,
    })
    plot_drug_comparison(comparison, kind="heatmap")
    plot_drug_comparison(comparison, kind="upset")
