Differential module
===================

Per-bin differential peak testing between resistant (R) and susceptible
(S) groups, with multiple-testing correction, log2 fold change,
Cohen's d effect size, and AMR-aware visualisations.

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

Visualisation
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

    import pandas as pd
    from maldiamrkit.differential import (
        DifferentialAnalysis,
        plot_volcano,
        plot_manhattan,
        plot_drug_comparison,
    )
    from maldiamrkit.susceptibility import LabelEncoder

    # Labels must be 0/1: encode R/S, with intermediate as NaN, and drop NaN
    y = pd.Series(
        LabelEncoder(intermediate="nan").fit_transform(
            maldi_set.get_y_single("Ceftriaxone")
        ),
        index=maldi_set.X.index,
    ).dropna().astype(int)
    X = maldi_set.X.loc[y.index]

    # Per-drug analysis: run Mann-Whitney + FDR-BH across all m/z bins
    analysis = DifferentialAnalysis(X, y).run(
        test="mann_whitney", correction="fdr_bh"
    )

    # On small datasets, narrow the hypothesis set before correction:
    from maldiamrkit.detection import MaldiPeakDetector
    analysis = DifferentialAnalysis(X, y).run(
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
    # analysis_cro, analysis_caz, analysis_mem: one analysis per drug, run as above
    comparison = DifferentialAnalysis.compare_drugs({
        "Ceftriaxone": analysis_cro,
        "Ceftazidime": analysis_caz,
        "Meropenem":   analysis_mem,
    })
    plot_drug_comparison(comparison, kind="heatmap")
    plot_drug_comparison(comparison, kind="upset")

``DifferentialAnalysis.from_maldi_set`` reads the label column as it is, so it
works only when the metadata already stores 0/1 labels.
