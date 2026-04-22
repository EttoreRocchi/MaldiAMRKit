Drift Module
============

Temporal drift monitoring for MALDI-TOF spectra.  The ``DriftMonitor``
class anchors a baseline on the earliest timestamps and reports
three complementary views of drift over subsequent time windows:
reference similarity, PCA centroid trajectory, and peak-selection
stability (plus per-peak Cohen's d tracking).

Monitor
-------

.. autoclass:: maldiamrkit.drift.DriftMonitor
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. autofunction:: maldiamrkit.drift.plot_reference_drift

.. autofunction:: maldiamrkit.drift.plot_pca_drift

.. autofunction:: maldiamrkit.drift.plot_peak_stability

.. autofunction:: maldiamrkit.drift.plot_effect_size_drift

Example
-------

.. code-block:: python

    from maldiamrkit import MaldiSet
    from maldiamrkit.differential import DifferentialAnalysis
    from maldiamrkit.drift import (
        DriftMonitor,
        plot_reference_drift,
        plot_pca_drift,
        plot_peak_stability,
        plot_effect_size_drift,
    )

    # MaldiSet with an acquisition-date metadata column
    data = MaldiSet.from_directory(
        "spectra/", "metadata.csv",
        aggregate_by=dict(antibiotics="Ceftriaxone"),
    )

    # Reference similarity + PCA drift (no labels required)
    monitor = DriftMonitor(
        time_column="acquisition_date", window="30D",
    ).fit(data)

    ref_df = monitor.monitor(data)
    pca_df = monitor.monitor_pca(data)

    plot_reference_drift(ref_df, title="Cosine distance to baseline median")
    plot_pca_drift(pca_df, title="Centroid trajectory")

    # Peak-selection stability + per-peak effect size drift
    baseline_analysis = DifferentialAnalysis.from_maldi_set(
        data, antibiotic="Ceftriaxone"
    ).run()
    stability_df = monitor.monitor_peak_stability(
        data, baseline_analysis, antibiotic="Ceftriaxone", n_top=20,
    )
    effect_df = monitor.monitor_effect_sizes(
        data,
        peaks=list(baseline_analysis.top_peaks(n=5)["mz_bin"]),
        antibiotic="Ceftriaxone",
    )

    plot_peak_stability(stability_df)
    plot_effect_size_drift(effect_df)
