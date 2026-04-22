"""Temporal drift monitoring for MALDI-TOF spectra.

`DriftMonitor` establishes a baseline from early timestamps and tracks
instrument or biological drift via three complementary views:

- reference similarity (distance of window-median to baseline reference)
- PCA centroid trajectory (movement in a baseline-fitted PCA space)
- peak-selection stability (Jaccard overlap of top-k discriminative
  peaks vs. baseline) and per-peak Cohen's d over time
"""

from .monitor import DriftMonitor
from .plots import (
    plot_effect_size_drift,
    plot_pca_drift,
    plot_peak_stability,
    plot_reference_drift,
)

__all__ = [
    "DriftMonitor",
    "plot_effect_size_drift",
    "plot_pca_drift",
    "plot_peak_stability",
    "plot_reference_drift",
]
