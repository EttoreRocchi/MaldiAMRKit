# Changelog

All notable changes to MaldiAMRKit are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.7.0] - 2026-02-10

First changelog-tracked release of MaldiAMRKit.

- **Spectral preprocessing**: composable pipeline of transformers
  (smoothing, baseline correction, normalization, trimming) with
  multiple binning strategies (uniform, logarithmic, adaptive, custom)
  and raw-spectrum alignment (shift, linear, piecewise, DTW).
- **AMR evaluation**: VME/ME rates, sensitivity, specificity, and
  classification reports following EUCAST/CLSI conventions; species-drug
  stratified and case-based splitting to prevent data leakage.
- **Composable filters**: `SpeciesFilter`, `DrugFilter`, `QualityFilter`,
  and `MetadataFilter` combinable with `&`, `|`, `~` operators for
  flexible dataset subsetting.
- **CLI**: `maldiamrkit preprocess` and `maldiamrkit quality` for batch
  processing and quality assessment from the command line.

See the [documentation](https://maldiamrkit.readthedocs.io/) for the
complete API reference and the full set of package functionalities.
