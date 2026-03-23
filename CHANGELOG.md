# Changelog

All notable changes to MaldiAMRKit are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.8.0] - 2026-03-23

### Added

- **DRIAMS dataset builder**: `build_driams_dataset()` function and `maldiamrkit build-driams` CLI command for creating DRIAMS-like dataset directories from raw spectra and metadata, with year-based subfolders, configurable output ID column, dynamic binned folder naming, and extra processing handlers (`ProcessingHandler`).
- **Visualization module** (`maldiamrkit.visualization`): standalone `plot_spectrum`, `plot_pseudogel`, `plot_peaks`, and `plot_alignment` functions extracted from data classes for better separation of concerns.
- **Alignment strategies** (`maldiamrkit.alignment.strategies`): `AlignmentStrategy` ABC with `ShiftStrategy`, `LinearStrategy`, `PiecewiseStrategy`, and `DTWStrategy` implementations. New strategies can be registered via `ALIGNMENT_REGISTRY`.
- **Binning registry** (`BINNING_REGISTRY`): extensible registry for binning methods - custom methods can be added without modifying `bin_spectrum()`.
- `MaldiSpectrum.get_data()` method for accessing spectrum data at the best available processing stage without touching private attributes.
- `MaldiSpectrum.has_bin_metadata` property.

### Changed

- Plotting functions (`plot_spectrum`, `plot_pseudogel`, `plot_alignment`, `plot_peaks`) moved to dedicated `maldiamrkit.visualization` module.
- `Warping` and `RawWarping` now delegate alignment logic to shared strategy classes, eliminating code duplication.

## [0.7.0] - 2026-02-10

First changelog-tracked release of MaldiAMRKit.

- **Spectral preprocessing**: composable pipeline of transformers (smoothing, baseline correction, normalization, trimming) with multiple binning strategies (uniform, logarithmic, adaptive, custom) and raw-spectrum alignment (shift, linear, piecewise, DTW).
- **AMR evaluation**: VME/ME rates, sensitivity, specificity, and classification reports following EUCAST/CLSI conventions; species-drug stratified and case-based splitting to prevent data leakage.
- **Composable filters**: `SpeciesFilter`, `DrugFilter`, `QualityFilter`, and `MetadataFilter` combinable with `&`, `|`, `~` operators for flexible dataset subsetting.
- **CLI**: `maldiamrkit preprocess` and `maldiamrkit quality` for batch processing and quality assessment from the command line.

See the [documentation](https://maldiamrkit.readthedocs.io/) for the complete API reference and the full set of package functionalities.
