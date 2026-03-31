# Changelog

All notable changes to MaldiAMRKit are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.10.0] - 2026-03-31

### Added

- **Exploratory visualizations**: `plot_pca`, `plot_tsne`, `plot_umap` functions for dimensionality reduction scatter plots colored by metadata columns (species, resistance phenotype, batch). UMAP requires the `[batch]` optional extra.
- **Multi-drug AMR evaluation**: `amr_multilabel_report()` computes per-drug VME, ME, sensitivity, specificity, and categorical agreement with macro-average, supporting `as_dataframe=True`.
- **LabelEncoder 2-D support**: `LabelEncoder` now accepts DataFrames (multi-drug labels) and returns encoded DataFrames. New `intermediate="nan"` mode maps intermediate labels to `NaN` for independent per-drug handling.
- **`[batch]` install extra**: `pip install maldiamrkit[batch]` installs `combatlearn` for ComBat-based batch effect correction and `umap-learn` for UMAP plots.
- **`[all]` install extra**: `pip install maldiamrkit[all]` installs all optional dependencies (formats + batch).

### Fixed

- Notebook images on ReadTheDocs: fixed broken image extraction on ReadTheDocs.

### Changed

- Updated installation guide with `[batch]` and `[all]` extras, combatlearn links and reference.
- Added exploratory plots and multi-drug evaluation sections to API reference, quickstart, and README.
- Tutorial notebook `05_exploration.ipynb` demonstrating PCA, t-SNE, UMAP, and batch correction workflow.

## [0.9.0] - 2026-03-25

### Added

- **DRIAMS dataset loader**: `load_driams_dataset()` function for loading DRIAMS-formatted directories into `MaldiSet` objects, with auto-detection of processing stage (binned > preprocessed > raw), year-based layouts, and ID column.
- **mzML/mzXML support**: `read_spectrum()` now reads `.mzML` and `.mzXML` files via optional `pyteomics` dependency.
- **`[formats]` install extra**: `pip install maldiamrkit[formats]` installs `pyteomics` and `lxml` for mzML/mzXML parsing.

### Changed

- Updated API reference, installation guide, and quickstart for the new loader and mzML/mzXML support.

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

### Added

- **Spectral preprocessing**: composable pipeline of transformers (smoothing, baseline correction, normalization, trimming) with multiple binning strategies (uniform, logarithmic, adaptive, custom) and raw-spectrum alignment (shift, linear, piecewise, DTW).
- **AMR evaluation**: VME/ME rates, sensitivity, specificity, and classification reports following EUCAST/CLSI conventions; species-drug stratified and case-based splitting to prevent data leakage.
- **Composable filters**: `SpeciesFilter`, `DrugFilter`, `QualityFilter`, and `MetadataFilter` combinable with `&`, `|`, `~` operators for flexible dataset subsetting.
- **CLI**: `maldiamrkit preprocess` and `maldiamrkit quality` for batch processing and quality assessment from the command line.

See the [documentation](https://maldiamrkit.readthedocs.io/) for the complete API reference and the full set of package functionalities.
