# Changelog

All notable changes to MaldiAMRKit are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.15.0] - 2026-05-14

### Added

- **New `maldiamrkit.susceptibility` submodule** consolidating MIC encoding, clinical breakpoints, and resistance label handling:
    - `MICEncoder(breakpoints=None)` - sklearn-style transformer producing a tidy DataFrame with `log2_mic`, `censored`, and (when `breakpoints` is provided) `category` (`S`/`I`/`R`), `atu` (Area of Technical Uncertainty flag), and `source` (provenance) columns. One encoder, both regression and classification targets.
    - `BreakpointTable` - clinical breakpoint table for MIC interpretation. Constructors: `from_version("16.0")`, `from_year(2026)`, `from_yaml(path)`, `from_latest()`, plus `BreakpointTable.list_available()` for discovery. `bp.apply(species, drug, mic)` returns a `BreakpointResult`; `bp.apply_batch(...)` is the vectorised path used by `MICEncoder`.
    - `BreakpointResult` - dataclass with `category`, `atu`, `source` fields. ATU is treated as an *assay-quality flag* orthogonal to S/I/R (EUCAST `I` = "Susceptible, increased exposure", not "uncertain").
- **`mic_regression_report(y_true, y_pred, breakpoints=...)`** in `maldiamrkit.evaluation` - regression counterpart to `amr_classification_report`: RMSE / MAE / bias in log2 dilutions, essential agreement (±1 dilution), and (with breakpoints) clinical categorical agreement, very-major-error rate, and major-error rate.
- **19 vendored EUCAST clinical breakpoint tables** (v1.0 to v16.0) under `maldiamrkit/data/breakpoints/eucast/`, auto-converted from the official EUCAST Excel workbooks.
- **Self-describing dataset manifest (`site_info.json`)**. `DatasetBuilder.build()` writes a versioned manifest at the dataset root recording loader settings (`id_column`, `metadata_dir`, `metadata_suffix`, `spectrum_ext`, `spectra_folders`, `mz_range`, `bin_width`) plus an optional `build_info` block with provenance. `DRIAMSLayout` reads it at construction and fills in any kwarg the caller omitted; explicit kwargs win, missing manifests fall back to defaults. Format is integer-versioned with a lenient reader (future versions warn but still load if all v1 fields are present).

### Changed

- **`LabelEncoder` and `IntermediateHandling` moved** from `maldiamrkit.evaluation` to `maldiamrkit.susceptibility`. Both old import paths (`from maldiamrkit.evaluation import LabelEncoder` and `from maldiamrkit.evaluation.label_encoder import LabelEncoder`) still work via lazy re-exports that emit `DeprecationWarning`. The shims will be removed in v0.17.

## [0.14.0] - 2026-04-27

### Added

- **New preprocessing transformers**: `TopHatBaseline` (morphological top-hat), `ConvexHullBaseline` (parameter-free lower-convex-hull), `MedianBaseline` (iterative rolling-median), and `MovingAverageSmooth` (uniform moving-average), all `PreprocessingPipeline`-serialisable.
- **MAD noise estimation**: `SpectrumQuality.estimate_mad_noise()` with configurable scaling `constant` and per-call `mz_region` override.
- **New warping methods**: `method="quadratic"` / `"cubic"` polynomial recalibration via `numpy.polyfit` (with shift-alignment fallback), and `method="lowess"` non-linear LOWESS recalibration with `lowess_frac` and `lowess_it` parameters.
- **`DatasetLayout.postprocess_spectrum` hook**: virtual method called by `DatasetLoader.load()` after each spectrum, letting layouts apply dataset-specific fix-ups without touching the loader.
- **DRIAMS m/z fixes**: `DRIAMSLayout` rewrites `binned_N/` bin indices to real m/z (`mz_min` / `mz_max` kwargs), fixing every downstream m/z-aware API; new `normalize_tic` kwarg for opt-in per-spectrum TIC re-normalization, and new `id_transform` kwarg for metadata ID canonicalisation.

### Changed

- **Plot API overhaul** across all 15 public plot functions: consistent `(fig, ax)` returns and `show=True`, default titles, dynamic `figsize`, legend sample counts, susceptibility-aware group ordering (S, then I, then R) with human-readable labels, unified `random_state` / `label_map` / `legend_loc` on dimensionality-reduction plots, metric-aware colourbar bounds and optional clustering on `plot_distance_heatmap`, `annotate_top_k` on `plot_volcano` / `plot_manhattan`, and reference lines / baseline markers on the drift plots.
- **`Warping.transform` non-monotonic warning** aggregated once per call instead of one warning per sample.
- **Backwards-compatible deprecations**: `plot_spectrum(binned=...)` and `plot_pseudogel(sort_by_intensity=...)` emit `DeprecationWarning`.

### Fixed

- **MARISMa loading**: `DatasetLoader.load()` matches spectrum files by `MaldiSpectrum._infer_id(...)` instead of `Path.name`, so MARISMa datasets with `duplicate_strategy="keep_all"` load end-to-end.

## [0.13.0] - 2026-04-22

### Added

- **Differential analysis module** (`maldiamrkit.differential`): `DifferentialAnalysis` class for per-bin statistical testing (Mann-Whitney U, Welch's t-test) between resistant and susceptible groups, with multiple-testing correction (Benjamini-Hochberg and Benjamini-Yekutieli FDR, Bonferroni), log2 fold change, and Cohen's d effect size; `from_maldi_set()` constructor, `top_peaks()` / `significant_peaks()` selectors, and `compare_drugs()` for multi-drug significance matrices.
- **Pre-test filtering in `DifferentialAnalysis.run()`**: `mz_ranges` (single tuple or list of `(low, high)` tuples) restricts testing to m/z windows of interest; `peak_detector` (a `MaldiPeakDetector` instance) further restricts testing to bins that are peaks in at least one spectrum.
- **Differential visualizations**: `plot_volcano()` (log2FC vs. -log10 adjusted p-value with direction-coloured points and threshold lines), `plot_manhattan()` (p-value significance along the m/z axis), and `plot_drug_comparison()` with `kind="heatmap"` (compact boolean matrix) or `kind="upset"` (pure-matplotlib UpSet-style intersection plot across drugs).
- **Drift monitoring module** (`maldiamrkit.drift`): `DriftMonitor` anchors a baseline on the earliest timestamps (defaulting to the first 20% of sorted samples) and reports three views of temporal drift: 
    - `monitor()` - cosine/Wasserstein distance of per-window median spectra to the baseline reference,
    - `monitor_pca()` - centroid trajectory and dispersion in a baseline-fitted PCA space,
    - `monitor_peak_stability()` / `monitor_effect_sizes()` - Jaccard overlap of top-k differential peaks per window vs. baseline plus per-peak Cohen's d over time.
- **Drift visualizations**: `plot_reference_drift`, `plot_pca_drift` (centroid trajectory + time-colored points and dispersion-scaled markers), `plot_peak_stability`, `plot_effect_size_drift`.

## [0.12.0] - 2026-04-08

### Added

- **Spectral similarity module** (`maldiamrkit.similarity`): pairwise distance matrices with Wasserstein, DTW, cosine, spectral contrast angle, and Pearson metrics; hierarchical, HDBSCAN, and K-medoids clustering via `cluster_spectra()` interface; silhouette scores and metadata concordance (ARI, NMI) evaluation; distance heatmap and dendrogram plotting utilities.
- **Duplicate handling module** (`maldiamrkit.data.duplicates`): `DuplicateStrategy` enum (`first`, `last`, `drop`, `keep_all`, `average`) with `apply_metadata_strategy()` and `apply_index_strategy()` helpers, replacing `deduplicate` booleans across layouts.
- **Replicate averaging in `DatasetLoader`**: when `duplicate_strategy="average"`, spectra sharing the same original ID are interpolated to a common m/z grid and averaged.
- **`deprecated()` decorator**: marks callables as deprecated with configurable `removed_in` version.

### Changed

- **Enum-based configuration across the library**: all string method/strategy parameters now accept typed enums (`BinningMethod`, `MergingMethod`, `SignalMethod`, `AlignmentMethod`, `PeakMethod`, `IntermediateHandling`) alongside plain strings.
- **CLI `build` command**: `--deduplicate` flag replaced by `--duplicate-strategy` option.
- **`BrukerTreeLayout` and `MARISMaLayout`**: `deduplicate` parameter replaced by `duplicate_strategy`.

### Fixed

- **`ShiftStrategy`**: apply `np.round()` on median shifts to avoid sub-sample drift.
- **`_build_strata()`**: preserve resistance label when merging rare strata to maintain class balance.
- **`MedianNormalizer`**: warn when median intensity is zero (empty/flat spectrum).

## [0.11.2] - 2026-04-07

### Added

- **Metadata pre-filtering in `DatasetLoader.load()`**: When `aggregate_by` specifies species or antibiotics, metadata rows are now filtered *before* matching spectrum files, avoiding unnecessary I/O on large datasets.
- **`verbose` parameter on `DatasetLoader`**: When `True` (with `n_jobs=1`), shows a tqdm progress bar during spectrum loading. Passed through to `MaldiSet`.
- **tqdm progress bar in `MaldiSet`**: When `verbose=True`, displays a progress bar during per-spectrum preprocessing and binning.

### Fixed

- **`MaldiSet.meta` / `.X` / `.y` row alignment**: `.meta` now contains only the sample IDs present in `.X` (i.e. spectra that exist on disk and pass `aggregate_by` filters). Previously `.meta` could have extra rows for samples whose spectrum files were missing.
- **`MARISMaLayout.collect_spectrum_files()` path resolution**: Metadata paths with a leading segment that duplicates `root_dir.name` no longer produce doubled paths.

### Dependencies

- Added `tqdm`.

## [0.11.1] - 2026-04-03

### Fixed

- **String intensity TypeError**: `read_spectrum()` now coerces `mass` and `intensity` columns to numeric and drops unparseable rows (this fixes embedded headers in spectra). Raises `ValueError` when a file contains no valid numeric data.
- **Species column case mismatch**: `DRIAMSLayout.discover_metadata()` now normalizes the species column to `"Species"` via case-insensitive auto-detection, fixing `KeyError: 'Species'` on DRIAMS sites that use lowercase `"species"`.

### Added

- `species_column` parameter on `DRIAMSLayout` for explicitly specifying the metadata species column name.

## [0.11.0] - 2026-04-02

### Added

- **Bruker format support**: `read_spectrum()` now reads Bruker flexAnalysis binary data directories (fid/1r + acqus) natively. Default reads the processed `1r` from `pdata/1/`; raw `fid` available via `bruker_source="fid"`.
- **MIC parsing utility**: `parse_mic_column()` in `maldiamrkit.io` for parsing MIC strings (e.g. `"<=8"`, `">16"`, `"0,5"`) into numeric values and qualifiers.
- **`DatasetBuilder` class**: Replaces `build_driams_dataset()`. Accepts an `InputLayout` adapter for flexible input format support.
- **`DatasetLoader` class**: Replaces `load_driams_dataset()`. Accepts a `DatasetLayout` adapter for navigating different dataset structures.
- **Input layout adapters**: `FlatLayout` for flat text file directories, `BrukerTreeLayout` for hierarchical Bruker binary trees with optional empty/duplicate spectrum validation.
- **Dataset layout adapters**: `DRIAMSLayout` for loading DRIAMS-like built datasets, `MARISMaLayout` for loading directly from MARISMa-style raw Bruker trees.
- **`maldiamrkit/data/` subpackage**: Centralises all data building and loading concerns.

### Changed

- `MaldiSpectrum` now supports Bruker directory paths (ID set to directory name).

### Removed

- **`build_driams_dataset()` function**: Replaced by `DatasetBuilder(FlatLayout(...), ...).build()`.
- **`load_driams_dataset()` function**: Replaced by `DatasetLoader(DRIAMSLayout(...)).load()`.
- **mzML/mzXML support**: Removed in favor of native Bruker format support.
- **`[formats]` install extra**: Removed (mzML/mzXML support dropped).
- **`[all]` install extra**: Removed (`[formats]` no longer exists).

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

- **Spectral preprocessing**: composable pipeline of transformers (smoothing, baseline correction, normalization, trimming) with multiple binning strategies (uniform, proportional, adaptive, custom) and raw-spectrum alignment (shift, linear, piecewise, DTW).
- **AMR evaluation**: VME/ME rates, sensitivity, specificity, and classification reports following EUCAST conventions; species-drug stratified and case-based splitting to prevent data leakage.
- **Composable filters**: `SpeciesFilter`, `DrugFilter`, `QualityFilter`, and `MetadataFilter` combinable with `&`, `|`, `~` operators for flexible dataset subsetting.
- **CLI**: `maldiamrkit preprocess` and `maldiamrkit quality` for batch processing and quality assessment from the command line.

See the [documentation](https://maldiamrkit.readthedocs.io/) for the complete API reference and the full set of package functionalities.
