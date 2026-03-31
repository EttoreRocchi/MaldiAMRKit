# MaldiAMRKit

[![CI](https://github.com/EttoreRocchi/MaldiAMRKit/actions/workflows/ci.yml/badge.svg)](https://github.com/EttoreRocchi/MaldiAMRKit/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/github/EttoreRocchi/MaldiAMRKit/branch/main/graph/badge.svg)](https://codecov.io/github/EttoreRocchi/MaldiAMRKit)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://maldiamrkit.readthedocs.io/)

[![PyPI Version](https://img.shields.io/pypi/v/maldiamrkit)](https://pypi.org/project/maldiamrkit/)
[![Python](https://img.shields.io/pypi/pyversions/maldiamrkit)](https://pypi.org/project/maldiamrkit/)
[![License](https://img.shields.io/github/license/EttoreRocchi/MaldiAMRKit)](https://github.com/EttoreRocchi/MaldiAMRKit/blob/main/LICENSE)

<p align="center">
  <img src="docs/maldiamrkit.png" alt="MaldiAMRKit" width="320"/>
</p>

<p align="center">
  <strong>A comprehensive toolkit for MALDI-TOF mass spectrometry data preprocessing for antimicrobial resistance (AMR) prediction purposes</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="https://maldiamrkit.readthedocs.io/">Documentation</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#citing">Citing</a> •
  <a href="#license">License</a>
</p>

## Installation

```bash
pip install maldiamrkit
```

### Optional: mzML/mzXML Format Support

```bash
pip install maldiamrkit[formats]
```

Installs [`pyteomics`](https://pyteomics.readthedocs.io/) and `lxml` for reading standard mass spectrometry data formats.

### Optional: Batch Correction & UMAP

```bash
pip install maldiamrkit[batch]
```

Installs [`combatlearn`](https://github.com/EttoreRocchi/combatlearn) for ComBat-based batch effect correction and `umap-learn` for UMAP exploratory plots.

### Optional: All Extras

```bash
pip install maldiamrkit[all]
```

### Development Installation

```bash
git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
cd MaldiAMRKit
pip install -e .[dev]
```

## Features

### Preprocessing
- **Composable Pipeline**: Build custom `PreprocessingPipeline` from individual transformers (smoothing, baseline correction, normalization, trimming), serializable to JSON/YAML
- **Multiple Binning Strategies**: Uniform, logarithmic, adaptive, and custom bin edges
- **Quality Metrics**: SNR estimation, comprehensive quality reports, and alignment assessment
- **Replicate Merging**: Mean/median/weighted merging with correlation-based outlier detection

### Alignment & Detection
- **Spectral Alignment**: Shift, linear, piecewise, and DTW warping for both binned and raw full-resolution spectra
- **Peak Detection**: Local maxima and persistent homology methods

### Evaluation
- **AMR Metrics**: VME, ME, sensitivity, specificity, categorical agreement, and `amr_classification_report` following EUCAST/CLSI conventions
- **Label Encoding**: `LabelEncoder` for mapping R/I/S to binary with configurable intermediate handling
- **Stratified Splitting**: Species-drug stratified and case-based (patient-grouped) splitting to prevent data leakage

### Data Management
- **DRIAMS Dataset Building & Loading**: Build and load DRIAMS-like dataset directories via `build_driams_dataset()` / `load_driams_dataset()`
- **Composable Filters**: `SpeciesFilter`, `DrugFilter`, `QualityFilter`, `MetadataFilter` combinable with `&`/`|`/`~` operators
- **mzML/mzXML Support**: Read standard mass spectrometry formats via optional `pyteomics` dependency
- **Spectrum Export**: Save spectra to CSV or TXT via `MaldiSet.save_spectra()`

### Visualization & Tools
- **Exploratory Plots**: PCA, t-SNE, and UMAP scatter plots colored by species, resistance phenotype, or any metadata column
- **Batch Effect Correction**: Multi-site/multi-instrument correction via [`combatlearn`](https://github.com/EttoreRocchi/combatlearn) (`pip install maldiamrkit[batch]`)
- **CLI**: `maldiamrkit preprocess`, `maldiamrkit quality`, and `maldiamrkit build-driams` for batch processing
- **Parallel Processing**: Multi-core support via `n_jobs` parameter
- **ML-Ready**: Direct integration with scikit-learn pipelines

## Documentation

Full documentation is available at [maldiamrkit.readthedocs.io](https://maldiamrkit.readthedocs.io/).

## Quick Start

### Load and Preprocess a Single Spectrum

```python
from maldiamrkit import MaldiSpectrum

# Load spectrum from file
spec = MaldiSpectrum("data/spectrum.txt")

# Preprocess: smoothing, baseline removal, normalization
spec.preprocess()

# Optional: bin to reduce dimensions
spec.bin(bin_width=3)  # 3 Da bins

# Visualize
from maldiamrkit.visualization import plot_spectrum
plot_spectrum(spec, binned=True)
```

### Build a Dataset from Multiple Spectra

```python
from maldiamrkit import MaldiSet

# Load multiple spectra with metadata
data = MaldiSet.from_directory(
    spectra_dir="data/spectra/",
    meta_file="data/metadata.csv",
    aggregate_by=dict(antibiotics="Drug", species="Escherichia coli"),
    bin_width=3
)

# Access features and labels
X = data.X  # Feature matrix
y = data.get_y_single("Drug")  # Target labels
```

### Machine Learning Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from maldiamrkit.alignment import Warping
from maldiamrkit.detection import MaldiPeakDetector

# Create ML pipeline
pipe = Pipeline([
    ("peaks", MaldiPeakDetector(binary=False, prominence=0.05)),
    ("warp", Warping(method="shift")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
```

For more examples covering alignment, filtering, evaluation, CLI usage, and more, see the
[Quickstart Guide](https://maldiamrkit.readthedocs.io/quickstart.html) and
[API Reference](https://maldiamrkit.readthedocs.io/api/index.html).

## Tutorials

For more detailed examples, see the notebooks:

- [Quick Start](notebooks/01_quick_start.ipynb) - Loading, preprocessing, binning, and quality assessment
- [Peak Detection](notebooks/02_peak_detection.ipynb) - Local maxima and persistent homology methods
- [Alignment](notebooks/03_alignment.ipynb) - Warping methods and alignment quality
- [Evaluation](notebooks/04_evaluation.ipynb) - AMR metrics, label encoding, and stratified splitting
- [Exploration](notebooks/05_exploration.ipynb) - PCA, t-SNE, UMAP visualizations and batch correction

## Contributing

Pull requests, bug reports, and feature ideas are welcome. See the [Contributing Guide](CONTRIBUTING.md) for how to get started.

## Citing

If you use MaldiAMRKit in your research, please cite:

> Rocchi, E., Nicitra, E., Calvo, M. et al. *Combining mass spectrometry and machine learning models for predicting Klebsiella pneumoniae antimicrobial resistance: a multicenter experience from clinical isolates in Italy*. **BMC Microbiol** (2026). [doi:10.1186/s12866-025-04657-2](https://link.springer.com/article/10.1186/s12866-025-04657-2)

See the [full publications list](https://maldiamrkit.readthedocs.io/papers.html) for more papers using MaldiAMRKit.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This toolkit is inspired by:

> **Weis, C., Cuénod, A., Rieck, B., et al.** (2022). *Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning*. **Nature Medicine**, 28, 164-174. [https://doi.org/10.1038/s41591-021-01619-9](https://doi.org/10.1038/s41591-021-01619-9)
