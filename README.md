# MaldiAMRKit

[![CI](https://github.com/EttoreRocchi/MaldiAMRKit/actions/workflows/ci.yml/badge.svg)](https://github.com/EttoreRocchi/MaldiAMRKit/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/github/EttoreRocchi/MaldiAMRKit/branch/main/graph/badge.svg)](https://codecov.io/github/EttoreRocchi/MaldiAMRKit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
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
  <a href="https://maldiamrkit.readthedocs.io/">Documentation</a> •
  <a href="#license">License</a> 
</p>

## Installation

```bash
pip install maldiamrkit
```

### Development Installation

```bash
git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
cd MaldiAMRKit
pip install -e .[dev]
```

## Features

- **Spectrum Processing**: Load, smooth, baseline correct, and normalize MALDI-TOF spectra
- **Dataset Management**: Process multiple spectra with metadata integration
- **Peak Detection**: Local maxima and persistent homology methods
- **Spectral Alignment (Warping)**: Multiple alignment methods (shift, linear, piecewise, DTW)
- **Raw Spectra Warping**: Full m/z resolution alignment before binning
- **Quality Metrics**: SNR estimation, comprehensive quality reports, and alignment assessment
- **Replicate Merging**: Mean/median/weighted merging of spectral replicates with correlation-based outlier detection
- **Composable Preprocessing Pipeline**: Build custom `PreprocessingPipeline` from individual transformers, serializable to JSON/YAML
- **Composable Filter System**: `SpeciesFilter`, `DrugFilter`, `QualityFilter`, `MetadataFilter` with `&`/`|`/`~` operators for flexible dataset filtering
- **Evaluation Metrics**: VME, ME, sensitivity, specificity, categorical agreement, and `amr_classification_report`
- **Stratified Splitting**: Species-drug stratified and case-based (patient-grouped) splitting to prevent data leakage
- **Label Encoding**: `LabelEncoder` for mapping R/I/S to binary with configurable intermediate handling
- **Spectrum Export**: Save individual spectra (raw, preprocessed, or binned) to CSV or TXT via `MaldiSet.save_spectra()`
- **CLI**: `maldiamrkit preprocess` and `maldiamrkit quality` commands for batch processing
- **Parallel Processing**: Multi-core support via `n_jobs` parameter for faster processing
- **ML-Ready**: Direct integration with scikit-learn pipelines

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
spec.plot(binned=True)
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

### Binning Methods

MaldiAMRKit supports multiple binning strategies:

```python
from maldiamrkit import MaldiSpectrum

spec = MaldiSpectrum("data/spectrum.txt").preprocess()

# Uniform binning (default)
spec.bin(bin_width=3)

# Logarithmic binning (width scales with m/z)
spec.bin(bin_width=3, method="logarithmic")

# Adaptive binning (smaller bins in peak-dense regions)
spec.bin(method="adaptive", adaptive_min_width=1.0, adaptive_max_width=10.0)

# Custom binning (user-defined edges)
spec.bin(method="custom", custom_edges=[2000, 5000, 10000, 15000, 20000])

# Access bin metadata
print(spec.bin_metadata.head())
#    bin_index  bin_start  bin_end  bin_width
# 0          0     2000.0   2003.0        3.0
# 1          1     2003.0   2006.0        3.0
```

**Binning Methods:**
- `uniform`: Fixed width bins (default)
- `logarithmic`: Bin width scales with m/z (matches instrument resolution)
- `adaptive`: Smaller bins where peaks are dense, larger bins elsewhere
- `custom`: User-defined bin edges for domain-specific analysis

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

# Cross-validation (recommended over train accuracy)
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### Spectral Alignment

Align spectra to correct for mass calibration drift:

```python
from maldiamrkit.alignment import Warping

# Create warping transformer
warper = Warping(
    method='piecewise',  # or 'shift', 'linear', 'dtw'
    reference='median',
    n_segments=5
)

# Fit on training data and transform
warper.fit(X_train)
X_aligned = warper.transform(X_test)

# Check alignment quality
quality = warper.get_alignment_quality(X_test, X_aligned)
print(f"Mean improvement: {quality['improvement'].mean():.4f}")

# Visualize
warper.plot_alignment(X_test, X_aligned, indices=[0], show_peaks=True)
```

### Raw Spectra Warping

For higher precision, use RawWarping which operates at full m/z resolution:

```python
from maldiamrkit.alignment import RawWarping, create_raw_input

# Create input DataFrame from spectrum files
X_raw = create_raw_input("data/spectra/")

# Raw warping loads original files for warping
warper = RawWarping(
    method="piecewise",
    bin_width=3,
    max_shift_da=10.0,
    n_jobs=-1  # Parallel processing
)

# Outputs binned data for pipeline compatibility
warper.fit(X_raw)
X_aligned = warper.transform(X_raw)
```

**Alignment Methods:**
- `shift`: Global median shift (fast, simple)
- `linear`: Least-squares linear transformation
- `piecewise`: Local shifts across spectrum segments (most flexible)
- `dtw`: Dynamic Time Warping (best for non-linear drift)

### Quality Assessment

```python
from maldiamrkit import MaldiSpectrum
from maldiamrkit.preprocessing import estimate_snr, SpectrumQuality

# Estimate signal-to-noise ratio
spec = MaldiSpectrum("spectrum.txt").preprocess()
snr = estimate_snr(spec)
print(f"SNR: {snr:.1f}")

# Comprehensive quality report
qc = SpectrumQuality()  # Uses high m/z region (19500-20000) by default
report = qc.assess(spec)
print(f"SNR: {report.snr:.1f}")
print(f"Peak count: {report.peak_count}")
print(f"Dynamic range: {report.dynamic_range:.2f}")
```

### Replicate Merging

Merge multiple spectral replicates per isolate into a single consensus spectrum:

```python
from maldiamrkit import MaldiSpectrum
from maldiamrkit.preprocessing import merge_replicates, detect_outlier_replicates

# Load replicates as MaldiSpectrum objects
spectra = [MaldiSpectrum(f"data/isolate_rep{i}.txt") for i in range(1, 4)]

# Detect and remove outlier replicates
keep = detect_outlier_replicates(spectra)
clean = [s for s, k in zip(spectra, keep) if k]

# Merge into a single consensus spectrum
merged = merge_replicates(clean, method="mean")
```

### Composable Preprocessing Pipeline

Build a composable, serializable preprocessing pipeline:

```python
from maldiamrkit.preprocessing import (
    PreprocessingPipeline,
    ClipNegatives, SqrtTransform, SavitzkyGolaySmooth,
    SNIPBaseline, MzTrimmer, TICNormalizer,
)

# Use the default pipeline
pipe = PreprocessingPipeline.default()

# Or build a custom pipeline
pipe = PreprocessingPipeline([
    ("clip", ClipNegatives()),
    ("sqrt", SqrtTransform()),
    ("smooth", SavitzkyGolaySmooth(window_length=15, polyorder=2)),
    ("baseline", SNIPBaseline(half_window=30)),
    ("trim", MzTrimmer(mz_min=2000, mz_max=20000)),
    ("norm", TICNormalizer()),
])

# Serialize to JSON/YAML for reproducibility
pipe.to_json("my_pipeline.json")
pipe = PreprocessingPipeline.from_json("my_pipeline.json")

# Apply to a spectrum
spec = MaldiSpectrum("data/spectrum.txt", pipeline=pipe)
spec.preprocess().bin(3)
```

### Dataset Filtering

Use composable filters to select subsets of a `MaldiSet`:

```python
from maldiamrkit import MaldiSet
from maldiamrkit.filters import SpeciesFilter, DrugFilter, QualityFilter, MetadataFilter

data = MaldiSet.from_directory("spectra/", "metadata.csv",
    aggregate_by=dict(antibiotics="Drug"))

# Filter by species
ecoli = data.filter(SpeciesFilter("Escherichia coli"))

# Combine filters with & (and), | (or), ~ (not)
f = SpeciesFilter("Escherichia coli") & QualityFilter(min_snr=5.0)
high_quality_ecoli = data.filter(f)

# Filter by antibiotic resistance status
f = SpeciesFilter("Escherichia coli") & DrugFilter("Ceftriaxone", status="R")
resistant_ecoli = data.filter(f)

# Custom metadata filter
f = MetadataFilter("batch_id", lambda v: v == "batch_1")
batch1 = data.filter(f)
```

### Evaluation Metrics

AMR-specific evaluation following EUCAST/CLSI conventions:

```python
from maldiamrkit.evaluation import (
    very_major_error_rate, major_error_rate,
    amr_classification_report, vme_scorer, me_scorer,
    LabelEncoder,
)

# Encode R/I/S labels to binary
enc = LabelEncoder(intermediate="susceptible")
y_binary = enc.fit_transform(y_raw)

# Compute individual metrics
vme = very_major_error_rate(y_true, y_pred)
me = major_error_rate(y_true, y_pred)

# Full classification report
report = amr_classification_report(y_true, y_pred)
# {'vme': 0.1, 'me': 0.05, 'sensitivity': 0.9, 'specificity': 0.95, ...}

# Use as sklearn scorers in cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5, scoring=vme_scorer)
```

### Stratified Splitting

Prevent data leakage with species-aware and patient-grouped splits:

```python
from maldiamrkit.evaluation import (
    stratified_species_drug_split,
    case_based_split,
    SpeciesDrugStratifiedKFold,
    CaseGroupedKFold,
)

# Single split stratified by species + drug label
X_train, X_test, y_train, y_test = stratified_species_drug_split(
    X, y, species=species_labels, test_size=0.2, random_state=42
)

# Patient-grouped split (no patient in both train and test)
X_train, X_test, y_train, y_test = case_based_split(
    X, y, case_ids=patient_ids, test_size=0.2
)

# Cross-validation splitters (sklearn-compatible)
cv = SpeciesDrugStratifiedKFold(n_splits=5)
for train_idx, test_idx in cv.split(X, y, species=species_labels):
    ...

cv = CaseGroupedKFold(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
    ...
```

### Command-Line Interface

Batch preprocess spectra or generate quality reports from the terminal:

```bash
# Preprocess and bin to a CSV feature matrix
maldiamrkit preprocess --input-dir data/ --output processed.csv --bin-width 3

# Also save individual preprocessed spectra as TXT files
maldiamrkit preprocess --input-dir data/ --output processed.csv --save-spectra-dir processed/

# Use a custom pipeline config
maldiamrkit preprocess --input-dir data/ --output processed.csv --pipeline config.yaml

# Generate quality report
maldiamrkit quality --input-dir data/ --output report.csv
```

### Parallel Processing

Use `n_jobs` parameter for multi-core processing:

```python
from maldiamrkit import MaldiSet
from maldiamrkit.alignment import Warping
from maldiamrkit.detection import MaldiPeakDetector

# Parallel dataset loading
data = MaldiSet.from_directory("spectra/", "meta.csv", n_jobs=-1)

# Parallel peak detection
detector = MaldiPeakDetector(prominence=0.01, n_jobs=-1)
peaks = detector.fit_transform(X)

# Parallel alignment
warper = Warping(method="piecewise", n_jobs=-1)
X_aligned = warper.fit_transform(X)
```

## Tutorials

For more detailed examples, see the notebooks:

- [Quick Start](notebooks/01_quick_start.ipynb) - Loading, preprocessing, binning, and quality assessment
- [Peak Detection](notebooks/02_peak_detection.ipynb) - Local maxima and persistent homology methods
- [Alignment](notebooks/03_alignment.ipynb) - Warping methods and alignment quality
- [Evaluation](notebooks/04_evaluation.ipynb) - AMR metrics, label encoding, and stratified splitting

## Contributing

Pull requests, bug reports, and feature ideas are welcome: feel free to open a PR!

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Papers

Publications using `MaldiAMRKit`:

> Rocchi, E., Nicitra, E., Calvo, M. et al. *Combining mass spectrometry and machine learning models for predicting Klebsiella pneumoniae antimicrobial resistance: a multicenter experience from clinical isolates in Italy*. **BMC Microbiol** (2026). https://doi.org/10.1186/s12866-025-04657-2

## Acknowledgements

This toolkit is inspired by:

> **Weis, C., Cuénod, A., Rieck, B., et al.** (2022). *Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning*. **Nature Medicine**, 28, 164–174. [https://doi.org/10.1038/s41591-021-01619-9](https://doi.org/10.1038/s41591-021-01619-9)

Please consider citing this work if you find `MaldiAMRKit` useful.
