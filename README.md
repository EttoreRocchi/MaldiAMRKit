# MaldiAMRKit

[![PyPI Version](https://img.shields.io/pypi/v/maldiamrkit?cacheSeconds=300)](https://pypi.org/project/maldiamrkit/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://maldiamrkit.readthedocs.io/)
[![License](https://img.shields.io/github/license/EttoreRocchi/MaldiAMRKit)](https://github.com/EttoreRocchi/MaldiAMRKit/blob/main/LICENSE)

<p align="center">
  <img src="docs/maldiamrkit.png" alt="MaldiAMRKit" width="450"/>
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
pip install -e .
```

### Install with Documentation Dependencies

```bash
pip install -e ".[docs]"
```

## Features

- **Spectrum Processing**: Load, smooth, baseline correct, and normalize MALDI-TOF spectra
- **Dataset Management**: Process multiple spectra with metadata integration
- **Peak Detection**: Local maxima and persistent homology methods
- **Spectral Alignment (Warping)**: Multiple alignment methods (shift, linear, piecewise, DTW)
- **Raw Spectra Warping**: Full m/z resolution alignment before binning
- **Quality Metrics**: SNR estimation, comprehensive quality reports, and alignment assessment
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
    aggregate_by=dict(antibiotics="Drug", species="Species"),
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
from maldiamrkit import MaldiPeakDetector, Warping

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
from maldiamrkit import Warping

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
from maldiamrkit import RawWarping, create_raw_input

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
from maldiamrkit import estimate_snr, SpectrumQuality, MaldiSpectrum

# Estimate signal-to-noise ratio
spec = MaldiSpectrum("spectrum.txt").preprocess()
snr = estimate_snr(spec.preprocessed)
print(f"SNR: {snr:.1f}")

# Comprehensive quality report
qc = SpectrumQuality()  # Uses high m/z region (19500-20000) by default
report = qc.assess(spec.preprocessed)
print(f"SNR: {report.snr:.1f}")
print(f"Peak count: {report.peak_count}")
print(f"Dynamic range: {report.dynamic_range:.2f}")
```

### Parallel Processing

Use `n_jobs` parameter for multi-core processing:

```python
from maldiamrkit import MaldiSet, MaldiPeakDetector, Warping

# Parallel dataset loading
data = MaldiSet.from_directory("spectra/", "meta.csv", n_jobs=-1)

# Parallel peak detection
detector = MaldiPeakDetector(prominence=0.01, n_jobs=-1)
peaks = detector.fit_transform(X)

# Parallel alignment
warper = Warping(method="piecewise", n_jobs=-1)
X_aligned = warper.fit_transform(X)
```

## Project Structure

```
maldiamrkit/
├── core/           # Core data structures (MaldiSpectrum, MaldiSet)
├── preprocessing/  # Preprocessing functions (pipeline, binning, quality)
├── alignment/      # Warping transformers (Warping, RawWarping)
├── detection/      # Peak detection (MaldiPeakDetector)
├── io/             # File I/O utilities
└── utils/          # Validation and plotting helpers
```

## Tutorials

For more detailed examples, see the notebooks:

- [Quick Start](notebooks/01_quick_start.ipynb) - Loading, preprocessing, binning, and quality assessment
- [Peak Detection](notebooks/02_peak_detection.ipynb) - Local maxima and persistent homology methods
- [Alignment](notebooks/03_alignment.ipynb) - Warping methods and alignment quality

## Contributing

Pull requests, bug reports, and feature ideas are welcome: feel free to open a PR!

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This toolkit is inspired by and builds upon the methodology described in:

> **Weis, C., Cuénod, A., Rieck, B., et al.** (2022). *Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning*. **Nature Medicine**, 28, 164–174. [https://doi.org/10.1038/s41591-021-01619-9](https://doi.org/10.1038/s41591-021-01619-9)

Please consider citing this work if you find `MaldiAMRKit` useful.
