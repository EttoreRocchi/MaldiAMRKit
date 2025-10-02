# MaldiAMRKit

[![PyPI Version](https://img.shields.io/pypi/v/maldiamrkit?cacheSeconds=300)](https://pypi.org/project/maldiamrkit/)
[![PyPI Downloads](https://static.pepy.tech/badge/maldiamrkit)](https://pepy.tech/projects/maldiamrkit)
[![License](https://img.shields.io/github/license/EttoreRocchi/MaldiAMRKit)](https://github.com/EttoreRocchi/MaldiAMRKit/blob/main/LICENSE)

<p align="center">
  <img src="docs/maldiamrkit.png" alt="MaldiAMRKit" width="250"/>
</p>

<p align="center">
  <strong>A comprehensive toolkit for MALDI-TOF mass spectrometry data preprocessing and antimicrobial resistance (AMR) prediction</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#license">License</a> •
  <a href="#contributing">Contributing</a>
</p>

## Installation

```bash
pip install maldiamrkit
```

## Features

- **📊 Spectrum Processing**: Load, smooth, baseline correct, and normalize MALDI-TOF spectra
- **📦 Dataset Management**: Process multiple spectra with metadata integration
- **🔍 Peak Detection**: Automated peak finding with customizable parameters
- **📈 Spectral Alignment (Warping)**: Multiple alignment methods (shift, linear, piecewise, DTW)
- **🤖 ML-Ready**: Direct integration with scikit-learn pipelines

## Quick Start

### Load and Preprocess a Single Spectrum

```python
from maldiamrkit.spectrum import MaldiSpectrum

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
from maldiamrkit.dataset import MaldiSet

# Load multiple spectra with metadata
data = MaldiSet.from_directory(
    spectra_dir="data/spectra/",
    metadata_path="data/metadata.csv",
    aggregate_by={"antibiotic": "Drug", "species": "Species"},
    bin_width=3
)

# Access features and labels
X = data.X  # Feature matrix
y = data.y["Drug"]  # Target labels
```

### Machine Learning Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from maldiamrkit.peak_detector import MaldiPeakDetector

# Create ML pipeline
pipe = Pipeline([
    ("peaks", MaldiPeakDetector(binary=False, prominence=0.05)),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### Align spectra to correct for mass calibration drift:

```python
from maldiamrkit.warping import Warping

# Create warping transformer with shift method
warper = Warping(
    method='shift',  # or 'linear', 'piecewise', 'dtw'
    reference='median',  # use median spectrum as reference
    max_shift=50
)

# Fit on training data and transform
warper.fit(X_train)
X_aligned = warper.transform(X_test)

# Visualize alignment results
fig, axes = warper.plot_alignment(
    X_original=X_test,
    X_aligned=X_aligned,
    indices=[0, 5, 10],  # plot multiple spectra
    xlim=(2000, 10000),  # zoom to m/z range
    show_peaks=True
)
```

**Alignment Methods:**
- `shift`: Global median shift (fast, simple)
- `linear`: Least-squares linear transformation
- `piecewise`: Local shifts across spectrum segments (most flexible)
- `dtw`: Dynamic Time Warping (best for non-linear drift)


For further details please see the [quick guide notebook](docs/quick_guide.ipynb).

## Contributing

Pull requests, bug reports, and feature ideas are welcome: feel free to open a PR!

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This toolkit is inspired by and builds upon the methodology described in:

> **Weis, C., Cuénod, A., Rieck, B., et al.** (2022). *Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning*. **Nature Medicine**, 28, 164–174. [https://doi.org/10.1038/s41591-021-01619-9](https://doi.org/10.1038/s41591-021-01619-9)

Please consider citing this work if you find `MaldiAMRKit` useful.