# MaldiAMRKit

<p align="center">
  <img src="docs/maldiamrkit.png" alt="MaldiAMRKit" width="250"/>
</p>
<p align="center">
  <strong>Toolkit to read and preprocess MALDI-TOF mass-spectra for AMR analyses</strong>
</p>

## 🚀 Installation

Clone  the repository:

```bash
git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
cd MaldiAMRKit
```

Install the package and dependencies:

```bash
pip install .
```

## 🏃 Quick Start

```python
from maldiamrkit.spectrum import MaldiSpectrum
from maldiamrkit.dataset import MaldiSet
from maldiamrkit.peak_detector import MaldiPeakDetector

# Load and preprocess a single spectrum
spec = MaldiSpectrum("data/1s.txt").preprocess() # smoothing, baseline removal, normalisation
spec.bin(3) # [optional] bin width 3 Da
spec.plot(binned=True) # plot

# Build a dataset from a directory of spectra + metadata CSV
data = MaldiSet.from_directory(
  "data/", "data/metadata/metadata.csv",
  aggregate_by=dict(antibiotic="Drug"),
  bin_width=3
)
X, y = data.X, data.y

# Machine learning pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("peaks", MaldiPeakDetector(binary=False, prominence=0.05)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500))
])
pipe.fit(X, y)
```
For further details please see the [quick guide](docs/quick_guide.ipynb).

## 🤝 Contributing

Pull requests, bug reports, and feature ideas are welcome: feel free to open a PR!

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
