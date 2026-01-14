"""
MaldiAMRKit: MALDI-TOF mass spectrometry preprocessing toolkit for AMR prediction.

A Python toolkit for preprocessing MALDI-TOF mass spectrometry data with
scikit-learn compatible transformers for antimicrobial resistance (AMR) prediction.

Examples
--------
>>> from maldiamrkit import MaldiSpectrum, MaldiSet, Warping
>>> spec = MaldiSpectrum("spectrum.txt")
>>> spec.preprocess().bin(3)
>>> spec.plot()
"""

# Core data structures
from maldiamrkit.core.config import PreprocessingSettings
from maldiamrkit.core.spectrum import MaldiSpectrum
from maldiamrkit.core.dataset import MaldiSet

# Preprocessing functions
from maldiamrkit.preprocessing.pipeline import preprocess
from maldiamrkit.preprocessing.binning import bin_spectrum
from maldiamrkit.preprocessing.quality import (
    estimate_snr,
    SpectrumQuality,
    SpectrumQualityReport,
)

# I/O utilities
from maldiamrkit.io.readers import read_spectrum

# Alignment transformers
from maldiamrkit.alignment.warping import Warping
from maldiamrkit.alignment.raw_warping import RawWarping, create_raw_input

# Detection transformers
from maldiamrkit.detection.peak_detector import MaldiPeakDetector

__version__ = "0.6.0"
__author__ = "Ettore Rocchi"

__all__ = [
    # Core
    "MaldiSpectrum",
    "MaldiSet",
    "PreprocessingSettings",
    # Preprocessing
    "preprocess",
    "bin_spectrum",
    "estimate_snr",
    "SpectrumQuality",
    "SpectrumQualityReport",
    # I/O
    "read_spectrum",
    # Alignment
    "Warping",
    "RawWarping",
    "create_raw_input",
    # Detection
    "MaldiPeakDetector",
    # Metadata
    "__version__",
    "__author__",
]
