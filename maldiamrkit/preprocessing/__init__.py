"""Preprocessing functions for MALDI-TOF spectra."""

from .binning import bin_spectrum, get_bin_metadata
from .merging import detect_outlier_replicates, merge_replicates
from .pipeline import preprocess
from .preprocessing_pipeline import PreprocessingPipeline
from .quality import SpectrumQuality, SpectrumQualityReport, estimate_snr
from .transformers import (
    ClipNegatives,
    LogTransform,
    MedianNormalizer,
    MzMultiTrimmer,
    MzTrimmer,
    PQNNormalizer,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    SqrtTransform,
    TICNormalizer,
)

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    "preprocess",
    # Binning
    "bin_spectrum",
    "get_bin_metadata",
    # Merging
    "merge_replicates",
    "detect_outlier_replicates",
    # Quality
    "estimate_snr",
    "SpectrumQuality",
    "SpectrumQualityReport",
    # Transformers
    "ClipNegatives",
    "SqrtTransform",
    "LogTransform",
    "SavitzkyGolaySmooth",
    "SNIPBaseline",
    "MzTrimmer",
    "TICNormalizer",
    "MedianNormalizer",
    "PQNNormalizer",
    "MzMultiTrimmer",
]
