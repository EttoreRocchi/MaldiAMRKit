"""Preprocessing functions for MALDI-TOF spectra."""

from .binning import (
    BinningMethod,
    bin_spectrum,
    get_bin_metadata,
    register_binning_method,
    unregister_binning_method,
)
from .merging import MergingMethod, detect_outlier_replicates, merge_replicates
from .pipeline import preprocess
from .preprocessing_pipeline import PreprocessingPipeline
from .quality import SignalMethod, SpectrumQuality, SpectrumQualityReport, estimate_snr
from .transformers import (
    ClipNegatives,
    ConvexHullBaseline,
    LogTransform,
    MedianBaseline,
    MedianNormalizer,
    MovingAverageSmooth,
    MzMultiTrimmer,
    MzTrimmer,
    PQNNormalizer,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    SqrtTransform,
    TICNormalizer,
    TopHatBaseline,
)

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    "preprocess",
    # Binning
    "BinningMethod",
    "bin_spectrum",
    "get_bin_metadata",
    "register_binning_method",
    "unregister_binning_method",
    # Merging
    "MergingMethod",
    "merge_replicates",
    "detect_outlier_replicates",
    # Quality
    "SignalMethod",
    "estimate_snr",
    "SpectrumQuality",
    "SpectrumQualityReport",
    # Transformers
    "ClipNegatives",
    "SqrtTransform",
    "LogTransform",
    "SavitzkyGolaySmooth",
    "MovingAverageSmooth",
    "SNIPBaseline",
    "TopHatBaseline",
    "ConvexHullBaseline",
    "MedianBaseline",
    "MzTrimmer",
    "TICNormalizer",
    "MedianNormalizer",
    "PQNNormalizer",
    "MzMultiTrimmer",
]
