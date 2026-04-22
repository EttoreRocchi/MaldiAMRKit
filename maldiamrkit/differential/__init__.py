"""Differential analysis: per-bin statistical tests with AMR-aware plots."""

from .analysis import CorrectionMethod, DifferentialAnalysis, StatisticalTest
from .plots import (
    DrugComparisonKind,
    plot_drug_comparison,
    plot_manhattan,
    plot_volcano,
)

__all__ = [
    "CorrectionMethod",
    "DifferentialAnalysis",
    "DrugComparisonKind",
    "StatisticalTest",
    "plot_drug_comparison",
    "plot_manhattan",
    "plot_volcano",
]
