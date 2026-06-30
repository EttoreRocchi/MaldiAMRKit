"""Evaluation utilities for AMR prediction.

Binary AMR metrics, MIC-regression reporting, and CV splitting helpers.
"""

from __future__ import annotations

from .metrics import (
    amr_classification_report,
    amr_multilabel_report,
    categorical_agreement,
    major_error_rate,
    me_scorer,
    sensitivity_score,
    specificity_score,
    very_major_error_rate,
    vme_me_curve,
    vme_scorer,
)
from .mic_regression import mic_regression_report
from .splitting import (
    CaseGroupedKFold,
    SpeciesDrugStratifiedKFold,
    case_based_split,
    stratified_species_drug_split,
)

__all__ = [
    "very_major_error_rate",
    "major_error_rate",
    "sensitivity_score",
    "specificity_score",
    "categorical_agreement",
    "vme_me_curve",
    "amr_classification_report",
    "amr_multilabel_report",
    "mic_regression_report",
    "vme_scorer",
    "me_scorer",
    "stratified_species_drug_split",
    "case_based_split",
    "SpeciesDrugStratifiedKFold",
    "CaseGroupedKFold",
]
