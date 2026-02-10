"""Evaluation utilities for AMR prediction."""

from .label_encoder import LabelEncoder
from .metrics import (
    amr_classification_report,
    categorical_agreement,
    major_error_rate,
    me_scorer,
    sensitivity_score,
    specificity_score,
    very_major_error_rate,
    vme_me_curve,
    vme_scorer,
)
from .splitting import (
    CaseGroupedKFold,
    SpeciesDrugStratifiedKFold,
    case_based_split,
    stratified_species_drug_split,
)

__all__ = [
    # Metrics
    "very_major_error_rate",
    "major_error_rate",
    "sensitivity_score",
    "specificity_score",
    "categorical_agreement",
    "vme_me_curve",
    "amr_classification_report",
    "vme_scorer",
    "me_scorer",
    # Splitting
    "stratified_species_drug_split",
    "case_based_split",
    "SpeciesDrugStratifiedKFold",
    "CaseGroupedKFold",
    # Label encoding
    "LabelEncoder",
]
