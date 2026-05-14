"""Evaluation utilities for AMR prediction.

The label encoder previously lived here; in v0.15 it moved to
:mod:`maldiamrkit.susceptibility` along with the new MIC encoder and
breakpoint table. Access via ``from maldiamrkit.evaluation import LabelEncoder``
still works for one release with a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings

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

_LAZY_DEPRECATED = {
    "LabelEncoder": "maldiamrkit.susceptibility.LabelEncoder",
    "IntermediateHandling": "maldiamrkit.susceptibility.IntermediateHandling",
}


def __getattr__(name: str):
    if name in _LAZY_DEPRECATED:
        new_path = _LAZY_DEPRECATED[name]
        warnings.warn(
            f"{name} has moved to {new_path}; "
            f"importing from maldiamrkit.evaluation is deprecated and "
            f"will be removed in v0.17.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..susceptibility.label_encoder import (
            IntermediateHandling,
            LabelEncoder,
        )

        return {
            "LabelEncoder": LabelEncoder,
            "IntermediateHandling": IntermediateHandling,
        }[name]
    raise AttributeError(f"module 'maldiamrkit.evaluation' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(__all__) + list(_LAZY_DEPRECATED.keys()))


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
