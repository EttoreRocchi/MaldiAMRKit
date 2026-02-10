"""AMR-specific evaluation metrics for clinical microbiology.

Provides Very Major Error (VME), Major Error (ME), sensitivity, specificity,
and categorical agreement metrics following EUCAST/CLSI conventions.

In AMR prediction:
- VME (Very Major Error): resistant isolates classified as susceptible (dangerous)
- ME (Major Error): susceptible isolates classified as resistant (wasteful)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer


def _get_confusion_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resistant_label: int = 1,
) -> tuple[int, int, int, int]:
    """Extract TP, TN, FP, FN from predictions.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    resistant_label : int, default=1
        Label value representing the resistant class.

    Returns
    -------
    tp, tn, fp, fn : int
        Confusion matrix values where positive = resistant.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true) | set(y_pred))

    if len(labels) < 2:
        # Single-class edge case
        if labels[0] == resistant_label:
            tp = np.sum(y_pred == resistant_label)
            fn = np.sum(y_pred != resistant_label)
            return tp, 0, 0, fn
        else:
            tn = np.sum(y_pred != resistant_label)
            fp = np.sum(y_pred == resistant_label)
            return 0, tn, fp, 0

    if resistant_label not in labels:
        # resistant_label never appears - treat as all-susceptible
        tn = len(y_true)
        return 0, tn, 0, 0

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Find index of resistant label
    r_idx = labels.index(resistant_label)
    s_idx = 1 - r_idx if len(labels) == 2 else 0

    tp = cm[r_idx, r_idx]
    fn = cm[r_idx, s_idx]
    fp = cm[s_idx, r_idx]
    tn = cm[s_idx, s_idx]

    return tp, tn, fp, fn


def very_major_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resistant_label: int = 1,
) -> float:
    """Very Major Error rate: resistant isolates classified as susceptible.

    VME = FN / (FN + TP), i.e., the miss rate for resistant samples.
    This is the most dangerous error type in clinical microbiology.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    resistant_label : int, default=1
        Label value representing the resistant class.

    Returns
    -------
    float
        VME rate in [0, 1]. Returns 0.0 if no resistant samples exist.

    Examples
    --------
    >>> very_major_error_rate([1, 1, 0, 0], [0, 1, 0, 0])
    0.5
    """
    tp, _, _, fn = _get_confusion_values(y_true, y_pred, resistant_label)
    denom = fn + tp
    return fn / denom if denom > 0 else 0.0


def major_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resistant_label: int = 1,
) -> float:
    """Major Error rate: susceptible isolates classified as resistant.

    ME = FP / (FP + TN), i.e., the false alarm rate for susceptible samples.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    resistant_label : int, default=1
        Label value representing the resistant class.

    Returns
    -------
    float
        ME rate in [0, 1]. Returns 0.0 if no susceptible samples exist.

    Examples
    --------
    >>> major_error_rate([1, 1, 0, 0], [1, 1, 1, 0])
    0.5
    """
    _, tn, fp, _ = _get_confusion_values(y_true, y_pred, resistant_label)
    denom = fp + tn
    return fp / denom if denom > 0 else 0.0


def sensitivity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resistant_label: int = 1,
) -> float:
    """Sensitivity (recall) for the resistant class.

    Sensitivity = TP / (TP + FN) = 1 - VME.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    resistant_label : int, default=1
        Label value representing the resistant class.

    Returns
    -------
    float
        Sensitivity in [0, 1]. Returns 0.0 if no resistant samples exist.
    """
    tp, _, _, fn = _get_confusion_values(y_true, y_pred, resistant_label)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def specificity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resistant_label: int = 1,
) -> float:
    """Specificity (true negative rate) for the susceptible class.

    Specificity = TN / (TN + FP) = 1 - ME.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    resistant_label : int, default=1
        Label value representing the resistant class.

    Returns
    -------
    float
        Specificity in [0, 1]. Returns 0.0 if no susceptible samples exist.
    """
    _, tn, fp, _ = _get_confusion_values(y_true, y_pred, resistant_label)
    denom = tn + fp
    return tn / denom if denom > 0 else 0.0


def categorical_agreement(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Categorical agreement (accuracy) as reported in AST studies.

    CA = (TP + TN) / N.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Agreement rate in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return np.mean(y_true == y_pred)


def vme_me_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    resistant_label: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VME and ME rates at varying decision thresholds.

    Useful for selecting an optimal threshold balancing VME against ME.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted scores (e.g., probabilities for the resistant class).
    resistant_label : int, default=1
        Label value representing the resistant class.

    Returns
    -------
    vme_rates : np.ndarray
        VME rates at each threshold.
    me_rates : np.ndarray
        ME rates at each threshold.
    thresholds : np.ndarray
        Decision thresholds (sorted ascending).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = np.sort(np.unique(y_score))

    vme_rates = np.empty(len(thresholds))
    me_rates = np.empty(len(thresholds))

    for i, t in enumerate(thresholds):
        y_pred = (y_score >= t).astype(int)
        # Map back to original labels if needed
        if resistant_label != 1:
            labels = sorted(set(y_true))
            susceptible_label = [lab for lab in labels if lab != resistant_label][0]
            y_pred = np.where(y_pred == 1, resistant_label, susceptible_label)

        vme_rates[i] = very_major_error_rate(y_true, y_pred, resistant_label)
        me_rates[i] = major_error_rate(y_true, y_pred, resistant_label)

    return vme_rates, me_rates, thresholds


def amr_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resistant_label: int = 1,
) -> dict:
    """Full AMR classification report.

    Returns all clinical metrics in a single dictionary.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    resistant_label : int, default=1
        Label value representing the resistant class.

    Returns
    -------
    dict
        Dictionary with keys: vme, me, sensitivity, specificity, categorical_agreement,
        n_resistant, n_susceptible, n_total.

    Examples
    --------
    >>> report = amr_classification_report([1, 1, 0, 0], [1, 0, 0, 1])
    >>> report["vme"]
    0.5
    """
    y_true = np.asarray(y_true)
    tp, tn, fp, fn = _get_confusion_values(y_true, y_pred, resistant_label)

    return {
        "vme": very_major_error_rate(y_true, y_pred, resistant_label),
        "me": major_error_rate(y_true, y_pred, resistant_label),
        "sensitivity": sensitivity_score(y_true, y_pred, resistant_label),
        "specificity": specificity_score(y_true, y_pred, resistant_label),
        "categorical_agreement": categorical_agreement(y_true, y_pred),
        "n_resistant": int(tp + fn),
        "n_susceptible": int(tn + fp),
        "n_total": len(y_true),
    }


# Pre-built sklearn scorers for cross_val_score / GridSearchCV
vme_scorer = make_scorer(very_major_error_rate, greater_is_better=False)
"""Scorer that minimizes VME. Use with ``cross_val_score`` or ``GridSearchCV``."""

me_scorer = make_scorer(major_error_rate, greater_is_better=False)
"""Scorer that minimizes ME. Use with ``cross_val_score`` or ``GridSearchCV``."""
