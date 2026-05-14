"""Regression-style metrics for MIC prediction.

Reports the metrics clinicians and ML practitioners actually look at when
predicting MIC values on dilution series: RMSE in log2 dilutions, essential
agreement (within ±1 dilution), and -- if breakpoints are provided --
categorical agreement after re-binning to S/I/R.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..susceptibility.breakpoint import BreakpointTable


def mic_regression_report(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
    *,
    breakpoints: BreakpointTable | None = None,
    species: str | Sequence[str] | None = None,
    drug: str | Sequence[str] | None = None,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Compute MIC regression metrics on log2-MIC predictions.

    Parameters
    ----------
    y_true : array-like
        True ``log2(MIC)`` values.
    y_pred : array-like
        Predicted ``log2(MIC)`` values.
    breakpoints : BreakpointTable or None, default=None
        When provided, the report also includes categorical agreement
        after re-binning both ``y_true`` and ``y_pred`` to S/I/R. Requires
        ``species`` and ``drug``.
    species : str or array-like, optional
        Species per sample (or a single species applied to all). Required
        when ``breakpoints`` is provided.
    drug : str or array-like, optional
        Drug per sample (or a single drug applied to all). Required when
        ``breakpoints`` is provided.
    sample_weight : array-like, optional
        Per-sample weights for the regression metrics. Ignored for
        categorical agreement.

    Returns
    -------
    dict
        Keys: ``n``, ``rmse_log2``, ``mae_log2``, ``bias_log2``,
        ``essential_agreement`` (fraction within ±1 dilution), and when
        breakpoints are provided also ``categorical_agreement``,
        ``very_major_error_rate`` (R predicted as S), ``major_error_rate``
        (S predicted as R), and per-category sample counts.

    Notes
    -----
    "Essential agreement" is the standard clinical benchmark for MIC
    prediction accuracy: a prediction is essential-agreement-correct if
    it is within one log2 dilution of the true value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_ok = y_true[mask]
    y_pred_ok = y_pred[mask]
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)[mask]
    else:
        weights = np.ones_like(y_true_ok)

    n = int(mask.sum())
    if n == 0:
        return {
            "n": 0,
            "rmse_log2": float("nan"),
            "mae_log2": float("nan"),
            "bias_log2": float("nan"),
            "essential_agreement": float("nan"),
        }

    diff = y_pred_ok - y_true_ok
    rmse = float(np.sqrt(np.average(diff**2, weights=weights)))
    mae = float(np.average(np.abs(diff), weights=weights))
    bias = float(np.average(diff, weights=weights))
    essential = float(np.average((np.abs(diff) <= 1.0).astype(float), weights=weights))

    report: dict[str, float | int] = {
        "n": n,
        "rmse_log2": rmse,
        "mae_log2": mae,
        "bias_log2": bias,
        "essential_agreement": essential,
    }

    if breakpoints is not None:
        if species is None or drug is None:
            raise ValueError(
                "species and drug are required when breakpoints is provided."
            )
        cat_metrics = _categorical_block(
            breakpoints=breakpoints,
            species=species,
            drug=drug,
            log2_true=y_true,
            log2_pred=y_pred,
            mask=mask,
        )
        report.update(cat_metrics)
    return report


def _categorical_block(
    *,
    breakpoints: BreakpointTable,
    species,
    drug,
    log2_true: np.ndarray,
    log2_pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    mic_true = np.power(2.0, log2_true)
    mic_pred = np.power(2.0, log2_pred)
    cat_true = breakpoints.apply_batch(species, drug, mic_true)["category"].to_numpy()
    cat_pred = breakpoints.apply_batch(species, drug, mic_pred)["category"].to_numpy()
    both = (
        mask
        & np.array([t is not None for t in cat_true])
        & np.array([p is not None for p in cat_pred])
    )
    n_cat = int(both.sum())
    if n_cat == 0:
        return {
            "categorical_agreement": float("nan"),
            "very_major_error_rate": float("nan"),
            "major_error_rate": float("nan"),
            "n_categorical": 0,
        }
    t = cat_true[both]
    p = cat_pred[both]
    agreement = float(np.mean(t == p))
    resistant_mask = t == "R"
    susceptible_mask = t == "S"
    vme = (
        float(np.mean(p[resistant_mask] == "S"))
        if resistant_mask.any()
        else float("nan")
    )
    me = (
        float(np.mean(p[susceptible_mask] == "R"))
        if susceptible_mask.any()
        else float("nan")
    )
    return {
        "categorical_agreement": agreement,
        "very_major_error_rate": vme,
        "major_error_rate": me,
        "n_categorical": n_cat,
        "n_resistant_true": int(resistant_mask.sum()),
        "n_susceptible_true": int(susceptible_mask.sum()),
    }
