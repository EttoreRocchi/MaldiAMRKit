"""Internal statistical helpers for differential analysis.

All functions in this module are implementation details of
:class:`~maldiamrkit.differential.DifferentialAnalysis` and are not part of
the public API.
"""

from __future__ import annotations

import numpy as np


def _mann_whitney_test(group_r: np.ndarray, group_s: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test between two groups.

    Parameters
    ----------
    group_r, group_s : ndarray
        1-D intensity samples for the resistant (R) and susceptible (S)
        groups.

    Returns
    -------
    tuple[float, float]
        ``(U_statistic, p_value)``.  Returns ``(nan, 1.0)`` if either
        group is empty or both groups are constant (no rank information).
    """
    from scipy.stats import mannwhitneyu

    if group_r.size == 0 or group_s.size == 0:
        return float("nan"), 1.0
    try:
        stat, p_value = mannwhitneyu(
            group_r, group_s, alternative="two-sided", method="auto"
        )
    except ValueError:
        return float("nan"), 1.0
    return float(stat), float(p_value)


def _t_test(group_r: np.ndarray, group_s: np.ndarray) -> tuple[float, float]:
    """Welch's two-sample t-test (unequal variances).

    Parameters
    ----------
    group_r, group_s : ndarray
        1-D intensity samples for the R and S groups.

    Returns
    -------
    tuple[float, float]
        ``(t_statistic, p_value)``.  Returns ``(nan, 1.0)`` if either
        group has fewer than two samples or both groups are constant.
    """
    from scipy.stats import ttest_ind

    if group_r.size < 2 or group_s.size < 2:
        return float("nan"), 1.0
    result = ttest_ind(group_r, group_s, equal_var=False)
    stat = float(result.statistic)
    p_value = float(result.pvalue)
    if np.isnan(stat) or np.isnan(p_value):
        return float("nan"), 1.0
    return stat, p_value


def _correct_pvalues(p_values: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """Apply multiple-testing correction to an array of p-values.

    NaN entries represent untestable features and are excluded from the
    correction denominator; they are returned as ``1.0`` in the output.

    Parameters
    ----------
    p_values : ndarray
        Raw p-values.  NaN entries are excluded from the correction (they
        do not contribute to ``m`` in Benjamini-Hochberg / Bonferroni) and
        are set to ``1.0`` in the output.
    method : {"fdr_bh", "fdr_by", "bonferroni"}, default="fdr_bh"
        Correction method.

        - ``"fdr_bh"``: Benjamini-Hochberg FDR via
          :func:`scipy.stats.false_discovery_control`.
        - ``"fdr_by"``: Benjamini-Yekutieli FDR via
          :func:`scipy.stats.false_discovery_control`.
        - ``"bonferroni"``: ``p * m`` clipped to ``[0, 1]`` where ``m`` is
          the number of non-NaN p-values.

    Returns
    -------
    ndarray
        Adjusted p-values, same shape as input.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    from scipy.stats import false_discovery_control

    p_values = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p_values)
    out = np.ones_like(p_values, dtype=float)

    if not valid.any():
        return out

    tested = p_values[valid]
    m = tested.size

    if method == "fdr_bh":
        adjusted = false_discovery_control(tested, method="bh")
    elif method == "fdr_by":
        adjusted = false_discovery_control(tested, method="by")
    elif method == "bonferroni":
        adjusted = np.clip(tested * m, 0.0, 1.0)
    else:
        raise ValueError(
            f"Unknown correction method '{method}'. "
            "Expected one of: 'fdr_bh', 'fdr_by', 'bonferroni'."
        )

    out[valid] = np.asarray(adjusted, dtype=float)
    return out


def _compute_fold_change(
    mean_r: np.ndarray,
    mean_s: np.ndarray,
    pseudocount: float | None = None,
) -> np.ndarray:
    """Log2 fold change between resistant and susceptible group means.

    Parameters
    ----------
    mean_r, mean_s : ndarray
        Per-feature group means.
    pseudocount : float or None, default=None
        Small positive constant added to both means to avoid division by
        zero and log of zero.  When ``None`` (default), the pseudocount
        is scaled to the input's dynamic range as
        ``1e-3 * median(|mean_r| + |mean_s|)`` (with a floor of
        ``1e-12``), which keeps the guard proportional to the data
        whether the caller passes TIC-normalised intensities (sum = 1)
        or raw counts.

    Returns
    -------
    ndarray
        ``log2((mean_r + pseudocount) / (mean_s + pseudocount))``.
    """
    mean_r = np.asarray(mean_r, dtype=float)
    mean_s = np.asarray(mean_s, dtype=float)
    if pseudocount is None:
        scale = float(np.median(np.abs(mean_r) + np.abs(mean_s)))
        pseudocount = max(1e-3 * scale, 1e-12)
    return np.log2((mean_r + pseudocount) / (mean_s + pseudocount))


def _compute_effect_size(group_r: np.ndarray, group_s: np.ndarray) -> float:
    """Cohen's d effect size between two groups (pooled SD).

    Parameters
    ----------
    group_r, group_s : ndarray
        1-D intensity samples for the R and S groups.

    Returns
    -------
    float
        ``(mean_r - mean_s) / pooled_std``.  Returns ``0.0`` when the
        pooled standard deviation is zero or either group has fewer
        than two samples.
    """
    n_r = group_r.size
    n_s = group_s.size
    if n_r < 2 or n_s < 2:
        return 0.0

    mean_r = float(np.mean(group_r))
    mean_s = float(np.mean(group_s))
    var_r = float(np.var(group_r, ddof=1))
    var_s = float(np.var(group_s, ddof=1))

    pooled_var = ((n_r - 1) * var_r + (n_s - 1) * var_s) / (n_r + n_s - 2)
    if pooled_var <= 0.0:
        return 0.0
    return (mean_r - mean_s) / float(np.sqrt(pooled_var))
