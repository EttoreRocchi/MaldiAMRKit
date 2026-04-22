"""DifferentialAnalysis: per-bin differential peak testing between R and S groups."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .stats import (
    _compute_effect_size,
    _compute_fold_change,
    _correct_pvalues,
    _mann_whitney_test,
    _t_test,
)

if TYPE_CHECKING:
    from maldiamrkit.dataset import MaldiSet
    from maldiamrkit.detection import MaldiPeakDetector


MzRange = tuple[float, float]


class StatisticalTest(str, Enum):
    """Supported statistical tests for :meth:`DifferentialAnalysis.run`.

    Attributes
    ----------
    mann_whitney : str
        Two-sided Mann-Whitney U test (non-parametric).
    t_test : str
        Welch's two-sample t-test (unequal variances).
    """

    mann_whitney = "mann_whitney"
    t_test = "t_test"


class CorrectionMethod(str, Enum):
    """Supported multiple-testing corrections for :meth:`DifferentialAnalysis.run`.

    Attributes
    ----------
    fdr_bh : str
        Benjamini-Hochberg false discovery rate.
    fdr_by : str
        Benjamini-Yekutieli false discovery rate.
    bonferroni : str
        Bonferroni family-wise correction.
    """

    fdr_bh = "fdr_bh"
    fdr_by = "fdr_by"
    bonferroni = "bonferroni"


def _normalize_mz_ranges(
    mz_ranges: MzRange | list[MzRange],
) -> list[MzRange]:
    """Normalize input into a list of ``(low, high)`` tuples with ``low <= high``.

    Accepts a single ``(low, high)`` tuple or a list of such tuples.
    """
    if (
        isinstance(mz_ranges, tuple)
        and len(mz_ranges) == 2
        and not isinstance(mz_ranges[0], (list, tuple))
    ):
        ranges: list[MzRange] = [mz_ranges]
    else:
        ranges = [tuple(r) for r in mz_ranges]  # type: ignore[arg-type]

    out: list[MzRange] = []
    for r in ranges:
        if len(r) != 2:
            raise ValueError(f"Each m/z range must be a (low, high) tuple; got {r!r}.")
        low, high = float(r[0]), float(r[1])
        if low > high:
            low, high = high, low
        out.append((low, high))
    return out


def _mz_range_mask(columns: pd.Index, mz_ranges: MzRange | list[MzRange]) -> np.ndarray:
    """Boolean mask of columns whose numeric m/z label falls in any range.

    Non-numeric column labels are excluded (mask entry is ``False``).
    """
    ranges = _normalize_mz_ranges(mz_ranges)
    mz_values = pd.to_numeric(pd.Index(columns), errors="coerce").to_numpy()
    mask = np.zeros(len(mz_values), dtype=bool)
    valid = ~np.isnan(mz_values)
    for low, high in ranges:
        mask |= valid & (mz_values >= low) & (mz_values <= high)
    return mask


class DifferentialAnalysis:
    """Identify discriminative m/z peaks between resistant and susceptible groups.

    Given a binned feature matrix and binary labels (0 = susceptible,
    1 = resistant), the analysis iterates over each m/z bin and computes
    a statistical test, a log2 fold change, and Cohen's d effect size
    comparing the two groups.  Multiple-testing correction is applied
    across bins.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of shape ``(n_samples, n_features)``.  Column
        names are m/z bin identifiers (numeric or string).
    y : pd.Series or ndarray
        Binary labels aligned with ``X`` rows: ``0`` = susceptible,
        ``1`` = resistant.  Any sample with a missing / NaN label is
        dropped before analysis.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix (possibly subset to rows with non-missing labels).
    y : pd.Series
        Labels aligned with ``X``.
    results : pd.DataFrame or None
        Populated by :meth:`run` with columns ``mz_bin``, ``mean_r``,
        ``mean_s``, ``fold_change``, ``p_value``, ``adjusted_p_value``,
        ``effect_size``.

    Examples
    --------
    >>> analysis = DifferentialAnalysis(X, y).run()
    >>> analysis.top_peaks(n=10)
    >>> analysis.significant_peaks(fc_threshold=1.0, p_threshold=0.05)
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if isinstance(y, pd.Series):
            y_series = y.copy()
            if not y_series.index.equals(X.index):
                y_series = y_series.reindex(X.index)
        else:
            y_arr = np.asarray(y)
            if y_arr.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Length of y ({y_arr.shape[0]}) does not match "
                    f"number of rows in X ({X.shape[0]})."
                )
            y_series = pd.Series(y_arr, index=X.index)

        mask = y_series.notna()
        if not mask.all():
            y_series = y_series[mask]
            X = X.loc[y_series.index]

        y_numeric = pd.to_numeric(y_series, errors="coerce")
        if y_numeric.isna().any():
            raise ValueError("y contains values that cannot be cast to numeric labels.")

        unique = np.unique(np.asarray(y_numeric.values))
        if not bool(np.all(np.isin(unique, [0, 1]))):
            raise ValueError(
                f"y must contain only binary labels 0 and 1; got {unique.tolist()}."
            )

        self.X: pd.DataFrame = X
        self.y: pd.Series = y_numeric.astype(int)
        self._results: pd.DataFrame | None = None

    @classmethod
    def from_maldi_set(
        cls, maldi_set: MaldiSet, antibiotic: str | None = None
    ) -> DifferentialAnalysis:
        """Build a :class:`DifferentialAnalysis` from a :class:`MaldiSet`.

        Extracts the feature matrix via ``maldi_set.X`` and labels via
        ``maldi_set.get_y_single(antibiotic)``.

        Parameters
        ----------
        maldi_set : MaldiSet
            Dataset providing ``X`` and ``get_y_single``.
        antibiotic : str or None, default=None
            Antibiotic label to analyse.  If ``None``, the first
            configured antibiotic is used.

        Returns
        -------
        DifferentialAnalysis
            Unrun analysis (call :meth:`run` next).
        """
        X = maldi_set.X
        y = maldi_set.get_y_single(antibiotic)
        return cls(X, y)

    def run(
        self,
        test: str | StatisticalTest = StatisticalTest.mann_whitney,
        correction: str | CorrectionMethod = CorrectionMethod.fdr_bh,
        mz_ranges: MzRange | list[MzRange] | None = None,
        peak_detector: MaldiPeakDetector | None = None,
    ) -> DifferentialAnalysis:
        """Run per-bin statistical analysis.

        For each kept column of ``X``, splits samples by label, computes
        the requested test statistic and p-value, the log2 fold change of
        group means, and Cohen's d.  Multiple-testing correction is then
        applied across the kept bins and the result is stored in
        :attr:`results`.

        Pre-test filters reduce the number of hypotheses - this is often
        decisive on small datasets where a full 1k-10k bin scan would
        exceed FDR power.

        Parameters
        ----------
        test : {"mann_whitney", "t_test"} or StatisticalTest
            Statistical test to apply per bin.
        correction : {"fdr_bh", "fdr_by", "bonferroni"} or CorrectionMethod
            Multiple-testing correction.
        mz_ranges : tuple, list of tuples, or None, default=None
            Restrict analysis to bins whose m/z value falls within the
            given range(s).  Pass a single ``(low, high)`` tuple or a
            list of such tuples for a union of intervals.  Endpoints are
            inclusive.  Column labels are coerced to ``float`` for range
            comparison; non-numeric columns are excluded.  ``None``
            disables the filter.
        peak_detector : MaldiPeakDetector or None, default=None
            Restrict analysis to bins that are peaks in at least one
            sample according to the provided detector.  The detector's
            ``fit_transform`` is run on the (range-filtered) feature
            matrix and any bin that is non-zero in at least one row is
            kept.  ``None`` disables the filter.

        Returns
        -------
        DifferentialAnalysis
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If ``y`` does not contain both classes, or if the combined
            filters leave no bins to test.
        """
        test = StatisticalTest(test)
        correction = CorrectionMethod(correction)

        y_arr = np.asarray(self.y.values)
        r_mask = y_arr == 1
        s_mask = y_arr == 0
        if not bool(r_mask.any()) or not bool(s_mask.any()):
            raise ValueError(
                "DifferentialAnalysis requires at least one sample in each class "
                "(0 = susceptible, 1 = resistant)."
            )

        keep_mask = self._build_feature_mask(mz_ranges, peak_detector)
        if not bool(keep_mask.any()):
            raise ValueError(
                "No bins remain after applying 'mz_ranges' / 'peak_detector' filters."
            )

        kept_columns = self.X.columns[keep_mask]
        X_values = self.X.values[:, keep_mask].astype(float, copy=False)
        X_r = X_values[r_mask]
        X_s = X_values[s_mask]

        mean_r = X_r.mean(axis=0)
        mean_s = X_s.mean(axis=0)
        fold_change = _compute_fold_change(mean_r, mean_s)

        if test == StatisticalTest.mann_whitney:
            test_fn = _mann_whitney_test
        else:
            test_fn = _t_test

        n_features = X_values.shape[1]
        p_values = np.empty(n_features, dtype=float)
        effect_sizes = np.empty(n_features, dtype=float)

        for j in range(n_features):
            col_r = X_r[:, j]
            col_s = X_s[:, j]
            _, p_values[j] = test_fn(col_r, col_s)
            effect_sizes[j] = _compute_effect_size(col_r, col_s)

        adjusted = _correct_pvalues(p_values, method=correction.value)

        self._results = pd.DataFrame(
            {
                "mz_bin": kept_columns.to_numpy(),
                "mean_r": mean_r,
                "mean_s": mean_s,
                "fold_change": fold_change,
                "p_value": p_values,
                "adjusted_p_value": adjusted,
                "effect_size": effect_sizes,
            }
        )
        return self

    def _build_feature_mask(
        self,
        mz_ranges: MzRange | list[MzRange] | None,
        peak_detector: MaldiPeakDetector | None,
    ) -> np.ndarray:
        """Combine m/z range and peak-detector masks into one boolean array.

        Returns an array of shape ``(n_features,)`` with ``True`` for
        columns of :attr:`X` that should be included in the test.
        """
        n_features = self.X.shape[1]
        mask = np.ones(n_features, dtype=bool)

        if mz_ranges is not None:
            mask &= _mz_range_mask(self.X.columns, mz_ranges)

        if peak_detector is not None:
            if not bool(mask.any()):
                return mask
            sub = self.X.loc[:, mask]
            detected = peak_detector.fit_transform(sub)
            peak_mask_sub = (detected.to_numpy() != 0).any(axis=0)
            full_peak_mask = np.zeros(n_features, dtype=bool)
            full_peak_mask[np.flatnonzero(mask)] = peak_mask_sub
            mask &= full_peak_mask

        return mask

    @property
    def results(self) -> pd.DataFrame:
        """Per-bin results table.

        Returns
        -------
        pd.DataFrame
            Columns: ``mz_bin``, ``mean_r``, ``mean_s``,
            ``fold_change``, ``p_value``, ``adjusted_p_value``,
            ``effect_size``.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet.
        """
        if self._results is None:
            raise RuntimeError(
                "DifferentialAnalysis has not been run yet. Call .run() first."
            )
        return self._results.copy()

    def top_peaks(self, n: int = 20) -> pd.DataFrame:
        """Return the top ``n`` peaks sorted by adjusted p-value ascending.

        Parameters
        ----------
        n : int, default=20
            Number of peaks to return.

        Returns
        -------
        pd.DataFrame
            Sub-table with the ``n`` lowest adjusted p-values.
        """
        df = self.results
        return df.sort_values(
            "adjusted_p_value", ascending=True, kind="mergesort"
        ).head(n)

    def significant_peaks(
        self, fc_threshold: float = 1.0, p_threshold: float = 0.05
    ) -> pd.DataFrame:
        """Return peaks passing both fold-change and adjusted p-value filters.

        Parameters
        ----------
        fc_threshold : float, default=1.0
            Absolute log2 fold-change threshold (inclusive).
        p_threshold : float, default=0.05
            Adjusted p-value threshold (inclusive).

        Returns
        -------
        pd.DataFrame
            Peaks where ``|fold_change| >= fc_threshold`` AND
            ``adjusted_p_value <= p_threshold``.
        """
        df = self.results
        mask = (df["fold_change"].abs() >= fc_threshold) & (
            df["adjusted_p_value"] <= p_threshold
        )
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def compare_drugs(
        analyses: dict[str, DifferentialAnalysis],
        fc_threshold: float = 1.0,
        p_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Build a boolean significance matrix across multiple drug analyses.

        Parameters
        ----------
        analyses : dict[str, DifferentialAnalysis]
            Mapping from drug name to a fitted :class:`DifferentialAnalysis`.
        fc_threshold : float, default=1.0
            Absolute log2 fold-change threshold for significance.
        p_threshold : float, default=0.05
            Adjusted p-value threshold for significance.

        Returns
        -------
        pd.DataFrame
            Boolean matrix indexed by the union of significant m/z bins
            across all drugs; columns are drug names; ``True`` indicates
            the peak is significant for that drug.

        Raises
        ------
        ValueError
            If *analyses* is empty.
        """
        if not analyses:
            raise ValueError("'analyses' must contain at least one entry.")

        per_drug: dict[str, pd.Series] = {}
        for drug, analysis in analyses.items():
            sig = analysis.significant_peaks(
                fc_threshold=fc_threshold, p_threshold=p_threshold
            )
            per_drug[drug] = pd.Series(
                True, index=pd.Index(sig["mz_bin"].to_numpy(), name="mz_bin")
            )

        if not per_drug:
            return pd.DataFrame()

        union_index = pd.Index([], name="mz_bin")
        for series in per_drug.values():
            union_index = union_index.union(series.index)

        data = {
            drug: series.reindex(union_index, fill_value=False).astype(bool).values
            for drug, series in per_drug.items()
        }
        return pd.DataFrame(data, index=union_index)
